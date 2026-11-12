"""Khung tool: đăng ký, kiểm tra quyền + chế độ, kiểm tra vào/ra, timeout, log tự động (T-10).

Gọi tool:  call("T-03", agent_id="AG-02", mode="demo", session_id="s1", measurements={...})
- Agent không được cấp tool (configs/agents.yaml) ⇒ ToolNotAllowed (lỗi lập trình, ném ra).
- Agent không chạy ở chế độ `mode` ⇒ ToolNotAllowed.
- Input sai kiểu / exception / quá thời gian ⇒ Out(ok=False, error=…) — không ném ra agent.
"""
from __future__ import annotations

import concurrent.futures as cf
import contextvars
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from diabetes_mas.agents.registry import ToolNotAllowed, agents, assert_allowed
from diabetes_mas.config import PROJECT_ROOT, settings
from diabetes_mas.tools.errors import classify_exception
from diabetes_mas.tools.pii import redact

LOG_DIR = PROJECT_ROOT / "logs" / "agent_runs"
_EXECUTOR = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")


@dataclass(frozen=True)
class ToolSpec:
    id: str
    name: str
    fn: Callable[[BaseModel], BaseModel]
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    timeout_s: float
    summarize: Callable[[BaseModel], str]


TOOLS: dict[str, ToolSpec] = {}
# Vết tool call của lượt hiện tại (API trả về `tool_calls`). Đặt list vào biến này để bắt đầu ghi.
TRACE: contextvars.ContextVar[list | None] = contextvars.ContextVar("tool_trace", default=None)


def _default_summary(out: BaseModel) -> str:
    data = out.model_dump(exclude={"ok", "error"})
    return redact(json.dumps(data, ensure_ascii=False, default=str), 200)


def tool(tool_id: str, name: str, input_model: type[BaseModel], output_model: type[BaseModel],
         timeout_s: float = 10.0, summarize: Callable[[BaseModel], str] | None = None):
    def deco(fn: Callable[[BaseModel], BaseModel]):
        if tool_id in TOOLS:
            raise ValueError(f"{tool_id} đã được đăng ký")
        TOOLS[tool_id] = ToolSpec(tool_id, name, fn, input_model, output_model, timeout_s,
                                  summarize or _default_summary)
        return fn
    return deco


def _safe_args(data: dict[str, Any]) -> dict[str, Any]:
    """Tham số đưa vào log: chuỗi đi qua bộ lọc PII và bị cắt ngắn (NFR-04)."""
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, str):
            out[k] = redact(v, 120)
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = redact(json.dumps(v, ensure_ascii=False, default=str), 160)
    return out


def log_event(session_id: str | None, agent: str, tool_id: str, args: dict, result_summary: str,
              latency_ms: int, ok: bool) -> Path:
    """T-10: ghi một dòng JSONL. Được gọi tự động sau mỗi tool call."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{datetime.now():%Y-%m-%d}_{session_id or 'no-session'}.jsonl"
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "session_id": session_id,
           "agent": agent, "tool": tool_id, "args": args, "result_summary": result_summary,
           "latency_ms": latency_ms, "ok": ok}
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def call(tool_id: str, agent_id: str, mode: str = "demo", session_id: str | None = None,
         timeout_s: float | None = None, **kwargs: Any) -> BaseModel:
    spec = TOOLS[tool_id]
    assert_allowed(agent_id, tool_id)
    if mode not in agents()[agent_id]["modes"]:
        raise ToolNotAllowed(f"{agent_id} không chạy ở chế độ '{mode}'")

    if session_id is not None and "session_id" in spec.input_model.model_fields:
        kwargs.setdefault("session_id", session_id)       # session_id vừa để log vừa là input của tool
    if timeout_s is not None and "timeout_s" in spec.input_model.model_fields:
        # Tool có tham số timeout_s riêng (T-13: thời hạn request HTTP) ⇒ chuyển vào input, thời hạn của cả
        # tool giữ theo đặc tả. Trước đây giá trị bị nuốt làm thời hạn tool: T-13 với 8 s luôn "timeout"
        # vì gồm cả nạp + nhúng bge-m3 (phát hiện GĐ 6 với SearXNG thật).
        kwargs["timeout_s"], timeout_s = timeout_s, None
    t0 = time.perf_counter()
    try:
        inp = spec.input_model(**kwargs)
    except ValidationError as exc:
        out = spec.output_model(ok=False, error=f"invalid_input: {exc.errors()[0]['msg']}")
        inp = None
    else:
        fut = _EXECUTOR.submit(spec.fn, inp)
        try:
            out = fut.result(timeout=timeout_s or spec.timeout_s)
            if not isinstance(out, spec.output_model):
                out = spec.output_model.model_validate(out)
        except cf.TimeoutError:
            out = spec.output_model(ok=False, error="timeout")
        except Exception as exc:  # noqa: BLE001 — tool không được ném lỗi ra agent
            code = classify_exception(exc) or type(exc).__name__
            out = spec.output_model(ok=False, error=f"{code}: {exc}")
    latency = int((time.perf_counter() - t0) * 1000)
    trace = TRACE.get()
    if trace is not None:
        trace.append({"agent": agent_id, "tool": tool_id, "name": spec.name, "ok": out.ok, "latency_ms": latency,
                      "summary": spec.summarize(out) if out.ok else (out.error or "")[:160]})
    if mode == "train" and not settings().get("logging", {}).get("train_calls", False):
        return out                                    # huấn luyện: hàng triệu call — không ghi log (settings.logging)
    log_event(session_id, agent_id, tool_id, _safe_args(inp.model_dump() if inp else kwargs),
              spec.summarize(out) if out.ok else (out.error or ""), latency, out.ok)
    return out

import dataclasses
import json
import time

import pytest

from diabetes_mas.agents.registry import ToolNotAllowed
from diabetes_mas.tools import base
from diabetes_mas.tools.base import TOOLS, call


@pytest.fixture(autouse=True)
def tmp_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "LOG_DIR", tmp_path)
    return tmp_path


def _patch_fn(monkeypatch, tool_id, fn, timeout_s=None):
    spec = TOOLS[tool_id]
    monkeypatch.setitem(TOOLS, tool_id, dataclasses.replace(spec, fn=fn, timeout_s=timeout_s or spec.timeout_s))


def test_agent_without_permission_is_rejected():
    with pytest.raises(ToolNotAllowed):
        call("T-11", agent_id="AG-07", mode="demo", prev_entropy=1, entropy=1, asked=True, hit=False, terminal=False)


def test_agent_wrong_mode_is_rejected():
    with pytest.raises(ToolNotAllowed):          # AG-12 Research không được chạy khi huấn luyện
        call("T-12", agent_id="AG-12", mode="train", query="metformin")


def test_invalid_input_becomes_ok_false():
    out = call("T-11", agent_id="AG-11", mode="train", prev_entropy="abc", entropy=1, asked=True, hit=False,
               terminal=False)
    assert out.ok is False and out.error.startswith("invalid_input")


def test_exception_becomes_ok_false(monkeypatch):
    def boom(inp):
        raise RuntimeError("hỏng")
    _patch_fn(monkeypatch, "T-11", boom)
    out = call("T-11", agent_id="AG-11", mode="train", prev_entropy=1, entropy=1, asked=True, hit=False, terminal=False)
    assert out.ok is False and "RuntimeError" in out.error


def test_timeout_becomes_ok_false(monkeypatch):
    def slow(inp):
        time.sleep(1.0)
    _patch_fn(monkeypatch, "T-11", slow, timeout_s=0.1)
    out = call("T-11", agent_id="AG-11", mode="train", prev_entropy=1, entropy=1, asked=True, hit=False, terminal=False)
    assert out.ok is False and out.error == "timeout"


def test_every_call_is_logged_with_pii_redacted(tmp_logs):
    call("T-02", agent_id="AG-06", mode="demo", session_id="s-log",
         text="Tôi tên là Nguyễn Văn An, SĐT 0912345678, bị tiểu đường tuýp 2")
    rows = [json.loads(l) for f in tmp_logs.glob("*s-log.jsonl") for l in f.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert r["tool"] == "T-02" and r["agent"] == "AG-06" and r["ok"] is True
    blob = json.dumps(r, ensure_ascii=False)
    assert "0912345678" not in blob and "Nguyễn Văn An" not in blob
    assert "DTD_TIP_2" in r["result_summary"]


@pytest.mark.parametrize("exc_factory, code", [
    (lambda: __import__("neo4j").exceptions.ServiceUnavailable("Neo4j tắt"), "kb_unavailable"),
    (lambda: __import__("requests").ConnectionError("mất mạng"), "net_unavailable"),
    (lambda: __import__("diabetes_mas.tools.errors", fromlist=["x"]).LLMUnavailable("Ollama tắt"), "llm_unavailable"),
])
def test_infra_failure_maps_to_stable_code(monkeypatch, exc_factory, code):
    """Luồng D (04_SEQUENCE.md): lỗi hạ tầng → mã ổn định → agent trả UNAVAILABLE_TEXT, không đoán."""
    from diabetes_mas.tools.errors import UNAVAILABLE_TEXT, is_unavailable

    def down(inp):
        raise exc_factory()
    _patch_fn(monkeypatch, "T-04", down)
    out = call("T-04", agent_id="AG-07", mode="demo", question="Metformin có tác dụng phụ gì?")
    assert out.ok is False and out.error.startswith(code + ":")
    assert is_unavailable(out.error) and "không truy cập được dữ liệu" in UNAVAILABLE_TEXT


def test_logic_error_is_not_reported_as_unavailable(monkeypatch):
    from diabetes_mas.tools.errors import is_unavailable

    def bug(inp):
        raise KeyError("x")
    _patch_fn(monkeypatch, "T-04", bug)
    out = call("T-04", agent_id="AG-07", mode="demo", question="Metformin?")
    assert out.ok is False and not is_unavailable(out.error)

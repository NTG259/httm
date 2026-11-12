"""Client Ollama (K3) + nạp prompt có phiên bản (05_PROMPTS.md §1).

- `chat()` ném `LLMUnavailable` khi không kết nối được / quá thời gian (Luồng D, mã `llm_unavailable`).
- `chat_json()` dùng `format: "json"`, validate bằng Pydantic, lỗi parse ⇒ thử lại 1 lần rồi ném `LLMBadOutput`.
- Prompt nằm ở `agents/prompts/P-xx.md`; dòng đầu `<!-- P-05@v2 -->` là phiên bản. Placeholder `{ten}` được
  thay bằng `render()` (không dùng str.format vì prompt chứa ngoặc nhọn JSON).
- `LLM` là biến module: test thay bằng client giả để chạy không cần Ollama.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from diabetes_mas.config import env, settings
from diabetes_mas.tools.errors import LLMUnavailable

PROMPT_DIR = Path(__file__).with_name("prompts")
T = TypeVar("T", bound=BaseModel)
_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLMBadOutput(ValueError):
    """LLM trả lời nhưng không đúng định dạng sau khi đã thử lại."""


@lru_cache(maxsize=None)
def prompt(pid: str) -> tuple[str, str]:
    """Trả (phiên bản, nội dung) của prompt `pid` (vd. "P-05")."""
    text = (PROMPT_DIR / f"{pid}.md").read_text(encoding="utf-8")
    m = re.match(r"<!--\s*(P-\d+@v\d+)\s*-->\n", text)
    if not m:
        raise ValueError(f"{pid}.md thiếu dòng phiên bản <!-- P-xx@vN -->")
    return m.group(1), text[m.end():].strip()


def render(pid: str, **values: Any) -> str:
    _, tpl = prompt(pid)
    for k, v in values.items():
        tpl = tpl.replace("{" + k + "}", str(v))
    left = re.findall(r"\{([a-z_]+)\}", tpl)
    if left:
        raise KeyError(f"{pid}: thiếu giá trị cho {left}")
    return tpl


class OllamaClient:
    def __init__(self, url: str | None = None, model: str | None = None, timeout_s: float | None = None):
        cfg = settings().get("llm", {})
        self.url = (url or env("OLLAMA_URL") or cfg.get("url", "http://localhost:11434")).rstrip("/")
        self.model = model or env("LLM_MODEL") or cfg.get("model", "qwen2.5:3b")
        self.timeout_s = timeout_s or cfg.get("timeout_s", 60)
        self.num_ctx = cfg.get("num_ctx", 4096)
        self.num_predict = cfg.get("num_predict", 512)
        self.num_predict_json = cfg.get("num_predict_json", 256)
        self.keep_alive = cfg.get("keep_alive", "30m")

    def chat(self, user: str, system: str | None = None, temperature: float = 0.0, json_mode: bool = False) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        body = {"model": self.model, "messages": msgs, "stream": False, "keep_alive": self.keep_alive,
                "options": {"temperature": temperature, "num_ctx": self.num_ctx,
                            "num_predict": self.num_predict_json if json_mode else self.num_predict}}
        if json_mode:
            body["format"] = "json"
        try:
            r = requests.post(f"{self.url}/api/chat", json=body, timeout=self.timeout_s)
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise LLMUnavailable(f"Ollama không phản hồi: {exc}") from exc
        if r.status_code != 200:
            raise LLMUnavailable(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
        return _THINK.sub("", r.json()["message"]["content"]).strip()

    def healthy(self) -> bool:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=3)
            return r.ok and any(m.get("name", "").startswith(self.model) for m in r.json().get("models", []))
        except requests.RequestException:
            return False


def chat_json(model_cls: type[T], user: str, temperature: float = 0.0, client: OllamaClient | None = None) -> T:
    cli = client or LLM
    last: Exception | None = None
    for _ in range(2):                                     # thử lại 1 lần (05 §1)
        raw = cli.chat(user, temperature=temperature, json_mode=True)
        try:
            return model_cls.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            last = exc
    raise LLMBadOutput(str(last)[:200])


LLM = OllamaClient()

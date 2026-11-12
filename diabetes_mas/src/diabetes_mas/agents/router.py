"""AG-00 Router — phân loại ý định (P-01).

Luật chạy TRƯỚC LLM cho hai trường hợp không được phép sai:
- Cấp cứu: câu chứa dấu hiệu nhóm `canh_bao` (cùng từ điển NLU) ⇒ "emergency" (không cần LLM).
- Câu kể số đo / triệu chứng của bản thân ("tôi", "mình", "em" + slot) ⇒ "diagnose".
Còn lại: LLM P-01. Ollama lỗi ⇒ heuristic: câu hỏi kiến thức ("là gì", "có tác dụng", "nên"...) ⇒ consult,
ngược lại diagnose (02 §8 ghi "mặc định diagnose"; bổ sung heuristic để câu hỏi kiến thức không bị
đưa vào hội thoại chẩn đoán).
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from diabetes_mas.agents import llm
from diabetes_mas.agents.slots import catalog
from diabetes_mas.tools.errors import LLMUnavailable
from diabetes_mas.utils.text import norm_key

Intent = Literal["diagnose", "consult", "emergency", "out_of_scope"]
SELF = re.compile(r"(?<!\w)(toi|minh|em|chau|con|bo toi|me toi|vo toi|chong toi)(?!\w)")
KNOWLEDGE = re.compile(r"(la gi|the nao|nhu the nao|tai sao|vi sao|co tac dung|tac dung phu|nen an|nen lam|"
                       r"bao nhieu la|phong ngua|dieu tri|khac nhau|nguyen nhan|bien chung)")


DOMAIN = re.compile(r"(tieu duong|dai thao duong|duong huyet|duong mau|insulin|metformin|hba1c|glucose|tuy|"
                    r"benh|trieu chung|thuoc|xet nghiem|bac si|an kieng|che do an|bien chung|suc khoe|huyet ap|"
                    r"can nang|beo phi|tiem|kham)")


def fallback(text: str, slots: dict) -> Intent:
    """Khi không có LLM: không slot + không từ khóa y khoa ⇒ out_of_scope; câu hỏi kiến thức ⇒ consult."""
    t = norm_key(text)
    if not slots and not DOMAIN.search(t):
        return "out_of_scope"
    if SELF.search(t) and not KNOWLEDGE.search(t):
        return "diagnose"                                  # "Tôi có bị tiểu đường không?"
    return "consult" if KNOWLEDGE.search(t) or ("?" in text and not slots) else "diagnose"


class RouterOut(BaseModel):
    intent: Intent


def classify_rules(text: str, slots: dict) -> Intent | None:
    alarm = {s.id for s in catalog().slots if s.group == "canh_bao"}
    if any(k in alarm and v.status == "yes" for k, v in slots.items()):
        return "emergency"
    t = norm_key(text)
    if slots and SELF.search(t) and not KNOWLEDGE.search(t):
        return "diagnose"
    return None


def classify(text: str, slots: dict | None = None) -> tuple[Intent, str]:
    """Trả (intent, phương pháp: rules | llm | fallback)."""
    hit = classify_rules(text, slots or {})
    if hit:
        return hit, "rules"
    try:
        return llm.chat_json(RouterOut, llm.render("P-01", user_text=text)).intent, "llm"
    except (LLMUnavailable, llm.LLMBadOutput):
        return fallback(text, slots or {}), "fallback"

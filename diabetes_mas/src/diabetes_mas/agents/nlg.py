"""AG-09 NLG — soạn câu hỏi cho một slot (P-03). Lấy câu hỏi mẫu + đơn vị từ T-01.

Kiểm đầu ra: một câu, kết thúc bằng "?", ≤ 200 ký tự, không chứa lời khuyên; sai ⇒ dùng nguyên `question_vi`
(cũng là dự phòng khi Ollama lỗi — 02 §8). `use_llm=False` cho câu hỏi tất định (test, demo offline).
"""
from __future__ import annotations

import re
from functools import lru_cache

from diabetes_mas.agents import llm
from diabetes_mas.agents.slots import catalog
from diabetes_mas.config import settings
from diabetes_mas.tools import call
from diabetes_mas.tools.errors import LLMUnavailable

ADVICE = re.compile(r"(nên|hãy|cần phải|khuyên)", re.I)


@lru_cache(maxsize=None)
def _slot_info(worker: str, slot_id: str):
    out = call("T-01", agent_id="AG-09", worker=worker)
    return next((s for s in out.slots if s.slot_id == slot_id), None)


def _name_vi(slot_id: str) -> str:
    from diabetes_mas.kb.seed import load_seed
    for rows in load_seed().nodes.values():
        for r in rows:
            if r["id"] == slot_id:
                return r.get("name_vi") or r.get("name") or slot_id
    return slot_id.replace("_", " ")


QUANTITY = re.compile(r"(bao nhiêu|mấy|bao lâu|khi nào|ở đâu)", re.I)


def ok_question(q: str, kind: str | None = None) -> bool:
    """Câu LLM dùng được khi: 1 câu hỏi, không khuyên. Slot có/không thì không được hỏi số lượng — GĐ 7: P-03 đổi
    "Anh/chị có đang mang thai không?" thành "...mang thai được bao nhiêu tuần không?" ⇒ bệnh nhân đáp "chưa biết"
    ⇒ NLU ghi unknown dù sự thật là có."""
    q = q.strip()
    if kind == "binary" and QUANTITY.search(q):
        return False
    return 5 < len(q) <= 200 and q.endswith("?") and q.count("?") == 1 and not ADVICE.search(q)


def question_for(slot_id: str, use_llm: bool | None = None) -> tuple[str, str]:
    """Trả (câu hỏi, phương pháp: llm | template)."""
    slot = catalog().by_id()[slot_id]
    info = _slot_info(slot.worker, slot_id) if slot.worker else None
    base = (info.question_vi if info and info.question_vi else slot.question_vi) or f"Anh/chị có {_name_vi(slot_id)} không?"
    if info and info.unit and slot.source == "LabTest" and info.unit not in base:
        base = base.rstrip("?") + f" (đơn vị {info.unit}; nếu không nhớ, anh/chị cứ nói không biết)?"
    use_llm = settings().get("nlg", {}).get("use_llm", True) if use_llm is None else use_llm
    if not use_llm:
        return base, "template"
    try:
        q = llm.LLM.chat(llm.render("P-03", slot_name_vi=_name_vi(slot_id), question_vi=base,
                                    unit=(info.unit if info and info.unit else "không có")), temperature=0.0)
        q = q.strip().strip('"').splitlines()[0] if q.strip() else ""
        return (q, "llm") if ok_question(q, slot.kind) else (base, "template")
    except LLMUnavailable:
        return base, "template"

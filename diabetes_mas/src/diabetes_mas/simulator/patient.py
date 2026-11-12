"""AG-10 Patient Simulator — chế độ ngôn ngữ (P-04, demo/kiểm thử end-to-end).

Chế độ slot (huấn luyện) nằm ở T-07 kênh simulator. Chế độ này biến câu trả lời slot thành lời nói đời thường
để kiểm thử NLU trên câu tự nhiên. LLM lỗi hoặc lỡ nói tên bệnh ⇒ dùng câu mẫu theo slot.
"""
from __future__ import annotations

import json
import re

from diabetes_mas.agents import llm
from diabetes_mas.agents.state import SlotValue
from diabetes_mas.tools.errors import LLMUnavailable

DISEASE_WORDS = re.compile(r"(tiểu đường|đái tháo đường|tiền đái|típ|tuýp|type)", re.I)


CATEGORY_TEXT = {"nam": "Tôi là nam.", "nu": "Tôi là nữ.", "khac": "Giới tính khác ạ.",
                 "chua_bao_gio": "Tôi chưa bao giờ hút thuốc.", "da_bo": "Tôi đã bỏ thuốc rồi.",
                 "dang_hut": "Tôi đang hút thuốc.", "hien_khong_hut": "Hiện tôi không hút thuốc.",
                 "tung_hut": "Tôi từng hút thuốc."}


def _num(v: float) -> str:
    """55.0 → "55", 7.1 → "7,1" (cách người Việt nói)."""
    return str(int(v)) if float(v).is_integer() else f"{v:g}".replace(".", ",")


HEIGHT_M = 1.65   # ca bệnh chỉ lưu BMI; bệnh nhân nói chiều cao + cân nặng tương ứng, NLU tự tính lại BMI


def value_text(slot_id: str | None, v: float) -> str:
    if slot_id == "tuoi":
        return f"{_num(v)} tuổi"
    if slot_id == "bmi":
        return f"cao 1m65, nặng {_num(round(v * HEIGHT_M ** 2, 1))} kg"
    return _num(v)


def template_answer(sv: SlotValue, slot_id: str | None = None) -> str:
    if sv.status == "yes" and sv.value is not None:
        if isinstance(sv.value, str):
            return CATEGORY_TEXT.get(sv.value, sv.value)
        if slot_id in ("tuoi", "bmi"):
            return f"Tôi {value_text(slot_id, sv.value)}."
        return f"Khoảng {_num(sv.value)} ạ."
    return {"yes": "Có ạ.", "no": "Không ạ.", "unknown": "Tôi không rõ, chưa đo bao giờ."}.get(sv.status,
                                                                                             "Tôi không để ý.")


def speak(case: dict, slot_id: str, question: str, use_llm: bool = True) -> str:
    raw = case.get("explicit", {}).get(slot_id) or case.get("implicit", {}).get(slot_id) or {"status": "unknown"}
    sv = SlotValue(**raw)
    if not use_llm:
        return template_answer(sv, slot_id)
    profile = {k: v for k, v in {**case.get("implicit", {}), **case.get("explicit", {})}.items()
               if v.get("status") != "not_asked"}
    try:
        fact = (CATEGORY_TEXT.get(sv.value, sv.value) if sv.status == "yes" and isinstance(sv.value, str) else
                f"có, {value_text(slot_id, sv.value)}" if sv.status == "yes" and sv.value is not None else
                {"yes": "có", "no": "không", "unknown": "không biết"}.get(sv.status, "không biết"))
        out = llm.LLM.chat(llm.render("P-04", case_json=json.dumps(profile, ensure_ascii=False), answer_fact=fact,
                                      question=question), temperature=0.7).strip()
    except LLMUnavailable:
        return template_answer(sv, slot_id)
    return template_answer(sv, slot_id) if not out or DISEASE_WORDS.search(out) or len(out) > 300 else out

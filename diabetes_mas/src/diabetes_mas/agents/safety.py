"""AG-08 Safety — kiểm MỌI câu ra (02 §7 quy tắc 3).

1. T-09 check_red_flags trước câu ra ⇒ cấp cứu thì thay toàn bộ bằng `action_text`.
2. Luật cứng (không cần LLM): kê liều cụ thể, khẳng định chẩn đoán chắc chắn, khuyên ngừng thuốc
   ⇒ thay bằng câu an toàn; trích dẫn [n] ngoài context đã được AG-07 kiểm.
3. P-07 (LLM) chỉ là lớp kiểm bổ sung, bật bằng settings.safety.use_llm; lỗi LLM không chặn câu đã qua luật.
4. Nối disclaimer cố định (05 §3).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel

from diabetes_mas.agents import llm
from diabetes_mas.config import settings
from diabetes_mas.tools import call
from diabetes_mas.tools.errors import LLMUnavailable

DISCLAIMER = "Thông tin chỉ mang tính tham khảo, không thay thế chẩn đoán và điều trị của bác sĩ."
# "126 mg/dL", "48 mmol/mol" là đơn vị nồng độ, không phải liều (GĐ 7: tiêu chuẩn chẩn đoán bị chặn nhầm)
DOSE = re.compile(r"\d+(?:[.,]\d+)?\s*(mg|mcg|µg|đơn vị|ui|iu|viên|ml)\b(?!\s*/\s*(dl|l|mol)\b)"
                  r".{0,30}(mỗi|/|lần|ngày|sáng|tối)", re.I)
CERTAIN = re.compile(r"(chắc chắn|khẳng định|100\s*%)\s*(bạn|anh|chị|bệnh nhân)?\s*(bị|mắc|có)", re.I)
STOP_MED = re.compile(r"(ngừng|ngưng|bỏ)\s+(uống\s+|dùng\s+)?(thuốc|insulin|metformin)", re.I)
SAFE_REPLACEMENT = ("Tôi không thể đưa ra chỉ định điều trị hay liều dùng cụ thể. "
                    "Anh/chị vui lòng trao đổi trực tiếp với bác sĩ điều trị.")


class SafetyOut(BaseModel):
    pass_: bool = True
    issues: list[str] = []

    model_config = {"populate_by_name": True}


@dataclass
class Checked:
    text: str
    emergency: bool = False
    issues: list[str] = field(default_factory=list)


def rule_issues(draft: str) -> list[str]:
    issues = []
    if DOSE.search(draft):
        issues.append("kê liều thuốc cụ thể")
    if CERTAIN.search(draft):
        issues.append("khẳng định chẩn đoán chắc chắn")
    if STOP_MED.search(draft):
        issues.append("khuyên ngừng thuốc")
    return issues


def check(draft: str, session_id: str | None = None, recent: str = "", add_disclaimer: bool = True,
          mode: str = "demo") -> Checked:
    if session_id:
        rf = call("T-09", agent_id="AG-08", mode=mode, session_id=session_id)
        if rf.ok and rf.red_flag:
            return Checked(text=rf.action_text + "\n\n" + DISCLAIMER, emergency=True, issues=rf.reasons)
    issues = rule_issues(draft)
    if not issues and settings().get("safety", {}).get("use_llm", False):
        try:
            raw = llm.chat_json(_P07, llm.render("P-07", recent=recent[-500:], draft=draft))
            if not raw.pass_:
                issues += [f"P-07: {i}" for i in raw.issues]
        except (LLMUnavailable, llm.LLMBadOutput):
            pass
    text = SAFE_REPLACEMENT if issues else draft
    if add_disclaimer and DISCLAIMER not in text:
        text = f"{text}\n\n{DISCLAIMER}"
    return Checked(text=text, issues=issues)


class _P07(BaseModel):
    pass_: bool = True
    issues: list[str] = []

    model_config = {"populate_by_name": True}

    @classmethod
    def model_validate(cls, obj, *a, **k):  # LLM trả khóa "pass"
        if isinstance(obj, dict) and "pass" in obj:
            obj = {**obj, "pass_": obj.pop("pass")}
        return super().model_validate(obj, *a, **k)

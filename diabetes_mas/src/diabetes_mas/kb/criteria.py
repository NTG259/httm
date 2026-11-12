"""Đối chiếu chỉ số xét nghiệm với DiagnosticCriterion (lõi của tool T-03).

Thuần Python, không phụ thuộc Neo4j: nhận danh sách tiêu chuẩn (từ seed hoặc từ Neo4j).
"""
from __future__ import annotations

from dataclasses import dataclass

DTD_FAMILY = {"DTD", "DTD_TIP_1", "DTD_TIP_2", "DTD_KHAC"}


@dataclass(frozen=True)
class Criterion:
    id: str
    disease_id: str
    test_id: str
    op: str
    value: float
    value_max: float | None
    unit: str
    condition: str | None
    source_id: str

    @classmethod
    def from_row(cls, r: dict) -> "Criterion":
        return cls(
            id=r["id"], disease_id=r["disease_id"], test_id=r["test_id"], op=r["op"],
            value=float(r["value"]), value_max=float(r["value_max"]) if r.get("value_max") not in (None, "") else None,
            unit=r.get("unit", ""), condition=r.get("condition") or None, source_id=r["source_id"],
        )

    def rule(self) -> str:
        if self.op == "between":
            return f"{self.test_id} {self.value}–{self.value_max} {self.unit}"
        return f"{self.test_id} {self.op} {self.value} {self.unit}"


def _match(op: str, x: float, v: float, vmax: float | None) -> bool:
    if op == ">=":
        return x >= v
    if op == ">":
        return x > v
    if op == "<=":
        return x <= v
    if op == "<":
        return x < v
    if op == "between":
        return v <= x <= (vmax if vmax is not None else v)
    raise ValueError(op)


def check(measurements: dict[str, float], context: dict[str, bool], criteria: list[Criterion]) -> list[Criterion]:
    """Trả về các tiêu chuẩn đạt, đã áp quy tắc ưu tiên:
    - Tiêu chuẩn có `condition` chỉ xét khi context[condition] là True.
    - Đạt tiêu chuẩn ĐTĐ ⇒ bỏ hit Tiền ĐTĐ và ĐTĐ thai kỳ (ĐTĐ rõ trong thai kỳ không gọi là ĐTĐ thai kỳ).
    - Đạt tiêu chuẩn ĐTĐ thai kỳ ⇒ bỏ hit Tiền ĐTĐ. Căn cứ VMC-0003: "Nếu không đủ tiêu chuẩn trên
      thì làm nghiệm pháp dung nạp glucose" — thai phụ chưa đạt ngưỡng ĐTĐ được đánh giá theo ngưỡng thai kỳ.
    """
    hits = []
    for c in criteria:
        if c.test_id not in measurements:
            continue
        if c.condition and not context.get(c.condition, False):
            continue
        if _match(c.op, measurements[c.test_id], c.value, c.value_max):
            hits.append(c)
    diseases = {h.disease_id for h in hits}
    if diseases & DTD_FAMILY:
        hits = [h for h in hits if h.disease_id not in {"TIEN_DTD", "DTD_THAI_KY"}]
    elif "DTD_THAI_KY" in diseases:
        hits = [h for h in hits if h.disease_id != "TIEN_DTD"]
    return sorted(hits, key=lambda h: h.id)


# Lớp bị LOẠI khi trúng tiêu chuẩn của bệnh tương ứng (cùng thứ tự ưu tiên với check()):
#  - ĐTĐ  ⇒ loại Không ĐTĐ, Tiền ĐTĐ, ĐTĐ thai kỳ (còn lại típ 1 / típ 2)
#  - ĐTĐ thai kỳ ⇒ loại Không ĐTĐ, Tiền ĐTĐ
#  - Tiền ĐTĐ ⇒ chỉ loại Không ĐTĐ (ĐTĐ vẫn có thể nếu xét nghiệm khác chưa biết)
RULE_EXCLUDES = {
    "DTD": {"KHONG_DTD", "TIEN_DTD", "DTD_THAI_KY"},
    "DTD_THAI_KY": {"KHONG_DTD", "TIEN_DTD"},
    "TIEN_DTD": {"KHONG_DTD"},
}


def apply_rules(probs: dict[str, float], hit_diseases: set[str]) -> dict[str, float]:
    """Hợp nhất luật KB vào phân phối của classifier: đặt 0 cho lớp bị loại rồi chuẩn hóa lại."""
    excluded: set[str] = set()
    for d in hit_diseases:
        key = "DTD" if d in DTD_FAMILY else d
        excluded |= RULE_EXCLUDES.get(key, set())
    out = {k: (0.0 if k in excluded else v) for k, v in probs.items()}
    total = sum(out.values())
    return {k: v / total for k, v in out.items()} if total > 0 else probs

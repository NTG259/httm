"""AG-13 Prediction — chạy mô hình DL (T-15) rồi đối chiếu luật KB (T-03), ghi risk_dl/dl_conflict (ADR-002).

Được AG-01 kích hoạt qua K1 (nhận DialogState từ graph state; AG-13 không có quyền T-06).
Mâu thuẫn (dl_conflict) chỉ xét khi đã có xét nghiệm:
- luật trúng ĐTĐ / ĐTĐ thai kỳ nhưng risk < threshold; hoặc
- risk ≥ threshold nhưng có xét nghiệm và luật không trúng ĐTĐ.
Luật KB luôn được ưu tiên; AG-13 không ghi đè p_t.
"""
from __future__ import annotations

from diabetes_mas.agents.state import DialogState
from diabetes_mas.tools import call
from diabetes_mas.tools.dialog_tools import _measurements
from diabetes_mas.tools.predict_tools import features_from_state

DIABETES_RULES = {"DTD", "DTD_TIP_1", "DTD_TIP_2", "DTD_THAI_KY"}


def run(st: DialogState, mode: str = "demo") -> dict:
    feats, assumptions = features_from_state(st)
    out = call("T-15", agent_id="AG-13", mode=mode, session_id=st.session_id, **feats.model_dump())
    res = {"ok": out.ok, "error": out.error, "variant": out.model_variant, "missing": out.missing_features,
           "assumptions": assumptions + list(out.assumptions)}
    if not out.ok:
        return res
    meas, ctx = _measurements(st)
    hits = call("T-03", agent_id="AG-13", mode=mode, session_id=st.session_id, measurements=meas, context=ctx)
    rule_dm = {h.disease_id for h in hits.hits} & DIABETES_RULES if hits.ok else set()
    positive = out.risk >= out.threshold
    conflict = bool(meas) and ((bool(rule_dm) and not positive) or (positive and not rule_dm))
    st.risk_dl, st.dl_conflict = out.risk, conflict
    res.update({"risk": out.risk, "threshold": out.threshold, "positive": positive,
                "rule_disease_hits": sorted(rule_dm), "conflict": conflict})
    return res


def dl_signature(st: DialogState) -> tuple:
    """Đặc trưng DL hiện tại — AG-01 kích hoạt lại AG-13 khi chữ ký này đổi (02 §6 bước 3)."""
    f, _ = features_from_state(st)
    return tuple(f.model_dump().values())

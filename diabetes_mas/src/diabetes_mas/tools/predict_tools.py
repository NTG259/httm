"""T-15 predict_diabetes_risk — ánh xạ slot → đặc trưng dataset SRC-KAGGLE (ADR-002) + chạy MLP.

Mô hình được huấn luyện ở Giai đoạn 5 (`09_MODEL_NOTES.md` §2b). Khi chưa có checkpoint, tool trả
ok=False, error="model_not_trained" nhưng vẫn báo biến thể và đặc trưng thiếu để kiểm thử luồng.
"""
from __future__ import annotations

from typing import Callable

from diabetes_mas.agents.state import DialogState, SlotValue
from diabetes_mas.tools.base import tool
from diabetes_mas.tools.schemas import PredictIn, PredictOut

GENDER = {"nam": "Male", "nu": "Female", "khac": "Other"}
SMOKING = {"chua_bao_gio": "never", "da_bo": "former", "dang_hut": "current",
           "hien_khong_hut": "not current", "tung_hut": "ever"}
MMOL_TO_MGDL = 18.0
LAB_FEATURES = ("HbA1c_level", "blood_glucose_level")


def _yes(st: DialogState, sid: str) -> bool:
    return st.slots.get(sid, SlotValue()).status == "yes"


def _num(st: DialogState, sid: str) -> float | None:
    sv = st.slots.get(sid)
    return float(sv.value) if sv and sv.status == "yes" and isinstance(sv.value, (int, float)) else None


def _known(st: DialogState, sid: str) -> bool:
    sv = st.slots.get(sid)
    return sv is not None and sv.status in ("yes", "no")


def features_from_state(st: DialogState) -> tuple[PredictIn, list[str]]:
    """DialogState → PredictIn + danh sách giả định đã dùng (để giải thích khi vấn đáp)."""
    assumptions: list[str] = []
    glucose = _num(st, "GLUCOSE_BAT_KY")
    if glucose is None and _num(st, "GLUCOSE_DOI") is not None:
        glucose = _num(st, "GLUCOSE_DOI")
        assumptions.append("blood_glucose_level lấy từ glucose lúc đói (dataset không ghi thời điểm đo)")
    elif glucose is not None:
        assumptions.append("blood_glucose_level lấy từ glucose bất kỳ (dataset không ghi thời điểm đo)")
    gender = st.slots.get("gioi_tinh")
    smoke = st.slots.get("hut_thuoc")
    feats = PredictIn(
        gender=GENDER.get(gender.value) if gender and gender.status == "yes" else None,
        age=_num(st, "tuoi"),
        hypertension=int(_yes(st, "tang_huyet_ap")) if _known(st, "tang_huyet_ap") else None,
        heart_disease=int(_yes(st, "benh_tim")) if _known(st, "benh_tim") else None,
        smoking_history=SMOKING.get(smoke.value, "No Info") if smoke and smoke.status == "yes" else "No Info",
        bmi=_num(st, "bmi"),
        HbA1c_level=_num(st, "HBA1C"),
        blood_glucose_level=round(glucose * MMOL_TO_MGDL, 1) if glucose is not None else None,
    )
    return feats, assumptions


def plan(feats: PredictIn) -> tuple[str | None, list[str], str | None]:
    """Chọn biến thể và liệt kê đặc trưng thiếu. Trả (variant, missing, error)."""
    missing = [k for k, v in feats.model_dump().items() if v is None]
    if feats.age is None or feats.bmi is None:
        return None, missing, "insufficient_features"
    variant = "full" if any(getattr(feats, f) is not None for f in LAB_FEATURES) else "no_labs"
    return variant, missing, None


# Hàm suy luận: (variant, feats) -> (risk, threshold). None ⇒ nạp checkpoint experiments/best/dl/ (dl.model.load).
PREDICTOR: Callable[[str, PredictIn], tuple[float, float]] | None = None


def _predictor():
    if PREDICTOR is not None:
        return PREDICTOR
    from diabetes_mas.dl.model import load
    return load()


@tool("T-15", "predict_diabetes_risk", PredictIn, PredictOut,
      summarize=lambda o: f"risk={o.risk} ({o.model_variant})")
def predict_diabetes_risk(inp: PredictIn) -> PredictOut:
    variant, missing, err = plan(inp)
    if err:
        return PredictOut(ok=False, error=err, missing_features=missing)
    pred = _predictor()
    if pred is None:
        return PredictOut(ok=False, error="model_not_trained", model_variant=variant, missing_features=missing)
    assumptions = []
    if hasattr(pred, "predict"):
        risk, threshold, filled = pred.predict(variant, inp.model_dump())
        assumptions = [f"{c} chưa biết → điền giá trị trung bình/mặc định của tập huấn luyện" for c in filled]
    else:
        risk, threshold = pred(variant, inp)
    return PredictOut(risk=risk, threshold=threshold, model_variant=variant, missing_features=missing,
                      assumptions=assumptions)

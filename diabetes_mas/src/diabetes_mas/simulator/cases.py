"""Sinh ca bệnh mô phỏng (user goal) cho Patient Simulator — SRC-CASE.

Chạy:  python -m diabetes_mas.simulator.cases

Nguồn tham số:
- P(triệu chứng | bệnh), P(yếu tố nguy cơ | bệnh): quan hệ HAS_SYMPTOM / HAS_RISK_FACTOR trong seed KB
  (prob_source = team_estimate — ước lượng của nhóm, cần chuyên gia hiệu chỉnh).
- Ngưỡng xét nghiệm: DiagnosticCriterion trong seed KB → mọi ca được kiểm tra nhất quán với nhãn.
- Quan hệ HbA1c ↔ glucose trung bình: phụ lục 3, QĐ 3280/QĐ-BYT (≈ 1,59·HbA1c − 2,59 mmol/L).
- Tỉ lệ nền và khoảng HbA1c tiềm ẩn theo lớp: GIẢ ĐỊNH của nhóm, khai báo tại PRIORS bên dưới.

DTD_KHAC không được sinh: KB chưa có dữ liệu phân biệt cho thể này.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter

from diabetes_mas.config import CASES_DIR, settings
from diabetes_mas.kb.criteria import Criterion, check
from diabetes_mas.kb.seed import load_seed
from diabetes_mas.utils.text import write_jsonl

ASKABLE_SYMPTOM_GROUPS = {"trieu_chung_kinh_dien", "bien_chung"}

# ---- GIẢ ĐỊNH CỦA NHÓM (không có nguồn) — chỉnh tại đây, ghi lý do vào docs/08_EVALUATION.md ----
PRIORS = {
    "class_weights": {"KHONG_DTD": 0.25, "TIEN_DTD": 0.20, "DTD_TIP_2": 0.30, "DTD_TIP_1": 0.12, "DTD_THAI_KY": 0.13},
    "hba1c_range": {                       # HbA1c tiềm ẩn (%) theo lớp
        "KHONG_DTD": (4.5, 5.6), "TIEN_DTD": (5.6, 6.4), "DTD_TIP_2": (6.3, 11.0),
        "DTD_TIP_1": (7.5, 13.0), "DTD_THAI_KY": (5.2, 6.4),
    },
    "symptom_background": 0.05,            # P(triệu chứng) khi bệnh không có cạnh HAS_SYMPTOM
    "risk_background": {                   # tỉ lệ nền yếu tố nguy cơ trong dân số
        "tuoi_tu_45": 0.35, "tuoi_duoi_30": 0.25, "bmi_tu_23": 0.35, "tang_huyet_ap": 0.2,
        "tien_su_gia_dinh_dtd": 0.12, "tien_su_dtd_thai_ky": 0.02, "roi_loan_lipid": 0.25,
        "it_van_dong": 0.35, "buong_trung_da_nang": 0.03, "benh_tu_mien": 0.03, "dang_mang_thai": 0.04,
    },
    # P(yếu tố | bệnh) ghi đè khi KB không có cạnh nhưng tỉ lệ nền dân số sai rõ với bệnh đó. KHÔNG thêm vào KB
    # (câu "tuổi ≥ 45 là yếu tố nguy cơ của típ 1" sai). GĐ 8 (TC-30): tỉ lệ nền 0,35 khiến ~26 % ca típ 1 ≥ 45 tuổi,
    # nên classifier chọn típ 1 cho người 55 tuổi BMI 27 có sụt cân. VMC-0001: típ 1 thường khởi phát < 30 tuổi.
    "risk_override": {("DTD_TIP_1", "tuoi_tu_45"): 0.10},
    "lab_noise_sd": 0.6,                   # độ lệch chuẩn nhiễu quanh glucose trung bình (mmol/L)
    # Slot hồ sơ (FR-19, ADR-002) — sinh nhất quán với cờ tuoi_*/bmi_tu_23 đã lấy mẫu
    "age_ranges": {"tuoi_tu_45": (45, 80), "tuoi_duoi_30": (12, 29), "mid": (30, 44), "pregnant": (18, 44)},
    "bmi_ranges": {"high": (23.0, 38.0), "low": (17.0, 22.9)},
    "p_male": 0.5,
    "smoking": {"chua_bao_gio": 0.55, "da_bo": 0.10, "dang_hut": 0.15, "hien_khong_hut": 0.07, "tung_hut": 0.05,
                "_unknown": 0.08},
    "p_heart_disease": {"DTD_TIP_2": 0.10, "_default": 0.04},
}


def _sample_profile(rng: random.Random, disease: str, rf: dict[str, bool]) -> dict[str, float | str | None]:
    """tuoi, bmi (số) nhất quán với cờ; giới tính nhất quán với thai kỳ/PCOS; hút thuốc; bệnh tim."""
    if rf["dang_mang_thai"]:
        rf["tuoi_tu_45"] = False                       # thai phụ: 18–44 tuổi
        lo, hi = PRIORS["age_ranges"]["tuoi_duoi_30"] if rf["tuoi_duoi_30"] else (30, 44)
        lo = max(lo, 18)
    elif rf["tuoi_tu_45"]:
        lo, hi = PRIORS["age_ranges"]["tuoi_tu_45"]
    elif rf["tuoi_duoi_30"]:
        lo, hi = PRIORS["age_ranges"]["tuoi_duoi_30"]
    else:
        lo, hi = PRIORS["age_ranges"]["mid"]
    female_only = rf["dang_mang_thai"] or rf["tien_su_dtd_thai_ky"] or rf["buong_trung_da_nang"]
    gender = "nu" if female_only or rng.random() >= PRIORS["p_male"] else "nam"
    smoke_opts = list(PRIORS["smoking"])
    smoke = rng.choices(smoke_opts, [PRIORS["smoking"][k] for k in smoke_opts])[0]
    p_heart = PRIORS["p_heart_disease"].get(disease, PRIORS["p_heart_disease"]["_default"])
    bmi_lo, bmi_hi = PRIORS["bmi_ranges"]["high" if rf["bmi_tu_23"] else "low"]
    return {
        "tuoi": float(rng.randint(lo, hi)),
        "bmi": round(rng.uniform(bmi_lo, bmi_hi), 1),
        "gioi_tinh": gender,
        "hut_thuoc": None if smoke == "_unknown" else smoke,
        "benh_tim": rng.random() < p_heart,
    }


def eag_mmol(hba1c: float) -> float:
    """Glucose trung bình ước tính từ HbA1c (phụ lục 3, QĐ 3280/QĐ-BYT)."""
    return 1.59 * hba1c - 2.59


def _sample_labs(rng: random.Random, disease: str, pregnant: bool) -> dict[str, float]:
    lo, hi = PRIORS["hba1c_range"][disease]
    a1c = rng.uniform(lo, hi)
    g = eag_mmol(a1c)
    sd = PRIORS["lab_noise_sd"]
    labs = {
        "HBA1C": round(a1c, 1),
        "GLUCOSE_DOI": round(max(3.5, g * 0.88 + rng.gauss(0, sd)), 1),
        "OGTT_2H": round(max(4.0, g * 1.35 + rng.gauss(0, sd * 1.5)), 1),
        "GLUCOSE_BAT_KY": round(max(4.0, g * 1.15 + rng.gauss(0, sd * 1.5)), 1),
    }
    if pregnant:
        labs["OGTT_1H"] = round(max(4.0, g * 1.45 + rng.gauss(0, sd * 1.5)), 1)
    return labs


def consistent(disease: str, labs: dict[str, float], ctx: dict[str, bool], criteria: list[Criterion]) -> bool:
    hit_d = {h.disease_id for h in check(labs, ctx, criteria)}
    dtd_hit = bool(hit_d & {"DTD"})
    if disease in {"DTD_TIP_1", "DTD_TIP_2"}:
        return dtd_hit
    if disease == "TIEN_DTD":
        return hit_d == {"TIEN_DTD"}          # không được trúng ĐTĐ hay ĐTĐ thai kỳ
    if disease == "DTD_THAI_KY":
        return hit_d == {"DTD_THAI_KY"}
    if disease == "KHONG_DTD":
        return not hit_d
    raise ValueError(disease)


def _sample_case(rng, disease, risks, symptoms, p_risk, p_sym, criteria):
    """Một lần lấy mẫu cho lớp `disease`; trả None nếu xét nghiệm không nhất quán với nhãn."""
    rf = {r: rng.random() < p_risk.get((disease, r), PRIORS["risk_override"].get(
        (disease, r), PRIORS["risk_background"].get(r, 0.05))) for r in risks}
    if rf["tuoi_tu_45"] and rf["tuoi_duoi_30"]:
        rf["tuoi_duoi_30" if rng.random() < 0.5 else "tuoi_tu_45"] = False
    if disease == "DTD_THAI_KY":
        rf["dang_mang_thai"] = True
    sym = {s: rng.random() < p_sym.get((disease, s), PRIORS["symptom_background"]) for s in symptoms}
    ctx = {
        "dang_mang_thai": rf["dang_mang_thai"],
        "co_trieu_chung_kinh_dien": any(sym[s] for s in ("an_nhieu", "uong_nhieu", "tieu_nhieu", "sut_can")),
    }
    labs = _sample_labs(rng, disease, rf["dang_mang_thai"])
    if not consistent(disease, labs, ctx, criteria):
        return None
    return rf, sym, labs, ctx


def generate(n_total: int, seed_val: int, p_unknown: float, p_explicit: float) -> list[dict]:
    rng = random.Random(seed_val)
    kb = load_seed()
    criteria = [Criterion.from_row(r) for r in kb.criteria]
    symptoms = [s["id"] for s in kb.nodes["Symptom"] if s["group"] in ASKABLE_SYMPTOM_GROUPS]
    risks = [r["id"] for r in kb.nodes["RiskFactor"]]
    p_sym: dict[tuple[str, str], float] = {}
    p_risk: dict[tuple[str, str], float] = {}
    for e in kb.edges:
        if e["rel"] == "HAS_SYMPTOM" and e["prob"]:
            p_sym[(e["from_id"], e["to_id"])] = float(e["prob"])
        if e["rel"] == "HAS_RISK_FACTOR" and e["prob"]:
            p_risk[(e["from_id"], e["to_id"])] = float(e["prob"])

    classes = list(PRIORS["class_weights"])
    weights = [PRIORS["class_weights"][c] for c in classes]
    cases = []
    while len(cases) < n_total:
        # Chọn lớp TRƯỚC rồi lấy mẫu lại trong cùng lớp đến khi nhất quán → giữ đúng tỉ lệ lớp
        disease = rng.choices(classes, weights)[0]
        for _attempt in range(500):
            case = _sample_case(rng, disease, risks, symptoms, p_risk, p_sym, criteria)
            if case is not None:
                break
        else:
            raise RuntimeError(f"Không sinh được ca nhất quán cho {disease} — kiểm tra PRIORS / criteria")
        rf, sym, labs, ctx = case
        profile = _sample_profile(rng, disease, rf)        # có thể chỉnh rf (thai phụ ⇒ không ≥ 45 tuổi)

        explicit, implicit = {}, {}
        for k, v in profile.items():
            if v is None:
                implicit[k] = {"status": "unknown", "value": None}
            elif isinstance(v, bool):
                implicit[k] = {"status": "yes" if v else "no", "value": None}
            else:
                implicit[k] = {"status": "yes", "value": v}
        for s, v in sym.items():
            if v and rng.random() < p_explicit:
                explicit[s] = {"status": "yes", "value": None}
            else:
                implicit[s] = {"status": "yes" if v else "no", "value": None}
        for r, v in rf.items():
            implicit[r] = {"status": "yes" if v else "no", "value": None}
        for t, v in labs.items():
            implicit[t] = {"status": "unknown", "value": None} if rng.random() < p_unknown else {"status": "yes", "value": v}

        cases.append({
            "case_id": f"C-{len(cases):05d}", "disease": disease,
            "explicit": dict(sorted(explicit.items())), "implicit": dict(sorted(implicit.items())),
            "true_labs": labs, "rule_hits": [h.id for h in check(labs, ctx, criteria)],
            "source": "sampled_from_seed_kb", "seed": seed_val,
        })
    return cases


def split(cases: list[dict], ratios: dict[str, float], seed_val: int) -> dict[str, list[dict]]:
    """Chia phân tầng theo bệnh để mỗi tập giữ tỉ lệ lớp."""
    rng = random.Random(seed_val + 1)
    out = {k: [] for k in ratios}
    by_cls: dict[str, list[dict]] = {}
    for c in cases:
        by_cls.setdefault(c["disease"], []).append(c)
    for cls in sorted(by_cls):
        items = by_cls[cls][:]
        rng.shuffle(items)
        n = len(items)
        n_train = round(n * ratios["train"])
        n_val = round(n * ratios["val"])
        out["train"] += items[:n_train]
        out["val"] += items[n_train:n_train + n_val]
        out["test"] += items[n_train + n_val:]
    return {k: sorted(v, key=lambda c: c["case_id"]) for k, v in out.items()}


def main() -> int:
    cfg = settings()["cases"]
    cases = generate(cfg["n_total"], cfg["seed"], cfg["p_lab_unknown"], cfg["p_explicit"])
    parts = split(cases, cfg["split"], cfg["seed"])
    stats = {"n_total": len(cases), "by_split": {}, "by_disease": dict(sorted(Counter(c["disease"] for c in cases).items()))}
    for name, rows in parts.items():
        write_jsonl(CASES_DIR / f"{name}.jsonl", rows)
        stats["by_split"][name] = dict(sorted(Counter(c["disease"] for c in rows).items()))
    all_unknown = sum(1 for c in cases if all(c["implicit"][t]["status"] == "unknown" for t in c["true_labs"]))
    stats["cases_all_labs_unknown"] = all_unknown
    stats["avg_explicit"] = round(sum(len(c["explicit"]) for c in cases) / len(cases), 2)
    (CASES_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

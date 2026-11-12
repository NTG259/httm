"""E2E hội thoại chẩn đoán (07_TEST_PLAN.md §2.3, TC-30…35, TC-40, TC-43).

Bệnh nhân mô phỏng (AG-10, câu mẫu — không cần Ollama) trả lời qua toàn bộ hệ thống: NLU → Router → Tracker →
Main/Worker (HRL) → NLG → Safety → Consult. Chạy:  pytest tests/e2e -m slow
"""
import json

import pytest

from diabetes_mas.agents import graph
from diabetes_mas.agents.slots import catalog
from diabetes_mas.config import CASES_DIR
from diabetes_mas.devtools.run_dialogs import run_case, summarize
from diabetes_mas.tools import base, kg_tools
from diabetes_mas.utils.text import read_jsonl

pytestmark = pytest.mark.slow
DTD_GROUP = {"DTD_TIP_1", "DTD_TIP_2", "DTD_THAI_KY"}
LABS = {"GLUCOSE_DOI", "OGTT_2H", "OGTT_1H", "GLUCOSE_BAT_KY", "HBA1C"}


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "LOG_DIR", tmp_path / "logs")
    kg_tools._CRIT_CACHE.clear()


def make_case(disease: str, **slots) -> dict:
    """Ca bệnh: slot nhị phân mặc định "no", số/phân loại mặc định "unknown", rồi ghi đè bằng `slots`."""
    implicit = {}
    for s in catalog().slots:
        if s.source == "derived":
            continue
        implicit[s.id] = {"status": "no" if s.kind == "binary" else "unknown", "value": None}
    for k, v in slots.items():
        implicit[k] = {"status": v, "value": None} if v in ("yes", "no", "unknown") else {"status": "yes", "value": v}
    return {"case_id": f"TC-{disease}", "disease": disease, "explicit": {}, "implicit": implicit}


def policy_questions(r: dict) -> list[str]:
    intake = {"tuoi", "bmi", "gioi_tinh"}
    return [s for s in r["asked"] if s not in intake]


def _tc30():
    case = make_case("DTD_TIP_2", tuoi=55, bmi=27.0, gioi_tinh="nam", uong_nhieu="yes", tieu_nhieu="yes",
                     sut_can="yes", GLUCOSE_DOI=8.4, HBA1C=7.6)
    return run_case(case, opening="Dạo này tôi khát nước suốt, đi tiểu nhiều, sụt 3 kg trong tháng.")


def test_tc30_story1_classic_symptoms_asks_labs_early():
    """GĐ 8 đã sửa (luật require_lab_before_classify + train lại HRL): hỏi xét nghiệm trong ≤ 4 lượt."""
    r = _tc30()
    assert r["slots"]["uong_nhieu"]["status"] == "yes" and r["slots"]["sut_can"]["status"] == "yes"
    assert set(policy_questions(r)[:4]) & LABS, r["asked"]
    assert r["stage"] == "concluded" and r["top1"] in DTD_GROUP


def test_tc30_story1_concludes_type2_for_55_years_bmi_27():
    """GĐ 8 đã sửa bằng PRIORS.risk_override (típ 1 ≥ 45 tuổi 21 % → 7 %) + train lại classifier."""
    assert _tc30()["top1"] == "DTD_TIP_2"


def test_tc31_pregnant_fasting_5_3_is_gestational():
    case = make_case("DTD_THAI_KY", tuoi=29, bmi=24.0, gioi_tinh="nu", dang_mang_thai="yes", GLUCOSE_DOI=5.3)
    r = run_case(case, opening="Tôi đang mang thai 28 tuần, muốn kiểm tra xem có bị tiểu đường không.")
    assert r["slots"]["dang_mang_thai"]["status"] == "yes"
    assert r["top1"] == "DTD_THAI_KY", r["asked"]
    assert "CRIT_DTD_THAI_KY_DOI" in r["outcome"]["rule_hits"]


def test_tc32_young_autoimmune_weight_loss_is_type1():
    case = make_case("DTD_TIP_1", tuoi=22, bmi=18.5, gioi_tinh="nam", sut_can="yes", uong_nhieu="yes",
                     tieu_nhieu="yes", benh_tu_mien="yes", GLUCOSE_DOI=9.5)
    r = run_case(case, opening="Tôi 22 tuổi, gần đây sụt cân nhanh, khát nước và đi tiểu nhiều.")
    assert r["top1"] == "DTD_TIP_1", (r["asked"], r["outcome"])


def test_tc33_no_symptoms_normal_labs_is_no_diabetes_and_ends_early():
    case = make_case("KHONG_DTD", tuoi=35, bmi=22.0, gioi_tinh="nu", HBA1C=5.3)
    r = run_case(case)
    assert r["top1"] == "KHONG_DTD", (r["asked"], r["outcome"])
    assert r["turns"] <= 6, r["asked"]                                     # "kết thúc sớm": ≤ ½ T_max


def test_tc34_story5_emergency_stops_immediately():
    sid = graph.new_session()
    out = graph.handle(sid, "Tôi đường huyết 25 mmol/L, đang buồn nôn, thở dốc", use_llm=False)
    assert out["stage"] == "emergency" and out["state"]["red_flag"] and out["state"]["done"]
    assert "115" in out["reply"] and out["path"][-1] == "emergency"
    assert out["state"]["turn"] == 0                                       # không hỏi thêm câu nào


def test_tc40_profile_sentence_fills_numeric_and_derived_slots():
    sid = graph.new_session()
    out = graph.handle(sid, "Tôi nam 52 tuổi, nặng 78 kg cao 1m65, có bị tiểu đường không?", use_llm=False)
    s = out["state"]["slots"]
    assert s["tuoi"]["value"] == 52 and s["bmi"]["value"] == pytest.approx(28.7, abs=0.1)
    assert s["gioi_tinh"]["value"] == "nam"
    assert s["tuoi_tu_45"]["status"] == "yes" and s["bmi_tu_23"]["status"] == "yes"


def test_tc43_story6_dl_signal_is_called_by_ag13_and_explained():
    case = make_case("TIEN_DTD", tuoi=52, bmi=28.7, gioi_tinh="nam", hut_thuoc="dang_hut", HBA1C=6.2)
    r = run_case(case, opening="Tôi nam 52 tuổi, nặng 78 kg cao 1m65, đang hút thuốc, HbA1c 6,2%.")
    t15 = [c for c in r["tool_calls"] if c["tool"] == "T-15"]
    assert t15 and all(c["agent"] == "AG-13" for c in t15)                 # AG-13 gọi T-15 (Main chỉ kích hoạt)
    assert r["risk_dl"] is not None
    assert "Mô hình học sâu" in r["reply"]                                 # giải thích nêu tín hiệu DL
    assert r["top1"] not in DTD_GROUP or r["outcome"]["rule_hits"]         # luật KB không kết luận ĐTĐ khi HbA1c 6,2


def test_tc35_test_split_never_exceeds_t_max_or_repeats_questions():
    cases = list(read_jsonl(CASES_DIR / "test.jsonl"))[:200]
    rows = [run_case(c) for c in cases]
    s = summarize(rows)
    print(json.dumps(s, ensure_ascii=False))
    assert s["over_T_max"] == 0 and s["with_repeats"] == 0
    assert s["concluded"] + s["stages"].get("emergency", 0) == s["n"]


@pytest.mark.xfail(strict=True, reason="GĐ 8: ca biên — chỉ biết glucose lúc đói 6,5 (HbA1c không biết), classifier cho "
                                       "típ 2 0,44 so với tiền ĐTĐ 0,40; luật Bộ Y tế bắt đúng CRIT_TIEN_DTD nhưng không loại "
                                       "ĐTĐ được. Thử luật 'không đạt tiêu chuẩn ĐTĐ ⇒ loại típ 1/2': E2E 0,796 → 0,742 nên "
                                       "không dùng. Xem 08_EVALUATION.md §7")
def test_tc21_story2_fasting_6_5_routes_to_diagnosis_and_concludes_prediabetes_by_moh_rule():
    """Router xếp câu này là diagnose (đúng ví dụ P-01: người dùng nói chỉ số CỦA MÌNH) ⇒ kết luận theo luật KB."""
    case = make_case("TIEN_DTD", tuoi=45, bmi=24.0, gioi_tinh="nam", GLUCOSE_DOI=6.5)
    r = run_case(case, opening="Đường huyết lúc đói 6,5 mmol/L có sao không?")
    assert r["slots"]["GLUCOSE_DOI"]["value"] == 6.5
    assert r["top1"] == "TIEN_DTD" and "CRIT_TIEN_DTD_GLUCOSE_DOI" in r["outcome"]["rule_hits"]
    assert "MOH-3280-2011" in r["reply"] or any(c["tool"] == "T-08" for c in r["tool_calls"])

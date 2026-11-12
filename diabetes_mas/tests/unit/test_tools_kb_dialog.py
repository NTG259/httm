import json

import pytest

from diabetes_mas.agents.state import SlotValue
from diabetes_mas.kb.criteria import apply_rules
from diabetes_mas.config import PROJECT_ROOT
from diabetes_mas.tools import base, dialog_tools, kg_tools, session
from diabetes_mas.tools.base import call

U = {"KHONG_DTD": 0.2, "TIEN_DTD": 0.2, "DTD_TIP_1": 0.2, "DTD_TIP_2": 0.2, "DTD_THAI_KY": 0.2}


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "LOG_DIR", tmp_path)
    # Unit test không phụ thuộc Neo4j: ép T-03 dùng seed
    monkeypatch.setattr(kg_tools, "_criteria_from_neo4j", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    kg_tools._CRIT_CACHE.clear()
    session.reset()


# ---------- T-01 ----------
def test_t01_action_spaces_match_design():
    sizes = {w: len(call("T-01", agent_id=f"AG-{w}", worker=w).slots) for w in ("W1", "W2", "W3", "W4")}
    assert sizes == {"W1": 4, "W2": 5, "W3": 13, "W4": 6}
    w3 = {s.slot_id: s for s in call("T-01", agent_id="AG-W3", worker="W3").slots}
    assert w3["tuoi"].kind == "numeric" and w3["tuoi"].unit == "tuổi"
    assert w3["gioi_tinh"].values == ["nam", "nu", "khac"]


# ---------- T-02 ----------
def test_t02_lexicon_links_vietnamese_variants():
    out = call("T-02", agent_id="AG-06", text="Bị tiểu đường tuýp 2 thì uống metformin được không?")
    ids = {e.node_id for e in out.entities}
    assert {"DTD_TIP_2", "METFORMIN"} <= ids
    assert all(e.method == "lexicon" for e in out.entities)


# ---------- T-03 (TC-01 … TC-04b) ----------
@pytest.mark.parametrize("meas,ctx,expected", [
    ({"GLUCOSE_DOI": 7.0}, {}, {"DTD"}),                                   # TC-01
    ({"HBA1C": 6.5}, {}, set()),                                           # TC-01: nguồn ghi "> 6,5%"
    ({"HBA1C": 6.6}, {}, {"DTD"}),
    ({"GLUCOSE_DOI": 6.5}, {}, {"TIEN_DTD"}),                              # TC-02
    ({"GLUCOSE_DOI": 5.4}, {}, set()),                                     # TC-03
    ({"OGTT_1H": 10.2}, {"dang_mang_thai": False}, set()),                 # TC-04
    ({"GLUCOSE_DOI": 6.5}, {"dang_mang_thai": True}, {"DTD_THAI_KY"}),     # TC-04b
])
def test_t03_criteria(meas, ctx, expected):
    out = call("T-03", agent_id="AG-02", measurements=meas, context=ctx)
    assert out.ok and out.kb_backend == "seed"
    assert {h.disease_id for h in out.hits} == expected
    for h in out.hits:
        assert h.source_id and h.evidence and h.source_title


def test_apply_rules_precedence():
    assert apply_rules(U, {"DTD"})["DTD_THAI_KY"] == 0 and apply_rules(U, {"DTD"})["TIEN_DTD"] == 0
    p = apply_rules(U, {"DTD_THAI_KY"})
    assert p["KHONG_DTD"] == p["TIEN_DTD"] == 0 and p["DTD_TIP_2"] > 0
    p = apply_rules(U, {"TIEN_DTD"})
    assert p["KHONG_DTD"] == 0 and p["DTD_TIP_2"] > 0          # ĐTĐ vẫn có thể khi xét nghiệm khác chưa biết
    assert abs(sum(p.values()) - 1) < 1e-9


# ---------- T-05 / T-06 / T-07 ----------
CASE = {"case_id": "C-T", "disease": "DTD_TIP_1",
        "explicit": {"uong_nhieu": {"status": "yes", "value": None}},
        "implicit": {"HBA1C": {"status": "yes", "value": 9.0}, "tuoi": {"status": "yes", "value": 22.0},
                     "GLUCOSE_DOI": {"status": "unknown", "value": None}, "sut_can": {"status": "yes", "value": None}}}


def test_t07_simulator_answers_from_case_and_defaults():
    session.attach_case("s1", CASE)
    a = call("T-07", agent_id="AG-W2", session_id="s1", slot_id="HBA1C")
    assert a.parsed["HBA1C"].value == 9.0
    assert call("T-07", agent_id="AG-W2", session_id="s1", slot_id="GLUCOSE_DOI").parsed["GLUCOSE_DOI"].status == "unknown"
    assert call("T-07", agent_id="AG-W4", session_id="s1", slot_id="nhin_mo").parsed["nhin_mo"].status == "no"
    assert call("T-07", agent_id="AG-W2", session_id="no-case", slot_id="HBA1C").error.startswith("no_case_attached")


def test_t05_intake_not_counted_but_policy_questions_are():
    call("T-05", agent_id="AG-05", session_id="s2", updates={"tuoi": {"status": "yes", "value": 50}},
         asked_slot="tuoi", count_turn=False)
    out = call("T-05", agent_id="AG-05", session_id="s2", updates={"HBA1C": {"status": "yes", "value": 7.0}},
               asked_slot="HBA1C")
    assert out.turn == 1
    st = call("T-06", agent_id="AG-01", session_id="s2").state
    assert st.slots["tuoi_tu_45"].status == "yes"


# ---------- T-08 ----------
def test_t08_rules_override_naive_bayes(monkeypatch):
    monkeypatch.setattr(dialog_tools, "VARIANT", "kb_naive_bayes")
    call("T-05", agent_id="AG-05", session_id="s3",
         updates={"uong_nhieu": {"status": "yes"}, "tieu_nhieu": {"status": "yes"}, "sut_can": {"status": "yes"},
                  "HBA1C": {"status": "yes", "value": 9.0}})
    out = call("T-08", agent_id="AG-05", session_id="s3")
    assert out.variant == "kb_naive_bayes"
    assert out.probs["KHONG_DTD"] == out.probs["TIEN_DTD"] == out.probs["DTD_THAI_KY"] == 0
    assert out.top1 in ("DTD_TIP_1", "DTD_TIP_2")
    assert [h.criterion_id for h in out.rule_hits] == ["CRIT_DTD_HBA1C"]
    assert session.get_or_create("s3").entropy == out.entropy


def test_t08_classic_symptoms_shift_mass_toward_diabetes(monkeypatch):
    monkeypatch.setattr(dialog_tools, "VARIANT", "kb_naive_bayes")
    call("T-05", agent_id="AG-05", session_id="s4",
         updates={"uong_nhieu": {"status": "yes"}, "tieu_nhieu": {"status": "yes"}, "sut_can": {"status": "yes"}})
    p = call("T-08", agent_id="AG-05", session_id="s4").probs
    assert p["DTD_TIP_1"] > p["KHONG_DTD"] and p["DTD_TIP_2"] > p["KHONG_DTD"]


# ---------- T-09 (TC-07) ----------
def test_t09_alarm_symptoms_trigger_red_flag_with_dka_reason():
    call("T-05", agent_id="AG-05", session_id="s5",
         updates={"buon_non": {"status": "yes"}, "tho_nhanh": {"status": "yes"},
                  "GLUCOSE_BAT_KY": {"status": "yes", "value": 25.0}})
    out = call("T-09", agent_id="AG-08", session_id="s5")
    assert out.red_flag and out.action_text
    assert any("13,9" in r for r in out.reasons)


def test_t09_high_glucose_alone_is_not_emergency():
    call("T-05", agent_id="AG-05", session_id="s6", updates={"GLUCOSE_BAT_KY": {"status": "yes", "value": 25.0}})
    assert call("T-09", agent_id="AG-05", session_id="s6").red_flag is False


# ---------- T-11 (TC-08) ----------
def test_t11_reward():
    out = call("T-11", agent_id="AG-11", mode="train", prev_entropy=1.2, entropy=0.7, asked=True, hit=True,
               terminal=False)
    assert abs(out.reward - 1.0) < 1e-9 and set(out.parts) == {"turn", "hit", "entropy"}


@pytest.mark.skipif(not (PROJECT_ROOT / "experiments/best/classifier.pt").exists(), reason="chưa huấn luyện classifier")
def test_t08_mlp_variant_still_obeys_rules(monkeypatch):
    monkeypatch.setattr(dialog_tools, "VARIANT", "mlp")
    call("T-05", agent_id="AG-05", session_id="s-mlp",
         updates={"tuoi": {"status": "yes", "value": 55}, "HBA1C": {"status": "yes", "value": 9.0}})
    out = call("T-08", agent_id="AG-05", session_id="s-mlp")
    assert out.variant == "mlp" and abs(sum(out.probs.values()) - 1) < 1e-6
    assert out.probs["KHONG_DTD"] == out.probs["TIEN_DTD"] == out.probs["DTD_THAI_KY"] == 0
    assert set(out.sub_probs) == {"W1", "W2", "W3", "W4"}

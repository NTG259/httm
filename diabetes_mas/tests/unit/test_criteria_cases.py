from diabetes_mas.kb.criteria import Criterion, check
from diabetes_mas.kb.seed import load_seed
from diabetes_mas.simulator.cases import consistent, eag_mmol, generate

CRIT = [Criterion.from_row(r) for r in load_seed().criteria]


def ids(hits):
    return {h.id for h in hits}


def test_tc01_fasting_7_is_diabetes():
    assert "CRIT_DTD_GLUCOSE_DOI" in ids(check({"GLUCOSE_DOI": 7.0}, {}, CRIT))


def test_tc02_ifg_is_prediabetes_only():
    hits = check({"GLUCOSE_DOI": 6.5}, {}, CRIT)
    assert {h.disease_id for h in hits} == {"TIEN_DTD"}


def test_tc03_normal_fasting_no_hit():
    assert check({"GLUCOSE_DOI": 5.4}, {}, CRIT) == []


def test_tc04_gdm_requires_pregnancy():
    assert check({"OGTT_1H": 10.2}, {"dang_mang_thai": False}, CRIT) == []
    assert "CRIT_DTD_THAI_KY_1H" in ids(check({"OGTT_1H": 10.2}, {"dang_mang_thai": True}, CRIT))


def test_random_glucose_requires_symptoms():
    assert check({"GLUCOSE_BAT_KY": 12.0}, {}, CRIT) == []
    assert ids(check({"GLUCOSE_BAT_KY": 12.0}, {"co_trieu_chung_kinh_dien": True}, CRIT)) == {"CRIT_DTD_GLUCOSE_BAT_KY"}


def test_pregnant_ifg_is_gdm_not_prediabetes():
    hits = check({"GLUCOSE_DOI": 6.5}, {"dang_mang_thai": True}, CRIT)
    assert {h.disease_id for h in hits} == {"DTD_THAI_KY"}


def test_overt_diabetes_in_pregnancy_suppresses_gdm():
    hits = check({"GLUCOSE_DOI": 7.5}, {"dang_mang_thai": True}, CRIT)
    assert {h.disease_id for h in hits} == {"DTD"}


def test_hba1c_6_5_is_not_a_hit_under_sourced_strict_operator():
    assert check({"HBA1C": 6.5}, {}, CRIT) == []
    assert ids(check({"HBA1C": 6.6}, {}, CRIT)) == {"CRIT_DTD_HBA1C"}


def test_diabetes_hit_suppresses_prediabetes():
    hits = check({"GLUCOSE_DOI": 6.5, "OGTT_2H": 12.0}, {}, CRIT)
    assert {h.disease_id for h in hits} == {"DTD"}


def test_eag_matches_moh_appendix():
    for a1c, mmol in [(6, 7.0), (8, 10.2), (12, 16.5)]:
        assert abs(eag_mmol(a1c) - mmol) < 0.1


def test_generated_cases_are_label_consistent_and_deterministic():
    a = generate(200, seed_val=7, p_unknown=0.5, p_explicit=0.35)
    b = generate(200, seed_val=7, p_unknown=0.5, p_explicit=0.35)
    assert a == b
    for c in a:
        ctx = {
            "dang_mang_thai": c["implicit"]["dang_mang_thai"]["status"] == "yes",
            "co_trieu_chung_kinh_dien": any(
                (c["explicit"].get(s) or c["implicit"].get(s))["status"] == "yes"
                for s in ("an_nhieu", "uong_nhieu", "tieu_nhieu", "sut_can")),
        }
        assert consistent(c["disease"], c["true_labs"], ctx, CRIT), c["case_id"]
        assert not (set(c["explicit"]) & set(c["implicit"]))


def test_profile_slots_consistent_with_derived_flags():          # FR-19
    from diabetes_mas.agents.slots import derive_flags

    for c in generate(300, seed_val=3, p_unknown=0.5, p_explicit=0.35):
        imp = c["implicit"]
        flags = derive_flags({"tuoi": imp["tuoi"]["value"], "bmi": imp["bmi"]["value"]})
        for f, v in flags.items():
            assert (imp[f]["status"] == "yes") == v, (c["case_id"], f)
        if imp["dang_mang_thai"]["status"] == "yes":
            assert imp["gioi_tinh"]["value"] == "nu" and 18 <= imp["tuoi"]["value"] <= 44


def test_every_askable_slot_has_an_answer_in_cases():
    from diabetes_mas.agents.slots import catalog

    c = generate(50, seed_val=5, p_unknown=0.5, p_explicit=0.35)[0]
    answers = set(c["explicit"]) | set(c["implicit"])
    missing = [s for s in catalog().askable() if s not in answers and s != "OGTT_1H"]
    assert missing == []

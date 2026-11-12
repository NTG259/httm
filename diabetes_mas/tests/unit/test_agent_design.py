import math

from diabetes_mas.agents.slots import DISEASES, WORKERS, catalog
from diabetes_mas.agents.state import (DialogState, SlotValue, delegate, encode, end_option, main_mask,
                                       pending_intake, state_dim, worker_mask)
from diabetes_mas.rl.reward import compute_reward, entropy, is_hit, option_return


def test_every_askable_slot_belongs_to_exactly_one_worker():
    cat = catalog()
    seen = [a for w in WORKERS for a in cat.actions(w)]
    assert len(seen) == len(set(seen)) == len(cat.askable())


def test_alarm_symptoms_and_derived_flags_are_not_askable():
    by = catalog().by_id()
    for sid in ("buon_non", "tho_nhanh", "roi_loan_y_thuc", "hoi_tho_mui_tao_thoi", "tuoi_tu_45", "bmi_tu_23"):
        assert by[sid].worker is None


def test_encode_length_matches_state_dim_and_is_bounded():
    st = DialogState(session_id="t")
    st.update({"HBA1C": SlotValue(status="yes", value=7.1), "uong_nhieu": SlotValue(status="yes"),
               "gioi_tinh": SlotValue(status="yes", value="nam"), "GLUCOSE_DOI": SlotValue(status="unknown")})
    v = encode(st)
    assert len(v) == state_dim()
    assert all(0.0 <= x <= 1.0 for x in v)


def test_derived_flags_from_numeric_slots():
    st = DialogState(session_id="t")
    st.update({"tuoi": SlotValue(status="yes", value=52), "bmi": SlotValue(status="yes", value=28.7)})
    assert st.slots["tuoi_tu_45"].status == "yes"
    assert st.slots["tuoi_duoi_30"].status == "no"
    assert st.slots["bmi_tu_23"].status == "yes"


def test_worker_mask_blocks_known_slots_and_ogtt1h_unless_pregnant():   # TC-09
    st = DialogState(session_id="t")
    st.update({"HBA1C": SlotValue(status="yes", value=6.0)})
    acts = catalog().actions("W2")
    m = worker_mask(st, "W2")
    assert m[acts.index("HBA1C")] is False
    assert m[acts.index("OGTT_1H")] is False
    assert m[-1] is False                                  # RETURN chưa mở khi chưa hỏi câu nào
    st.update({"dang_mang_thai": SlotValue(status="yes")})
    assert worker_mask(st, "W2")[acts.index("OGTT_1H")] is True


def test_worker_cannot_return_before_asking():
    st = DialogState(session_id="t")
    delegate(st, "W1")
    assert worker_mask(st, "W1")[-1] is False          # option rỗng (K = 0) bị chặn
    st.sub_turn = 1
    assert worker_mask(st, "W1")[-1] is True


def test_worker_may_return_when_nothing_left_to_ask():
    st = DialogState(session_id="t")
    st.update({s: SlotValue(status="no") for s in catalog().actions("W1")})
    delegate(st, "W1")
    assert worker_mask(st, "W1") == [False] * len(catalog().actions("W1")) + [True]


def test_main_mask_ignores_previous_workers_sub_turn():
    st = DialogState(session_id="t", turn=4)
    delegate(st, "W1")
    st.sub_turn = 4                                    # W1 vừa dùng hết lượt con
    end_option(st)
    assert main_mask(st)[:4] == [True, True, True, True]   # các worker khác vẫn được giao quyền mới


def test_intake_pending_until_profile_known():
    st = DialogState(session_id="t")
    assert pending_intake(st) == ["tuoi", "bmi", "gioi_tinh"]
    st.update({"tuoi": SlotValue(status="yes", value=40), "bmi": SlotValue(status="yes", value=22.0),
               "gioi_tinh": SlotValue(status="yes", value="nu")})
    assert pending_intake(st) == []


def test_main_mask_forces_classify_at_tmax():
    st = DialogState(session_id="t", turn=12)
    assert main_mask(st) == [False] * len(WORKERS) + [True]
    st0 = DialogState(session_id="t", turn=0)
    assert main_mask(st0)[-1] is False                     # chưa hỏi câu nào thì chưa được kết luận


def test_main_mask_requires_lab_before_classify():
    """GĐ 8 (TC-30): chưa có giá trị xét nghiệm và còn xét nghiệm chưa hỏi ⇒ chưa được CLASSIFY."""
    st = DialogState(session_id="t", turn=3)
    assert main_mask(st)[-1] is False
    st.update({"HBA1C": SlotValue(status="unknown")})
    assert main_mask(st)[-1] is False                       # "không biết" 1 chỉ số chưa đủ
    st.update({"GLUCOSE_DOI": SlotValue(status="yes", value=6.1)})
    assert main_mask(st)[-1] is True                        # có một giá trị
    st2 = DialogState(session_id="t", turn=4)
    st2.update({s: SlotValue(status="unknown") for s in catalog().actions("W2")})
    assert main_mask(st2)[-1] is True                       # hỏi hết, không biết chỉ số nào
    st_full = DialogState(session_id="t", turn=12)              # hết T_max vẫn bắt buộc CLASSIFY dù chưa có xét nghiệm
    assert main_mask(st_full)[-1] is True


def test_lab_worker_cannot_return_until_labs_ready():
    st = DialogState(session_id="t", turn=2)
    delegate(st, "W2")
    st.sub_turn = 2
    st.update({"HBA1C": SlotValue(status="unknown"), "GLUCOSE_BAT_KY": SlotValue(status="unknown")})
    assert worker_mask(st, "W2")[-1] is False                # 2 câu "không biết", còn xét nghiệm chưa hỏi
    st.update({"GLUCOSE_DOI": SlotValue(status="yes", value=6.5)})
    assert worker_mask(st, "W2")[-1] is True
    st.sub_turn = 4                                          # hết lượt con ⇒ không hỏi được nữa ⇒ RETURN mở
    st.slots.pop("GLUCOSE_DOI")
    assert worker_mask(st, "W2")[-1] is True


def test_reward_formula_tc08():
    r, parts = compute_reward(prev_entropy=1.2, cur_entropy=0.7, asked=True, hit=True, terminal=False, correct=None)
    assert math.isclose(r, -1 + 1 + 2 * 0.5)
    assert set(parts) == {"turn", "hit", "entropy"}
    r_end, _ = compute_reward(0.5, 0.5, asked=False, hit=False, terminal=True, correct=False)
    assert r_end == -20


def test_profile_answers_are_never_hits():
    assert is_hit("uong_nhieu", "yes", None)
    assert is_hit("HBA1C", "yes", 7.1)
    assert not is_hit("HBA1C", "unknown", None)
    assert not is_hit("uong_nhieu", "no", None)
    for slot, val in [("tuoi", 52.0), ("bmi", 28.0), ("gioi_tinh", "nam"), ("hut_thuoc", "chua_bao_gio"), ("benh_tim", None)]:
        assert not is_hit(slot, "yes", val), slot


def test_entropy_uniform_and_option_return():
    assert math.isclose(entropy([1 / len(DISEASES)] * len(DISEASES)), math.log(len(DISEASES)))
    total, k = option_return([1.0, 1.0, 1.0], gamma=0.5)
    assert (total, k) == (1.75, 3)

"""T-05 update_state · T-06 get_state · T-07 ask_patient · T-08 classify · T-09 check_red_flags
· T-10 log_event (đăng ký để xuất schema; được gọi tự động) · T-11 compute_reward."""
from __future__ import annotations

import math
from functools import lru_cache

from diabetes_mas.agents.slots import DISEASES, WORKERS, catalog
from diabetes_mas.agents.state import SlotValue, Turn, encode
from diabetes_mas.config import settings
from diabetes_mas.kb.criteria import apply_rules
from diabetes_mas.kb.seed import load_seed
from diabetes_mas.rl.reward import compute_reward as _reward
from diabetes_mas.rl.reward import entropy
from diabetes_mas.tools import session
from diabetes_mas.tools.base import log_event, tool
from diabetes_mas.tools.kg_tools import check_with_meta
from diabetes_mas.tools.schemas import (AskIn, AskOut, ClassifyIn, ClassifyOut, GetStateIn, GetStateOut, LogEventIn,
                                        LogEventOut, RedFlagIn, RedFlagOut, RewardIn, RewardOut, UpdateStateIn,
                                        UpdateStateOut)

CLASSIC = ("an_nhieu", "uong_nhieu", "tieu_nhieu", "sut_can")
ALARM_GROUP = "canh_bao"
EMERGENCY_TEXT = ("Các dấu hiệu bạn mô tả có thể là biến chứng cấp tính của đái tháo đường cần được cấp cứu. "
                  "Hãy gọi 115 hoặc đến cơ sở y tế gần nhất NGAY. Không tự điều trị tại nhà.")


# ---------------- T-05 / T-06 ----------------
@tool("T-05", "update_state", UpdateStateIn, UpdateStateOut, summarize=lambda o: f"turn={o.turn}")
def update_state(inp: UpdateStateIn) -> UpdateStateOut:
    st = session.get_or_create(inp.session_id)
    st.update(inp.updates)
    if inp.asked_slot:
        st.history.append(Turn(speaker="user", slot=inp.asked_slot))
        if inp.count_turn:
            st.turn += 1
            if st.current_worker:
                st.sub_turn += 1
    return UpdateStateOut(turn=st.turn)


@tool("T-06", "get_state", GetStateIn, GetStateOut, summarize=lambda o: f"turn={o.state.turn if o.state else '-'}")
def get_state(inp: GetStateIn) -> GetStateOut:
    st = session.get_or_create(inp.session_id)
    return GetStateOut(state=st.model_copy(deep=True), vector=encode(st))


# ---------------- T-07 ----------------
def _answer_text(sv: SlotValue) -> str:
    if sv.status == "yes" and sv.value is not None:
        return str(sv.value)
    return {"yes": "Có.", "no": "Không.", "unknown": "Tôi không rõ."}.get(sv.status, "Tôi không rõ.")


@tool("T-07", "ask_patient", AskIn, AskOut, summarize=lambda o: _answer_text(next(iter(o.parsed.values())))
      if o.parsed else "-")
def ask_patient(inp: AskIn) -> AskOut:
    """Kênh simulator: phiên có gắn ca bệnh ⇒ trả lời đúng slot được hỏi theo ca bệnh.
    Kênh UI: phiên không gắn ca ⇒ ghi câu hỏi (do NLG soạn) vào lịch sử, đánh dấu slot đang chờ trả lời;
    câu trả lời đến ở lượt sau qua NLU (AG-06) → T-05. Thiếu question_text ⇒ lỗi."""
    case = session.case_for(inp.session_id)
    if case is None:
        if not inp.question_text:
            return AskOut(ok=False, error="no_case_attached: kênh UI cần question_text", channel="ui")
        st = session.get_or_create(inp.session_id)
        st.history.append(Turn(speaker="system", text=inp.question_text, slot=inp.slot_id))
        return AskOut(answer_text="", parsed={}, channel="ui")
    raw = case["explicit"].get(inp.slot_id) or case["implicit"].get(inp.slot_id)
    if raw is None:
        slot = catalog().by_id().get(inp.slot_id)
        raw = {"status": "no" if slot is not None and slot.kind == "binary" else "unknown", "value": None}
    sv = SlotValue(**raw)
    return AskOut(answer_text=_answer_text(sv), parsed={inp.slot_id: sv})


def simulator_opening(session_id: str) -> dict[str, SlotValue]:
    """Câu mở đầu của bệnh nhân mô phỏng: các slot `explicit` (AG-10)."""
    case = session.case_for(session_id) or {"explicit": {}}
    return {k: SlotValue(**v) for k, v in case["explicit"].items()}


# ---------------- T-08 ----------------
@lru_cache(maxsize=1)
def _nb_params() -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    """P(slot = yes | bệnh) từ quan hệ KB; tỉ lệ nền khi bệnh không có cạnh.
    Tỉ lệ nền dùng chung PRIORS của simulator ⇒ baseline lạc quan (đã ghi trong 03 §3)."""
    from diabetes_mas.simulator.cases import PRIORS

    p: dict[tuple[str, str], float] = {}
    for e in load_seed().edges:
        if e["rel"] in ("HAS_SYMPTOM", "HAS_RISK_FACTOR") and e["prob"]:
            p[(e["from_id"], e["to_id"])] = float(e["prob"])
    bg = {s.id: PRIORS["symptom_background"] for s in catalog().slots if s.source == "Symptom"}
    bg.update(PRIORS["risk_background"])
    return p, bg


def _nb(st, slot_ids: list[str]) -> dict[str, float]:
    p, bg = _nb_params()
    logp = {d: math.log(1.0 / len(DISEASES)) for d in DISEASES}
    for sid in slot_ids:
        sv = st.slots.get(sid)
        if sv is None or sv.status not in ("yes", "no") or sid not in bg:
            continue
        for d in DISEASES:
            q = min(max(p.get((d, sid), bg[sid]), 1e-3), 1 - 1e-3)
            logp[d] += math.log(q if sv.status == "yes" else 1 - q)
    m = max(logp.values())
    z = sum(math.exp(v - m) for v in logp.values())
    return {d: math.exp(logp[d] - m) / z for d in DISEASES}


def _measurements(st) -> tuple[dict[str, float], dict[str, bool]]:
    labs = {s.id for s in catalog().slots if s.source == "LabTest"}
    meas = {k: float(v.value) for k, v in st.slots.items()
            if k in labs and v.status == "yes" and isinstance(v.value, (int, float))}
    ctx = {"dang_mang_thai": st.slots.get("dang_mang_thai", SlotValue()).status == "yes",
           "co_trieu_chung_kinh_dien": any(st.slots.get(s, SlotValue()).status == "yes" for s in CLASSIC)}
    return meas, ctx


@tool("T-08", "classify", ClassifyIn, ClassifyOut, summarize=lambda o: f"{o.top1} {o.probs.get(o.top1, 0):.2f} H={o.entropy:.2f} ({o.variant})")
def classify(inp: ClassifyIn) -> ClassifyOut:
    """Biến thể "mlp" (AG-02 + AG-03-h, checkpoint experiments/best/classifier.pt) nếu có; nếu không,
    "kb_naive_bayes" (Naive Bayes từ xác suất trong KB). Cả hai đều áp luật KB sau cùng."""
    st = session.get_or_create(inp.session_id)
    model = _mlp() if (VARIANT or settings().get("classifier", {}).get("variant", "mlp")) == "mlp" else None
    if model is not None:
        probs, sub = model.predict(st)
        variant = "mlp"
    else:
        binary = [s.id for s in catalog().slots if s.kind == "binary"]
        probs = _nb(st, binary)
        sub = {w: _nb(st, [a for a in catalog().actions(w) if a in binary]) for w in WORKERS}
        variant = "kb_naive_bayes"
    meas, ctx = _measurements(st)
    hits, _ = check_with_meta(meas, ctx)
    probs = apply_rules(probs, {h.disease_id for h in hits})
    h = entropy(probs)
    st.probs, st.entropy = probs, h
    top1 = max(probs, key=probs.get)
    return ClassifyOut(probs=probs, sub_probs=sub, entropy=h, top1=top1, rule_hits=hits, variant=variant)


VARIANT: str | None = None          # ghi đè settings.classifier.variant (test / so sánh baseline)


@lru_cache(maxsize=1)
def _mlp():
    from diabetes_mas.agents.classifiers import load
    return load()


# ---------------- T-09 ----------------
@tool("T-09", "check_red_flags", RedFlagIn, RedFlagOut, summarize=lambda o: "RED FLAG" if o.red_flag else "ổn")
def check_red_flags(inp: RedFlagIn) -> RedFlagOut:
    """Cấp cứu khi có bất kỳ dấu hiệu nhóm `canh_bao` (KB: triệu chứng của hôn mê nhiễm toan ceton /
    tăng áp lực thẩm thấu — VMC-0001: "Biến chứng này cần phải điều trị cấp cứu").
    KHÔNG dùng ngưỡng đường huyết đơn lẻ: 13,9 mmol/L (QĐ 3280 tr.9) là tiêu chí cận lâm sàng KÈM
    triệu chứng, không phải ngưỡng cấp cứu độc lập — chỉ dùng để bổ sung lý do."""
    st = session.get_or_create(inp.session_id)
    alarms = [s for s in catalog().slots if s.group == ALARM_GROUP and st.slots.get(s.id, SlotValue()).status == "yes"]
    if not alarms:
        return RedFlagOut(red_flag=False)
    seed_names = {r["id"]: r["name_vi"] for r in load_seed().nodes["Symptom"]}
    reasons = [f"Dấu hiệu cảnh báo: {seed_names.get(s.id, s.id)}" for s in alarms]
    glucose = [st.slots[k].value for k in ("GLUCOSE_BAT_KY", "GLUCOSE_DOI")
               if k in st.slots and isinstance(st.slots[k].value, (int, float))]
    if any(g > 13.9 for g in glucose):
        reasons.append("Kèm đường huyết > 13,9 mmol/L — một tiêu chí của nhiễm toan ceton (Bộ Y tế, QĐ 3280/2011)")
    st.red_flag = True
    return RedFlagOut(red_flag=True, reasons=reasons, action_text=EMERGENCY_TEXT)


# ---------------- T-10 ----------------
@tool("T-10", "log_event", LogEventIn, LogEventOut)
def log_event_tool(inp: LogEventIn) -> LogEventOut:
    """Đăng ký để có schema; trên thực tế base.call() ghi log tự động sau MỌI tool call."""
    path = log_event(inp.session_id, inp.agent, inp.tool, inp.args, inp.result_summary, inp.latency_ms, inp.ok)
    return LogEventOut(path=str(path))


# ---------------- T-11 ----------------
@tool("T-11", "compute_reward", RewardIn, RewardOut, summarize=lambda o: f"r={o.reward:+.2f} {o.parts}")
def compute_reward(inp: RewardIn) -> RewardOut:
    r, parts = _reward(inp.prev_entropy, inp.entropy, inp.asked, inp.hit, inp.terminal, inp.correct)
    return RewardOut(reward=r, parts=parts)

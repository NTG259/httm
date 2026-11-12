"""DialogState (AG-05 State Tracker) + mã hóa vector s_t và action mask cho agent học.

Hợp đồng chốt ở Giai đoạn 2 (02_AGENT_DESIGN.md §3). Thay đổi cách mã hóa ⇒ phải ghi ADR
vì làm mất hiệu lực mọi checkpoint đã huấn luyện.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

from diabetes_mas.agents.slots import DISEASES, WORKERS, catalog, derive_flags
from diabetes_mas.config import load_yaml

Status = Literal["yes", "no", "unknown", "not_asked"]


class SlotValue(BaseModel):
    status: Status = "not_asked"
    value: float | str | None = None      # số (numeric) hoặc nhãn (categorical)


class Turn(BaseModel):
    speaker: Literal["system", "user"]
    text: str = ""
    slot: str | None = None
    agent: str | None = None


class DialogState(BaseModel):
    session_id: str
    turn: int = 0                                   # số câu hỏi đã gửi bệnh nhân
    slots: dict[str, SlotValue] = Field(default_factory=dict)
    current_worker: str | None = None               # W1..W4 đang giữ quyền
    sub_turn: int = 0                               # số câu hỏi của worker hiện tại trong lần giao quyền này
    probs: dict[str, float] = Field(default_factory=dict)
    entropy: float = math.log(len(DISEASES))
    risk_dl: float | None = None                    # AG-13 (không vào vector — ADR-002)
    dl_conflict: bool = False
    red_flag: bool = False
    done: bool = False
    history: list[Turn] = Field(default_factory=list)

    def known(self, slot_id: str) -> bool:
        sv = self.slots.get(slot_id)
        return sv is not None and sv.status != "not_asked"

    def update(self, updates: dict[str, SlotValue]) -> None:
        """Ghi slot mới rồi suy ra cờ nhị phân từ slot số (tuoi → tuoi_tu_45…)."""
        self.slots.update(updates)
        numeric = {k: float(v.value) for k, v in self.slots.items()
                   if v.status == "yes" and isinstance(v.value, (int, float))}
        for flag, val in derive_flags(numeric).items():
            self.slots[flag] = SlotValue(status="yes" if val else "no")
        preg = self.slots.get("dang_mang_thai")
        if preg is not None and preg.status == "yes" and not self.known("gioi_tinh"):
            self.slots["gioi_tinh"] = SlotValue(status="yes", value="nu")   # không hỏi giới tính người đang mang thai


def _cfg() -> dict:
    return load_yaml("slots.yaml")["dialog"]


def slot_dims(kind: str, n_values: int) -> int:
    if kind == "binary":
        return 3                                    # yes, no, unknown  (not_asked = 000)
    if kind == "numeric":
        return 3                                    # giá trị chuẩn hóa, đã biết, không biết
    return n_values + 1                             # one-hot giá trị + không biết


def state_dim() -> int:
    d = sum(slot_dims(s.kind, len(s.values)) for s in catalog().slots)
    return d + 1 + 1 + (len(WORKERS) + 1) + len(DISEASES) + 1   # turn, sub_turn, worker(+none), p_t, H


def encode_slot(s, sv: SlotValue) -> list[float]:
    """Mã hóa một slot (dùng chung cho s_t và đầu vào classifier)."""
    if s.kind == "binary":
        return [float(sv.status == "yes"), float(sv.status == "no"), float(sv.status == "unknown")]
    if s.kind == "numeric":
        lo, hi = s.range
        known = sv.status == "yes" and isinstance(sv.value, (int, float))
        x = min(max((float(sv.value) - lo) / (hi - lo), 0.0), 1.0) if known else 0.0
        return [x, float(known), float(sv.status == "unknown")]
    return [float(sv.status == "yes" and sv.value == v) for v in s.values] + [float(sv.status == "unknown")]


def encode_slots(state: DialogState, slot_ids: list[str] | None = None) -> list[float]:
    """Phần slot của s_t (mặc định mọi slot, thứ tự catalog()); `slot_ids` để lấy một nhóm con."""
    keep = None if slot_ids is None else set(slot_ids)
    vec: list[float] = []
    for s in catalog().slots:
        if keep is None or s.id in keep:
            vec += encode_slot(s, state.slots.get(s.id, SlotValue()))
    return vec


def encode(state: DialogState) -> list[float]:
    """s_t → vector float có độ dài state_dim(); thứ tự slot cố định theo catalog()."""
    cfg = _cfg()
    vec = encode_slots(state)
    vec.append(state.turn / cfg["T_max"])
    vec.append(state.sub_turn / cfg["max_sub_turns"])
    vec += [float(state.current_worker == w) for w in WORKERS] + [float(state.current_worker is None)]
    vec += [state.probs.get(d, 1.0 / len(DISEASES)) for d in DISEASES]
    vec.append(state.entropy / math.log(len(DISEASES)))
    return vec


def unasked_slots(state: DialogState, worker: str) -> list[bool]:
    """Slot nào của worker còn hỏi được về mặt nội dung (chưa biết; OGTT_1H chỉ cho thai phụ).
    Không xét ngân sách lượt — dùng chung cho worker_mask và main_mask."""
    acts = catalog().actions(worker)
    ok = [not state.known(a) for a in acts]
    if "OGTT_1H" in acts and state.slots.get("dang_mang_thai", SlotValue()).status != "yes":
        ok[acts.index("OGTT_1H")] = False
    return ok


def worker_mask(state: DialogState, worker: str) -> list[bool]:
    """Mask của Work-Agent đang giữ quyền: [slot_1..slot_n, RETURN].

    - Slot đã biết bị chặn (FR-04); hết T_max hoặc max_sub_turns thì chặn mọi câu hỏi.
    - RETURN chỉ mở sau khi worker đã hỏi ≥ 1 câu trong lần giao quyền này (chống option rỗng K = 0),
      hoặc khi không còn câu nào hỏi được.
    - Worker xét nghiệm (W2) chưa được RETURN khi `labs_ready` sai và còn câu hỏi được — bản trước để W2 trả quyền
      sau 2 câu "không biết", 17/225 ca E2E đi vòng tới T_max mà chưa hỏi glucose lúc đói (acc 0,53).
    """
    cfg = _cfg()
    can_ask = state.turn < cfg["T_max"] and state.sub_turn < cfg["max_sub_turns"]
    mask = [can_ask and ok for ok in unasked_slots(state, worker)]
    can_return = state.sub_turn >= 1 or not any(mask)
    if cfg.get("require_lab_before_classify") and worker == _lab_worker() and any(mask) and not labs_ready(state):
        can_return = False          # GĐ 8: worker xét nghiệm hỏi tiếp tới khi có 1 giá trị hoặc hết lượt con
    return mask + [can_return]


def _lab_worker() -> str:
    return next(s.worker for s in catalog().slots if s.source == "LabTest")


def labs_ready(state: DialogState) -> bool:
    """Đủ điều kiện xét nghiệm để kết luận: đã có ít nhất một GIÁ TRỊ xét nghiệm (hỏi hoặc người dùng tự nói),
    hoặc đã hỏi hết các xét nghiệm hỏi được (người bệnh không biết chỉ số nào).
    Bản đầu chỉ cần "đã hỏi 1 xét nghiệm" ⇒ policy hỏi 1 câu, nhận "không biết" rồi kết luận ngay (E2E --llm, GĐ 8)."""
    labs = [s for s in catalog().slots if s.source == "LabTest"]
    if any(state.slots.get(s.id, SlotValue()).status == "yes" for s in labs):
        return True
    return not any(unasked_slots(state, _lab_worker()))


def main_mask(state: DialogState) -> list[bool]:
    """Mask của Main Agent: [W1..W4, CLASSIFY], tính cho một lần giao quyền MỚI (sub_turn sẽ về 0).
    Worker không còn slot hỏi được bị chặn; hết T_max hoặc không worker nào khả dụng ⇒ bắt buộc CLASSIFY.

    Luật điều phối `require_lab_before_classify` (GĐ 8, 08_EVALUATION.md §5): chưa CLASSIFY khi chưa có giá trị
    xét nghiệm nào VÀ còn xét nghiệm chưa hỏi (`labs_ready`) — chẩn đoán ĐTĐ cần căn cứ xét nghiệm."""
    cfg = _cfg()
    workers_ok = [state.turn < cfg["T_max"] and any(unasked_slots(state, w)) for w in WORKERS]
    lab_ok = not cfg.get("require_lab_before_classify", False) or labs_ready(state)
    stop = cfg.get("confident_stop")
    if stop and state.turn >= cfg["min_turns_before_classify"] and lab_ok and state.probs             and max(state.probs.values()) >= stop:
        workers_ok = [False] * len(WORKERS)        # GĐ 9: đã có xét nghiệm và classifier chắc ⇒ kết luận ngay
    classify_ok = (state.turn >= cfg["min_turns_before_classify"] and lab_ok) or not any(workers_ok)
    return workers_ok + [classify_ok]


def pending_intake(state: DialogState) -> list[str]:
    """Slot hồ sơ cố định chưa biết — điều phối hỏi trước khi policy bắt đầu (không tính turn)."""
    return [s for s in load_yaml("slots.yaml")["dialog"]["intake_slots"] if not state.known(s)]


def delegate(state: DialogState, worker: str) -> None:
    """Main giao quyền: đặt worker hiện tại, reset sub_turn."""
    state.current_worker, state.sub_turn = worker, 0


def end_option(state: DialogState) -> None:
    """Worker RETURN (hoặc hết sub_turn/T_max): trả quyền về Main."""
    state.current_worker, state.sub_turn = None, 0

"""Môi trường hội thoại chẩn đoán cho HRL (04_SEQUENCE.md Luồng C).

Mọi bước đi qua `tools.call` đúng agent/quyền của Luồng C:
  AG-10 (simulator) → AG-05 T-05 update_state → AG-05 T-08 classify → AG-W* T-07 ask_patient
  → AG-11 T-11 compute_reward. Điều phối cố định (intake, mask, end_option) theo 02_AGENT_DESIGN.md §6.

Cùng một lớp dùng cho demo-mô-phỏng (mode="demo", bật T-09) và kiểm thử end-to-end.
"""
from __future__ import annotations

import itertools

from diabetes_mas.agents.slots import DISEASES, WORKERS, catalog
from diabetes_mas.agents.state import DialogState, delegate, end_option, encode, main_mask, pending_intake, worker_mask
from diabetes_mas.config import load_yaml
from diabetes_mas.rl.reward import is_hit
from diabetes_mas.tools import call, session
from diabetes_mas.tools.dialog_tools import simulator_opening

CLASSIFY = len(WORKERS)                  # chỉ số hành động CLASSIFY của Main
_IDS = itertools.count()


class DiagnosisEnv:
    def __init__(self, mode: str = "train"):
        self.mode = mode
        self.dialog = load_yaml("slots.yaml")["dialog"]
        self.sid = ""
        self.case: dict = {}

    # ---- tiện ích ----
    def state(self) -> DialogState:
        return session.get_or_create(self.sid)

    def vector(self) -> list[float]:
        return encode(self.state())

    def _classify(self):
        return call("T-08", agent_id="AG-05", mode=self.mode, session_id=self.sid)

    def _ask(self, worker_agent: str, slot_id: str, count_turn: bool):
        ans = call("T-07", agent_id=worker_agent, mode=self.mode, session_id=self.sid, slot_id=slot_id)
        call("T-05", agent_id="AG-05", mode=self.mode, session_id=self.sid, updates=ans.parsed,
             asked_slot=slot_id, count_turn=count_turn)
        return ans.parsed.get(slot_id)

    # ---- vòng đời episode ----
    def reset(self, case: dict) -> list[float]:
        self.case = case
        self.sid = f"{self.mode}-{next(_IDS)}"
        session.attach_case(self.sid, case)
        call("T-05", agent_id="AG-05", mode=self.mode, session_id=self.sid,
             updates=simulator_opening(self.sid), count_turn=False)
        for slot in pending_intake(self.state()):          # bước 0: intake, không tính T_max, không reward
            self._ask("AG-W3", slot, count_turn=False)
        self._classify()
        return self.vector()

    def main_mask(self) -> list[bool]:
        return main_mask(self.state())

    def delegate(self, worker_idx: int) -> None:
        delegate(self.state(), WORKERS[worker_idx])

    def worker_mask(self) -> list[bool]:
        st = self.state()
        return worker_mask(st, st.current_worker)

    def worker_step(self, action: int) -> tuple[list[float], float, bool]:
        """Worker đang giữ quyền hỏi slot `action` hoặc RETURN. Trả (s', r, option_done)."""
        st = self.state()
        w = st.current_worker
        acts = catalog().actions(w)
        if action == len(acts):                             # RETURN
            end_option(st)
            return self.vector(), 0.0, True
        slot = acts[action]
        prev_h = st.entropy
        sv = self._ask(f"AG-{w}", slot, count_turn=True)
        out = self._classify()
        hit = sv is not None and is_hit(slot, sv.status, sv.value)
        r = call("T-11", agent_id="AG-11", mode="train", prev_entropy=prev_h, entropy=out.entropy, asked=True,
                 hit=hit, terminal=False).reward if self.mode == "train" else 0.0
        if self.mode == "demo":
            call("T-09", agent_id="AG-05", mode=self.mode, session_id=self.sid)
        st = self.state()
        done = st.turn >= self.dialog["T_max"] or st.sub_turn >= self.dialog["max_sub_turns"] or st.red_flag
        if done:
            end_option(st)
        return self.vector(), r, done

    def classify_final(self) -> tuple[str, bool, float]:
        """CLASSIFY: AG-02 ra top-1; trả (top1, đúng?, reward kết thúc)."""
        out = self._classify()
        correct = out.top1 == self.case.get("disease")
        r = call("T-11", agent_id="AG-11", mode="train", prev_entropy=0, entropy=0, asked=False, hit=False,
                 terminal=True, correct=correct).reward if self.mode == "train" else 0.0
        self.state().done = True
        return out.top1, correct, r

    def close(self) -> None:
        session.reset(self.sid)


def labels() -> list[str]:
    return DISEASES

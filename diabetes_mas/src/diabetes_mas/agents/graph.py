"""Đồ thị điều phối một lượt hội thoại (04_SEQUENCE.md Luồng A, B, D) — ADR-003.

Mỗi tin nhắn người dùng chạy đồ thị từ node `nlu` tới khi gặp END (đã có câu trả lời / câu hỏi tiếp theo).
Trạng thái chia sẻ (K1) = `DialogState` (kho phiên) + `Conversation` (slot đang chờ, chữ ký DL, chế độ).

Node:  nlu → router → { emergency | out_of_scope | consult | tracker }
       tracker → (red flag ⇒ emergency) → main → { ask (WAIT) | conclude }
Mọi câu ra đi qua `safety.check` (AG-08).
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable

from diabetes_mas.agents import consult, nlg, nlu, prediction, router, safety
from diabetes_mas.agents.policy import RulePolicy, default_policy
from diabetes_mas.agents.slots import DISEASES, WORKERS, catalog
from diabetes_mas.agents.state import (DialogState, Turn, delegate, encode, end_option, main_mask, pending_intake,
                                       worker_mask)
from diabetes_mas.config import load_yaml
from diabetes_mas.rl.env import CLASSIFY
from diabetes_mas.tools import base, call, session
from diabetes_mas.tools.dialog_tools import EMERGENCY_TEXT
from diabetes_mas.tools.errors import UNAVAILABLE_TEXT, is_unavailable

END = "__end__"
OUT_OF_SCOPE_TEXT = ("Xin lỗi, tôi chỉ hỗ trợ các câu hỏi liên quan đến bệnh đái tháo đường (tiểu đường). "
                     "Anh/chị có thể hỏi về triệu chứng, xét nghiệm, điều trị hoặc mô tả tình trạng của mình.")


class Graph:
    """Đồ thị trạng thái tối giản: node là hàm (ctx) → None, cạnh là tên node hoặc hàm chọn node."""

    def __init__(self):
        self.nodes: dict[str, Callable] = {}
        self.edges: dict[str, str | Callable] = {}

    def add_node(self, name: str, fn: Callable) -> None:
        self.nodes[name] = fn

    def add_edge(self, src: str, dst: str | Callable) -> None:
        self.edges[src] = dst

    def run(self, ctx: "TurnCtx", start: str, max_steps: int = 60) -> None:
        node = start
        for _ in range(max_steps):
            if node == END:
                return
            ctx.path.append(node)
            self.nodes[node](ctx)
            nxt = self.edges.get(node, END)
            node = nxt(ctx) if callable(nxt) else nxt
        raise RuntimeError(f"đồ thị không dừng sau {max_steps} bước: {ctx.path[-10:]}")


@dataclass
class Conversation:
    session_id: str
    mode: str | None = None                  # diagnose | consult
    awaiting_slot: str | None = None
    awaiting_counts: bool = True             # False với câu hỏi intake
    last_question: str = ""
    dl_sig: tuple | None = None
    dl: dict | None = None
    done: bool = False
    outcome: dict | None = None


@dataclass
class TurnCtx:
    conv: Conversation
    text: str
    use_llm: bool = True
    slots: dict = field(default_factory=dict)
    intent: str | None = None
    reply: str = ""
    stage: str = ""
    sources: list = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    @property
    def st(self) -> DialogState:
        return session.get_or_create(self.conv.session_id)


# ---------------- node ----------------
def n_nlu(c: TurnCtx) -> None:
    r = nlu.understand(c.text, c.conv.awaiting_slot, c.conv.last_question, session_id=c.conv.session_id,
                       use_llm=c.use_llm)
    c.slots = r.slots
    c.notes["nlu"] = {"rules": r.by_rules, "llm": r.by_llm, "llm_error": r.llm_error}


def n_router(c: TurnCtx) -> None:
    hit = router.classify_rules(c.text, c.slots)
    if hit is None and c.conv.mode == "diagnose" and c.conv.awaiting_slot and (c.slots or len(c.text) < 40):
        hit = "diagnose"                       # đang trả lời câu hỏi chẩn đoán
    if hit:
        c.intent, method = hit, "rules"
    elif c.use_llm:
        c.intent, method = router.classify(c.text, c.slots)
    else:
        c.intent, method = router.fallback(c.text, c.slots), "fallback"
    c.notes["router"] = {"intent": c.intent, "method": method}


def n_tracker(c: TurnCtx) -> None:
    """AG-05: T-05 → T-09 → T-08."""
    cv, sid = c.conv, c.conv.session_id
    cv.mode = "diagnose"
    asked = cv.awaiting_slot
    slot = catalog().by_id().get(asked) if asked else None
    if slot is not None and slot.kind == "numeric" and asked not in c.slots             and nlu.YES.search(nlu.norm_key(c.text)):              # "có" nhưng chưa nói con số
        c.notes["follow_up"] = asked
        if c.slots:
            call("T-05", agent_id="AG-05", session_id=cv.session_id, updates=c.slots, count_turn=False)
        return
    c.st.history.append(Turn(speaker="user", text=c.text))
    call("T-05", agent_id="AG-05", session_id=sid, updates=c.slots, asked_slot=asked,
         count_turn=cv.awaiting_counts if asked else False)
    cv.awaiting_slot = None
    rf = call("T-09", agent_id="AG-05", session_id=sid)
    c.notes["red_flag"] = bool(rf.ok and rf.red_flag)
    out = call("T-08", agent_id="AG-05", session_id=sid)
    c.notes["classify"] = {"ok": out.ok, "variant": getattr(out, "variant", None)}


def n_emergency(c: TurnCtx) -> None:
    if c.slots:
        call("T-05", agent_id="AG-05", session_id=c.conv.session_id, updates=c.slots, count_turn=False)
    chk = safety.check(EMERGENCY_TEXT, session_id=c.conv.session_id)
    c.reply = chk.text if chk.emergency else f"{EMERGENCY_TEXT}\n\n{safety.DISCLAIMER}"
    c.stage, c.conv.done = "emergency", True
    c.st.done = True


def n_out_of_scope(c: TurnCtx) -> None:
    c.reply, c.stage = safety.check(OUT_OF_SCOPE_TEXT, add_disclaimer=False).text, "out_of_scope"


def n_consult(c: TurnCtx) -> None:
    a = consult.answer(c.text, c.conv.session_id, use_llm=c.use_llm)
    chk = safety.check(a.text, session_id=c.conv.session_id)
    c.reply, c.stage, c.sources = chk.text, ("unavailable" if a.method == "unavailable" else "answered"), a.sources
    c.notes["consult"] = {"method": a.method, "enough": a.enough, "research": a.research, "safety": chk.issues}
    if c.conv.mode != "diagnose":
        c.conv.mode = "consult"
    elif c.conv.last_question and not c.conv.done:
        c.reply += f"\n\nQuay lại câu hỏi trước: {c.conv.last_question}"
        c.conv.awaiting_slot = c.notes.get("pending_slot") or c.conv.awaiting_slot


def _ask(c: TurnCtx, worker: str, slot_id: str, counts: bool) -> None:
    q, method = nlg.question_for(slot_id, use_llm=c.use_llm)
    out = call("T-07", agent_id=f"AG-{worker}", session_id=c.conv.session_id, slot_id=slot_id, question_text=q)
    chk = safety.check(q, session_id=c.conv.session_id, add_disclaimer=False)
    if chk.emergency:
        c.reply, c.stage, c.conv.done = chk.text, "emergency", True
        return
    c.conv.awaiting_slot, c.conv.awaiting_counts, c.conv.last_question = slot_id, counts, q
    c.reply, c.stage = chk.text, "asking"
    c.notes["ask"] = {"slot": slot_id, "worker": worker, "nlg": method, "channel": out.channel if out.ok else None}


def n_main(c: TurnCtx) -> None:
    """AG-01 — điều phối cố định (02 §6) + policy học được."""
    st, cv = c.st, c.conv
    pend = pending_intake(st)
    if pend:                                                      # bước 0: intake
        _ask(c, "W3", pend[0], counts=False)
        return
    sig = prediction.dl_signature(st)                             # bước 3: kích hoạt AG-13
    if sig != cv.dl_sig:
        cv.dl_sig, cv.dl = sig, prediction.run(st)
        c.notes["ag13"] = cv.dl
    pol = POLICY()
    s = encode(st)
    for _ in range(len(WORKERS) * 15):                            # bước 4–5
        if st.current_worker:
            h = WORKERS.index(st.current_worker)
            wm = worker_mask(st, st.current_worker)
            a = pol.worker_act(h, s, wm)
            acts = catalog().actions(st.current_worker)
            if a < len(acts):
                _ask(c, st.current_worker, acts[a], counts=True)
                return
            end_option(st)
            continue
        mm = main_mask(st)
        a = pol.main_act(s, mm)
        if a == CLASSIFY:
            n_conclude(c)
            return
        delegate(st, WORKERS[a])
    n_conclude(c)


def n_conclude(c: TurnCtx) -> None:
    st, sid = c.st, c.conv.session_id
    out = call("T-08", agent_id="AG-05", session_id=sid)
    if not out.ok:
        c.reply = UNAVAILABLE_TEXT if is_unavailable(out.error) else UNAVAILABLE_TEXT
        c.stage = "unavailable"
        return
    evidence = [k for k, v in st.slots.items() if v.status == "yes" and catalog().by_id().get(k)
                and catalog().by_id()[k].source != "derived"]
    a = consult.explain(sid, out.top1, out.probs, out.rule_hits, evidence, st.risk_dl, st.dl_conflict,
                        use_llm=c.use_llm)
    chk = safety.check(a.text, session_id=sid)
    c.reply, c.stage, c.sources = chk.text, "concluded", a.sources
    c.conv.done, st.done = True, True
    c.conv.outcome = {"top1": out.top1, "probs": out.probs, "rule_hits": [h.criterion_id for h in out.rule_hits],
                      "risk_dl": st.risk_dl, "dl_conflict": st.dl_conflict, "explain_method": a.method}


def _after_router(c: TurnCtx) -> str:
    return {"emergency": "emergency", "out_of_scope": "out_of_scope", "consult": "consult"}.get(c.intent, "tracker")


def n_follow_up(c: TurnCtx) -> None:
    """Người dùng xác nhận có chỉ số nhưng chưa nói số ⇒ hỏi tiếp, không tính lượt."""
    q = "Anh/chị cho biết con số cụ thể là bao nhiêu ạ? (nếu không nhớ, cứ nói không biết)"
    c.reply, c.stage = safety.check(q, session_id=c.conv.session_id, add_disclaimer=False).text, "asking"
    c.conv.awaiting_slot = c.notes["follow_up"]


def _after_tracker(c: TurnCtx) -> str:
    if c.notes.get("follow_up"):
        return "follow_up"
    return "emergency" if c.notes.get("red_flag") else "main"


def build() -> Graph:
    g = Graph()
    for name, fn in (("nlu", n_nlu), ("router", n_router), ("tracker", n_tracker), ("emergency", n_emergency),
                     ("out_of_scope", n_out_of_scope), ("consult", n_consult), ("main", n_main),
                     ("follow_up", n_follow_up)):
        g.add_node(name, fn)
    g.add_edge("nlu", "router")
    g.add_edge("router", _after_router)
    g.add_edge("tracker", _after_tracker)
    return g


# ---------------- API mức phiên ----------------
GRAPH = build()
_POLICY = None
_CONVS: dict[str, Conversation] = {}
_LOCK = threading.Lock()


def POLICY():
    global _POLICY
    if _POLICY is None:
        _POLICY = default_policy()
    return _POLICY


def new_session() -> str:
    sid = uuid.uuid4().hex[:12]
    with _LOCK:
        _CONVS[sid] = Conversation(sid)
    session.get_or_create(sid)
    return sid


def get_conv(sid: str) -> Conversation:
    with _LOCK:
        if sid not in _CONVS:
            _CONVS[sid] = Conversation(sid)
        return _CONVS[sid]


def summary(sid: str) -> dict:
    st = session.get_or_create(sid)
    top1 = max(st.probs, key=st.probs.get) if st.probs else None
    return {"turn": st.turn, "T_max": load_yaml("slots.yaml")["dialog"]["T_max"], "probs": st.probs,
            "top1": top1, "entropy": round(st.entropy, 4), "risk_dl": st.risk_dl, "dl_conflict": st.dl_conflict,
            "red_flag": st.red_flag, "done": st.done,
            "slots": {k: v.model_dump() for k, v in st.slots.items() if v.status != "not_asked"}}


def handle(sid: str, text: str, use_llm: bool = True) -> dict:
    """Một lượt: chạy đồ thị, trả câu trả lời + trạng thái + vết tool call."""
    conv = get_conv(sid)
    if conv.done:
        conv.__init__(sid)                                       # phiên đã kết thúc ⇒ bắt đầu hội thoại mới
        session.reset(sid)
    trace: list = []
    token = base.TRACE.set(trace)
    try:
        ctx = TurnCtx(conv, text, use_llm=use_llm)
        GRAPH.run(ctx, "nlu")
    finally:
        base.TRACE.reset(token)
    session.get_or_create(sid).history.append(Turn(speaker="system", text=ctx.reply))
    return {"session_id": sid, "reply": ctx.reply, "intent": ctx.intent, "stage": ctx.stage,
            "asking_slot": conv.awaiting_slot, "path": ctx.path, "notes": ctx.notes,
            "sources": [s.__dict__ for s in ctx.sources], "state": summary(sid), "outcome": conv.outcome,
            "tool_calls": trace, "policy": POLICY().name}


__all__ = ["handle", "new_session", "summary", "build", "Graph", "DISEASES", "RulePolicy"]

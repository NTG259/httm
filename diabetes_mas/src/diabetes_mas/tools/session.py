"""Kho phiên hội thoại trong bộ nhớ (một tiến trình). DialogState + ca bệnh gắn cho simulator."""
from __future__ import annotations

import threading

from diabetes_mas.agents.state import DialogState

_LOCK = threading.Lock()
_STATES: dict[str, DialogState] = {}
_CASES: dict[str, dict] = {}


def get_or_create(session_id: str) -> DialogState:
    with _LOCK:
        if session_id not in _STATES:
            _STATES[session_id] = DialogState(session_id=session_id)
        return _STATES[session_id]


def attach_case(session_id: str, case: dict) -> DialogState:
    """Gắn ca bệnh cho Patient Simulator (chế độ huấn luyện / kiểm thử)."""
    with _LOCK:
        _CASES[session_id] = case
        _STATES[session_id] = DialogState(session_id=session_id)
        return _STATES[session_id]


def case_for(session_id: str) -> dict | None:
    return _CASES.get(session_id)


def reset(session_id: str | None = None) -> None:
    with _LOCK:
        if session_id is None:
            _STATES.clear()
            _CASES.clear()
        else:
            _STATES.pop(session_id, None)
            _CASES.pop(session_id, None)

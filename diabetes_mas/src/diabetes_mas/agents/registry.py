"""Sổ đăng ký agent (configs/agents.yaml) + kiểm tra quyền gọi tool."""
from __future__ import annotations

from functools import lru_cache

from diabetes_mas.config import load_yaml


class ToolNotAllowed(PermissionError):
    pass


@lru_cache(maxsize=1)
def agents() -> dict[str, dict]:
    return load_yaml("agents.yaml")["agents"]


def allowed_tools(agent_id: str) -> set[str]:
    return set(agents()[agent_id]["tools"])


def assert_allowed(agent_id: str, tool_id: str) -> None:
    if tool_id not in allowed_tools(agent_id):
        raise ToolNotAllowed(f"{agent_id} không được phép gọi {tool_id}")

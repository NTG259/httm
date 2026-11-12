"""Nạp mọi module tool để đăng ký vào TOOLS (T-01 … T-15)."""
from diabetes_mas.tools import (dialog_tools, kg_tools, predict_tools, research_tools,  # noqa: F401
                                retrieve_tools)
from diabetes_mas.tools.base import TOOLS, call  # noqa: F401

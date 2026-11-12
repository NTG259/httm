"""Mã lỗi hạ tầng chuẩn hóa (04_SEQUENCE.md Luồng D).

`tools.call` đổi exception của driver/thư viện thành mã ổn định, để agent rẽ nhánh theo mã chứ không theo
tên class của thư viện. Agent gặp mã thuộc `UNAVAILABLE` ⇒ trả `UNAVAILABLE_TEXT`, KHÔNG đoán.
"""
from __future__ import annotations

KB_UNAVAILABLE = "kb_unavailable"      # Neo4j không phản hồi (K5)
LLM_UNAVAILABLE = "llm_unavailable"    # Ollama không phản hồi (K3) — client LLM ném LLMUnavailable
NET_UNAVAILABLE = "net_unavailable"    # mạng / SearXNG không phản hồi (K6)
TIMEOUT = "timeout"
UNAVAILABLE = (KB_UNAVAILABLE, LLM_UNAVAILABLE, NET_UNAVAILABLE, TIMEOUT)

UNAVAILABLE_TEXT = ("Hệ thống tạm thời không truy cập được dữ liệu y khoa nên chưa thể trả lời chính xác. "
                    "Vui lòng thử lại sau ít phút. Nếu có dấu hiệu bất thường, hãy liên hệ cơ sở y tế.")


class LLMUnavailable(RuntimeError):
    """Client Ollama ném lỗi này khi không kết nối được / quá thời gian."""


def classify_exception(exc: BaseException) -> str | None:
    """Exception hạ tầng → mã chuẩn; exception khác (lỗi logic) → None."""
    if isinstance(exc, LLMUnavailable):
        return LLM_UNAVAILABLE
    try:
        from neo4j.exceptions import ServiceUnavailable, SessionExpired
        if isinstance(exc, (ServiceUnavailable, SessionExpired)):
            return KB_UNAVAILABLE
    except ImportError:  # pragma: no cover
        pass
    try:
        import requests
        if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
            return NET_UNAVAILABLE
    except ImportError:  # pragma: no cover
        pass
    return None


def is_unavailable(error: str | None) -> bool:
    return bool(error) and error.split(":", 1)[0] in UNAVAILABLE

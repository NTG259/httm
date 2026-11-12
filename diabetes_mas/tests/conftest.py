import pytest


@pytest.fixture(autouse=True)
def _reset_tool_sessions():
    """Kho phiên của tool là toàn cục trong tiến trình — làm sạch giữa các test."""
    from diabetes_mas.tools import session

    session.reset()
    yield
    session.reset()

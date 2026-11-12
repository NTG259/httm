"""Contract checks for the zero-build demo interface."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup
from fastapi.testclient import TestClient

from diabetes_mas.api.app import app


def test_demo_page_exposes_accessible_chat_and_diagnostics() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")

    page = BeautifulSoup(response.text, "html.parser")
    assert page.html["lang"] == "vi"
    assert page.select_one("main") is not None
    assert page.select_one("#chat[role='log'][aria-live='polite']") is not None
    assert page.select_one("form#form input#text[name='message']") is not None
    assert page.select_one("button#sendBtn[type='submit']") is not None
    assert page.select_one("button#newBtn[type='button'][aria-label='Phiên mới']") is not None
    assert page.select_one("input#useLlm[type='checkbox'][aria-label='Dùng LLM']") is not None
    assert page.select_one("#probs") is not None
    assert page.select_one("#slots") is not None
    assert page.select_one("#tools") is not None
    assert page.select_one("#sources") is not None


def test_demo_page_includes_loading_error_and_responsive_states() -> None:
    html = TestClient(app).get("/").text

    assert 'id="composerStatus"' in html
    assert 'aria-busy="false"' in html
    assert "response.ok" in html
    assert "AbortController" in html
    assert re.search(r"@media\s*\(max-width:\s*720px\)", html)
    assert "prefers-reduced-motion" in html

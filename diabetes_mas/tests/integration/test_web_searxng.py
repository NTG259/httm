"""T-12 / T-13 / AG-12 với SearXNG + web thật (Giai đoạn 6). Tự bỏ qua nếu SearXNG không chạy.

Chạy:  docker compose up -d searxng   rồi   pytest -m network
"""
import pytest
import requests

from diabetes_mas.config import env
from diabetes_mas.tools import base, call, research_tools

pytestmark = pytest.mark.network


def _searxng_up() -> bool:
    try:
        return requests.get(f"{env('SEARXNG_URL', 'http://localhost:8888')}/healthz", timeout=3).ok
    except requests.RequestException:
        return False


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    if not _searxng_up():
        pytest.skip("SearXNG không chạy")
    monkeypatch.setattr(base, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(research_tools, "REVIEW_QUEUE", tmp_path / "review_queue.jsonl")
    monkeypatch.setattr(research_tools, "SEARCH_CACHE", tmp_path / "search")
    monkeypatch.setattr(research_tools, "WEB_CACHE", tmp_path / "web")


def test_t12_web_returns_only_whitelisted_domains():
    out = call("T-12", agent_id="AG-12", query="metformin tác dụng phụ", scope="web")
    assert out.ok and out.hits, out.error
    assert all(research_tools.domain_allowed(h.url) for h in out.hits)


def test_t13_fetches_and_scores_a_vinmec_article():
    hits = call("T-12", agent_id="AG-12", query="metformin tác dụng phụ", scope="web").hits
    vinmec = [h for h in hits if h.domain.endswith("vinmec.com")]
    if not vinmec:
        pytest.skip("SearXNG không trả bài vinmec lần này")
    art = call("T-13", agent_id="AG-12", url=vinmec[0].url, question="Metformin có tác dụng phụ gì?", timeout_s=15)
    assert art.ok, art.error
    assert art.chunks and all(c.web and not c.reviewed and 0 <= c.cosine <= 1 for c in art.chunks)

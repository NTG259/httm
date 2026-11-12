"""T-12 search_articles · T-13 fetch_article · T-14 queue_article_for_kb (AG-12 Research)."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from diabetes_mas.config import CACHE_DIR, DATA_DIR, PROCESSED_DIR, env, settings
from diabetes_mas.ingest.chunk import split_text
from diabetes_mas.ingest.http import PoliteClient, RobotsDisallowed
from diabetes_mas.ingest.parse_vinmec import _leaf_blocks, _sections_from_blocks, _strip, parse
from diabetes_mas.tools.base import tool
from diabetes_mas.tools.pii import find_pii
from diabetes_mas.tools.schemas import (ArticleHit, ChunkHit, FetchArticleIn, FetchArticleOut, QueueArticleIn,
                                        QueueArticleOut, SearchArticlesIn, SearchArticlesOut)
from diabetes_mas.utils.text import append_jsonl, collapse_ws, norm_key, read_jsonl, sha256_text

REVIEW_QUEUE = DATA_DIR / "review_queue.jsonl"
SEARCH_CACHE = CACHE_DIR / "search"
WEB_CACHE = CACHE_DIR / "web"
CACHE_TTL_S = 24 * 3600


# ---------------- whitelist ----------------
def allowed_domains() -> list[str]:
    return [d.lower() for d in settings()["search"]["allowed_domains"]]


def domain_allowed(url: str, extra: list[str] | None = None) -> bool:
    """Khớp host chính xác hoặc tên miền con ('.' + domain). Domain do agent truyền vào chỉ được THU HẸP
    whitelist, không mở rộng."""
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    allowed = allowed_domains()
    if extra:
        allowed = [d for d in allowed if d in {e.lower() for e in extra}]
    return any(host == d or host.endswith("." + d) for d in allowed)


def search_enabled() -> bool:
    flag = env("SEARCH_ENABLED")
    return settings()["search"]["enabled"] and (flag is None or flag.lower() not in ("0", "false", "no"))


def _kb_urls() -> dict[str, str]:
    return {s["url"]: s["id"] for s in read_jsonl(PROCESSED_DIR / "sources.jsonl") if s.get("url")}


# ---------------- T-12 ----------------
LOCAL_CYPHER = """
CALL db.index.fulltext.queryNodes('source_text', $q) YIELD node, score
OPTIONAL MATCH (node)-[:FROM_SOURCE]->(src:Source)
WITH coalesce(src, node) AS s, score, CASE WHEN node:Chunk THEN node.text ELSE '' END AS text
WHERE s:Source
RETURN s.id AS source_id, s.title AS title, s.url AS url, max(score) AS score, collect(text)[0] AS snippet
ORDER BY score DESC LIMIT $k
"""


def _lucene_escape(text: str) -> str:
    return "".join(f"\\{c}" if c in '+-&|!(){}[]^"~*?:\\/' else c for c in text)


def _search_local(inp: SearchArticlesIn) -> list[ArticleHit]:
    from diabetes_mas.kb.neo4j_io import driver

    with driver() as drv, drv.session() as s:
        rows = s.run(LOCAL_CYPHER, q=_lucene_escape(inp.query), k=inp.max_results).data()
    return [ArticleHit(url=r["url"], title=r["title"] or "", snippet=collapse_ws(r["snippet"] or "")[:240],
                       domain=(urlparse(r["url"]).hostname or "local") if r["url"] else "local",
                       in_kb=True, source_id=r["source_id"], score=round(r["score"], 3)) for r in rows]


SEARXNG_GET: Callable[..., requests.Response] = requests.get        # tiêm được khi test


def _search_web(inp: SearchArticlesIn) -> list[ArticleHit]:
    domains = [d for d in allowed_domains() if not inp.domains or d in {x.lower() for x in inp.domains}]
    if not domains:
        return []
    key = sha256_text(json.dumps([inp.query, domains, inp.max_results]))[:16]
    cache = SEARCH_CACHE / f"{key}.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < CACHE_TTL_S:
        results = json.loads(cache.read_text(encoding="utf-8"))
    else:
        q = f"{inp.query} (" + " OR ".join(f"site:{d}" for d in domains) + ")"
        resp = SEARXNG_GET(f"{env('SEARXNG_URL', 'http://localhost:8888')}/search",
                           params={"q": q, "format": "json", "language": "vi"},
                           timeout=settings()["search"]["timeout_s"])
        resp.raise_for_status()
        results = resp.json().get("results", [])
        SEARCH_CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    kb = _kb_urls()
    hits = []
    for r in results:
        url = r.get("url", "")
        if not domain_allowed(url, inp.domains):
            continue                                   # lọc lại dù đã có site: — không tin máy tìm kiếm
        hits.append(ArticleHit(url=url, title=r.get("title", ""), snippet=collapse_ws(r.get("content", ""))[:240],
                               domain=urlparse(url).hostname or "", in_kb=url in kb, source_id=kb.get(url),
                               score=float(r.get("score") or 0.0)))
        if len(hits) == inp.max_results:
            break
    return hits


@tool("T-12", "search_articles", SearchArticlesIn, SearchArticlesOut, timeout_s=15.0,
      summarize=lambda o: f"{len(o.hits)} bài: " + ", ".join((h.source_id or h.domain) for h in o.hits))
def search_articles(inp: SearchArticlesIn) -> SearchArticlesOut:
    if inp.scope == "local":
        return SearchArticlesOut(hits=_search_local(inp))
    if not search_enabled():
        return SearchArticlesOut(ok=False, error="search_disabled")
    pii = find_pii(inp.query)
    if pii:
        return SearchArticlesOut(ok=False, error=f"pii_detected: {', '.join(pii)}")   # không có request nào ra web
    return SearchArticlesOut(hits=_search_web(inp))


# ---------------- T-13 ----------------
def _generic_sections(html: str):
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("article") or soup.find("main") or soup.body or soup
    _strip(root)
    title = collapse_ws(soup.title.string) if soup.title and soup.title.string else ""
    return title, None, _sections_from_blocks(_leaf_blocks(root), default_heading=title)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    from diabetes_mas.rag.vector_search import embed_texts

    return embed_texts(texts)


EMBED_TEXTS: Callable[[list[str]], list[list[float]]] = _embed_texts
HTTP_CLIENT_FACTORY: Callable[[float], PoliteClient] = lambda t: PoliteClient(timeout_s=t, retries=0)


@tool("T-13", "fetch_article", FetchArticleIn, FetchArticleOut, timeout_s=90.0,
      summarize=lambda o: f"{o.title[:60]} → {len(o.chunks)} chunk")
def fetch_article(inp: FetchArticleIn) -> FetchArticleOut:
    if not domain_allowed(inp.url):
        return FetchArticleOut(ok=False, error="not_allowed_domain", url=inp.url)
    client = HTTP_CLIENT_FACTORY(inp.timeout_s)
    try:
        resp = client.get(inp.url)
    except RobotsDisallowed:
        return FetchArticleOut(ok=False, error="robots_disallow", url=inp.url)
    except requests.Timeout:
        return FetchArticleOut(ok=False, error="timeout", url=inp.url)
    if resp.status_code != 200:
        return FetchArticleOut(ok=False, error=f"http_{resp.status_code}", url=inp.url)
    html = resp.text
    sha8 = sha256_text(inp.url)[:8]
    WEB_CACHE.mkdir(parents=True, exist_ok=True)
    (WEB_CACHE / f"{sha8}.html").write_text(html, encoding="utf-8")

    host = urlparse(inp.url).hostname or ""
    if host == "vinmec.com" or host.endswith(".vinmec.com"):
        page = parse(html, "disease" if "/benh/" in inp.url else "post")
        title, published, sections = page.title, page.published, page.sections
    else:
        title, published, sections = _generic_sections(html)
    cfg = settings()["chunk"]
    pieces = [(s.heading, p) for s in sections for p in split_text(s.text, cfg["max_words"], cfg["overlap_words"])
              if len(p.split()) >= 8]
    if not pieces:
        return FetchArticleOut(ok=False, error="parse_failed", url=inp.url, title=title)
    texts = [f"{title} › {h}\n{p}" for h, p in pieces]
    vecs = EMBED_TEXTS([inp.question] + texts)
    q, docs = vecs[0], vecs[1:]
    scored = []
    for i, (t, v) in enumerate(zip(texts, docs)):
        cos = sum(a * b for a, b in zip(q, v))
        scored.append(((1 + cos) / 2, i, t))           # cùng thang với vector index Neo4j
    scored.sort(key=lambda x: (-x[0], x[1]))
    sid = f"WEB-{sha8}"
    chunks = [ChunkHit(chunk_id=f"{sid}#{i:03d}", text=t, score=round(s, 4), cosine=round(s, 4), source_id=sid,
                       source_title=title, url=inp.url, reviewed=False, web=True) for s, i, t in scored[:inp.k_chunks]]
    return FetchArticleOut(url=inp.url, title=title, published=(published or "")[:10] or None, chunks=chunks)


def prefilter(texts: list[str], question: str, cap: int) -> list[str]:
    """Giữ tối đa `cap` đoạn trùng nhiều từ (≥ 2 ký tự, bỏ dấu) với câu hỏi nhất, giữ thứ tự gốc khi hòa.
    Lý do: nhúng cả bài vinmec (vài chục đoạn) bằng bge-m3 trên CPU mất 30–60 s/bài (đo GĐ 6)."""
    if len(texts) <= cap:
        return texts
    words = {w for w in norm_key(question).split() if len(w) >= 2}
    overlap = [len(words & set(norm_key(t).split())) for t in texts]
    keep = sorted(range(len(texts)), key=lambda i: (-overlap[i], i))[:cap]
    return [texts[i] for i in sorted(keep)]


# ---------------- T-14 ----------------
@tool("T-14", "queue_article_for_kb", QueueArticleIn, QueueArticleOut,
      summarize=lambda o: "đã vào hàng chờ" if o.queued else "bỏ qua (đã có)")
def queue_article_for_kb(inp: QueueArticleIn) -> QueueArticleOut:
    if not domain_allowed(inp.url):
        return QueueArticleOut(ok=False, error="not_allowed_domain")
    if inp.url in _kb_urls() or any(r["url"] == inp.url for r in read_jsonl(REVIEW_QUEUE)):
        return QueueArticleOut(queued=False)
    append_jsonl(REVIEW_QUEUE, {
        "url": inp.url, "title": inp.title, "domain": urlparse(inp.url).hostname, "reason": inp.reason,
        "session_id": inp.session_id, "queued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "pending", "reviewer": None, "reviewed_at": None,
    })
    return QueueArticleOut(queued=True)

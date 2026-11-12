"""AG-12 Research — tìm bài viết bổ sung khi KB không đủ (04_SEQUENCE.md Luồng B, nhánh opt).

1. P-08 viết lại truy vấn, bỏ thông tin cá nhân (Ollama lỗi ⇒ bỏ số + đại từ nhân xưng bằng luật).
2. T-12 scope=local: kho có bài với cosine thô ≥ search.local_min_cosine ⇒ dùng snippet (nguồn đã duyệt), không lên web.
3. Không có ⇒ T-12 scope=web (SearXNG + whitelist) → T-13 tối đa `max_fetch_per_question` bài
   → giữ chunk có điểm ≥ rag.min_score → T-14 đưa bài vào hàng chờ duyệt.
Giới hạn `max_web_calls_per_session` lần lên web mỗi phiên. Mọi lỗi mạng ⇒ trả 0 chunk (Luồng D).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel

from diabetes_mas.agents import llm
from diabetes_mas.config import settings
from diabetes_mas.tools import call, research_tools
from diabetes_mas.tools.errors import LLMUnavailable
from diabetes_mas.tools.schemas import ChunkHit

_WEB_CALLS: dict[str, int] = {}
PERSONAL = re.compile(r"\b(tôi|mình|em|anh|chị|bố|mẹ|con|năm nay|tuổi)\b|\d+(?:[.,]\d+)?\s*%?", re.I)


class QueryOut(BaseModel):
    query_vi: str
    query_en: str = ""


@dataclass
class ResearchResult:
    chunks: list[ChunkHit] = field(default_factory=list)
    query: str = ""
    scope: str = ""
    queued: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def rewrite(question: str) -> tuple[str, str]:
    try:
        q = llm.chat_json(QueryOut, llm.render("P-08", question=question))
        if q.query_vi.strip() and not re.search(r"\d", q.query_vi):
            return q.query_vi.strip(), "llm"
    except (LLMUnavailable, llm.LLMBadOutput):
        pass
    q = re.sub(r"\s+", " ", PERSONAL.sub(" ", question)).strip(" ?.,")
    return q, "rules"


def _relevant_local(query: str, hits: list, min_score: float, errors: list[str]) -> list[ChunkHit]:
    """Bài trong kho chỉ được dùng khi cosine THÔ(truy vấn, tiêu đề + snippet) ≥ `search.local_min_cosine`.
    Fulltext khớp cụm chung ("người tiểu đường") nên trả cả bài lạc đề: đo 2026-11-12, câu tirzepatide
    nhận "Các món cháo…" cos 0,52; câu sầu riêng nhận đúng bài cos 0,77."""
    if not hits:
        return []
    try:
        vecs = research_tools.EMBED_TEXTS([query] + [f"{h.title} {h.snippet}" for h in hits])
    except Exception as exc:  # noqa: BLE001 — không nhúng được thì không dùng bài kho chưa kiểm
        errors.append(f"embed local: {type(exc).__name__}")
        return []
    q = vecs[0]
    out = []
    for h, v in zip(hits, vecs[1:]):
        cos = float(sum(a * b for a, b in zip(q, v)))
        if cos >= min_score:
            out.append(ChunkHit(chunk_id=f"{h.source_id}#snippet", text=h.snippet, score=round(cos, 4),
                                cosine=round(cos, 4), source_id=h.source_id or "", source_title=h.title, url=h.url,
                                reviewed=True))
    return out


def find(question: str, session_id: str) -> ResearchResult:
    cfg = settings()["search"]
    res = ResearchResult()
    res.query, _ = rewrite(question)
    local = call("T-12", agent_id="AG-12", session_id=session_id, query=res.query, scope="local", max_results=3)
    if local.ok and local.hits:
        good = _relevant_local(res.query, [h for h in local.hits if h.snippet], cfg.get("local_min_cosine", 0.75),
                               res.errors)
        if good:
            res.scope = "local"
            res.chunks = good
            return res
    if not local.ok:
        res.errors.append(f"T-12 local: {local.error}")
    if not cfg.get("enabled", True) or _WEB_CALLS.get(session_id, 0) >= cfg["max_web_calls_per_session"]:
        return res
    _WEB_CALLS[session_id] = _WEB_CALLS.get(session_id, 0) + 1
    res.scope = "web"
    web = call("T-12", agent_id="AG-12", session_id=session_id, query=res.query, scope="web", max_results=5)
    if not web.ok:
        res.errors.append(f"T-12 web: {web.error}")
        return res
    for hit in web.hits[: cfg.get("max_fetch_per_question", 2)]:
        art = call("T-13", agent_id="AG-12", session_id=session_id, url=hit.url, question=question,
                   timeout_s=cfg.get("timeout_s", 8))
        if not art.ok:
            res.errors.append(f"T-13 {hit.domain}: {art.error}")
            continue
        good = [c for c in art.chunks if c.cosine >= settings()["rag"]["min_score"]]   # thang (1+cos)/2
        res.chunks += good
        if good:
            q = call("T-14", agent_id="AG-12", session_id=session_id, url=hit.url, title=art.title or hit.title,
                     reason=f"bổ sung cho câu hỏi: {res.query}")
            if q.ok and q.queued:
                res.queued.append(hit.url)
    return res

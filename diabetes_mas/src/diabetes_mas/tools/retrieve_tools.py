"""T-04 hybrid_retrieve — 8 bước ở 06_KB_RAG_INTEGRATION.md §2.

Backend Neo4j và hàm embedding được tiêm vào (BACKEND, EMBED) ⇒ unit test không cần Neo4j / bge-m3.
Điểm cosine thống nhất theo thang của vector index Neo4j: (1 + cos) / 2 ∈ [0, 1].
"""
from __future__ import annotations

import re
from typing import Callable, Protocol

from diabetes_mas.config import settings
from diabetes_mas.tools.base import tool
from diabetes_mas.tools.kg_tools import link_entities
from diabetes_mas.tools.schemas import ChunkHit, Fact, RetrieveIn, RetrieveOut

SEED_RELS = ["HAS_SYMPTOM", "HAS_RISK_FACTOR", "HAS_COMPLICATION", "DIAGNOSED_BY", "TREATS",
             "SUBTYPE_OF", "DIFFERENTIAL_OF"]
INTENT_RELS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"thuốc|điều trị|chữa|dùng gì|insulin|metformin|tác dụng phụ", re.I), ["TREATS"]),
    (re.compile(r"triệu chứng|dấu hiệu|biểu hiện", re.I), ["HAS_SYMPTOM"]),
    (re.compile(r"chẩn đoán|xét nghiệm|tiêu chuẩn|ngưỡng|chỉ số|bao nhiêu là", re.I), ["DIAGNOSED_BY"]),
    (re.compile(r"nguy cơ|yếu tố|ai dễ|đối tượng", re.I), ["HAS_RISK_FACTOR"]),
    (re.compile(r"biến chứng", re.I), ["HAS_COMPLICATION"]),
    (re.compile(r"\bgen\b|di truyền", re.I), ["ASSOCIATED_GENE"]),
]
REL_TEMPLATES = {
    "HAS_SYMPTOM": "{s} có triệu chứng {o}.",
    "HAS_RISK_FACTOR": "{o} là yếu tố nguy cơ của {s}.",
    "HAS_COMPLICATION": "{s} có thể gây biến chứng {o}.",
    "TREATS": "{s} được dùng điều trị {o}.",
    "SUBTYPE_OF": "{s} là một thể của {o}.",
    "DIFFERENTIAL_OF": "Cần phân biệt {s} với {o}.",
    "ASSOCIATED_GENE": "{s} liên quan đến gen {o}.",
    "AFFECTS": "{s} ảnh hưởng đến {o}.",
}

FACTS_CYPHER = """
UNWIND $ids AS eid
MATCH (e {id: eid})-[r]-(x)
WHERE type(r) IN $rels AND r.source_id IS NOT NULL
WITH DISTINCT r
WITH r, startNode(r) AS a, endNode(r) AS b
OPTIONAL MATCH (b)-[:USES_TEST]->(t:LabTest)
RETURN a.id AS s_id, coalesce(a.name_vi, a.name, a.symbol) AS s, type(r) AS rel,
       b.id AS o_id, coalesce(b.name_vi, b.name, b.symbol, b.name_en) AS o,
       r.source_id AS source_id, r.evidence AS evidence,
       b.op AS op, b.value AS value, b.value_max AS vmax, b.unit AS unit, t.name_vi AS test
ORDER BY rel, s_id, o_id
LIMIT $limit
"""

VECTOR_CYPHER = """
CALL db.index.vector.queryNodes('chunk_embedding', $k, $q_emb) YIELD node AS c, score
MATCH (c)-[:FROM_SOURCE]->(s:Source)
OPTIONAL MATCH (c)-[:MENTIONS]->(e) WHERE e.id IN $ids
RETURN c.id AS chunk_id, c.text AS text, score, s.id AS source_id, s.title AS title, s.url AS url,
       s.reviewed AS reviewed, count(e) AS n_mention
"""

GUIDED_CYPHER = """
MATCH (c:Chunk)-[:MENTIONS]->(e) WHERE e.id IN $ids
WITH c, count(e) AS n_mention ORDER BY n_mention DESC, c.id LIMIT $k
MATCH (c)-[:FROM_SOURCE]->(s:Source)
RETURN c.id AS chunk_id, c.text AS text, vector.similarity.cosine(c.embedding, $q_emb) AS score,
       s.id AS source_id, s.title AS title, s.url AS url, s.reviewed AS reviewed, n_mention
"""


class KBBackend(Protocol):
    def facts(self, ids: list[str], rels: list[str], limit: int) -> list[dict]: ...
    def vector(self, q_emb: list[float], ids: list[str], k: int) -> list[dict]: ...
    def guided(self, q_emb: list[float], ids: list[str], k: int) -> list[dict]: ...


class Neo4jBackend:
    def _run(self, q: str, **params) -> list[dict]:
        from diabetes_mas.kb.neo4j_io import driver

        with driver() as drv, drv.session() as s:
            return s.run(q, **params).data()

    def facts(self, ids, rels, limit):
        return self._run(FACTS_CYPHER, ids=ids, rels=rels, limit=limit) if ids and limit else []

    def vector(self, q_emb, ids, k):
        return self._run(VECTOR_CYPHER, q_emb=q_emb, ids=ids, k=k)

    def guided(self, q_emb, ids, k):
        return self._run(GUIDED_CYPHER, q_emb=q_emb, ids=ids, k=k) if ids else []


def _default_embed(text: str) -> list[float]:
    from diabetes_mas.rag.vector_search import embed_query

    return embed_query(text)


BACKEND: KBBackend = Neo4jBackend()
EMBED: Callable[[str], list[float]] = _default_embed


def relations_for(question: str) -> list[str]:
    rels = [r for pat, rs in INTENT_RELS if pat.search(question) for r in rs]
    return sorted(set(rels)) if rels else SEED_RELS


def _fmt_num(x) -> str:
    return f"{x:g}".replace(".", ",") if isinstance(x, (int, float)) else str(x)


def fact_text(row: dict) -> str:
    if row["rel"] == "DIAGNOSED_BY":
        if row.get("op") == "between":
            cond = f"từ {_fmt_num(row['value'])} đến {_fmt_num(row['vmax'])}"
        else:
            cond = f"{row.get('op', '')} {_fmt_num(row.get('value'))}"
        return f"Tiêu chuẩn chẩn đoán {row['s']}: {row.get('test') or row['o']} {cond} {row.get('unit') or ''}".strip() + "."
    return REL_TEMPLATES.get(row["rel"], "{s} —{rel}→ {o}.").format(**row)


def rank_chunks(rows: list[dict], k: int) -> list[ChunkHit]:
    cfg = settings()["rag"]
    w = cfg["weights"]
    best: dict[str, dict] = {}
    for r in rows:                              # gộp ứng viên từ vector + graph-guided
        if r["chunk_id"] not in best or r["score"] > best[r["chunk_id"]]["score"]:
            best[r["chunk_id"]] = r
    scored = []
    for r in best.values():
        final = (w["cosine"] * r["score"] + w["mention"] * min(r.get("n_mention", 0), 2) / 2
                 + w["reviewed"] * (1.0 if r.get("reviewed") is True else 0.0))
        scored.append((final, r))
    scored.sort(key=lambda t: (-t[0], t[1]["chunk_id"]))
    per_source: dict[str, int] = {}
    out: list[ChunkHit] = []
    for final, r in scored:
        if per_source.get(r["source_id"], 0) >= cfg["max_chunks_per_source"]:
            continue
        per_source[r["source_id"]] = per_source.get(r["source_id"], 0) + 1
        out.append(ChunkHit(chunk_id=r["chunk_id"], text=r["text"], score=round(final, 4), cosine=round(r["score"], 4),
                            source_id=r["source_id"], source_title=r.get("title") or "", url=r.get("url"),
                            reviewed=r.get("reviewed")))
        if len(out) == k:
            break
    return out


def decide_enough(n_entities: int, n_facts: int, max_cos: float) -> tuple[bool, str]:
    """FR-08 — đủ căn cứ khi MỘT trong hai:
    (a) có thực thể KB liên kết được hoặc graph fact, VÀ cosine ≥ `min_score`;
    (b) không có thực thể nhưng cosine ≥ `min_score_no_entity` (ngưỡng cao hơn).
    (b) thêm ở GĐ 8 (08_EVALUATION.md §3, §5): chỉ dùng (a) thì 7/66 câu hợp lệ không chứa thực thể
    (SGLT2, Charcot, GI, LADA…) bị từ chối nhầm (FR 0,106); thêm (b) ⇒ FR 0,03, Correct rejection giữ 0,909."""
    cfg = settings()["rag"]
    min_score, hi = cfg["min_score"], cfg.get("min_score_no_entity", 1.01)
    if n_entities == 0 and n_facts == 0:
        if max_cos >= hi:
            return True, f"không có thực thể KB, nhưng cosine tốt nhất {max_cos:.3f} ≥ min_score_no_entity {hi}"
        return False, f"không liên kết được thực thể KB nào và cosine tốt nhất {max_cos:.3f} < min_score_no_entity {hi}"
    if max_cos < min_score:
        return False, f"cosine tốt nhất {max_cos:.3f} < min_score {min_score}"
    return True, f"{n_entities} thực thể, {n_facts} fact, cosine tốt nhất {max_cos:.3f}"


@tool("T-04", "hybrid_retrieve", RetrieveIn, RetrieveOut, timeout_s=60.0,
      summarize=lambda o: f"enough={o.enough} ({o.enough_reason}); facts={len(o.facts)}; "
                          f"chunks={[c.chunk_id for c in o.chunks]}")
def hybrid_retrieve(inp: RetrieveIn) -> RetrieveOut:
    cfg = settings()["rag"]
    entities = link_entities(inp.question)                                   # bước 1
    ids = [e.node_id for e in entities]
    rels = relations_for(inp.question)
    fact_rows = BACKEND.facts(ids, rels, inp.max_facts)                       # bước 3
    seen, facts = set(), []
    for r in fact_rows:
        key = (r["s_id"], r["rel"], r["o_id"])
        if key in seen:
            continue
        seen.add(key)
        facts.append(Fact(text=fact_text(r), triple=key, source_id=r["source_id"], evidence=r.get("evidence")))
    q_emb = EMBED(inp.question)                                               # bước 2
    cands = BACKEND.vector(q_emb, ids, cfg["vector_k"]) + BACKEND.guided(q_emb, ids, cfg["guided_k"])  # bước 4–5
    chunks = rank_chunks(cands, inp.k_chunks)                                 # bước 6
    max_cos = max((c["score"] for c in cands), default=0.0)
    enough, reason = decide_enough(len(entities), len(facts), max_cos)        # bước 8
    return RetrieveOut(entities=entities, facts=facts, chunks=chunks, enough=enough, enough_reason=reason,
                       max_cosine=round(max_cos, 4))

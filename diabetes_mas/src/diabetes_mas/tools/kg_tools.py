"""T-01 kg_get_group_slots · T-02 kg_link_entities · T-03 kg_check_criteria."""
from __future__ import annotations

import time
from functools import lru_cache

from diabetes_mas.agents.slots import catalog
from diabetes_mas.config import PROCESSED_DIR, load_yaml
from diabetes_mas.kb.criteria import Criterion, check
from diabetes_mas.kb.link import build_lexicon, link_text
from diabetes_mas.kb.seed import load_seed
from diabetes_mas.tools.base import tool
from diabetes_mas.tools.schemas import (CheckCriteriaIn, CheckCriteriaOut, CriterionHit, GetGroupSlotsIn,
                                        GetGroupSlotsOut, LinkedEntity, LinkIn, LinkOut, SlotInfo)
from diabetes_mas.utils.text import read_jsonl


# ---------------- T-01 ----------------
@tool("T-01", "kg_get_group_slots", GetGroupSlotsIn, GetGroupSlotsOut,
      summarize=lambda o: f"{len(o.slots)} slot")
def kg_get_group_slots(inp: GetGroupSlotsIn) -> GetGroupSlotsOut:
    """Không gian hành động của Work-Agent. Lấy từ kho slot (KB seed + slot hồ sơ) — cùng nguồn với KB
    trong Neo4j, nhưng gồm cả slot hồ sơ vốn không phải node KB."""
    seed = load_seed()
    units = {t["id"]: t["unit"] for t in seed.nodes["LabTest"]}
    profile = load_yaml("slots.yaml")["profile"]
    by = catalog().by_id()
    out = []
    for sid in catalog().actions(inp.worker):
        s = by[sid]
        unit = units.get(sid) or profile.get(sid, {}).get("unit")
        out.append(SlotInfo(slot_id=sid, kind=s.kind, source=s.source, question_vi=s.question_vi,
                            values=list(s.values), unit=unit))
    return GetGroupSlotsOut(slots=out)


# ---------------- T-02 ----------------
@lru_cache(maxsize=1)
def _lexicon():
    seed = load_seed()
    names = {r["id"]: (r.get("name_vi") or r.get("name", "")) for rows in seed.nodes.values() for r in rows}
    return build_lexicon(seed), names


def link_entities(text: str, top_k: int = 8) -> list[LinkedEntity]:
    lex, names = _lexicon()
    found = link_text(text, lex)
    ents = [LinkedEntity(node_id=nid, label=label, name_vi=names.get(nid, nid), count=n)
            for (nid, label), n in found.items()]
    ents.sort(key=lambda e: (-e.count, e.node_id))
    return ents[:top_k]


_FT_DOWN_UNTIL = 0.0          # Neo4j lỗi ⇒ bỏ qua fulltext trong FT_RETRY_S giây (như criteria())
FT_RETRY_S = 30.0


def _fulltext_fallback(text: str, top_k: int) -> list[LinkedEntity]:
    from diabetes_mas.kb.neo4j_io import driver

    q = ("CALL db.index.fulltext.queryNodes('entity_names', $q) YIELD node, score "
         "RETURN node.id AS id, labels(node)[0] AS label, coalesce(node.name_vi, node.name) AS name, score "
         "ORDER BY score DESC LIMIT $k")
    escaped = "".join(f"\\{c}" if c in '+-&|!(){}[]^"~*?:\\/' else c for c in text)
    with driver() as drv, drv.session() as s:
        rows = s.run(q, q=escaped, k=top_k).data()
    return [LinkedEntity(node_id=r["id"], label=r["label"], name_vi=r["name"], method="fulltext",
                         score=round(r["score"], 3)) for r in rows]


@tool("T-02", "kg_link_entities", LinkIn, LinkOut,
      summarize=lambda o: ", ".join(e.node_id for e in o.entities) or "(không có)")
def kg_link_entities(inp: LinkIn) -> LinkOut:
    """Chính: từ điển (cùng mã dùng tạo quan hệ MENTIONS ⇒ hai phía khớp nhau, tất định).
    Dự phòng: fulltext Neo4j (điểm Lucene không chuẩn hóa → chỉ dùng khi từ điển không khớp gì)."""
    global _FT_DOWN_UNTIL
    ents = link_entities(inp.text, inp.top_k)
    if not ents and inp.fulltext and time.monotonic() >= _FT_DOWN_UNTIL:
        try:
            ents = _fulltext_fallback(inp.text, inp.top_k)
        except Exception:  # noqa: BLE001 — Neo4j không sẵn sàng thì trả rỗng, tạm ngừng thử FT_RETRY_S giây
            _FT_DOWN_UNTIL = time.monotonic() + FT_RETRY_S
            ents = []
    return LinkOut(entities=ents)


# ---------------- T-03 ----------------
_CRIT_CACHE: dict[str, tuple[float, list[Criterion], dict[str, dict]]] = {}

CRIT_CYPHER = """
MATCH (d:Disease)-[r:DIAGNOSED_BY]->(c:DiagnosticCriterion)-[:USES_TEST]->(t:LabTest)
OPTIONAL MATCH (s:Source {id: r.source_id})
RETURN c.id AS id, d.id AS disease_id, t.id AS test_id, c.op AS op, c.value AS value,
       c.value_max AS value_max, c.unit AS unit, c.condition AS condition,
       r.source_id AS source_id, r.evidence AS evidence, s.title AS source_title
ORDER BY id
"""


def _criteria_from_neo4j() -> tuple[list[Criterion], dict[str, dict]]:
    from diabetes_mas.kb.neo4j_io import driver

    with driver() as drv, drv.session() as s:
        rows = s.run(CRIT_CYPHER).data()
    if not rows:
        raise RuntimeError("Neo4j chưa có DiagnosticCriterion")
    return [Criterion.from_row(r) for r in rows], {r["id"]: r for r in rows}


def _criteria_from_seed() -> tuple[list[Criterion], dict[str, dict]]:
    titles = {s["id"]: s["title"] for s in read_jsonl(PROCESSED_DIR / "sources.jsonl")}
    rows = [{**r, "source_title": titles.get(r["source_id"], "")} for r in load_seed().criteria]
    return [Criterion.from_row(r) for r in rows], {r["id"]: r for r in rows}


def criteria(ttl_s: float = 300.0, retry_s: float = 30.0) -> tuple[list[Criterion], dict[str, dict], str]:
    """Tiêu chuẩn từ Neo4j (KB thật, đúng tinh thần 'agent truy vấn KB'); lỗi thì dùng seed.
    Sau một lần Neo4j lỗi, dùng seed trong `retry_s` giây rồi mới thử lại — tránh mỗi lần gọi phải chờ
    hết connection_timeout (3 s) khi Neo4j tắt (phát hiện khi huấn luyện classifier: 4 s/lần gọi)."""
    now = time.monotonic()
    for backend, ttl in (("neo4j", ttl_s), ("seed", retry_s)):
        cached = _CRIT_CACHE.get(backend)
        if cached and now - cached[0] < ttl:
            return cached[1], cached[2], backend
    try:
        crits, meta = _criteria_from_neo4j()
        _CRIT_CACHE.pop("seed", None)
        _CRIT_CACHE["neo4j"] = (now, crits, meta)
        return crits, meta, "neo4j"
    except Exception:  # noqa: BLE001
        crits, meta = _criteria_from_seed()
        _CRIT_CACHE["seed"] = (now, crits, meta)
        return crits, meta, "seed"


def check_with_meta(measurements: dict[str, float], context: dict[str, bool]) -> tuple[list[CriterionHit], str]:
    crits, meta, backend = criteria()
    hits = []
    for h in check(measurements, context, crits):
        m = meta[h.id]
        hits.append(CriterionHit(disease_id=h.disease_id, criterion_id=h.id, test_id=h.test_id,
                                 value=measurements[h.test_id], rule=h.rule(), source_id=h.source_id,
                                 source_title=m.get("source_title") or "", evidence=m.get("evidence") or ""))
    return hits, backend


@tool("T-03", "kg_check_criteria", CheckCriteriaIn, CheckCriteriaOut,
      summarize=lambda o: f"[{o.kb_backend}] " + (", ".join(h.criterion_id for h in o.hits) or "không đạt tiêu chuẩn nào"))
def kg_check_criteria(inp: CheckCriteriaIn) -> CheckCriteriaOut:
    hits, backend = check_with_meta(inp.measurements, inp.context)
    return CheckCriteriaOut(hits=hits, kb_backend=backend)

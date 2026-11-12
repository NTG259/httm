"""Sinh kb/kb_diabetes.cypher từ seed + Hetionet + chunks (+ MENTIONS).

Chạy:  python -m diabetes_mas.kb.build_cypher

- Từ chối build nếu seed không qua kiểm chứng (kb.seed.validate).
- Dùng MERGE + UNWIND theo lô → nạp lại nhiều lần không nhân bản, nhanh.
- Đầu ra tất định (sắp xếp mọi thứ) → chạy 2 lần cho file giống hệt byte.
- Embedding KHÔNG nằm trong file (quá lớn) — ghi sau bằng rag.embed_chunks.
"""
from __future__ import annotations

import json
import math
import sys
from typing import Any, Iterable

from diabetes_mas.config import KB_DIR, PROCESSED_DIR
from diabetes_mas.kb import hetionet
from diabetes_mas.kb.link import build_lexicon, link_text
from diabetes_mas.kb.seed import REL_ENDPOINTS, Seed, load_seed, split_aliases, validate
from diabetes_mas.utils.text import read_jsonl, strip_accents

OUT = KB_DIR / "kb_diabetes.cypher"
STATS = PROCESSED_DIR / "kb_stats.json"
BATCH = 100
CHUNK_BATCH = 25

_ESC = {"\\": "\\\\", "'": "\\'", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def lit(v: Any) -> str:
    """Giá trị Python → literal Cypher."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            raise ValueError("NaN/Inf không hợp lệ trong Cypher")
        return repr(v)
    if isinstance(v, str):
        return "'" + "".join(_ESC.get(c, c) for c in v) + "'"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(lit(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {lit(v[k])}" for k in sorted(v)) + "}"
    raise TypeError(f"Không hỗ trợ kiểu {type(v)}")


def _batches(rows: list[dict], size: int) -> Iterable[list[dict]]:
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def _unwind(rows: list[dict], body: str, size: int = BATCH) -> list[str]:
    return [f"UNWIND {lit(b)} AS row\n{body};" for b in _batches(rows, size)]


def _num(x: str | None) -> float | None:
    return float(x) if x not in (None, "") else None


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v not in (None, "", [])}


def _aliases_text(names: list[str]) -> str:
    forms = {n for n in names if n} | {strip_accents(n) for n in names if n}
    return " | ".join(sorted(forms))


def node_rows(seed: Seed) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for label, rows in seed.nodes.items():
        conv = []
        for r in rows:
            aliases = split_aliases(r.get("aliases"))
            props = {k: v for k, v in r.items() if k not in {"aliases", "hetionet_id", "hetionet_name"}}
            if "in_scope" in props:
                props["in_scope"] = props["in_scope"].lower() == "true"
            props["aliases"] = aliases
            props["aliases_text"] = _aliases_text([r.get("name_vi") or r.get("name", ""), *aliases])
            conv.append(_clean(props))
        out[label] = sorted(conv, key=lambda x: x["id"])
    return out


def criterion_rows(seed: Seed) -> list[dict]:
    rows = []
    for c in seed.criteria:
        rows.append(_clean({
            "id": c["id"], "disease_id": c["disease_id"], "test_id": c["test_id"], "op": c["op"],
            "value": _num(c["value"]), "value_max": _num(c["value_max"]), "unit": c["unit"],
            "condition": c["condition"], "note": c["note"], "source_id": c["source_id"], "evidence": c["evidence"],
        }))
    return sorted(rows, key=lambda x: x["id"])


def edge_rows(seed: Seed) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for e in seed.edges:
        props = _clean({
            "prob": _num(e["prob"]), "strength": e["strength"], "line": e["line"],
            "confidence": _num(e["confidence"]), "prob_source": e["prob_source"],
            "source_id": e["source_id"], "evidence": e["evidence"],
        })
        out.setdefault(e["rel"], []).append({"from": e["from_id"], "to": e["to_id"], "props": props})
    return {k: sorted(v, key=lambda x: (x["from"], x["to"])) for k, v in sorted(out.items())}


def build(seed: Seed) -> tuple[str, dict]:
    stmts: list[str] = ["// SINH TỰ ĐỘNG bởi diabetes_mas.kb.build_cypher — KHÔNG SỬA TAY", ""]
    stats: dict[str, Any] = {"nodes": {}, "relationships": {}}

    # --- Source ---
    sources = sorted(read_jsonl(PROCESSED_DIR / "sources.jsonl"), key=lambda s: s["id"])
    stmts.append("// ---- Source ----")
    stmts += _unwind([_clean(s) for s in sources], "MERGE (s:Source {id: row.id}) SET s += row")
    stats["nodes"]["Source"] = len(sources)

    # --- Seed nodes ---
    for label, rows in node_rows(seed).items():
        stmts.append(f"// ---- {label} ----")
        stmts += _unwind(rows, f"MERGE (n:{label} {{id: row.id}}) SET n += row")
        stats["nodes"][label] = len(rows)

    # --- Criteria ---
    crit = criterion_rows(seed)
    stmts.append("// ---- DiagnosticCriterion + DIAGNOSED_BY + USES_TEST ----")
    stmts += _unwind(crit, (
        "MERGE (c:DiagnosticCriterion {id: row.id})\n"
        "SET c.test_id = row.test_id, c.op = row.op, c.value = row.value, c.value_max = row.value_max,\n"
        "    c.unit = row.unit, c.condition = row.condition, c.note = row.note\n"
        "WITH c, row\n"
        "MATCH (d:Disease {id: row.disease_id}), (t:LabTest {id: row.test_id})\n"
        "MERGE (d)-[r:DIAGNOSED_BY]->(c) SET r.source_id = row.source_id, r.evidence = row.evidence\n"
        "MERGE (c)-[:USES_TEST]->(t)"
    ))
    stats["nodes"]["DiagnosticCriterion"] = len(crit)
    stats["relationships"]["DIAGNOSED_BY"] = len(crit)
    stats["relationships"]["USES_TEST"] = len(crit)

    # --- Seed edges ---
    for rel, rows in edge_rows(seed).items():
        a, b = REL_ENDPOINTS[rel]
        stmts.append(f"// ---- {rel} ----")
        stmts += _unwind(rows, (
            f"MATCH (a:{a} {{id: row.from}}), (b:{b} {{id: row.to}})\n"
            f"MERGE (a)-[r:{rel}]->(b) SET r += row.props"
        ))
        stats["relationships"][rel] = stats["relationships"].get(rel, 0) + len(rows)

    # --- Hetionet ---
    disease_map = {r["hetionet_id"]: r["id"] for r in seed.nodes["Disease"] if r.get("hetionet_id")}
    drug_map = {r["hetionet_name"].lower(): r["id"] for r in seed.nodes["Drug"] if r.get("hetionet_name")}
    het = hetionet.extract(disease_map, drug_map)
    stmts.append("// ---- Hetionet: Gene / Anatomy / Drug ----")
    stmts += _unwind(het.genes, "MERGE (n:Gene {id: row.id}) SET n += row")
    stmts += _unwind(het.anatomies, "MERGE (n:Anatomy {id: row.id}) SET n += row")
    stmts += _unwind(het.drugs, "MERGE (n:Drug {id: row.id}) SET n += row")
    stats["nodes"]["Gene"] = len(het.genes)
    stats["nodes"]["Anatomy"] = len(het.anatomies)
    stats["nodes"]["Drug"] += len(het.drugs)
    het_by_rel: dict[str, list[dict]] = {}
    for e in het.edges:
        het_by_rel.setdefault(e["rel"], []).append(e)
    het_labels = {"ASSOCIATED_GENE": ("Disease", "Gene"), "AFFECTS": ("Disease", "Anatomy"), "TREATS": ("Drug", "Disease")}
    seed_pairs = {(e["rel"], e["from_id"], e["to_id"]) for e in seed.edges}
    stats["hetionet_overlap_with_seed"] = sum((e["rel"], e["from_id"], e["to_id"]) in seed_pairs for e in het.edges)
    for rel, rows in sorted(het_by_rel.items()):
        new_rows = [r for r in rows if (rel, r["from_id"], r["to_id"]) not in seed_pairs]
        a, b = het_labels[rel]
        stmts.append(f"// ---- Hetionet {rel} ----")
        stmts += _unwind(rows, (
            f"MATCH (a:{a} {{id: row.from_id}}), (b:{b} {{id: row.to_id}})\n"
            f"MERGE (a)-[r:{rel}]->(b) SET r.source_id = coalesce(r.source_id, row.source_id),\n"
            f"    r.also_in_hetionet = true"
        ))
        stats["relationships"][rel] = stats["relationships"].get(rel, 0) + len(new_rows)

    # --- Chunks + FROM_SOURCE ---
    chunks = sorted(read_jsonl(PROCESSED_DIR / "chunks.jsonl"), key=lambda c: c["chunk_id"])
    chunk_rows = [_clean({
        "id": c["chunk_id"], "source_id": c["source_id"], "section": c["section"], "text": c["text"],
        "lang": c["lang"], "page": c["page"], "n_words": c["n_words"],
    }) for c in chunks]
    stmts.append("// ---- Chunk + FROM_SOURCE ----")
    stmts += _unwind(chunk_rows, (
        "MERGE (c:Chunk {id: row.id}) SET c += row\n"
        "WITH c, row MATCH (s:Source {id: row.source_id}) MERGE (c)-[:FROM_SOURCE]->(s)"
    ), size=CHUNK_BATCH)
    stats["nodes"]["Chunk"] = len(chunk_rows)
    stats["relationships"]["FROM_SOURCE"] = len(chunk_rows)

    # --- MENTIONS ---
    lexicon = build_lexicon(seed)
    mentions: dict[str, list[dict]] = {}
    for c in chunks:
        if c["lang"] != "vi":
            continue
        for (node_id, label), n in sorted(link_text(c["text"], lexicon).items()):
            mentions.setdefault(label, []).append({"chunk": c["chunk_id"], "node": node_id, "count": n})
    n_mentions = 0
    for label, rows in sorted(mentions.items()):
        stmts.append(f"// ---- MENTIONS → {label} ----")
        stmts += _unwind(rows, (
            f"MATCH (c:Chunk {{id: row.chunk}}), (e:{label} {{id: row.node}})\n"
            "MERGE (c)-[m:MENTIONS]->(e) SET m.count = row.count"
        ), size=500)
        n_mentions += len(rows)
    stats["relationships"]["MENTIONS"] = n_mentions
    stats["mentions_by_label"] = {k: len(v) for k, v in sorted(mentions.items())}

    stats["nodes"] = dict(sorted(stats["nodes"].items()))
    stats["relationships"] = dict(sorted(stats["relationships"].items()))
    return "\n".join(stmts) + "\n", stats


def main() -> int:
    seed = load_seed()
    errors = validate(seed)
    if errors:
        print("[build] seed chưa hợp lệ — chạy `python -m diabetes_mas.kb.seed` để xem lỗi", file=sys.stderr)
        return 1
    text, stats = build(seed)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build] {OUT} ({len(text.encode('utf-8')) / 1e6:.2f} MB)")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

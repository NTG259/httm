"""Nạp file .cypher vào Neo4j và chạy kiểm tra chất lượng KB (docs/00_KB_GRAPH.md mục 8).

Chạy:
  python -m diabetes_mas.kb.neo4j_io load      # schema.cypher + kb_diabetes.cypher
  python -m diabetes_mas.kb.neo4j_io check     # kiểm tra chất lượng + thống kê

Cách thay thế khi demo (đúng yêu cầu "nạp KB từ file .cypher"):
  cat kb/schema.cypher kb/kb_diabetes.cypher | docker compose exec -T neo4j cypher-shell -u neo4j -p <pass>
"""
from __future__ import annotations

import re
import sys

from diabetes_mas.config import KB_DIR, env

QUALITY_CHECKS = {
    "Quan hệ y khoa thiếu source_id (phải = 0)":
        "MATCH ()-[r]->() WHERE NOT type(r) IN ['MENTIONS','FROM_SOURCE','USES_TEST'] "
        "AND r.source_id IS NULL RETURN count(r) AS n",
    "Bệnh trong phạm vi không có triệu chứng (trừ KHONG_DTD, TIEN_DTD, DTD_KHAC)":
        "MATCH (d:Disease {in_scope:true}) WHERE NOT (d)-[:HAS_SYMPTOM]->() "
        "AND NOT d.id IN ['KHONG_DTD','TIEN_DTD','DTD_KHAC'] RETURN count(d) AS n",
    "Chunk chưa có embedding": "MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN count(c) AS n",
    "Chunk mồ côi (không FROM_SOURCE)": "MATCH (c:Chunk) WHERE NOT (c)-[:FROM_SOURCE]->() RETURN count(c) AS n",
}


def driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        env("NEO4J_URI", "bolt://localhost:7687"),
        auth=(env("NEO4J_USER", "neo4j"), env("NEO4J_PASSWORD", "")),
        connection_timeout=3.0,            # Neo4j tắt ⇒ tool rơi về dự phòng nhanh, không treo
        max_transaction_retry_time=2.0,
    )


def split_statements(text: str) -> list[str]:
    """Tách câu lệnh theo ';' cuối dòng, bỏ dòng chú thích //."""
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))
    return [s.strip() for s in re.split(r";\s*\n", body + "\n") if s.strip()]


def load() -> None:
    with driver() as drv, drv.session() as s:
        for fname in ("schema.cypher", "kb_diabetes.cypher"):
            stmts = split_statements((KB_DIR / fname).read_text(encoding="utf-8"))
            for i, stmt in enumerate(stmts, 1):
                s.run(stmt).consume()
                if i % 50 == 0:
                    print(f"[load] {fname}: {i}/{len(stmts)}", flush=True)
            print(f"[load] {fname}: xong {len(stmts)} câu lệnh")


def check() -> int:
    bad = 0
    with driver() as drv, drv.session() as s:
        for label, q in QUALITY_CHECKS.items():
            n = s.run(q).single()["n"]
            print(f"  {'✓' if n == 0 else '✗'} {label}: {n}")
            bad += n > 0 and "embedding" not in label
        print("[thống kê node]")
        for r in s.run("MATCH (n) RETURN labels(n)[0] AS l, count(*) AS n ORDER BY l"):
            print(f"  {r['l']}: {r['n']}")
        print("[thống kê quan hệ]")
        for r in s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY t"):
            print(f"  {r['t']}: {r['n']}")
    return 1 if bad else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "load":
        load()
        sys.exit(check())
    sys.exit(check())

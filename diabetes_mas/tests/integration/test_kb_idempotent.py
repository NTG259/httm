"""TC-10 (NFR-03): nạp lại schema.cypher + kb_diabetes.cypher không đổi số node / quan hệ, không mất embedding.

An toàn khi chạy trên KB thật: file chỉ dùng MERGE (tests/unit/test_build_cypher.py) — lần chạy đầu 2026-11-12:
3.164 node, 8.133 quan hệ trước = sau, ~18 s.
"""
import pytest

pytestmark = pytest.mark.neo4j


def _counts():
    from diabetes_mas.kb.neo4j_io import driver
    with driver() as d, d.session() as s:
        n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        r = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        e = s.run("MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN count(c) AS c").single()["c"]
    return n, r, e


def test_tc10_reloading_kb_is_idempotent():
    try:
        before = _counts()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j không chạy: {type(exc).__name__}")
    from diabetes_mas.kb.neo4j_io import load
    load()
    assert _counts() == before

import pytest

from diabetes_mas.config import PROCESSED_DIR
from diabetes_mas.kb.build_cypher import build
from diabetes_mas.kb.seed import load_seed, validate

pytestmark = pytest.mark.skipif(
    not (PROCESSED_DIR / "chunks.jsonl").exists(), reason="cần chạy ingest.clean + ingest.chunk trước"
)


def test_seed_valid_and_build_deterministic():
    seed = load_seed()
    assert validate(seed) == []
    text1, stats1 = build(seed)
    text2, stats2 = build(load_seed())
    assert text1 == text2
    assert stats1 == stats2


def test_every_medical_relationship_has_source():
    text, _ = build(load_seed())
    # mỗi lô quan hệ seed đều SET r += row.props, và mọi props đều có source_id
    for e in load_seed().edges:
        assert e["source_id"] and e["evidence"]
    assert "MERGE (a)-[r:HAS_SYMPTOM]->(b) SET r += row.props" in text

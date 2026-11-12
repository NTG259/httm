"""Test tích hợp với Neo4j thật (KB đã nạp). Tự bỏ qua nếu Neo4j không chạy.

Không tải bge-m3: dùng embedding của một chunk có sẵn làm vector truy vấn.
"""
import pytest

from diabetes_mas.tools import base, kg_tools, retrieve_tools
from diabetes_mas.tools.base import call

pytestmark = pytest.mark.neo4j


@pytest.fixture(scope="module")
def neo():
    from diabetes_mas.kb.neo4j_io import driver

    try:
        with driver() as d, d.session() as s:
            s.run("RETURN 1").consume()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j không sẵn sàng: {exc}")
    return driver


@pytest.fixture(autouse=True)
def tmp_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "LOG_DIR", tmp_path)


def _chunk_embedding(neo, chunk_id):
    with neo() as d, d.session() as s:
        return s.run("MATCH (c:Chunk {id: $id}) RETURN c.embedding AS e", id=chunk_id).single()["e"]


def test_t03_neo4j_backend_matches_seed(neo):
    kg_tools._CRIT_CACHE.clear()
    from_neo, meta_neo = kg_tools._criteria_from_neo4j()
    from_seed, _ = kg_tools._criteria_from_seed()
    assert sorted(c.id for c in from_neo) == sorted(c.id for c in from_seed)
    assert {c.id: (c.op, c.value, c.value_max, c.condition) for c in from_neo} == \
           {c.id: (c.op, c.value, c.value_max, c.condition) for c in from_seed}
    out = call("T-03", agent_id="AG-02", measurements={"GLUCOSE_DOI": 7.2})
    assert out.kb_backend == "neo4j" and out.hits[0].source_title.startswith("Hướng dẫn chẩn đoán")


def test_t04_live_facts_and_vector(neo, monkeypatch):
    emb = _chunk_embedding(neo, "VMC-0003#006")          # đoạn tiêu chuẩn chẩn đoán ĐTĐ thai kỳ
    monkeypatch.setattr(retrieve_tools, "EMBED", lambda t: emb)
    out = call("T-04", agent_id="AG-07", question="Tiêu chuẩn chẩn đoán tiểu đường thai kỳ là gì?")
    assert out.ok, out.error
    assert any(e.node_id == "DTD_THAI_KY" for e in out.entities)
    assert any("Tiêu chuẩn chẩn đoán Đái tháo đường thai kỳ" in f.text for f in out.facts)
    assert all(f.source_id for f in out.facts)
    assert out.chunks[0].chunk_id == "VMC-0003#006" and out.enough


def test_t04_gene_facts_only_when_asked(neo, monkeypatch):
    monkeypatch.setattr(retrieve_tools, "EMBED", lambda t: _chunk_embedding(neo, "VMC-0002#000"))
    plain = call("T-04", agent_id="AG-07", question="Tiểu đường tuýp 2 là gì?", max_facts=50)
    genes = call("T-04", agent_id="AG-07", question="Tiểu đường tuýp 2 có di truyền không, liên quan gen nào?")
    assert not any(f.triple[1] == "ASSOCIATED_GENE" for f in plain.facts)
    assert any(f.triple[1] == "ASSOCIATED_GENE" for f in genes.facts)


def test_t02_fulltext_fallback_when_lexicon_misses(neo):
    ents = kg_tools._fulltext_fallback("glucose", 5)
    assert ents and all(e.method == "fulltext" for e in ents)


def test_t12_local_search(neo):
    out = call("T-12", agent_id="AG-12", query="tiểu đường thai kỳ", scope="local")
    assert out.ok and out.hits and all(h.in_kb and h.source_id for h in out.hits)

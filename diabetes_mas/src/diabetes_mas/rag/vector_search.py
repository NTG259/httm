"""Vector search trên :Chunk (bước 4–6 của docs/06_KB_RAG_INTEGRATION.md), dạng tối giản.

Chạy thử:  python -m diabetes_mas.rag.vector_search "Metformin có tác dụng phụ gì?"
"""
from __future__ import annotations

import sys
from functools import lru_cache

from diabetes_mas.config import env
from diabetes_mas.kb.neo4j_io import driver

QUERY = """
CALL db.index.vector.queryNodes('chunk_embedding', $k, $q_emb) YIELD node AS c, score
MATCH (c)-[:FROM_SOURCE]->(s:Source)
OPTIONAL MATCH (c)-[:MENTIONS]->(e)
RETURN c.id AS chunk_id, score, s.publisher AS publisher, s.title AS title, s.url AS url,
       s.reviewed AS reviewed, collect(DISTINCT e.id)[..6] AS entities, left(c.text, 220) AS preview
ORDER BY score DESC
"""


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(env("EMBED_MODEL", "BAAI/bge-m3"))


def embed_query(text: str) -> list[float]:
    return _model().encode([text], normalize_embeddings=True)[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _model().encode(texts, normalize_embeddings=True)]


def search(question: str, k: int = 5) -> list[dict]:
    with driver() as drv, drv.session() as s:
        return s.run(QUERY, k=k, q_emb=embed_query(question)).data()


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "Metformin có tác dụng phụ gì?",
        "Tiêu chuẩn chẩn đoán đái tháo đường thai kỳ",
        "Cách chữa ung thư phổi",
    ]
    for q in questions:
        print(f"\n=== {q}")
        for r in search(q):
            print(f"  {r['score']:.3f} {r['chunk_id']:<16} [{r['publisher']}] {r['title'][:60]}")
            print(f"        thực thể: {r['entities']}")

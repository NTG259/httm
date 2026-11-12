"""Ghi embedding (bge-m3, 1024 chiều, đã chuẩn hóa) cho các :Chunk chưa có embedding.

Chạy:  python -m diabetes_mas.rag.embed_chunks [--batch 32]

Yêu cầu: Neo4j đã nạp KB; `pip install -e .[embed]`.
Lần đầu sẽ tải model BAAI/bge-m3 (~2,3 GB) về cache của HuggingFace.
"""
from __future__ import annotations

import argparse
import sys

from diabetes_mas.config import env
from diabetes_mas.kb.neo4j_io import driver

DIM = 1024


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args(argv)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(env("EMBED_MODEL", "BAAI/bge-m3"))
    with driver() as drv, drv.session() as s:
        rows = s.run("MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN c.id AS id, c.text AS text ORDER BY id").data()
        print(f"[embed] {len(rows)} chunk cần embedding")
        for i in range(0, len(rows), args.batch):
            part = rows[i:i + args.batch]
            vecs = model.encode([r["text"] for r in part], normalize_embeddings=True, batch_size=args.batch)
            if vecs.shape[1] != DIM:
                raise ValueError(f"Model trả {vecs.shape[1]} chiều, vector index cần {DIM}")
            s.run(
                "UNWIND $rows AS row MATCH (c:Chunk {id: row.id}) SET c.embedding = row.v",
                rows=[{"id": r["id"], "v": v.tolist()} for r, v in zip(part, vecs)],
            ).consume()
            print(f"[embed] {min(i + args.batch, len(rows))}/{len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

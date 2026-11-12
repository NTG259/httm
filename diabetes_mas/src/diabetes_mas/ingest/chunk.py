"""Chunking theo mục: docs.jsonl → data/processed/chunks.jsonl.

Chạy:  python -m diabetes_mas.ingest.chunk

- Mỗi mục (section) là đơn vị ý trọn vẹn; chỉ cắt tiếp khi dài hơn max_words.
- Cắt theo ranh giới dòng/câu, overlap ~15%.
- Mục ngắn hơn min_words được gộp với mục kế tiếp cùng nguồn.
- Chunk mang tiền tố "Tiêu đề › Mục" để embedding có ngữ cảnh.
"""
from __future__ import annotations

import re
import sys

from diabetes_mas.config import INTERIM_DIR, PROCESSED_DIR, settings
from diabetes_mas.utils.text import read_jsonl, sha256_text, write_jsonl

CHUNKS = PROCESSED_DIR / "chunks.jsonl"
_SENT = re.compile(r"(?<=[.!?;:])\s+|\n+")


def _units(text: str) -> list[str]:
    return [u.strip() for u in _SENT.split(text) if u and u.strip()]


def split_text(text: str, max_words: int, overlap_words: int) -> list[str]:
    units = _units(text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for u in units:
        n = len(u.split())
        if n > max_words:                       # câu quá dài: cắt cứng theo từ
            words = u.split()
            for i in range(0, len(words), max_words - overlap_words):
                chunks.append(" ".join(words[i:i + max_words]))
            continue
        if cur and cur_len + n > max_words:
            chunks.append("\n".join(cur))
            tail, tail_len = [], 0
            for prev in reversed(cur):          # overlap: giữ các câu cuối
                tail_len += len(prev.split())
                if tail_len > overlap_words:
                    break
                tail.insert(0, prev)
            cur, cur_len = tail, sum(len(t.split()) for t in tail)
        cur.append(u)
        cur_len += n
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _merge_short(docs: list[dict], min_words: int) -> list[dict]:
    out: list[dict] = []
    for d in docs:
        if out and out[-1]["source_id"] == d["source_id"] and len(out[-1]["text"].split()) < min_words:
            prev = out[-1]
            prev["text"] = f"{prev['text']}\n{d['section']}\n{d['text']}"
            continue
        out.append(dict(d))
    return out


def build_chunks(docs: list[dict], max_words: int, min_words: int, overlap_words: int) -> list[dict]:
    rows, seen = [], set()
    counters: dict[str, int] = {}
    for d in _merge_short(docs, min_words):
        for piece in split_text(d["text"], max_words, overlap_words):
            if len(piece.split()) < 8:
                continue
            h = sha256_text(piece)
            if h in seen:
                continue
            seen.add(h)
            idx = counters.get(d["source_id"], 0)
            counters[d["source_id"]] = idx + 1
            rows.append({
                "chunk_id": f"{d['source_id']}#{idx:03d}",
                "doc_id": d["doc_id"], "source_id": d["source_id"], "lang": d["lang"], "page": d["page"],
                "section": d["section"],
                "text": f"{d['title']} › {d['section']}\n{piece}",
                "n_words": len(piece.split()),
            })
    return rows


def main() -> int:
    cfg = settings()["chunk"]
    docs = list(read_jsonl(INTERIM_DIR / "docs.jsonl"))
    rows = build_chunks(docs, cfg["max_words"], cfg["min_words"], cfg["overlap_words"])
    n = write_jsonl(CHUNKS, rows)
    print(f"[chunk] {len(docs)} mục → {n} chunk → {CHUNKS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

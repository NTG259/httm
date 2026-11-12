from diabetes_mas.ingest.chunk import build_chunks, split_text


def test_short_text_single_chunk():
    assert split_text("Câu một. Câu hai.", max_words=50, overlap_words=5) == ["Câu một.\nCâu hai."]


def test_long_text_split_with_overlap_and_limit():
    text = " ".join(f"Câu số {i} có năm từ." for i in range(100))
    chunks = split_text(text, max_words=60, overlap_words=10)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 60 for c in chunks)
    # overlap: câu cuối chunk trước xuất hiện đầu chunk sau
    assert chunks[0].splitlines()[-1] == chunks[1].splitlines()[0]


def test_build_chunks_ids_dedup_and_prefix():
    docs = [
        {"doc_id": "S1#s00", "source_id": "S1", "title": "Bài A", "section": "Mục 1",
         "text": "Nội dung đủ dài để thành một chunk riêng biệt có nghĩa.", "lang": "vi", "page": None},
        {"doc_id": "S1#s01", "source_id": "S1", "title": "Bài A", "section": "Mục 2",
         "text": "Nội dung đủ dài để thành một chunk riêng biệt có nghĩa.", "lang": "vi", "page": None},
    ]
    rows = build_chunks(docs, max_words=300, min_words=3, overlap_words=10)
    assert [r["chunk_id"] for r in rows] == ["S1#000"]       # mục 2 trùng nội dung → bỏ
    assert rows[0]["text"].startswith("Bài A › Mục 1\n")

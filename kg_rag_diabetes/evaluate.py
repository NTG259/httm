"""Đánh giá: Recall@k cho truy xuất + tỷ lệ từ chối đúng. Chạy: python evaluate.py
Mỗi dòng eval_set.csv có cột expect (cụm từ phải xuất hiện trong chunk) và/hoặc expect_graph
(cụm từ phải xuất hiện trong sự kiện đồ thị). Gắn kỳ vọng vào NỘI DUNG chứ không vào số thứ tự chunk,
nên đổi chunk size hay cập nhật tài liệu thì bộ test vẫn dùng được."""
import csv
import json

from config import PROC
from hybrid import hybrid_context
from kg import graph_facts, load_graph


def load_rows(path="eval_set.csv"):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate(rows, corpus_text, graph_text):
    """Bước bảo trì: dòng nào có cụm kỳ vọng không còn tồn tại trong dữ liệu thì đánh dấu hỏng."""
    good, bad = [], []
    for r in rows:
        ok = (not r["expect"] or r["expect"].lower() in corpus_text) and \
             (not r["expect_graph"] or r["expect_graph"].lower() in graph_text)
        (good if ok else bad).append(r)
    return good, bad


def main(ks=(1, 3, 5)) -> None:
    rows = load_rows()
    G = load_graph()
    with open(PROC / "chunks.jsonl", encoding="utf-8") as f:
        corpus_text = " ".join(json.loads(l)["text"].lower() for l in f)
    graph_text = " ".join(f.lower() for n in G.nodes for f in graph_facts(G, n, per_relation=10**6))

    answerable = [r for r in rows if r["type"] != "no_answer"]
    unanswerable = [r for r in rows if r["type"] == "no_answer"]
    good, bad = validate(answerable, corpus_text, graph_text)
    for r in bad:
        print(f"[DÒNG HỎNG] {r['id']}: cụm kỳ vọng không còn trong dữ liệu -> sửa hoặc xóa dòng này")

    print(f"\nRecall@k trên {len(good)} câu hợp lệ")
    for k in ks:
        hits = 0
        for r in good:
            ctx = hybrid_context(r["question"], top_k=k)
            text_ok = (not r["expect"]) or any(r["expect"].lower() in d["text"].lower() for d in ctx["docs"])
            graph_ok = (not r["expect_graph"]) or any(r["expect_graph"].lower() in g.lower() for g in ctx["graph"])
            hits += text_ok and graph_ok
        print(f"  Recall@{k} = {hits}/{len(good)} = {hits / max(len(good), 1):.2f}")

    if unanswerable:
        refused = sum(not hybrid_context(r["question"])["enough"] for r in unanswerable)
        print(f"\nTừ chối đúng câu không có đáp án: {refused}/{len(unanswerable)}")


if __name__ == "__main__":
    main()

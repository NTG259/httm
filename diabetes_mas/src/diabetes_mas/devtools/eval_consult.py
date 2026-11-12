"""TC-25 — đánh giá tư vấn (RAG) trên data/eval/consult_eval.csv theo 08_EVALUATION.md §1.2, §3, §4.

Chạy (cần Neo4j; bước --answer cần thêm Ollama):
    python -m diabetes_mas.devtools.eval_consult                 # truy hồi + quét min_score (không LLM, không web)
    python -m diabetes_mas.devtools.eval_consult --answer        # thêm AG-07 trả lời: trích dẫn, độ trễ, phương thức
    python -m diabetes_mas.devtools.eval_consult --answer --research   # cho phép AG-12 tìm web khi enough = false

Chấm theo NỘI DUNG (đổi chunking không làm hỏng bộ đánh giá):
- Hit@k: một trong k chunk đầu chứa một cụm trong `expect_chunk_substr` (phân cách `|`, không phân biệt hoa thường).
- Source hit@k: một trong k chunk đầu thuộc `expect_source` (tham khảo).
- Fact hit: một graph fact chứa `expect_fact` (chỉ tính câu có cột này).
- Correct rejection: câu no_answer có enough = false.  False rejection: câu answerable có enough = false.

Ba retriever cho bảng §4:
- vector: chỉ vector index; enough = cosine ≥ min_score (không cổng thực thể).
- graph:  chỉ chunk MENTIONS thực thể (graph-guided) + facts; enough = có thực thể hoặc fact.
- hybrid: T-04 đầy đủ (06 §2); enough = decide_enough: (thực thể/fact VÀ cosine ≥ min_score)
          HOẶC cosine ≥ min_score_no_entity. Quét `min_score_no_entity`; `hybrid_gate` là luật cũ (chỉ vế đầu).

Kết quả: experiments/eval/consult.json (+ consult_rows.csv, consult_sweep.csv).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from statistics import mean

from diabetes_mas.config import PROJECT_ROOT, settings
from diabetes_mas.tools import call, retrieve_tools as rt
from diabetes_mas.tools.kg_tools import link_entities

EVAL_CSV = PROJECT_ROOT / "data" / "eval" / "consult_eval.csv"
OUT_DIR = PROJECT_ROOT / "experiments" / "eval"
KS = (1, 3, 5)
SWEEP = [round(0.60 + 0.01 * i, 2) for i in range(36)]          # thang (1+cos)/2 của Neo4j: 0,60 … 0,95


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "").lower()).strip()


def alts(cell: str) -> list[str]:
    return [norm(x) for x in (cell or "").split("|") if x.strip()]


def load_rows(path: Path = EVAL_CSV) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def hit_rank(texts: list[str], expect: list[str]) -> int | None:
    """Hạng (1-based) của chunk đầu tiên chứa một cụm kỳ vọng; None nếu không có."""
    for i, t in enumerate(texts, 1):
        nt = norm(t)
        if any(e in nt for e in expect):
            return i
    return None


def retrieve_all(question: str, k: int = 5) -> dict:
    """Chạy 3 retriever trên cùng embedding câu hỏi. Trả dict mode → {chunks, facts, max_cos, n_ent, enough_rule}."""
    cfg = settings()["rag"]
    t0 = time.perf_counter()
    out = call("T-04", agent_id="AG-07", session_id="eval-consult", question=question, k_chunks=k)
    lat = time.perf_counter() - t0
    if not out.ok:
        raise RuntimeError(f"T-04 lỗi: {out.error}")
    ents = link_entities(question)
    ids = [e.node_id for e in ents]
    q_emb = rt.EMBED(question)
    vec = rt.BACKEND.vector(q_emb, [], cfg["vector_k"])
    gui = rt.BACKEND.guided(q_emb, ids, cfg["guided_k"])
    return {
        "hybrid": {"chunks": out.chunks, "facts": [f.text for f in out.facts], "max_cos": out.max_cosine,
                   "n_ent": len(out.entities), "latency_s": lat},
        "vector": {"chunks": rt.rank_chunks(vec, k), "facts": [], "max_cos": max((r["score"] for r in vec), default=0.0),
                   "n_ent": 0},
        "graph": {"chunks": rt.rank_chunks(gui, k), "facts": [f.text for f in out.facts],
                  "max_cos": max((r["score"] for r in gui), default=0.0), "n_ent": len(ents)},
    }


def enough(mode: str, r: dict, th: float) -> bool:
    """th: ngưỡng được quét — min_score cho vector/hybrid_gate, min_score_no_entity cho hybrid."""
    grounded = r["n_ent"] > 0 or bool(r["facts"])
    if mode == "vector":
        return r["max_cos"] >= th
    if mode == "graph":
        return grounded
    if mode == "hybrid_gate":
        return grounded and r["max_cos"] >= th
    return (grounded and r["max_cos"] >= settings()["rag"]["min_score"]) or r["max_cos"] >= th


def score_mode(rows: list[dict], res: list[dict], mode: str, th: float) -> dict:
    key = "hybrid" if mode == "hybrid_gate" else mode
    ans = [(row, r[key]) for row, r in zip(rows, res) if row["type"] == "answerable"]
    noa = [(row, r[key]) for row, r in zip(rows, res) if row["type"] == "no_answer"]
    out: dict = {}
    ranks = [hit_rank([c.text for c in r["chunks"]], alts(row["expect_chunk_substr"])) for row, r in ans]
    for k in KS:
        out[f"R@{k}"] = round(mean(x is not None and x <= k for x in ranks), 4)
    out["source@5"] = round(mean(any(c.source_id in row["expect_source"].split("|") for c in r["chunks"][:5])
                                 for row, r in ans), 4)
    fq = [(row, r) for row, r in ans if row["expect_fact"].strip()]
    if fq and mode != "vector":
        out["fact_hit"] = round(mean(any(any(e in norm(f) for e in alts(row["expect_fact"])) for f in r["facts"])
                                     for row, r in fq), 4)
        out["n_fact_q"] = len(fq)
    out["correct_rejection"] = round(mean(not enough(mode, r, th) for _, r in noa), 4) if noa else None
    out["false_rejection"] = round(mean(not enough(mode, r, th) for _, r in ans), 4)
    return out


def sweep(rows: list[dict], res: list[dict]) -> list[dict]:
    table = []
    for th in SWEEP:
        row = {"min_score": th}
        for mode in ("vector", "hybrid_gate", "hybrid"):
            s = score_mode(rows, res, mode, th)
            row[f"{mode}_CR"], row[f"{mode}_FR"] = s["correct_rejection"], s["false_rejection"]
        table.append(row)
    return table


def pick_threshold(table: list[dict], mode: str = "hybrid", max_fr: float = 0.10) -> dict | None:
    """08 §3: ưu tiên False rejection ≤ 0,10; trong các ngưỡng thỏa, chọn Correct rejection cao nhất (hòa ⇒ ngưỡng thấp)."""
    ok = [r for r in table if r[f"{mode}_FR"] <= max_fr]
    return max(ok, key=lambda r: (r[f"{mode}_CR"], -r["min_score"])) if ok else None


def p95(xs: list[float]) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(0.95 * (len(xs) - 1) + 0.5))], 2) if xs else 0.0


def run_answers(rows: list[dict], use_llm: bool, research: bool) -> tuple[dict, list[dict]]:
    from diabetes_mas.agents import consult

    out_rows = []
    for row in rows:
        t0 = time.perf_counter()
        a = consult.answer(row["question"], f"eval-{row['id']}", use_llm=use_llm, allow_research=research)
        lat = time.perf_counter() - t0
        cites = [int(n) for n in consult.CITE.findall(a.text)]
        out_rows.append({"id": row["id"], "type": row["type"], "method": a.method, "enough": a.enough,
                         "latency_s": round(lat, 2), "web": bool(a.research and a.research.get("n_chunks")),
                         "cite_valid": all(1 <= n <= len(a.sources) for n in cites) and (bool(cites) or not a.enough),
                         "answer": a.text, "reference": row["reference_answer"]})
        print(f"  {row['id']} {a.method:<10} {lat:5.1f}s {a.text[:70]!r}")
    answered = [r for r in out_rows if r["enough"]]
    no_web = [r["latency_s"] for r in out_rows if not r["web"]]
    web = [r["latency_s"] for r in out_rows if r["web"]]
    noa = [r for r in out_rows if r["type"] == "no_answer"]
    ans = [r for r in out_rows if r["type"] == "answerable"]
    summary = {
        "use_llm": use_llm, "research": research, "n": len(out_rows),
        "methods": {m: sum(r["method"] == m for r in out_rows) for m in sorted({r["method"] for r in out_rows})},
        "citation_validity": round(mean(r["cite_valid"] for r in answered), 4) if answered else None,
        "correct_rejection": round(mean(not r["enough"] for r in noa), 4) if noa else None,
        "false_rejection": round(mean(not r["enough"] for r in ans), 4) if ans else None,
        "latency_p95_s": p95(no_web), "latency_p95_web_s": p95(web) if web else None,
        "n_web": len(web),
    }
    return summary, out_rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=EVAL_CSV)
    ap.add_argument("--answer", action="store_true", help="chạy thêm AG-07 trả lời (Ollama nếu không --no-llm)")
    ap.add_argument("--no-llm", action="store_true", help="AG-07 dùng câu trả lời trích xuất")
    ap.add_argument("--research", action="store_true", help="cho phép AG-12 tìm web khi enough = false")
    args = ap.parse_args(argv)

    rows = load_rows(args.csv)
    cfg = settings()["rag"]
    th_of = {"vector": cfg["min_score"], "graph": cfg["min_score"], "hybrid_gate": cfg["min_score"],
             "hybrid": cfg.get("min_score_no_entity", 1.01)}
    th = th_of["hybrid"]
    print(f"[consult] {len(rows)} câu ({sum(r['type'] == 'answerable' for r in rows)} answerable), "
          f"min_score = {cfg['min_score']}, min_score_no_entity = {th}")
    res = []
    for row in rows:
        res.append(retrieve_all(row["question"]))
    report = {"n": len(rows), "min_score": cfg["min_score"], "min_score_no_entity": th,
              "retrievers": {m: score_mode(rows, res, m, th_of[m]) for m in ("vector", "graph", "hybrid_gate", "hybrid")},
              "latency_retrieve_p95_s": p95([r["hybrid"]["latency_s"] for r in res])}
    table = sweep(rows, res)
    report["sweep_pick_hybrid"] = pick_threshold(table, "hybrid")
    report["sweep_pick_vector"] = pick_threshold(table, "vector")
    report["sweep_pick_hybrid_gate"] = pick_threshold(table, "hybrid_gate")
    report["misses_hybrid_R@5"] = [row["id"] for row, r in zip(rows, res) if row["type"] == "answerable"
                                   and not (hit_rank([c.text for c in r["hybrid"]["chunks"]],
                                                     alts(row["expect_chunk_substr"])) or 99) <= 5]
    report["false_accept_hybrid"] = [row["id"] for row, r in zip(rows, res) if row["type"] == "no_answer"
                                     and enough("hybrid", r["hybrid"], th)]
    for m, s in report["retrievers"].items():
        print(f"[consult] {m:<7} {s}")
    print(f"[consult] chọn min_score_no_entity (hybrid): {report['sweep_pick_hybrid']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "consult_rows.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "n_ent", "n_facts", "max_cos", "enough_now", "hit_rank_hybrid", "hit_rank_vector",
                    "top_sources"])
        for row, r in zip(rows, res):
            h = r["hybrid"]
            w.writerow([row["id"], row["type"], h["n_ent"], len(h["facts"]), h["max_cos"], enough("hybrid", h, th),
                        hit_rank([c.text for c in h["chunks"]], alts(row["expect_chunk_substr"])),
                        hit_rank([c.text for c in r["vector"]["chunks"]], alts(row["expect_chunk_substr"])),
                        "|".join(c.source_id for c in h["chunks"])])
    with open(OUT_DIR / "consult_sweep.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    if args.answer:
        summary, ans_rows = run_answers(rows, use_llm=not args.no_llm, research=args.research)
        report["answers"] = summary
        (OUT_DIR / "consult_answers.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in ans_rows) + "\n", encoding="utf-8")
        print(f"[consult] trả lời: {summary}")
    (OUT_DIR / "consult.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[consult] → {(OUT_DIR / 'consult.json').relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

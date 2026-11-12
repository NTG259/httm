"""TC-25: bộ consult_eval.csv hợp lệ + logic chấm của devtools/eval_consult (không cần Neo4j)."""
import json
from types import SimpleNamespace

from diabetes_mas.config import PROJECT_ROOT
from diabetes_mas.devtools import eval_consult as ec

CHUNKS = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def test_dataset_size_and_columns():
    rows = ec.load_rows()
    ans = [r for r in rows if r["type"] == "answerable"]
    noa = [r for r in rows if r["type"] == "no_answer"]
    assert len(ans) >= 60 and len(noa) >= 20 and len(rows) == len(ans) + len(noa)   # 08 §1.2
    assert len({r["id"] for r in rows}) == len(rows)
    for r in ans:
        assert r["expect_chunk_substr"].strip() and r["expect_source"].strip() and r["reference_answer"].strip(), r["id"]


def test_every_answerable_has_evidence_in_expected_source():
    """Mỗi câu trong phạm vi có ít nhất một chunk thuộc expect_source chứa cụm kỳ vọng — chấm được."""
    chunks = [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in ec.load_rows():
        if r["type"] != "answerable":
            continue
        subs, srcs = ec.alts(r["expect_chunk_substr"]), set(r["expect_source"].split("|"))
        assert any(c["source_id"] in srcs and any(s in ec.norm(c["text"]) for s in subs) for c in chunks), r["id"]


def _c(text, src="S"):
    return SimpleNamespace(text=text, source_id=src)


def test_scoring_hit_rank_and_rejection():
    rows = [{"type": "answerable", "expect_chunk_substr": "126 mg/dL|7 mmol", "expect_source": "A", "expect_fact": "Tiêu chuẩn"},
            {"type": "answerable", "expect_chunk_substr": "sầu riêng", "expect_source": "B", "expect_fact": ""},
            {"type": "no_answer", "expect_chunk_substr": "", "expect_source": "", "expect_fact": ""}]
    mk = lambda chunks, facts, cos, n: {"chunks": chunks, "facts": facts, "max_cos": cos, "n_ent": n}  # noqa: E731
    res = [{"hybrid": mk([_c("x"), _c("Glucose ≥ 126  MG/DL", "A")], ["Tiêu chuẩn chẩn đoán ĐTĐ"], 0.85, 1)},
           {"hybrid": mk([_c("y")], [], 0.70, 1)},
           {"hybrid": mk([_c("z")], [], 0.90, 0)}]
    assert ec.hit_rank([c.text for c in res[0]["hybrid"]["chunks"]], ec.alts(rows[0]["expect_chunk_substr"])) == 2
    s = ec.score_mode(rows, res, "hybrid", 0.95)     # th = min_score_no_entity; min_score (0,75) từ settings
    assert s["R@1"] == 0 and s["R@3"] == 0.5 and s["source@5"] == 0.5 and s["fact_hit"] == 1.0
    assert s["correct_rejection"] == 1.0            # không có thực thể và cosine 0,90 < 0,95 ⇒ từ chối
    assert s["false_rejection"] == 0.5              # câu 2: cosine 0,70 < min_score 0,75
    assert ec.score_mode(rows, res, "hybrid", 0.85)["correct_rejection"] == 0.0   # 0,90 ≥ 0,85 ⇒ nhận
    assert ec.score_mode(rows, res, "hybrid_gate", 0.75)["correct_rejection"] == 1.0
    table = [{"min_score": 0.7, "hybrid_CR": 0.5, "hybrid_FR": 0.0}, {"min_score": 0.8, "hybrid_CR": 0.9, "hybrid_FR": 0.3}]
    assert ec.pick_threshold(table)["min_score"] == 0.7

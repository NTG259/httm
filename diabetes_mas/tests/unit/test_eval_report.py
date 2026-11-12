"""TC-45: báo cáo FR-21 đủ 6 nhóm; nhóm thiếu nguồn bị đánh dấu, không bịa số."""
import json

from diabetes_mas.devtools import eval_report as er


def test_tool_log_stats_error_rate(tmp_path):
    rows = [{"ts": "2026-11-12T08:00:00+00:00", "session_id": "a", "tool": "T-04", "ok": True, "latency_ms": 100},
            {"ts": "2026-11-12T08:00:01+00:00", "session_id": "a", "tool": "T-04", "ok": False, "latency_ms": 4000,
             "result_summary": "kb_unavailable: down"},
            {"ts": "2026-11-12T08:00:02+00:00", "session_id": "b", "tool": "T-08", "ok": True, "latency_ms": 2}]
    (tmp_path / "x.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    s = er.tool_log_stats(None, tmp_path)
    assert s["n_calls"] == 3 and s["n_sessions"] == 2 and s["error_rate"] == round(1 / 3, 4)
    assert s["error_kinds"] == {"kb_unavailable": 1} and s["per_tool"]["T-04"]["error_rate"] == 0.5


def test_missing_sources_are_reported_not_invented(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "EXP", tmp_path)
    monkeypatch.setattr(er, "EVAL", tmp_path / "eval")
    monkeypatch.setattr(er, "LOGS", tmp_path / "logs")
    rep = er.build(None)
    assert rep["missing"] == list(er.GROUPS)
    assert "Chưa có số liệu" in er.to_markdown(rep)


def test_faithfulness_sample_counts(tmp_path):
    f = tmp_path / "f.csv"
    f.write_text("id,grade,content_supported,citation_correct,note\nA,ok,yes,yes,\nB,minor,yes,partial,x\n"
                 "C,unfaithful,partial,no,y\nD,ok,yes,yes,\n", encoding="utf-8")
    s = er.faithfulness_sample(f)
    assert s == {"n": 4, "grades": {"ok": 2, "minor": 1, "unfaithful": 1}, "content_supported": 0.75,
                 "citation_correct": 0.5}
    assert er.faithfulness_sample(tmp_path / "missing.csv") is None

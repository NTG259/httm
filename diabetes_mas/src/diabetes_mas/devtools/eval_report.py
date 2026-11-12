"""TC-45 / FR-21 — báo cáo đánh giá tổng hợp 6 nhóm số liệu, gom từ kết quả đã chạy (không chạy lại mô hình).

Chạy:  python -m diabetes_mas.devtools.eval_report [--since "2026-11-12 15:00"]

| Nhóm FR-21                  | Nguồn                                                                 |
|-----------------------------|-----------------------------------------------------------------------|
| 1. Deep Learning (AG-13)    | experiments/best/dl/metrics.json                                      |
| 2. Truy xuất                | experiments/eval/consult.json → retrievers (eval_consult)             |
| 3. Độ đúng câu trả lời      | consult.json → answers (trích dẫn, từ chối) + classifier_metrics.json |
| 4. Điều phối                | experiments/best/hrl/metrics.json + experiments/eval/e2e*.json        |
| 5. Thời gian phản hồi       | consult.json (p95 truy hồi / trả lời) + log T-10 (latency_ms)          |
| 6. Độ tin cậy               | log T-10 trong logs/agent_runs/ từ `--since`: tỉ lệ tool call ok=false |

Nhóm thiếu file nguồn được đánh dấu "chưa chạy" — báo cáo không tự bịa số.
Kết quả: experiments/eval/report.json + report.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from diabetes_mas.config import PROJECT_ROOT

EXP = PROJECT_ROOT / "experiments"
EVAL = EXP / "eval"
LOGS = PROJECT_ROOT / "logs" / "agent_runs"
GROUPS = ("dl", "retrieval", "answer", "orchestration", "latency", "reliability")


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _p95(xs: list[float]) -> float | None:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.95 * (len(xs) - 1) + 0.5))] if xs else None


def tool_log_stats(since: datetime | None, log_dir: Path | None = None) -> dict | None:
    """Tỉ lệ lỗi và độ trễ theo tool từ log T-10 (mỗi dòng một tool call)."""
    log_dir = LOGS if log_dir is None else log_dir
    calls = []
    for f in sorted(log_dir.glob("*.jsonl")) if log_dir.exists() else []:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ts = datetime.fromisoformat(r["ts"])
            if since is None or ts >= since:
                calls.append(r)
    if not calls:
        return None
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for r in calls:
        by_tool[r["tool"]].append(r)
    kinds = Counter((r.get("result_summary") or "").split(":")[0] for r in calls if not r.get("ok", True))
    return {
        "n_calls": len(calls), "n_sessions": len({r.get("session_id") for r in calls}),
        "error_rate": round(sum(not r.get("ok", True) for r in calls) / len(calls), 4),
        "error_kinds": dict(kinds.most_common()),
        "per_tool": {t: {"n": len(rs), "error_rate": round(sum(not r.get("ok", True) for r in rs) / len(rs), 4),
                         "latency_p95_ms": _p95([r.get("latency_ms", 0) for r in rs])}
                     for t, rs in sorted(by_tool.items())},
    }


def faithfulness_sample(path: Path) -> dict | None:
    """Chấm tay MẪU (experiments/eval/faithfulness_sample.csv: id, grade ok|minor|unfaithful, content_supported,
    citation_correct). Trả số đếm — ghi rõ là mẫu, không phải toàn bộ."""
    import csv

    if not path.exists():
        return None
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    return {"n": n, "grades": dict(Counter(r["grade"] for r in rows)),
            "content_supported": round(sum(r["content_supported"] == "yes" for r in rows) / n, 4) if n else None,
            "citation_correct": round(sum(r["citation_correct"] == "yes" for r in rows) / n, 4) if n else None}


def build(since: datetime | None) -> dict:
    dl = _load(EXP / "best" / "dl" / "metrics.json")
    cls = _load(EXP / "best" / "classifier_metrics.json")
    hrl = _load(EXP / "best" / "hrl" / "metrics.json")
    consult = _load(EVAL / "consult.json")
    e2e = _load(EVAL / "e2e.json")
    e2e_llm = _load(EVAL / "e2e_llm.json")
    diag = _load(EVAL / "diag_final.json")              # mọi chính sách đo CÙNG cấu hình (08 §4.1 bảng chính)
    cls_cmp = _load(EVAL / "classifiers.json")          # NB vs MLP (devtools/compare_classifiers)
    faith = faithfulness_sample(EVAL / "faithfulness_sample.csv")
    logs = tool_log_stats(since)
    rep: dict = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "since": since.isoformat() if since else None}
    rep["dl"] = None if dl is None else {
        "full_test": dl["full"]["test"], "no_labs_test": dl["no_labs"]["test"], "D1_rules_test": dl["baseline_D1_rules_test"]}
    rep["retrieval"] = None if consult is None else {
        "n": consult["n"], "min_score": consult["min_score"], "min_score_no_entity": consult.get("min_score_no_entity"),
        "retrievers": consult["retrievers"]}
    rep["answer"] = None if consult is None and cls is None else {
        "consult": (consult or {}).get("answers"),
        "faithfulness_sample": faith,
        "classifier_compare": None if cls_cmp is None else {k: v for k, v in cls_cmp.items() if k != "n_test"},
        "classifier_test": None if cls is None else {k: {"acc": v["acc"], "macro_f1": v["macro_f1"]}
                                                     for k, v in cls.items() if k.startswith("test_")}}
    rep["orchestration"] = None if hrl is None and e2e is None else {
        "hrl": None if hrl is None else {"run_id": hrl["run_id"], "test": hrl["hrl"]["test"],
                                         "B0_test": hrl["baseline_B0"]["test"]},
        "same_conditions": diag,
        "e2e": None if e2e is None else {"ts": e2e["ts"], "policy": e2e["policy"], **e2e["summary"]},
        "e2e_llm": None if e2e_llm is None else {"ts": e2e_llm["ts"], **e2e_llm["summary"]}}
    rep["latency"] = None if consult is None and logs is None else {
        "retrieve_p95_s": (consult or {}).get("latency_retrieve_p95_s"),
        "answer_p95_s": ((consult or {}).get("answers") or {}).get("latency_p95_s"),
        "answer_p95_web_s": ((consult or {}).get("answers") or {}).get("latency_p95_web_s"),
        "tool_p95_ms": None if logs is None else {t: v["latency_p95_ms"] for t, v in logs["per_tool"].items()}}
    rep["reliability"] = logs
    rep["missing"] = [g for g in GROUPS if rep.get(g) is None]
    return rep


def to_markdown(rep: dict) -> str:
    def f(x, nd=3):
        if x is None:
            return "—"
        num = isinstance(x, (int, float)) and not isinstance(x, bool)
        return f"{float(x):.{nd}f}".replace(".", ",") if num else str(x)

    out = [f"# Báo cáo đánh giá (FR-21) — {rep['generated']}", ""]
    if rep["missing"]:
        out += [f"> Chưa có số liệu: {', '.join(rep['missing'])}", ""]
    if rep["dl"]:
        d = rep["dl"]
        out += ["## 1. Deep Learning (AG-13, test)", "", "| Mô hình | PR-AUC | ROC-AUC | Recall | Precision | F1 |",
                "|---|---|---|---|---|---|",
                f"| D3 MLP đủ đặc trưng | {f(d['full_test']['pr_auc'])} | {f(d['full_test']['roc_auc'])} | "
                f"{f(d['full_test']['recall'])} | {f(d['full_test']['precision'])} | {f(d['full_test']['f1'])} |",
                f"| D3-mask che xét nghiệm | {f(d['no_labs_test']['pr_auc'])} | {f(d['no_labs_test']['roc_auc'])} | "
                f"{f(d['no_labs_test']['recall'])} | {f(d['no_labs_test']['precision'])} | {f(d['no_labs_test']['f1'])} |",
                f"| D1 luật ngưỡng | — | — | {f(d['D1_rules_test']['recall'])} | {f(d['D1_rules_test']['precision'])} | "
                f"{f(d['D1_rules_test']['f1'])} |", ""]
    if rep["retrieval"]:
        r = rep["retrieval"]
        out += [f"## 2. Truy xuất ({r['n']} câu; min_score {r['min_score']}, min_score_no_entity {r['min_score_no_entity']})",
                "", "| Retriever | R@1 | R@3 | R@5 | Fact hit | Correct rej. | False rej. |", "|---|---|---|---|---|---|---|"]
        for m, s in r["retrievers"].items():
            out.append(f"| {m} | {f(s['R@1'])} | {f(s['R@3'])} | {f(s['R@5'])} | {f(s.get('fact_hit'))} | "
                       f"{f(s['correct_rejection'])} | {f(s['false_rejection'])} |")
        out.append("")
    if rep["answer"]:
        a = rep["answer"]
        out += ["## 3. Độ đúng câu trả lời", ""]
        if a["consult"]:
            c = a["consult"]
            out += [f"- Tư vấn (AG-07, LLM = {c['use_llm']}): trích dẫn hợp lệ {f(c['citation_validity'])}; "
                    f"phương thức {c['methods']}; Correct rej. {f(c['correct_rejection'])}; False rej. {f(c['false_rejection'])}."]
        if a.get("faithfulness_sample"):
            fs = a["faithfulness_sample"]
            out += [f"- Faithfulness — **chấm tay MẪU {fs['n']} câu**: nội dung có căn cứ {f(fs['content_supported'])}; "
                    f"trích dẫn đúng nguồn {f(fs['citation_correct'])}; mức {fs['grades']}."]
        if a.get("classifier_compare"):
            out += ["", "| Classifier (test) | che slot + luật | đủ slot + luật |", "|---|---|---|"]
            for k, v in a["classifier_compare"].items():
                out.append(f"| {k} | {f(v['masked_rules']['acc'])} / F1 {f(v['masked_rules']['macro_f1'])} | "
                           f"{f(v['full_rules']['acc'])} / F1 {f(v['full_rules']['macro_f1'])} |")
            out.append("")
        if a["classifier_test"]:
            out += ["- Classifier AG-02/03 (test): " + "; ".join(
                f"{k[5:]} acc {f(v['acc'])} / macro-F1 {f(v['macro_f1'])}" for k, v in a["classifier_test"].items()) + "."]
        out.append("")
    if rep["orchestration"]:
        o = rep["orchestration"]
        out += ["## 4. Điều phối (hội thoại chẩn đoán)", ""]
        if o.get("same_conditions"):
            out += ["Cùng cấu hình (`diag_final.json`, môi trường huấn luyện, 225 ca, 1 seed):", "",
                    "| Chính sách | Val acc | Val lượt | Test acc | Test lượt |", "|---|---|---|---|---|"]
            for k, v in o["same_conditions"].items():
                out.append(f"| {k} | {f(v['val']['success'])} | {f(v['val']['avg_turns'], 2)} | "
                           f"{f(v['test']['success'])} | {f(v['test']['avg_turns'], 2)} |")
            out += ["", "Số ghi lúc huấn luyện (luật điều phối tại thời điểm train) và qua toàn hệ thống:", ""]
        out += ["| Cấu hình | Success / acc | Số lượt TB | Ghi chú |", "|---|---|---|---|"]
        if o["hrl"]:
            out += [f"| HRL lúc train (môi trường, test) | {f(o['hrl']['test']['success'])} | {f(o['hrl']['test']['avg_turns'], 2)} | "
                    f"{o['hrl']['run_id']} |",
                    f"| B0' luật lúc train (môi trường, test) | {f(o['hrl']['B0_test']['success'])} | "
                    f"{f(o['hrl']['B0_test']['avg_turns'], 2)} | baseline |"]
        for key, label in (("e2e", "E2E câu mẫu"), ("e2e_llm", "E2E Qwen (xem ts: có thể đo trước sửa nlu.DURATION)")):
            if o[key]:
                e = o[key]
                out.append(f"| {label} ({e['n']} ca) | {f(e['accuracy'])} | {f(e['avg_turns'], 2)} | "
                           f"vượt T_max {e['over_T_max']}, hỏi lặp {e['with_repeats']} |")
        out.append("")
    if rep["latency"]:
        lt = rep["latency"]
        out += ["## 5. Thời gian phản hồi", "", f"- Truy hồi T-04 p95: {f(lt['retrieve_p95_s'], 2)} s; "
                f"trả lời AG-07 p95 (không web): {f(lt['answer_p95_s'], 2)} s; có web: {f(lt['answer_p95_web_s'], 2)} s."]
        if lt["tool_p95_ms"]:
            out.append("- p95 theo tool (ms): " + ", ".join(f"{t} {v}" for t, v in lt["tool_p95_ms"].items()) + ".")
        out.append("")
    if rep["reliability"]:
        rl = rep["reliability"]
        out += ["## 6. Độ tin cậy (log T-10)", "",
                f"- {rl['n_calls']} tool call / {rl['n_sessions']} phiên; tỉ lệ lỗi {f(rl['error_rate'])}; "
                f"loại lỗi: {rl['error_kinds'] or 'không có'}.",
                "- Tool có lỗi: " + (", ".join(f"{t} {f(v['error_rate'])}" for t, v in rl["per_tool"].items()
                                              if v["error_rate"]) or "không có") + ".", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="chỉ tính log T-10 từ thời điểm này (giờ máy, 'YYYY-MM-DD HH:MM')")
    args = ap.parse_args(argv)
    since = datetime.strptime(args.since, "%Y-%m-%d %H:%M").astimezone(timezone.utc) if args.since else None
    rep = build(since)
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    (EVAL / "report.md").write_text(to_markdown(rep), encoding="utf-8")
    print(to_markdown(rep))
    print(f"[report] → {(EVAL / 'report.md').relative_to(PROJECT_ROOT)}")
    return 0 if not rep["missing"] else 1


if __name__ == "__main__":
    sys.exit(main())

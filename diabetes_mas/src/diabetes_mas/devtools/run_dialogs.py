"""Chạy hội thoại đầu–cuối (E2E): AG-10 Patient Simulator trả lời qua TOÀN BỘ hệ thống (07_TEST_PLAN.md §2.3).

Khác với môi trường huấn luyện (`rl/env.py`, slot đi thẳng vào state), ở đây câu trả lời là văn bản và đi qua
NLU → Router → Tracker → Main/Worker → NLG → Safety, đúng như người dùng thật — đo được lỗi của lớp ngôn ngữ.

  python -m diabetes_mas.devtools.run_dialogs --n 200            # 200 ca đầu của test.jsonl, câu mẫu (không LLM)
  python -m diabetes_mas.devtools.run_dialogs --n 20 --llm       # NLU/NLG/simulator dùng Qwen
  python -m diabetes_mas.devtools.run_dialogs --case C-00004 --keep-logs
  python -m diabetes_mas.devtools.run_dialogs --n 20 --llm --resume   # ghi từng ca, chạy lại thì làm tiếp
                                                                      # (máy 16 GB RAM có thể bị dừng giữa chừng)

Kết quả: experiments/eval/e2e[_llm].json. Log tool call (T-10) mỗi phiên: logs/agent_runs/ khi --keep-logs.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from diabetes_mas.agents import graph
from diabetes_mas.agents.slots import catalog
from diabetes_mas.config import CASES_DIR, PROJECT_ROOT, load_yaml
from diabetes_mas.simulator.patient import speak
from diabetes_mas.tools import base
from diabetes_mas.utils.text import read_jsonl

OPENING = "Tôi muốn kiểm tra xem mình có bị tiểu đường không."
OUT_DIR = PROJECT_ROOT / "experiments" / "eval"


def opening_text(case: dict) -> str:
    """Câu mở đầu: lời đề nghị + các triệu chứng/yếu tố mà bệnh nhân tự kể (slot `explicit` = yes)."""
    names = {s.id: (s.question_vi or s.id) for s in catalog().slots}
    from diabetes_mas.tools.kg_tools import _lexicon
    vi = _lexicon()[1]
    told = [vi.get(k, names.get(k, k)).split("(")[0].strip().lower()
            for k, v in case.get("explicit", {}).items() if v.get("status") == "yes" and v.get("value") is None]
    return OPENING + (f" Dạo này tôi bị {', '.join(told)}." if told else "")


def run_case(case: dict, use_llm: bool = False, max_msgs: int = 40, opening: str | None = None) -> dict:
    sid = graph.new_session()
    first = opening or opening_text(case)
    transcript = [("user", first)]
    out = graph.handle(sid, first, use_llm=use_llm)
    calls = list(out["tool_calls"])
    asked: list[str] = []
    for _ in range(max_msgs):
        transcript.append(("system", out["reply"]))
        if out["stage"] != "asking" or not out["asking_slot"]:
            break
        slot = out["asking_slot"]
        asked.append(slot)
        ans = speak(case, slot, out["reply"], use_llm=use_llm)
        transcript.append(("user", ans))
        out = graph.handle(sid, ans, use_llm=use_llm)
        calls += out["tool_calls"]
    st = out["state"]
    return {"case_id": case["case_id"], "disease": case["disease"], "stage": out["stage"],
            "top1": (out["outcome"] or {}).get("top1"), "correct": (out["outcome"] or {}).get("top1") == case["disease"],
            "turns": st["turn"], "T_max": st["T_max"], "n_messages": len(asked) + 1, "asked": asked,
            "repeats": [s for s, n in Counter(asked).items() if n > 1], "risk_dl": st["risk_dl"],
            "dl_conflict": st["dl_conflict"], "session_id": sid, "reply": out["reply"], "outcome": out["outcome"],
            "slots": st["slots"], "tool_calls": calls, "transcript": transcript}


SAMPLES = {   # 01_REQUIREMENTS.md §11 — log mẫu để trình chiếu (07_TEST_PLAN.md §5)
    "story1-chan-doan": ("Dạo này tôi khát nước suốt, đi tiểu nhiều, sụt 3 kg trong tháng.",
                         dict(tuoi=55, bmi=27.0, gioi_tinh="nam", uong_nhieu="yes", tieu_nhieu="yes", sut_can="yes",
                              GLUCOSE_DOI=8.4, HBA1C=7.6, tang_huyet_ap="yes")),
    "story3-tu-van": ("Metformin có tác dụng phụ gì?", None),
    "story5-cap-cuu": ("Tôi đường huyết 25 mmol/L, đang buồn nôn, thở dốc", None),
}


def sample_case(slots: dict) -> dict:
    implicit = {s.id: {"status": "no" if s.kind == "binary" else "unknown", "value": None}
                for s in catalog().slots if s.source != "derived"}
    for k, v in slots.items():
        implicit[k] = {"status": v, "value": None} if v in ("yes", "no", "unknown") else {"status": "yes", "value": v}
    return {"case_id": "sample", "disease": "?", "explicit": {}, "implicit": implicit}


def run_samples(use_llm: bool = True) -> list[Path]:
    """Chạy 3 hội thoại mẫu với session_id cố định; log T-10 ở logs/agent_runs/, bản Markdown ở .../samples/."""
    from diabetes_mas.devtools.render_log import render

    out_dir = base.LOG_DIR / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, (opening, slots) in SAMPLES.items():
        sid = f"demo-{name}"
        log = base.LOG_DIR / f"{time.strftime('%Y-%m-%d')}_{sid}.jsonl"
        log.unlink(missing_ok=True)
        graph._CONVS.pop(sid, None)
        from diabetes_mas.tools import session
        session.reset(sid)
        transcript = [("Người dùng", opening)]
        out = graph.handle(sid, opening, use_llm=use_llm)
        case = sample_case(slots) if slots else None
        for _ in range(40):
            transcript.append(("Hệ thống", out["reply"]))
            if out["stage"] != "asking" or case is None:
                break
            ans = speak(case, out["asking_slot"], out["reply"], use_llm=use_llm)
            transcript.append(("Người dùng", ans))
            out = graph.handle(sid, ans, use_llm=use_llm)
        rows = list(read_jsonl(log))
        md = [f"# Hội thoại mẫu `{sid}`", "", f"- Ngày: {time.strftime('%Y-%m-%d %H:%M')} · LLM: {use_llm} · "
              f"policy: {graph.POLICY().name} · kết thúc: `{out['stage']}`", f"- Log gốc: `{log.name}`", "",
              "## Lời thoại", ""]
        md += [f"**{who}:** {text}".replace("\n", "  \n") + "\n" for who, text in transcript]
        md += ["## Tool call", "", render(rows), ""]
        dst = out_dir / f"{sid}.md"
        dst.write_text("\n".join(md), encoding="utf-8")
        paths.append(dst)
        print(f"[samples] {sid}: {out['stage']}, {len(rows)} tool call → {dst.relative_to(PROJECT_ROOT)}")
    return paths


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    concluded = [r for r in rows if r["stage"] == "concluded"]
    return {"n": n, "concluded": len(concluded), "accuracy": round(sum(r["correct"] for r in rows) / n, 4),
            "avg_turns": round(sum(r["turns"] for r in rows) / n, 3), "max_turns": max(r["turns"] for r in rows),
            "over_T_max": sum(r["turns"] > r["T_max"] for r in rows),
            "with_repeats": sum(bool(r["repeats"]) for r in rows),
            "stages": dict(Counter(r["stage"] for r in rows)),
            "avg_messages": round(sum(r["n_messages"] for r in rows) / n, 2)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--case", default="")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--keep-logs", action="store_true", help="giữ log T-10 trong logs/agent_runs/")
    ap.add_argument("--samples", action="store_true", help="chạy 3 hội thoại mẫu (01 §11) để trình chiếu")
    ap.add_argument("--resume", action="store_true",
                    help="ghi từng ca vào e2e[_llm].partial.jsonl; chạy lại thì bỏ qua ca đã xong")
    args = ap.parse_args(argv)
    if args.samples:
        run_samples(use_llm=args.llm)
        return 0
    cases = list(read_jsonl(CASES_DIR / f"{args.split}.jsonl"))
    cases = [c for c in cases if c["case_id"] == args.case] if args.case else cases[: args.n]
    if not cases:
        print("[e2e] không có ca nào", file=sys.stderr)
        return 2
    tmp = None
    if not args.keep_logs:
        tmp = tempfile.TemporaryDirectory()
        base.LOG_DIR = Path(tmp.name)
    t0 = time.time()
    keep = lambda r: {k: v for k, v in r.items()  # noqa: E731
                      if k not in ("transcript", "tool_calls", "slots", "reply", "outcome")}
    partial = OUT_DIR / f"e2e{'_llm' if args.llm else ''}.partial.jsonl"
    done: dict[str, dict] = {}
    if args.resume and partial.exists():
        done = {r["case_id"]: r for r in read_jsonl(partial)}
        print(f"[e2e] tiếp tục: {len(done)}/{len(cases)} ca đã xong ({partial.name})", flush=True)
    rows = []
    for i, c in enumerate(cases, 1):
        if c["case_id"] in done:
            rows.append(done[c["case_id"]])
            continue
        tc = time.time()
        r = run_case(c, use_llm=args.llm)
        r["seconds"] = round(time.time() - tc, 1)
        rows.append(r)
        if args.resume:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(partial, "a", encoding="utf-8") as f:
                f.write(json.dumps(keep(r), ensure_ascii=False) + "\n")
            print(f"[e2e] {i}/{len(cases)} {c['case_id']} {'đúng' if r['correct'] else 'sai'} "
                  f"{r['turns']} lượt {r['seconds']} s", flush=True)
        elif i % 20 == 0:
            print(f"[e2e] {i}/{len(cases)} acc={sum(x['correct'] for x in rows) / i:.3f}", flush=True)
    rep = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "split": args.split, "llm": args.llm,
           "policy": graph.POLICY().name, "T_max": load_yaml("slots.yaml")["dialog"]["T_max"],
           "summary": summarize(rows), "seconds": round(time.time() - t0, 1),
           "cases": [keep(r) for r in rows]}
    secs = [r["seconds"] for r in rows if "seconds" in r]
    if secs:
        rep["seconds_per_case"] = {"mean": round(sum(secs) / len(secs), 1), "max": max(secs)}
    if args.case:
        rep["transcript"] = rows[0]["transcript"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"e2e{'_llm' if args.llm else ''}{'_' + args.case if args.case else ''}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[e2e] {rep['summary']} ({rep['seconds']} s) → {out.relative_to(PROJECT_ROOT)}")
    if args.resume and partial.exists():
        partial.unlink()                                  # đã đủ ca ⇒ kết quả nằm ở file chính
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())

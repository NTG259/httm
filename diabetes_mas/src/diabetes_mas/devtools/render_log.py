"""Chuyển log tool call (T-10, JSONL) thành bảng Markdown để dán vào báo cáo / trình chiếu (07_TEST_PLAN.md §3).

  python -m diabetes_mas.devtools.render_log logs/agent_runs/2026-11-12_<session>.jsonl [-o out.md]

Cột "lượt" lấy từ kết quả T-05 gần nhất (`turn=N`), vì log ghi theo tool call chứ không theo lượt.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from diabetes_mas.utils.text import read_jsonl


def _short(x, n: int = 48) -> str:
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    s = re.sub(r"\s+", " ", s).replace("|", "\\|")
    return s if len(s) <= n else s[: n - 1] + "…"


def _args(args: dict) -> str:
    keep = {k: v for k, v in (args or {}).items() if k not in ("session_id", "mode") and v not in (None, [], {})}
    if not keep:
        return "—"
    if len(keep) == 1:
        return _short(next(iter(keep.values())))
    return _short(", ".join(f"{k}={v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}"
                            for k, v in keep.items()))


def render(rows: list[dict]) -> str:
    out = ["| # | lượt | agent | tool | tham số | kết quả | ms |", "|---|---|---|---|---|---|---|"]
    turn = 0
    for i, r in enumerate(rows, 1):
        summary = r.get("result_summary") or r.get("error") or ""
        if r.get("tool") == "T-05" and (m := re.search(r"turn=(\d+)", summary)):
            turn = int(m.group(1))
        ok = "" if r.get("ok", True) else "❌ "
        out.append(f"| {i} | {turn} | {r.get('agent', '')} | {r.get('tool', '')} | {_args(r.get('args'))} | "
                   f"{ok}{_short(summary, 70)} | {r.get('latency_ms', '')} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("-o", "--out")
    args = ap.parse_args(argv)
    rows = list(read_jsonl(Path(args.log)))
    if not rows:
        print(f"[render_log] {args.log} rỗng", file=sys.stderr)
        return 2
    md = render(rows)
    if args.out:
        Path(args.out).write_text(md + "\n", encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

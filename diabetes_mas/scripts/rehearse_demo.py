"""Chạy thử demo có bấm giờ qua API (giống web demo): Story 1, tư vấn TC-20/22/23, cấp cứu TC-34.

Cần uvicorn đang chạy:  ../.venv/Scripts/python -m uvicorn diabetes_mas.api.app:app --port 8000
Chạy:                   ../.venv/Scripts/python scripts/rehearse_demo.py [--no-llm] [--url http://127.0.0.1:8000]
Kết quả: bảng Markdown in ra màn hình + logs/rehearsal_<llm|nollm>_<giờ>.json
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import requests

# (nhóm, câu người dùng, hàm kiểm tra trên response → lỗi hoặc None)
def _has(*subs):
    def chk(r):
        low = r["reply"].lower()
        miss = [s for s in subs if s.lower() not in low]
        return f"thiếu {miss}" if miss else None
    return chk


def _slot(sid, val):
    def chk(r):
        got = r["state"]["slots"].get(sid, {}).get("value")
        return None if got is not None and abs(got - val) < 0.05 else f"{sid}={got}, cần {val}"
    return chk


def _all(*chks):
    def chk(r):
        errs = [e for c in chks if (e := c(r))]
        return "; ".join(errs) or None
    return chk


SCRIPT = [
    ("Story 1", [
        ("Dạo này tôi khát nước suốt, đi tiểu nhiều, sụt 3 kg trong tháng.", None),
        ("Tôi 52 tuổi.", _slot("tuoi", 52)),
        ("Tôi cao 1m65, nặng 78 kg.", _slot("bmi", 28.7)),
        ("Tôi là nam.", None),
        ("HbA1c tháng trước 7,1%.", _all(_slot("HBA1C", 7.1), _has("típ 2"))),
    ]),
    ("TC-20", [("Tiêu chuẩn chẩn đoán tiểu đường là gì?", _has("["))]),
    ("TC-22", [("Metformin có tác dụng phụ gì?", _has("metformin", "["))]),
    ("TC-23", [("Cách chữa ung thư phổi?", _has("chưa có đủ thông tin"))]),
    ("TC-34", [("Tôi đường huyết 25 mmol/L, đang buồn nôn, thở dốc", _has("115"))]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--timeout", type=float, default=180)
    a = ap.parse_args()
    use_llm = not a.no_llm
    s = requests.Session()
    print("health:", json.dumps(s.get(f"{a.url}/v1/health", timeout=30).json(), ensure_ascii=False)[:300])
    t0 = time.perf_counter()
    s.post(f"{a.url}/v1/consult", json={"question": "Tiền đái tháo đường là gì?", "use_llm": use_llm}, timeout=a.timeout)
    warm = time.perf_counter() - t0
    rows, records = [], []
    for group, turns in SCRIPT:
        sid = s.post(f"{a.url}/v1/sessions", timeout=30).json()["session_id"]
        for i, (text, chk) in enumerate(turns, 1):
            t = time.perf_counter()
            r = s.post(f"{a.url}/v1/sessions/{sid}/messages", json={"text": text, "use_llm": use_llm},
                       timeout=a.timeout).json()
            dt = time.perf_counter() - t
            err = chk(r) if chk else None
            tool_ms = sum(c.get("latency_ms") or 0 for c in r["tool_calls"])
            rows.append(f"| {group} | {i} | {text[:45]} | {dt:.1f} | {tool_ms / 1000:.1f} | {r['stage']} "
                        f"| {'✅' if not err else '❌ ' + err} | {r['reply'][:90].replace(chr(10), ' ')} |")
            records.append({"group": group, "turn": i, "text": text, "seconds": round(dt, 2), "tool_s": tool_ms / 1000,
                            "stage": r["stage"], "intent": r["intent"], "error": err, "reply": r["reply"],
                            "probs": r["state"]["probs"], "path": r["path"]})
    total = sum(x["seconds"] for x in records)
    print(f"\nChế độ: {'Qwen' if use_llm else 'không LLM'} · làm nóng {warm:.1f} s · tổng {total:.1f} s · "
          f"lỗi {sum(1 for x in records if x['error'])}/{len(records)}\n")
    print("| Nhóm | Lượt | Câu | Giây | Tool (s) | Stage | Kiểm tra | Trả lời (90 ký tự) |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    out = Path("logs") / f"rehearsal_{'llm' if use_llm else 'nollm'}_{datetime.now():%H%M}.json"
    out.write_text(json.dumps({"use_llm": use_llm, "warmup_s": round(warm, 2), "total_s": round(total, 2),
                               "turns": records}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()

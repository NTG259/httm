"""Xuất sơ đồ Mermaid trong docs/*.md ra PNG ở docs/assets/ (artifact sinh ra — không sửa tay).

Chạy:  python -m diabetes_mas.devtools.render_diagrams [--doc 04_SEQUENCE.md] [--prefix seq]

Cần mermaid-cli (`mmdc`). Đường dẫn lấy theo thứ tự: biến môi trường MMDC, rồi `mmdc` trên PATH.
Không tải Chromium: dùng Chrome đã cài (biến môi trường CHROME_PATH, mặc định đường dẫn Windows).
  npm i @mermaid-js/mermaid-cli   (đặt PUPPETEER_SKIP_DOWNLOAD=1 trước khi cài)
Mỗi khối mermaid nằm dưới tiêu đề "## <n>. Luồng <X> — …" ⇒ <prefix>_<X>.png; khối khác ⇒ <prefix>_<n>.png.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from diabetes_mas.config import PROJECT_ROOT

DOCS = PROJECT_ROOT.parent / "docs"
DEFAULT_CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def blocks(text: str) -> list[tuple[str, str]]:
    """[(tên, mã mermaid)] theo thứ tự xuất hiện."""
    out, heading, idx = [], "", 0
    for part in re.split(r"(?m)^(## .*)$", text):
        if part.startswith("## "):
            heading = part
            continue
        for code in re.findall(r"```mermaid\n(.*?)```", part, re.S):
            idx += 1
            m = re.search(r"Luồng ([A-Z])\b", heading)
            out.append((m.group(1) if m else str(idx), code))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="04_SEQUENCE.md")
    ap.add_argument("--prefix", default="seq")
    ap.add_argument("--width", type=int, default=1600)
    args = ap.parse_args(argv)

    mmdc = os.environ.get("MMDC") or shutil.which("mmdc")
    if not mmdc:
        print("[render] không tìm thấy mmdc — đặt biến MMDC hoặc cài @mermaid-js/mermaid-cli", file=sys.stderr)
        return 2
    chrome = os.environ.get("CHROME_PATH", DEFAULT_CHROME)
    assets = DOCS / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "puppeteer.json"
        cfg.write_text(json.dumps({"executablePath": chrome, "headless": "new"}), encoding="utf-8")
        for name, code in blocks((DOCS / args.doc).read_text(encoding="utf-8")):
            src = Path(tmp) / f"{name}.mmd"
            src.write_text(code, encoding="utf-8")
            dst = assets / f"{args.prefix}_{name}.png"
            cmd = [mmdc, "-i", str(src), "-o", str(dst), "-p", str(cfg), "-w", str(args.width), "-b", "white", "-s", "2"]
            res = subprocess.run(cmd, capture_output=True, text=True, shell=os.name == "nt")
            if res.returncode != 0 and "unknown option '-w'" in res.stderr:      # mermaid-cli ≥ 12: -w → --size
                cmd[cmd.index("-w")] = "--size"
                res = subprocess.run(cmd, capture_output=True, text=True, shell=os.name == "nt")
            if res.returncode != 0:
                print(f"[render] lỗi {name}: {res.stderr.strip()[-400:]}", file=sys.stderr)
                return 1
            print(f"[render] {dst.relative_to(DOCS.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

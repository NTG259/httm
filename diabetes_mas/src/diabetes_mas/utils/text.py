"""Tiện ích văn bản tiếng Việt và JSONL."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

_WS = re.compile(r"\s+")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", nfc(text)).strip()


def strip_accents(text: str) -> str:
    """'Đái tháo đường' -> 'dai thao duong' (dùng cho so khớp không dấu)."""
    text = nfc(text).replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def norm_key(text: str) -> str:
    """Khóa so khớp: thường, bỏ dấu, gom khoảng trắng."""
    return collapse_ws(strip_accents(text).lower())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

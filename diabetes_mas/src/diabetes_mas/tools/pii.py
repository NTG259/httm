"""Lọc thông tin định danh (PII) — dùng cho truy vấn ra web (T-12) và cho log (T-10, NFR-04)."""
from __future__ import annotations

import re

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("phone", re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s.-]?\d){8,10}(?!\d)")),
    ("id_number", re.compile(r"(?<!\d)\d{9}(?:\d{3})?(?!\d)")),
    ("date", re.compile(r"(?<!\d)\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}(?!\d)")),
    ("name", re.compile(r"(?i)\b(?:tôi|em|mình|con)\s+tên\s+(?:là\s+)?([A-ZÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*){0,4})")),
    ("address", re.compile(r"(?i)\b(?:địa chỉ|nhà (?:tôi|em|mình) ở|sống tại)\b[^.,;\n]{3,80}")),
]


def find_pii(text: str) -> list[str]:
    return [kind for kind, pat in PATTERNS if pat.search(text)]


def redact(text: str, max_len: int = 160) -> str:
    out = text
    for kind, pat in PATTERNS:
        out = pat.sub(f"<{kind}>", out)
    return out[:max_len] + ("…" if len(out) > max_len else "")

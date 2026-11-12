"""Liên kết thực thể trong chunk → quan hệ (:Chunk)-[:MENTIONS {count}]->(thực thể).

So khớp theo từ điển (tên + bí danh) trên văn bản đã bỏ dấu, có ranh giới từ.
Dùng bản bỏ dấu để bắt cả cách viết 'tuýp/típ/týp', nhưng chỉ với mục từ dài ≥ 2 âm tiết
hoặc thuốc/xét nghiệm (tránh khớp nhầm từ ngắn).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from diabetes_mas.kb.seed import Seed, split_aliases
from diabetes_mas.utils.text import norm_key

LINK_LABELS = ("Disease", "Symptom", "RiskFactor", "LabTest", "Drug")
# Mục từ quá chung chung, gây nhiễu nếu dùng để liên kết
STOPTERMS = {"bo bi tieu duong", "me bi tieu duong", "beo", "noi"}


@dataclass(frozen=True)
class Term:
    key: str
    node_id: str
    label: str


def build_lexicon(seed: Seed) -> list[Term]:
    terms: dict[str, Term] = {}
    for label in LINK_LABELS:
        for row in seed.nodes[label]:
            surface = [row.get("name_vi") or row.get("name", ""), *split_aliases(row.get("aliases"))]
            for s in surface:
                key = norm_key(re.sub(r"\(.*?\)", "", s))
                if not key or key in STOPTERMS:
                    continue
                if len(key.split()) < 2 and label not in {"Drug", "LabTest"}:
                    continue
                # mục từ dài hơn / cụ thể hơn thắng khi trùng khóa
                terms.setdefault(key, Term(key, row["id"], label))
    # khớp mục từ dài trước để "tiểu đường tuýp 2" không bị tính thành "tiểu đường"
    return sorted(terms.values(), key=lambda t: (-len(t.key), t.key))


def link_text(text: str, lexicon: list[Term]) -> dict[tuple[str, str], int]:
    hay = f" {norm_key(text)} "
    found: dict[tuple[str, str], int] = {}
    for t in lexicon:
        pattern = re.compile(rf"(?<![\w]){re.escape(t.key)}(?![\w])")
        n = len(pattern.findall(hay))
        if n:
            found[(t.node_id, t.label)] = found.get((t.node_id, t.label), 0) + n
            hay = pattern.sub(" ", hay)       # không đếm lại phần đã khớp bởi mục từ dài hơn
    return found

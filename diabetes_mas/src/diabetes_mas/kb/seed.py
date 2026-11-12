"""Nạp & kiểm chứng seed KB (kb/seed/*.csv).

Chạy:  python -m diabetes_mas.kb.seed      → in báo cáo, exit 1 nếu có lỗi

Kiểm tra:
1. Khóa ngoại: mọi id trong edges/criteria tồn tại.
2. Nguồn: mọi source_id có trong data/processed/sources.jsonl.
3. Chứng cứ: chuỗi `evidence` xuất hiện NGUYÊN VĂN (sau khi gom khoảng trắng, không phân biệt hoa thường)
   trong văn bản đã làm sạch của đúng nguồn đó → chặn trích dẫn bịa.
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from functools import lru_cache

from diabetes_mas.config import INTERIM_DIR, PROCESSED_DIR, SEED_DIR
from diabetes_mas.utils.text import collapse_ws, read_jsonl

NODE_FILES = {
    "Disease": "diseases.csv",
    "Symptom": "symptoms.csv",
    "RiskFactor": "risk_factors.csv",
    "LabTest": "lab_tests.csv",
    "Drug": "drugs.csv",
}

REL_ENDPOINTS = {
    "SUBTYPE_OF": ("Disease", "Disease"),
    "DIFFERENTIAL_OF": ("Disease", "Disease"),
    "HAS_SYMPTOM": ("Disease", "Symptom"),
    "HAS_COMPLICATION": ("Disease", "Disease"),
    "HAS_RISK_FACTOR": ("Disease", "RiskFactor"),
    "TREATS": ("Drug", "Disease"),
}


def _read_csv(name: str) -> list[dict]:
    with open(SEED_DIR / name, encoding="utf-8", newline="") as f:
        return [{k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()} for row in csv.DictReader(f)]


def split_aliases(value: str | None) -> list[str]:
    return [a.strip() for a in (value or "").split("|") if a.strip()]


@dataclass
class Seed:
    nodes: dict[str, list[dict]]
    criteria: list[dict]
    edges: list[dict]
    errors: list[str] = field(default_factory=list)

    def ids(self, label: str) -> set[str]:
        return {n["id"] for n in self.nodes[label]}


def load_seed() -> Seed:
    nodes = {label: _read_csv(fname) for label, fname in NODE_FILES.items()}
    return Seed(nodes=nodes, criteria=_read_csv("criteria.csv"), edges=_read_csv("edges.csv"))


@lru_cache(maxsize=1)
def _source_texts() -> dict[str, str]:
    texts: dict[str, list[str]] = {}
    for d in read_jsonl(INTERIM_DIR / "docs.jsonl"):
        texts.setdefault(d["source_id"], []).append(d["text"])
    return {sid: collapse_ws(" ".join(parts)).lower() for sid, parts in texts.items()}


def evidence_found(source_id: str, evidence: str) -> bool:
    text = _source_texts().get(source_id)
    return bool(text) and collapse_ws(evidence).lower() in text


def validate(seed: Seed) -> list[str]:
    errors: list[str] = []
    known_sources = {s["id"] for s in read_jsonl(PROCESSED_DIR / "sources.jsonl")}

    for label, rows in seed.nodes.items():
        ids = [r["id"] for r in rows]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            errors.append(f"{label}: id trùng {sorted(dup)}")

    def check_prov(where: str, row: dict) -> None:
        sid = row.get("source_id", "")
        if sid not in known_sources:
            errors.append(f"{where}: source_id '{sid}' không có trong sources.jsonl")
        elif not row.get("evidence"):
            errors.append(f"{where}: thiếu evidence")
        elif not evidence_found(sid, row["evidence"]):
            errors.append(f"{where}: evidence không tìm thấy trong {sid}: «{row['evidence']}»")

    disease_ids, test_ids = seed.ids("Disease"), seed.ids("LabTest")
    for c in seed.criteria:
        where = f"criteria {c['id']}"
        if c["disease_id"] not in disease_ids:
            errors.append(f"{where}: disease_id '{c['disease_id']}' không tồn tại")
        if c["test_id"] not in test_ids:
            errors.append(f"{where}: test_id '{c['test_id']}' không tồn tại")
        if c["op"] not in {">=", ">", "<", "<=", "between"}:
            errors.append(f"{where}: op '{c['op']}' không hợp lệ")
        if c["op"] == "between" and not c["value_max"]:
            errors.append(f"{where}: op between cần value_max")
        check_prov(where, c)

    for i, e in enumerate(seed.edges, start=2):
        where = f"edges dòng {i} {e['rel']} {e['from_id']}->{e['to_id']}"
        if e["rel"] not in REL_ENDPOINTS:
            errors.append(f"{where}: quan hệ không hỗ trợ")
            continue
        src_label, dst_label = REL_ENDPOINTS[e["rel"]]
        if e["from_id"] not in seed.ids(src_label):
            errors.append(f"{where}: '{e['from_id']}' không phải {src_label}")
        if e["to_id"] not in seed.ids(dst_label):
            errors.append(f"{where}: '{e['to_id']}' không phải {dst_label}")
        if e.get("prob") and not 0.0 <= float(e["prob"]) <= 1.0:
            errors.append(f"{where}: prob ngoài [0,1]")
        check_prov(where, e)
    return errors


def main() -> int:
    seed = load_seed()
    errors = validate(seed)
    counts = {k: len(v) for k, v in seed.nodes.items()}
    print(f"[seed] nodes={counts} criteria={len(seed.criteria)} edges={len(seed.edges)}")
    for err in errors:
        print(f"  ✗ {err}")
    print("[seed] OK — mọi trích dẫn đều khớp nguồn" if not errors else f"[seed] {len(errors)} lỗi")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

"""Đồ thị tri thức: nạp triple, nhận diện thực thể trong câu hỏi, lấy sự kiện (graph facts)."""
import csv
import re

import networkx as nx

from config import MAX_FACTS_PER_RELATION, PROC

# Cách đọc từng loại quan hệ của Hetionet thành câu (đây là từ vựng của lược đồ, không phải dữ liệu)
TEMPLATES = {
    "CtD": "{s} điều trị {o}.",
    "CpD": "{s} làm giảm triệu chứng {o}.",
    "DpS": "{s} biểu hiện triệu chứng {o}.",
    "DrD": "{s} tương tự với bệnh {o}.",
    "DlA": "{s} khu trú tại {o}.",
    "DaG": "{s} liên quan đến gen {o}.",
}


# Từ khóa trong câu hỏi cho biết người dùng muốn loại quan hệ nào (cũng là từ vựng của lược đồ)
INTENT = {
    "CtD": ("drug", "treat", "medicine", "medication", "therap",
            "thuốc", "điều trị", "liệu pháp"),
    "CpD": ("palliat", "relie", "giảm nhẹ", "giảm triệu chứng"),
    "DpS": ("symptom", "signs", "triệu chứng", "biểu hiện", "dấu hiệu"),
    "DaG": ("gene", "genetic", "gen", "di truyền"),
    "DlA": ("organ", "anatom", "body part", "cơ quan", "bộ phận"),
    "DrD": ("similar", "resembl", "related disease", "tương tự", "giống"),
}


def relations_for(question: str) -> set[str] | None:
    """Loại quan hệ mà câu hỏi nhắc tới; None nếu không có từ khóa nào."""
    q = question.lower()
    found = {rel for rel, words in INTENT.items()
             if any(re.search(r"\b" + w, q) for w in words)}     # khớp đầu từ: "drugs", "treats", ...
    return found or None


def load_graph() -> nx.MultiDiGraph:
    """MultiDiGraph: cho phép nhiều loại quan hệ giữa cùng một cặp nút (DiGraph sẽ ghi đè)."""
    G = nx.MultiDiGraph()
    with open(PROC / "diabetes_triples.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            G.add_node(r["subject"], kind=r["subject_kind"])
            G.add_node(r["object"], kind=r["object_kind"])
            G.add_edge(r["subject"], r["object"], relation=r["relation"])
    return G


def _norm(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip() + " "


def _aliases(name: str):
    key = _norm(name)
    yield key
    if " mellitus " in key:                    # "type 2 diabetes mellitus" -> "type 2 diabetes"
        yield key.replace(" mellitus ", " ")


def find_entities(G: nx.MultiDiGraph, question: str, min_len: int = 4) -> list[str]:
    """Nhận diện thực thể bằng cách so tên nút trong đồ thị với câu hỏi (không dùng danh sách viết tay)."""
    q = _norm(question)
    hits = []
    for node in G.nodes:
        if any(len(a.strip()) >= min_len and a in q for a in _aliases(node)):
            hits.append(node)
    hits.sort(key=len, reverse=True)           # giữ tên dài nhất, bỏ tên nằm trong tên dài hơn
    kept = []
    for h in hits:
        if not any(h.lower() in k.lower() for k in kept):
            kept.append(h)
    return kept


def graph_facts(G: nx.MultiDiGraph, entity: str, relations: set[str] | None = None,
                per_relation: int | None = None) -> list[str]:
    """Sự kiện 1-hop quanh một thực thể.
    - Có relations (câu hỏi nhắc tới loại quan hệ cụ thể): chỉ lấy các loại đó, tối đa MAX_FACTS_PER_RELATION dòng mỗi loại.
    - Không có relations: chỉ lấy vài dòng mỗi loại để làm 'hồ sơ' ngắn của thực thể, tránh đổ cả đồ thị."""
    if entity not in G:
        return []
    if per_relation is None:
        per_relation = MAX_FACTS_PER_RELATION if relations else 2
    buckets: dict[str, list[str]] = {}
    for _, o, d in G.out_edges(entity, data=True):
        if relations and d["relation"] not in relations:
            continue
        buckets.setdefault(d["relation"], []).append(TEMPLATES[d["relation"]].format(s=entity, o=o))
    for s, _, d in G.in_edges(entity, data=True):
        if relations and d["relation"] not in relations:
            continue
        buckets.setdefault(d["relation"], []).append(TEMPLATES[d["relation"]].format(s=s, o=entity))
    facts = []
    for rel in sorted(buckets):
        items = sorted(set(buckets[rel]))
        facts += items[:per_relation]
        if len(items) > per_relation:
            facts.append(f"({len(items) - per_relation} more '{rel}' facts about {entity} not shown)")
    return facts

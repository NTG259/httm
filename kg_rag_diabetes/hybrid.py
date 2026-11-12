"""Gộp truy xuất đồ thị và vector thành một ngữ cảnh, rồi dựng prompt."""
from config import MIN_SCORE, TOP_K
from kg import find_entities, graph_facts, load_graph, relations_for
from retriever import Retriever

_G = None
_R = None


def _load():
    global _G, _R
    if _G is None:
        _G = load_graph()
        _R = Retriever()
    return _G, _R


def hybrid_context(question: str, top_k: int = TOP_K) -> dict:
    G, R = _load()
    entities = find_entities(G, question)
    wanted = relations_for(question)
    graph_lines = []
    for e in entities:
        graph_lines += graph_facts(G, e, relations=wanted)
    docs = R.retrieve(question, top_k=top_k)
    best = docs[0]["score"] if docs else 0.0
    return {
        "question": question,
        "entities": entities,
        "graph": list(dict.fromkeys(graph_lines)),     # bỏ dòng trùng, giữ thứ tự
        "docs": docs,
        "enough": bool(graph_lines) or best >= MIN_SCORE,
    }


def format_context(ctx: dict) -> str:
    parts = []
    if ctx["graph"]:
        parts.append("GRAPH FACTS:\n" + "\n".join(ctx["graph"]))
    if ctx["docs"]:
        lines = []
        for d in ctx["docs"]:
            where = f"{d['source']}, p.{d['page']}" if d["page"] else d["source"]
            lines.append(f"[{where}] {d['text']}")
        parts.append("DOCUMENT EVIDENCE:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    return (
        "You are a diabetes information assistant for education only. You do not give medical advice.\n"
        "Answer only from the supplied context. If the context is insufficient, say that the knowledge "
        "base does not contain enough information. Do not invent facts, doses or numbers.\n"
        "Cite the source in brackets for each statement.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
    )

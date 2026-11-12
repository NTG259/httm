"""Streamlit demo: Diabetes Q&A với 3 chế độ RAG / Knowledge Graph / Hybrid."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
import streamlit as st

import hybrid as _hybrid_mod
from config import MIN_SCORE, TOP_K
from hybrid import build_prompt, format_context, hybrid_context
from kg import find_entities, graph_facts, relations_for

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


# ── Khởi tạo một lần, dùng chung cho cả 3 tab ────────────────────────────────
@st.cache_resource(show_spinner="Đang tải mô hình và đồ thị tri thức...")
def _init():
    from kg import load_graph
    from retriever import Retriever

    G = load_graph()
    R = Retriever()
    # Inject vào hybrid để tránh load SBERT 2 lần
    _hybrid_mod._G = G
    _hybrid_mod._R = R
    return G, R


# ── Xây context riêng cho từng tab ───────────────────────────────────────────
def _rag_context(question: str, top_k: int) -> dict:
    _, R = _init()
    docs = R.retrieve(question, top_k=top_k)
    best = docs[0]["score"] if docs else 0.0
    return {
        "question": question,
        "entities": [],
        "graph": [],
        "docs": docs,
        "enough": best >= MIN_SCORE,
    }


def _kg_context(question: str) -> dict:
    G, _ = _init()
    entities = find_entities(G, question)
    wanted = relations_for(question)
    lines = []
    for e in entities:
        lines += graph_facts(G, e, relations=wanted)
    return {
        "question": question,
        "entities": entities,
        "graph": list(dict.fromkeys(lines)),
        "docs": [],
        "enough": bool(lines),
    }


# ── Gọi Qwen qua Ollama API ───────────────────────────────────────────────────
def _call_qwen(prompt: str, model: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": 600},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ── Render một tab ────────────────────────────────────────────────────────────
def _render(ctx: dict, model: str) -> None:
    if not ctx["enough"]:
        st.warning("⚠️ Không đủ thông tin trong cơ sở tri thức để trả lời câu hỏi này.")
        return

    with st.expander("📋 Context đã truy xuất", expanded=True):
        if ctx["entities"]:
            st.caption(f"Entities nhận diện: **{', '.join(ctx['entities'])}**")

        if ctx["graph"]:
            st.markdown("**Graph Facts**")
            for line in ctx["graph"]:
                st.markdown(f"- {line}")

        if ctx["docs"]:
            if ctx["graph"]:
                st.divider()
            st.markdown("**Document Evidence**")
            for d in ctx["docs"]:
                loc = f"{d['source']}, trang {d['page']}" if d["page"] else d["source"]
                st.markdown(f"**[{loc}]** · score: `{d['score']:.3f}`")
                st.markdown(f"> {d['text'][:300]}{'...' if len(d['text']) > 300 else ''}")

    with st.spinner("Đang hỏi Qwen..."):
        try:
            answer = _call_qwen(
                build_prompt(ctx["question"], format_context(ctx)), model
            )
        except requests.exceptions.ConnectionError:
            st.error("⛔ Không kết nối được Ollama. Hãy chắc container đang chạy: `docker start ollama`")
            return
        except Exception as exc:
            st.error(f"⛔ Lỗi: {exc}")
            return

    st.markdown("### 💬 Câu trả lời")
    st.markdown(answer)


# ── Giao diện chính ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Diabetes Q&A", page_icon="🩺", layout="wide")
st.title("🩺 Diabetes Q&A Demo")
st.caption(
    "Nguồn: WHO Global Report on Diabetes 2016 · MedlinePlus · "
    "Bộ Y tế Việt Nam (QĐ 5481/2020)"
)

with st.sidebar:
    st.header("⚙️ Cấu hình")

    # Kiểm tra Ollama
    try:
        _status = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        _models = [m["name"] for m in _status.json().get("models", [])]
    except Exception:
        _models = []

    if not _models:
        st.warning(
            "⚠️ Ollama chưa sẵn sàng hoặc chưa có model nào.  \n"
            "Chạy lệnh sau để pull Qwen:  \n"
            "```\ndocker exec -it ollama ollama pull qwen2.5:3b\n```"
        )
        _models = ["qwen2.5:3b"]

    model = st.selectbox(
        "Qwen model (Ollama)",
        _models,
        help="Model đang chạy local qua Ollama",
    )
    top_k = st.slider("Top-K chunks (RAG / Hybrid)", min_value=1, max_value=10, value=TOP_K)
    st.divider()
    st.markdown("**Nguồn dữ liệu**")
    st.markdown("- 📄 WHO PDF (2016)")
    st.markdown("- 🌐 MedlinePlus XML")
    st.markdown("- 🇻🇳 Bộ Y tế VN (QĐ 5481)")
    st.markdown("- 🕸️ Hetionet Knowledge Graph")

col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "Câu hỏi",
        placeholder="What drugs treat type 2 diabetes?",
        label_visibility="collapsed",
    )
with col2:
    ask = st.button("Hỏi", type="primary", use_container_width=True)

if ask and question.strip():
    _init()
    tab_rag, tab_kg, tab_hybrid = st.tabs(
        ["📄 RAG (Vector Search)", "🕸️ Knowledge Graph", "⚡ Hybrid"]
    )
    with tab_rag:
        st.caption("Truy xuất từ văn bản bằng SBERT — không dùng đồ thị tri thức.")
        _render(_rag_context(question, top_k), model)
    with tab_kg:
        st.caption("Truy xuất từ đồ thị Hetionet — không dùng vector search.")
        _render(_kg_context(question), model)
    with tab_hybrid:
        st.caption("Kết hợp cả đồ thị tri thức và vector search.")
        _render(hybrid_context(question, top_k=top_k), model)

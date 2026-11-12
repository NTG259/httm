# Spec: Diabetes Q&A Demo App

## Objective

Xây dựng một web app Streamlit cho phép nhập câu hỏi về bệnh tiểu đường và xem câu trả lời
được tạo bởi Claude API theo **3 chế độ truy xuất riêng biệt**: RAG, Knowledge Graph, Hybrid.

**Người dùng:** Hội đồng đánh giá (giảng viên/người chấm) xem trực tiếp trong buổi demo.

**Thành công:** Hội đồng thấy rõ sự khác biệt về chất lượng và nguồn dữ liệu giữa 3 chế độ
cho cùng một câu hỏi, có câu trả lời đầy đủ từ Claude API kèm nguồn trích dẫn.

---

## Tech Stack

| Thành phần | Thư viện | Phiên bản |
|---|---|---|
| Web UI | `streamlit` | ≥ 1.35 |
| LLM | `anthropic` (Claude API) | ≥ 0.30 |
| Vector search | `sentence-transformers` + `scikit-learn` | đã có |
| Knowledge Graph | `networkx` | đã có |
| PDF/XML parsing | `pypdf`, `requests` | đã có |

**Lưu ý:** App chạy từ trong thư mục `kg_rag_diabetes/kg_rag_diabetes/` (nơi có `config.py`),
vì `ROOT = Path(__file__).resolve().parent` trong `config.py` pins tất cả đường dẫn data.

---

## Commands

```bash
# Cài thêm 2 dependency mới
cd kg_rag_diabetes/kg_rag_diabetes
pip install streamlit anthropic

# Chạy app (phải set ANTHROPIC_API_KEY trước)
set ANTHROPIC_API_KEY=sk-ant-...          # Windows
streamlit run app.py
```

---

## Project Structure

```
kg_rag_diabetes/
└── kg_rag_diabetes/
    ├── config.py          # đã có — không sửa
    ├── retriever.py       # đã có — không sửa
    ├── kg.py              # đã có — không sửa
    ├── hybrid.py          # đã có — không sửa
    ├── app.py             # MỚI — Streamlit entry point
    └── requirements.txt   # CẬP NHẬT — thêm streamlit, anthropic
```

---

## Thiết kế màn hình

```
┌────────────────────────────────────────────────────────────┐
│  🩺 Diabetes Q&A Demo                                       │
│  ─────────────────────────────────────────────────────     │
│  [ Nhập câu hỏi...                             ] [Hỏi]    │
│                                                             │
│  ┌─────────┬──────────────────┬───────────────┐            │
│  │   RAG   │  Knowledge Graph │    Hybrid     │            │
│  └─────────┴──────────────────┴───────────────┘            │
│                                                             │
│  ▶ Context đã truy xuất  (expandable)                      │
│  ┌─────────────────────────────────────────────┐           │
│  │ Câu trả lời từ Claude                       │           │
│  └─────────────────────────────────────────────┘           │
└────────────────────────────────────────────────────────────┘
```

**3 tabs song song** — người dùng nhập câu hỏi một lần, nhấn "Hỏi",
rồi có thể click qua lại giữa 3 tab để so sánh.

---

## Logic 3 chế độ

### Tab RAG
- Dùng `Retriever.retrieve(question)` → `docs`
- Prompt chỉ có `DOCUMENT EVIDENCE` (không có graph facts)
- Hiển thị: top-K chunks + cosine scores, rồi câu trả lời Claude

### Tab Knowledge Graph
- Dùng `load_graph()` + `find_entities()` + `graph_facts()` → `graph_lines`
- Nếu `graph_lines` rỗng → hiển thị cảnh báo "Không tìm thấy thực thể trong đồ thị"
- Prompt chỉ có `GRAPH FACTS` (không có document chunks)
- Hiển thị: entities tìm thấy + graph facts, rồi câu trả lời Claude

### Tab Hybrid
- Dùng `hybrid_context(question)` từ `hybrid.py`
- Nếu `ctx["enough"] == False` → hiển thị cảnh báo thay vì gọi Claude
- Prompt có cả `GRAPH FACTS` + `DOCUMENT EVIDENCE`
- Hiển thị: entities + graph facts + chunks, rồi câu trả lời Claude

**Hàm `build_prompt()`** dùng chung từ `hybrid.py` cho cả 3 mode.

---

## Claude API

```python
import anthropic

client = anthropic.Anthropic()  # đọc ANTHROPIC_API_KEY từ env

def call_claude(prompt: str) -> str:
    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text
```

---

## Code Style

```python
# app.py — toàn bộ state trong st.session_state, không dùng global
import streamlit as st
from retriever import Retriever
from kg import load_graph, find_entities, graph_facts, relations_for
from hybrid import hybrid_context, format_context, build_prompt

@st.cache_resource
def get_retriever():
    return Retriever()

@st.cache_resource
def get_graph():
    return load_graph()
```

- `@st.cache_resource` cho Retriever và Graph (tải một lần, tái dùng qua mọi request)
- Không có global mutable state
- Tên hàm: `snake_case`, tên component Streamlit: comment ngắn theo block

---

## Testing Strategy

Không viết unit test cho app.py (UI code). Verification là:

1. **Smoke test thủ công:** Chạy app, nhập câu hỏi có đáp án → 3 tab đều trả lời
2. **Edge case thủ công:** Nhập câu hỏi không liên quan → Hybrid tab hiển thị cảnh báo "not enough"
3. **KG miss case:** Nhập câu hỏi không có entity trong graph → KG tab hiển thị cảnh báo

---

## Boundaries

- **Always:** Đọc `ANTHROPIC_API_KEY` từ env — không hardcode key trong code
- **Ask first:** Thay đổi `config.py` (ảnh hưởng pipeline hiện có), thêm dependency mới ngoài `streamlit` và `anthropic`
- **Never:** Sửa `retriever.py`, `kg.py`, `hybrid.py`, `config.py` — app.py chỉ import, không sửa pipeline

---

## Success Criteria

1. `streamlit run app.py` chạy không lỗi sau khi pipeline đã được setup (chunks.jsonl + diabetes_triples.csv tồn tại)
2. Nhập "What drugs treat type 2 diabetes?" → KG tab hiển thị ít nhất 1 graph fact về drug + câu trả lời Claude
3. Nhập "What symptoms does diabetes cause?" → RAG tab trả về ≥ 3 chunks có cosine score > 0
4. Nhập "What is the weather today?" → Hybrid tab hiển thị cảnh báo "not enough info" (không gọi Claude)
5. 3 tab hiển thị câu trả lời khác nhau cho cùng một câu hỏi về tiểu đường

---

## Open Questions

- Không có — intent đã confirmed đầy đủ.

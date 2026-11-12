# Tài liệu phát triển — Hệ đa tác tử KG + RAG chẩn đoán & tư vấn tiểu đường

> **Mục đích:** quy định mỗi giai đoạn phát triển cần những tài liệu/artifact gì, lưu ở đâu, định dạng gì.
> **Phạm vi:** xây mới hoàn toàn (greenfield). Source cũ (`kg_rag_diabetes/`, `Assignment2-AI/`) chỉ còn giá trị tham khảo, không phải nền tảng.
> **Căn cứ:** yêu cầu của thầy trong [`TEACHER_REQUIREMENTS.md`](../Assignment2-AI/docs/TEACHER_REQUIREMENTS.md).
> **Cập nhật:** 2026-11-11

---

## 1. Quyết định nền tảng (chốt trước khi viết các tài liệu khác)

| # | Quyết định | Lựa chọn | Lý do |
|---|---|---|---|
| D1 | Ngôn ngữ | Python ≥ 3.11 (máy phát triển hiện dùng 3.15, venv `D:\kg_rag_diabetes\.venv`) | Hệ sinh thái ML/RL/LLM đầy đủ |
| D2 | Knowledge Base | **Neo4j 5** (Docker) | Thầy yêu cầu; nạp bằng file `.cypher` |
| D3 | Vector store | **Neo4j vector index** trên node `:Chunk` | KB và RAG nằm chung một CSDL → truy vấn kết hợp graph + vector trong 1 câu Cypher |
| D4 | Embedding | `BAAI/bge-m3` (1024 chiều, đa ngôn ngữ) | Dữ liệu chính là tiếng Việt (Vinmec, Bộ Y tế) |
| D5 | LLM | Qwen chạy local qua **Ollama** | Riêng tư, không tốn phí API, chạy offline khi demo |
| D6 | Điều phối agent | ~~LangGraph~~ → **đồ thị tự viết** (`agents/graph.py`, ADR-003) | Có state tường minh (khớp State Tracker), đồ thị điều phối phân cấp. LangGraph không cài được trên Python 3.15 nên thay bằng lớp `Graph` cùng API |
| D7 | Lõi chẩn đoán | **Học tăng cường phân cấp (HRL)** + bộ phân loại MLP | Đúng kiến trúc Main Agent / Work-Agents / Disease Classifiers / Reward entropy của thầy |
| D8 | Deep learning | PyTorch | Mạng policy (DQN) và classifier — nơi áp dụng **epoch/layer** |
| D9 | API | FastAPI | Gọn, có OpenAPI tự động |
| D10 | Quản lý phiên bản | Git (cần `git init`) | Workspace hiện chưa có git |
| D12 | Mô hình dự đoán Deep Learning | MLP PyTorch trên SRC-KAGGLE, gọi qua AG-13; v1 là tín hiệu tham khảo song song, không vào trạng thái HRL | Đề vấn đáp Q39–Q45 yêu cầu DL tích hợp vào MAS; dataset có rò rỉ ngưỡng (ADR-002) |
| D11 | Tìm bài viết trên web | **SearXNG** tự host + whitelist tên miền y khoa, qua AG-12 Research | Không cần API key, kiểm soát được nguồn; bài tìm được chỉ vào KB sau khi người duyệt |

Mọi thay đổi quyết định trên phải ghi vào `docs/decisions/ADR-xxx.md` (mẫu ở mục 5).

---

## 2. Cấu trúc thư mục dự án mới

```text
D:\kg_rag_diabetes\
├── docs\                               # TÀI LIỆU (mục 3)
│   ├── TAI_LIEU_PHAT_TRIEN.md          # File này
│   ├── 00_DATA_SOURCES.md … 09_*.md
│   ├── decisions\ADR-xxx.md            # Quyết định kiến trúc
│   └── assets\                         # Ảnh sơ đồ xuất cho slide
└── diabetes_mas\                       # SOURCE CODE MỚI
    ├── pyproject.toml
    ├── .env.example                    # Mẫu biến môi trường (không chứa bí mật thật)
    ├── docker-compose.yml              # neo4j + ollama
    ├── configs\                        # settings.yaml, train.yaml, eval.yaml
    ├── data\
    │   ├── raw\{vinmec,moh,who,hetionet}\   # Dữ liệu nguồn — chỉ thêm, không sửa
    │   ├── interim\                    # Văn bản đã làm sạch, triples thô
    │   ├── processed\                  # chunks.jsonl, triples.csv, entities.csv
    │   └── cases\                      # Ca bệnh mô phỏng: train/val/test.jsonl
    ├── kb\
    │   ├── schema.cypher               # Constraint + index (viết tay)
    │   └── kb_diabetes.cypher          # Toàn bộ node/relationship (SINH RA bởi script)
    ├── src\diabetes_mas\
    │   ├── ingest\    kb\    rag\    tools\
    │   ├── agents\    simulator\    rl\    api\
    │   └── config.py
    ├── tests\{unit,integration,e2e}\
    ├── experiments\runs\<run_id>\      # config, metrics.json, checkpoints, log
    └── logs\agent_runs\                # Log tool call (jsonl)
```

### Ba loại file

| Loại | Ở đâu | Quy tắc |
|---|---|---|
| Dữ liệu nguồn | `data/raw/` | Chỉ thêm. Mỗi file có bản ghi trong `00_DATA_SOURCES.md` (URL, ngày tải, giấy phép). |
| Artifact sinh ra | `data/interim/`, `data/processed/`, `data/cases/`, `kb/kb_diabetes.cypher`, `experiments/`, `logs/` | Không sửa tay; sửa script rồi chạy lại. Không commit file lớn (dùng `.gitignore`). |
| Bí mật | `.env` | Không commit; chỉ commit `.env.example`. |

---

## 3. Giai đoạn → tài liệu → nơi lưu

| GĐ | Tên | Input | Tài liệu đầu ra | Artifact đầu ra |
|---|---|---|---|---|
| 0 | Dữ liệu & KB | Yêu cầu nguồn Vinmec, Neo4j | [`00_DATA_SOURCES.md`](00_DATA_SOURCES.md), [`00_KB_GRAPH.md`](00_KB_GRAPH.md) | `data/raw/*`, `kb/schema.cypher`, `kb/kb_diabetes.cypher`, `data/processed/chunks.jsonl` |
| 1 | Phân tích yêu cầu | Bài toán nghiệp vụ | [`01_REQUIREMENTS.md`](01_REQUIREMENTS.md) | Danh sách tác vụ |
| 2 | Thiết kế agent | Danh sách tác vụ | [`02_AGENT_DESIGN.md`](02_AGENT_DESIGN.md) | Sơ đồ agent |
| 3 | Định nghĩa Tool/API | Vai trò agent | [`03_TOOL_SPEC.md`](03_TOOL_SPEC.md) | Pydantic schema của tool |
| 4 | Luồng giao tiếp | Agent + Tool | [`04_SEQUENCE.md`](04_SEQUENCE.md) | Sequence diagram |
| 5 | Triển khai agent | Tool spec + prompt | [`05_PROMPTS.md`](05_PROMPTS.md) | `src/diabetes_mas/agents/`, `simulator/`, `rl/` |
| 6 | Tích hợp KB/RAG | Agent cần dữ liệu y tế | [`06_KB_RAG_INTEGRATION.md`](06_KB_RAG_INTEGRATION.md) | `src/diabetes_mas/rag/`, `tools/` |
| 7 | Kiểm thử | Câu hỏi/ca bệnh mẫu | [`07_TEST_PLAN.md`](07_TEST_PLAN.md) | `tests/`, `logs/agent_runs/` |
| 8 | Đánh giá & cải tiến | Kết quả kiểm thử | [`08_EVALUATION.md`](08_EVALUATION.md) | `experiments/runs/*/metrics.json` |
| 9 | Kiến trúc, mô hình, demo | Toàn bộ | [`09_ARCHITECTURE.md`](09_ARCHITECTURE.md), [`09_MODEL_NOTES.md`](09_MODEL_NOTES.md), [`09_DEMO_SCRIPT.md`](09_DEMO_SCRIPT.md) | `docs/assets/*.png` |

**Cổng chuyển giai đoạn:** chỉ sang giai đoạn sau khi tài liệu của giai đoạn trước đã đủ các mục bắt buộc (mỗi tài liệu có mục "Tiêu chí hoàn thành").

---

## 4. Quy ước viết tài liệu

- Markdown, tiếng Việt có dấu; tên hàm/biến/thuật ngữ giữ nguyên tiếng Anh.
- Đầu mỗi file: **Mục đích · Giai đoạn · Cập nhật**.
- Sơ đồ: Mermaid trong file `.md`; bản xuất ảnh để ở `docs/assets/`.
- Mã định danh dùng xuyên suốt: agent `AG-xx`, tool `T-xx`, yêu cầu `FR-xx`/`NFR-xx`, test `TC-xx`. Tài liệu sau tham chiếu tài liệu trước bằng mã này (truy vết được từ yêu cầu → test).

## 5. Mẫu ADR (`docs/decisions/ADR-001-ten-quyet-dinh.md`)

```markdown
# ADR-001: <tên quyết định>
- Ngày: YYYY-MM-DD · Trạng thái: Đề xuất | Chấp nhận | Thay thế bởi ADR-xxx
## Bối cảnh
## Quyết định
## Hệ quả (được gì, mất gì)
```

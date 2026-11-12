# CLAUDE.md

Hướng dẫn cho Claude Code khi làm việc trong repo này.

## Tổng quan

Sản phẩm là **`diabetes_mas/`**: hệ đa tác tử (multi-agent) hội thoại tiếng Việt về tiểu đường, chạy offline trên laptop. Có ba năng lực:

1. **Chẩn đoán sơ bộ qua hội thoại.**
   - Main Agent và Work-Agents W1–W4 học bằng HRL/DQN để hỏi ít lượt nhất.
   - Classifier MLP kết hợp luật KB cho ra 5 lớp: Không ĐTĐ / Tiền ĐTĐ / típ 1 / típ 2 / thai kỳ.
2. **Dự đoán nguy cơ bằng Deep Learning** (AG-13, MLP PyTorch). Chỉ là tín hiệu tham khảo, không ghi đè luật KB.
3. **Tư vấn hỏi đáp KB + RAG.**
   - Lấy từ Neo4j (graph + vector index bge-m3), trả lời bằng Qwen 2.5 3B qua Ollama, có trích nguồn `[n]`.
   - Từ chối khi không đủ căn cứ.

Đây là đồ án môn *Intelligent System Development*. Yêu cầu của thầy nằm ở `Assignment2-AI/docs/TEACHER_REQUIREMENTS.md`. Đề vấn đáp là `examination_50_questions_v2.pdf`, bản ôn là `ON_TAP_VAN_DAP.md`.

**Trạng thái hiện tại và việc tiếp theo:** luôn đọc `docs/HANDOFF.md` trước.

## Cấu trúc

```
diabetes_mas/                 ← SẢN PHẨM CHÍNH. Chạy mọi lệnh từ thư mục này
  src/diabetes_mas/
    api/        FastAPI (app.py) + web demo (static/index.html)
    agents/     graph.py (đồ thị điều phối tự viết, ADR-003), router, nlu, nlg, policy, classifiers,
                prediction, consult, safety, research, state, slots, llm.py (client Ollama), prompts/P-01..P-08
    tools/      T-01..T-15 (kg_tools, predict_tools, errors…); mọi lời gọi tool được ghi log T-10
    rl/         env, reward, networks, train_hrl, train_classifier, baselines
    dl/         MLP dự đoán AG-13
    rag/        embed_chunks, vector_search
    kb/         build_cypher, link (liên kết thực thể), neo4j_io, seed
    ingest/     fetch_vinmec, fetch_static, clean, chunk
    simulator/  Patient Simulator, sinh ca bệnh
    devtools/   run_dialogs, eval_consult, eval_report, compare_classifiers, plot_eval, check_docs, render_*
  kb/           schema.cypher (viết tay), kb_diabetes.cypher (SINH TỰ ĐỘNG), seed/*.csv
  configs/      settings.yaml, slots.yaml (dialog: T_max, labs_ready, confident_stop), train.yaml, sources.yaml
  experiments/  best/ (checkpoint đang dùng), runs/<id>/, eval/ (số liệu đánh giá)
  tests/        unit, integration, e2e
docs/                         ← tài liệu theo quy trình 8 bước: 00_* → 09_*, decisions/ADR-*, HANDOFF.md, assets/
.venv/                        ← môi trường Python (ở thư mục gốc, nên dùng ../.venv/Scripts/python)
```

Dữ liệu nguồn là Vinmec (chính), QĐ 3280/2011 Bộ Y tế, WHO, Hetionet và Kaggle Diabetes Prediction (cho DL). Chi tiết ở `docs/00_DATA_SOURCES.md`.

## Môi trường (Windows 11, 16 GB RAM, chỉ CPU)

```bash
cd diabetes_mas
docker compose up -d neo4j searxng     # KHÔNG bật service `ollama`: Ollama cài trên máy (cổng 11434, model qwen2.5:3b)
```

- **Neo4j:**
  - Browser ở `http://localhost:7474`, Bolt ở `localhost:7687`.
  - Cổng chỉ bind `127.0.0.1`. Tài khoản lấy trong `diabetes_mas/.env` (không commit `.env`).
  - Dữ liệu nằm ở `diabetes_mas/neo4j_data/`.
- **SearXNG:** cổng **8888** (cổng 8080 bị `httpd` chiếm). Chỉ cần khi AG-12 tìm web; chẩn đoán và E2E không dùng.
- **RAM là giới hạn chính:**
  - Python + bge-m3 khoảng 3 GB, Qwen khoảng 2,5 GB, Neo4j khoảng 1,1 GB.
  - Đóng ứng dụng nặng trước khi chạy Qwen.
  - Lệnh dài (`run_dialogs --llm`) chạy theo đợt với `--resume`.
- **PowerShell 5.1:** không dùng `cat file | cypher-shell`, vì tiếng Việt sẽ thành `?` mà lệnh vẫn báo thành công. Thay vào đó dùng `cypher-shell -f /import/...` hoặc `python -m diabetes_mas.kb.neo4j_io load`.
- **LangGraph** không cài được trên Python 3.15 (thiếu wheel `zstandard`), nên đồ thị điều phối được tự viết (ADR-003).

## Lệnh thường dùng

```bash
# Chạy web demo
../.venv/Scripts/python -m uvicorn diabetes_mas.api.app:app --port 8000
curl http://localhost:8000/v1/health

# Kiểm tra (chạy trước và sau mỗi thay đổi)
../.venv/Scripts/python -m pytest -q -m "not network"     # marker: neo4j, network, slow, ollama
../.venv/Scripts/python -m diabetes_mas.devtools.check_docs   # đối chiếu tài liệu ↔ code; phải báo OK

# Đánh giá (chi tiết: docs/08_EVALUATION.md §2)
python -m diabetes_mas.devtools.run_dialogs --n 225                      # E2E câu mẫu → experiments/eval/e2e.json
python -m diabetes_mas.devtools.run_dialogs --n 20 --llm --resume        # E2E với Qwen (~2 phút/ca)
python -m diabetes_mas.devtools.eval_consult [--answer]                  # 88 câu tư vấn
python -m diabetes_mas.devtools.eval_report --since "YYYY-MM-DD HH:MM"   # → experiments/eval/report.md

# Huấn luyện
python -m diabetes_mas.rl.train_classifier
python -m diabetes_mas.rl.train_hrl          # chỉ ghi đè experiments/best/hrl khi val tốt hơn (--force để ép)
```

Pipeline dựng KB từ đầu (fetch → clean → chunk → seed → build_cypher → load → embed) có trong `diabetes_mas/README.md`.

## Quy tắc khi sửa code

- **KB:**
  - Chỉ sửa `kb/seed/*.csv`. Mỗi quan hệ phải có `source_id` và câu trích **nguyên văn**.
  - Sau đó chạy `kb.seed` → `kb.build_cypher` → nạp lại Neo4j.
  - **Không sửa tay** `kb/kb_diabetes.cypher`.
- **Luật trước, LLM sau:**
  - NLU đọc con số bằng luật. LLM không được ghi đè slot mà luật đã điền.
  - Mọi câu trả lời ra người dùng đều đi qua AG-08 Safety.
  - Các bước cấp cứu, T_max, gọi DL và Safety là **code cố định**. Chỉ việc chọn Worker, slot hay CLASSIFY là policy học được.
- **Thứ tự ưu tiên khi các nguồn mâu thuẫn:** luật KB > Classifier > DL. Khi đó bật cờ `dl_conflict`.
- **Hệ phải chạy được khi Ollama tắt** (chế độ `use_llm=False`: luật + câu mẫu / trích xuất). Không thêm phụ thuộc bắt buộc vào LLM.
- **Đổi tham số** (reward, ngưỡng, luật điều phối) thì ghi một dòng vào nhật ký `docs/08_EVALUATION.md` §5 (trước/sau). **Đổi dạng công thức** thì viết ADR.
- **Số liệu trong tài liệu** phải ghi rõ đo ở điều kiện nào. Khi hành vi hệ thay đổi, các số cũ phải được đo lại hoặc ghi chú là đã cũ.
- **Không dùng LLM bên ngoài** (người dùng đã quyết định). Giữ Qwen qua Ollama.

## Quy tắc trung thực (từ quy định thi)

- Tài liệu dùng ký hiệu ✅ đã làm (có lệnh hoặc test chứng minh) / 🟡 làm một phần / 📐 đã thiết kế / ⏳ kế hoạch (`docs/01_REQUIREMENTS.md` §4).
- Không được tuyên bố một agent, mô hình hay tích hợp là "đã có" nếu không demo hay kiểm chứng được.
- Phải nêu giới hạn của mô hình dự đoán và vai trò của nhân viên y tế. Mọi kết luận chẩn đoán đều kèm disclaimer.

## Git

- Repo gốc ở `D:\kg_rag_diabetes`, nhánh `main`, chỉ lưu local, không đẩy lên remote.
- Commit sau mỗi thay đổi có ý nghĩa. Không thêm dòng `Co-Authored-By` hay attribution.
- Bỏ khỏi git: `.env`, `.venv`, `neo4j_data/`, `backups/`, `kg_rag_diabetes/data/`, `Assignment2-AI/` (repo git riêng).
- Sao lưu KB: `backups/neo4j.dump`. Muốn khôi phục thì dừng neo4j, rồi chạy `neo4j-admin database load neo4j --from-path=/dump --overwrite-destination` qua container với cùng volume.

## Thư mục cũ, KHÔNG xóa khi chưa di chuyển phụ thuộc

| Thư mục | Là gì | Còn được dùng ở đâu |
|---|---|---|
| `kg_rag_diabetes/` | Nguyên mẫu cũ (Hetionet + NetworkX + Streamlit). Code đã bỏ | `configs/sources.yaml` lấy PDF Bộ Y tế, WHO, Hetionet từ `kg_rag_diabetes/docs` và `kg_rag_diabetes/data/raw` |
| `archive/` | CSV Kaggle gốc | `configs/train.yaml` dùng để train DL (AG-13) |
| `Assignment2-AI/` | App dự đoán cũ (React + NestJS + RandomForest), repo git riêng | Chứa `docs/TEACHER_REQUIREMENTS.md` |
| `SPEC-demo-app.md` | Đặc tả app Streamlit + Claude API | Không còn dùng (đã bỏ hướng này) |

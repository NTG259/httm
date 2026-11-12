# Bàn giao tiến độ — cập nhật 2026-11-12 17:00 (phiên 4, Giai đoạn 8)

> Người dùng yêu cầu làm tuần tự hết các giai đoạn 4 → 9. **GĐ 5–7 xong; GĐ 8 gần xong** (còn B1 Flat DQN đang train, đo lại độ trễ, `eval_report` cuối). Tiếp theo: **GĐ 9** (`09_ARCHITECTURE.md`, `09_DEMO_SCRIPT.md`, ảnh minh họa). **Hạn demo 07:00 13/11**, đóng băng code 23:30.

## Khởi động lại sau khi tắt máy

1. Bật Docker Desktop.
2. Bật dịch vụ: `cd diabetes_mas && docker compose up -d neo4j searxng`.
   - **Không** bật container `ollama`: Ollama chạy trực tiếp trên máy (cổng 11434), có sẵn `qwen2.5:3b`.
   - SearXNG ở cổng **8888** (8080 bị `httpd` chiếm). E2E/chẩn đoán không cần SearXNG — tắt để tiết kiệm RAM.
   - Dữ liệu Neo4j nằm ở `./neo4j_data`, khởi động lại không mất.
3. **Máy 16 GB RAM:** đóng PyCharm/Chrome/Zalo/Discord trước khi chạy Qwen (Python + bge-m3 ~3 GB, Qwen ~2,5 GB, Neo4j ~1,1 GB). Lệnh dài dùng `--resume` (`run_dialogs`) và chạy theo đợt; Claude Code từng dừng tác vụ nền vì thiếu RAM.
4. Kiểm tra:
   - `../.venv/Scripts/python -m pytest -q -m "not network"`: lần gần nhất **181 passed, 1 xfailed** (TC-21).
   - `python -m diabetes_mas.devtools.check_docs`: phải báo OK.

## Đã xong

### Giai đoạn 4 — Luồng giao tiếp ✅
- `docs/04_SEQUENCE.md` v2: mọi mũi tên ghi tool hoặc kênh; thêm Luồng D (lỗi hạ tầng); §6 liệt kê 9 chỗ sửa so với bản đầu.
- `check_docs` thêm kiểm tra [9]: sơ đồ ↔ quyền tool ↔ chế độ chạy (`tests/unit/test_check_docs.py`).
- `tools/errors.py`: mã `kb_unavailable` / `llm_unavailable` / `net_unavailable` + `UNAVAILABLE_TEXT`. Đã thử với Neo4j tắt thật.
- `devtools/render_diagrams.py` → `docs/assets/seq_A..D.png`. mermaid-cli cài ở scratchpad, nên lần sau phải cài lại với `PUPPETEER_SKIP_DOWNLOAD=1` và đặt biến `MMDC`.

### Giai đoạn 5 — Triển khai agent (phần lớn code đã chạy)

| Thành phần | File | Kết quả |
|---|---|---|
| Classifier AG-02/03 (MLP) | `agents/classifiers.py`, `rl/train_classifier.py` | test: masked + luật acc 0,707; đủ slot + luật acc 0,769; 74.585 tham số. T-08 dùng `variant = mlp` |
| DL AG-13 | `dl/model.py`, `dl/train.py`, `agents/prediction.py` | full PR-AUC 0,885, F1 0,625 (R 0,91); no_labs PR-AUC 0,336; baseline D1 F1 0,820 |
| HRL AG-01 + W1..W4 (DQN) | `rl/env.py`, `rl/networks.py`, `agents/policy.py`, `rl/train_hrl.py` | test success 0,698 với 3,6 lượt; B0' 0,716 với 12 lượt; reward 6,28 so với 1,15 |
| Agent LLM | `agents/llm.py`, `prompts/P-01..P-08.md`, `router.py`, `nlu.py`, `nlg.py`, `safety.py`, `consult.py`, `research.py`, `simulator/patient.py` | luật trước, LLM sau, có dự phòng khi Ollama lỗi |
| Đồ thị điều phối | `agents/graph.py` | Graph tự viết (LangGraph không cài được trên Python 3.15 vì thiếu wheel `zstandard`) |
| API + UI | `api/app.py`, `api/static/index.html` | `uvicorn diabetes_mas.api.app:app --port 8000` |
| Đánh giá prompt | `devtools/eval_prompts.py`, `tests/unit/prompts/P-xx.yaml` | P-01/P-02: JSON 100 %, acc 100 %; P-04 roundtrip 0,9; P-06 trích dẫn hợp lệ 100 %; P-07, P-08 100 %; P-05 chưa đo (cần Neo4j) |

Các sửa phụ:
- `kg_tools.criteria` cache cả seed khi Neo4j lỗi: trước đó mỗi lần gọi chờ 4 giây, nay hết chờ.
- Ở chế độ train không ghi log T-10 (`settings.logging.train_calls`).
- T-07 có kênh UI.
- `base.TRACE` ghi vết tool call theo từng lượt.
- `.env`: `LLM_MODEL=qwen2.5:3b`.

Môi trường:
- Ollama đã cài bằng winget, model `qwen2.5:3b` đã tải.
- Docker Desktop **chưa bật**, nên Neo4j chưa chạy. SearXNG chưa được thêm và chưa chạy, dù người dùng đã đồng ý tải.

### Giai đoạn 5 — phần hoàn tất ở phiên 2 ✅
- `tests/unit/test_agents_llm_graph_api.py`: 35 test (NLU, Router, Safety, Consult với LLM giả; đồ thị `use_llm=False`; API bằng TestClient). Toàn bộ: **129 test qua**.
- Sửa lỗi: T-02 gọi fulltext Neo4j mỗi khi từ điển không khớp, nên mỗi lượt chờ thêm 4 giây khi Neo4j tắt. Đã thêm tạm ngừng 30 giây (`kg_tools._FT_DOWN_UNTIL`), giống `criteria()`.
- `docs/decisions/ADR-003-do-thi-dieu-phoi-tu-viet.md`.
- Đã cập nhật tài liệu `01`, `02` (v3), `03` (v2.1), `04` (v3: NLU trước Router, Ollama tắt thì trả lời trích xuất, §6 thêm mục 10–12, đã vẽ lại `seq_*.png`), `05` (bảng phiên bản và §5 có số đo), `09_MODEL_NOTES` §2/§3/§5, `09_ARCHITECTURE` và `TAI_LIEU_PHAT_TRIEN` D6 (bỏ LangGraph).
- Đo lại P-01: JSON 100 %, đúng 100 %. P-03: 90,9 % câu do LLM sinh qua kiểm tra.
- `test_check_docs`: test đột biến chuyển sang dòng `N-->>ST: tuoi = 52…`, vì dòng cũ đã bị xóa khỏi sơ đồ.
- mermaid-cli ở phiên này cài tại scratchpad của phiên. Lần sau cài lại theo hướng dẫn ở trên; `puppeteer.json` dùng dấu `/` trong đường dẫn Chrome.

### Giai đoạn 6 — Neo4j + SearXNG thật ✅ (2026-11-12, phiên 3)
- Chỉ bật `docker compose up -d neo4j searxng`. **Không** bật container `ollama`, vì Ollama chạy trực tiếp trên máy ở cổng 11434 và sẽ trùng cổng.
- KB: 4/4 kiểm tra chất lượng đạt; 5/5 test tích hợp Neo4j qua.
- SearXNG chuyển sang cổng **8888** (`SEARXNG_PORT` trong `.env`) vì 8080 trên máy bị `httpd`/`AgentService` chiếm. Test web thật: `tests/integration/test_web_searxng.py` (`pytest -m network`).
- Ba lỗi phát hiện nhờ chạy thật, đã sửa (ghi ở `08_EVALUATION.md` §5):
  1. `tools.call` dùng `timeout_s` làm thời hạn cả tool, nên T-13 luôn timeout. Nay giá trị được chuyển vào input của tool.
  2. AG-12 nhận bài kho lạc đề do fulltext. Đã thêm cổng cosine thô `search.local_min_cosine`.
  3. Qwen trích dẫn dạng `(1)` hoặc `[3, 2]`. Thêm `normalize_citations`, nhờ đó P-05 dùng LLM 9/10 câu (trước là 3/9).
- T-13 chỉ nhúng tối đa `search.max_embed_chunks` đoạn. Một câu phải lên web vẫn mất khoảng 30–40 giây (bge-m3 trên CPU); đây là giới hạn đã biết.
- `data/review_queue.jsonl` có 2 bài vinmec (Mounjaro, peptide ruột) do lần chạy thử thêm vào, chưa duyệt.
- Toàn bộ test: **145 qua** (gồm Neo4j và web).

### Giai đoạn 7 — Kiểm thử ✅ (2026-11-12, phiên 3)
- Báo cáo, bảng TC → test và 9 lỗi phát hiện: `07_TEST_PLAN.md` §2.5, §4, §6.
- Công cụ mới:
  - `devtools/run_dialogs.py`: E2E qua toàn hệ thống; `--llm`; `--samples` tạo 3 log mẫu.
  - `devtools/render_log.py`: log T-10 → bảng Markdown.
- Test mới:
  - `tests/e2e/test_dialogs.py`: TC-21, 30–35, 40, 43.
  - `tests/e2e/test_train_repro.py`: TC-46.
  - `tests/integration/test_consult_kb.py`: TC-20, 22–24.
  - `tests/integration/test_kb_idempotent.py`: TC-10.
  - Marker mới: `slow`, `ollama`.
- Lỗi nghiêm trọng đã sửa:
  - NLU từng dùng fulltext Neo4j, khiến câu "Tôi đã bỏ thuốc rồi" bật cảnh báo cấp cứu nhầm.
  - Safety chặn nhầm "126 mg/dL".
  - NLU bỏ sót "sụt 3 kg".
  - Simulator nói chưa tự nhiên.
- **Chưa sửa, chuyển sang GĐ 8:**
  - Policy kết luận khi chưa hỏi xét nghiệm (44/225 ca; TC-30 đang đánh dấu xfail strict).
  - `train_hrl.main` luôn ghi đè `experiments/best/hrl/`.
- E2E 225 ca (câu mẫu): acc 0,693, 4,05 lượt, 0 vượt T_max, 0 hỏi lặp. E2E 20 ca với Qwen: acc 0,60, 1 hỏi lặp (trước khi sửa lỗi NLG #10). Toàn bộ: 170 passed, 1 xfailed.

### Giai đoạn 8 — Đánh giá (2026-11-12, phiên 4) — chi tiết ở `08_EVALUATION.md` v2, `07_TEST_PLAN.md` v3
- **TC-30 đã sửa** (07 §6 lỗi #8): luật điều phối `labs_ready` (chưa CLASSIFY khi chưa có giá trị xét nghiệm và còn xét nghiệm chưa hỏi) + W2 không RETURN sớm; `PRIORS.risk_override` P(tuổi ≥ 45 | típ 1) = 0,10 (sinh lại ca — bản cũ `data/cases_v1_gd7`); train lại classifier + HRL.
- **Cấu hình đang dùng:** `experiments/best/hrl` = run 20261112-142402 + luật W2 (bản train lại với W2 kém hơn, `train_hrl` giữ best). Test 0,782 / 3,40 lượt; B0' 0,853 / 12 lượt; B0 Random 0,716 / 12. E2E câu mẫu 225 ca **0,796** / 3,57; E2E Qwen 20 ca **0,75** / 3,70 (~2 phút/hội thoại).
- `train_hrl` chỉ ghi đè best khi val tốt hơn (lỗi #9).
- **TC-25:** `data/eval/consult_eval.csv` 88 câu + `devtools/eval_consult.py`. R@5 1,00; CR 0,909; FR 0,030 nhờ luật mới `rag.min_score_no_entity` = 0,825; trích dẫn hợp lệ 1,00; faithfulness chấm **mẫu** 20 câu: 0,95 nội dung, 0,65 trích dẫn đúng nguồn.
- **TC-45:** `devtools/eval_report.py` → `experiments/eval/report.md` (6 nhóm FR-21).
- Công cụ mới: `devtools/compare_classifiers.py` (NB vs MLP), `devtools/plot_eval.py` (matplotlib, nhóm `dev`), `rl/baselines.py` (B0 Random, B1 Flat DQN), `run_dialogs --resume`.
- Sửa thêm: `consult.explain_ok` chặn P-06 đảo ngược kết luận (lỗi #14).
- **TC-21 xfail:** ca biên (glucose đói 6,5, classifier típ 2 0,44 vs tiền ĐTĐ 0,40); đã thử luật loại típ 1/2 khi không đạt tiêu chuẩn ĐTĐ — E2E giảm 0,796 → 0,742 nên không dùng.

### Giai đoạn 8–9 — cập nhật 17:24
- Cấu hình cuối (08 §4.1 bảng chính): HRL test 0,782 / 3,32 lượt; B1 Flat DQN 0,751 / 3,54; B0' 0,853 / 10,7; B0 Random 0,698 / 11,5; E2E câu mẫu 0,796 / 3,48. Test: **183 passed, 1 xfailed**.
- Sửa thêm khi chạy thử demo: `nlu.DURATION` ("sau 2 giờ là 10,3" từng ghi 2,0), `confident_stop` 0,95, không hỏi giới tính khi đang mang thai, câu hỏi `it_van_dong` (đã đổi ở seed, `.cypher`, Neo4j), giải thích dùng tên tiếng Việt.
- `09_ARCHITECTURE.md` v2 và `09_DEMO_SCRIPT.md` v2 đã viết theo code thật; `02 §6` có bảng "khi nào hệ dừng hỏi".

## Việc còn lại

- [x] Độ trễ đo lại khi máy rảnh: p95 11,6 s (LLM trung vị 4,2 s) — không đạt ≤ 5 s, ghi 08 §4.2, §7.
- [x] `eval_report` bản cuối (17:21) → `experiments/eval/report.md`, 08 §4.4.
- [x] Ảnh minh họa trong `docs/assets/`: `arch_1/2.png`, `kb_schema.png`, `kb_dtd_tip2.png` (`devtools/plot_kb`), `demo_story1/tuvan/capcuu.png` (`scripts/shots.js`, chế độ không LLM). Ảnh Neo4j Browser thật: người dùng tự chụp nếu cần (cần đăng nhập).
- [ ] 21:30–23:30: chạy thử demo có bấm giờ với Qwen; đo thời gian mỗi lượt trên web.
- [ ] 23:30 đóng băng code → 2 lần chạy thử, sao lưu (`git init` + commit, Neo4j dump), video dự phòng.
- Không làm (ghi ở 08 §7): HRL − entropy, D2 LR, faithfulness đầy đủ, TC-21, hỏi "có mang thai không" trong intake, `review_queue`.
- E2E Qwen 0,750 đo **trước** khi sửa `nlu.DURATION`; chạy lại nếu còn thời gian (~40 phút, `run_dialogs --n 20 --llm --resume`).

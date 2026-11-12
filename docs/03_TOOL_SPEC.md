# 03 — Đặc tả Tool/API (v2)

> **Mục đích:** đặc tả mọi tool agent gọi: tên, tham số, kiểu trả về, lỗi, agent được phép dùng, hành vi.
> **Giai đoạn:** 3 · **Input:** vai trò agent (`02_AGENT_DESIGN.md`) · **Output:** đặc tả + cài đặt tool
> **Cập nhật:** 2026-11-12 (v2.1: T-07 kênh UI, T-08 MLP, T-15 có mô hình; API FastAPI đã làm)

> **Trạng thái triển khai:** 15/15 tool đã cài đặt. Ngày 2026-11-12: 143 test (unit + tích hợp Neo4j) và 2 test web thật qua. Mức độ từng tool xem cột "Trạng thái" ở §2:
> - ✅ chạy thật, có test;
> - 🟡 chạy một phần: chỉ có giao diện / baseline, hoặc mới test với HTTP giả lập;
> - 📐 chưa có code.
>
> API FastAPI (§5) đã làm ở Giai đoạn 5 (`api/app.py`).

---

## 1. Quy ước chung

| Quy ước | Cài đặt |
|---|---|
| **Nguồn duy nhất của schema** | `src/diabetes_mas/tools/schemas.py` (Pydantic). JSON schema xuất ra `docs/assets/tool_schemas.json` bằng `python -m diabetes_mas.tools.export_schemas`. Tài liệu này **không chép lại** class Pydantic, để tránh lệch với code. |
| Gọi tool | `tools.call("T-03", agent_id="AG-02", mode="demo", session_id="s1", measurements={...})` (`tools/base.py`) |
| Phân quyền | Agent không được cấp tool trong `configs/agents.yaml` ⇒ `ToolNotAllowed` (lỗi lập trình, ném ra). Agent chạy sai chế độ (vd. AG-12 ở `train`) ⇒ `ToolNotAllowed`. |
| Lỗi | Input sai kiểu, exception hoặc quá thời gian ⇒ `Out(ok=False, error=...)`, không ném ra agent. Lỗi hạ tầng được chuẩn hóa thành mã ổn định (`tools/errors.py`): `kb_unavailable` (Neo4j), `llm_unavailable` (Ollama), `net_unavailable` (mạng/SearXNG), `timeout`; agent kiểm bằng `is_unavailable()` và trả `UNAVAILABLE_TEXT` (`04_SEQUENCE.md` Luồng D) |
| Log (T-10) | **Tự động** sau mỗi lần gọi: `logs/agent_runs/<ngày>_<session>.jsonl`. Chuỗi trong tham số đi qua bộ lọc PII (`tools/pii.py`) và bị cắt ngắn (NFR-04). |
| Timeout | Mặc định 10 s; T-04: 60 s (lần đầu tải bge-m3); T-12: 15 s; T-13: 90 s (gồm embedding), riêng HTTP theo `timeout_s` của từng lần gọi (mặc định 8 s, không retry) |
| Neo4j tắt | Driver `connection_timeout = 3 s`; T-03 tự dùng seed; T-02 bỏ qua fallback fulltext |
| Thang điểm cosine | Theo vector index Neo4j: **(1 + cos) / 2** ∈ [0, 1] (đã kiểm: cos gốc 0,562 ↔ 0,781). T-13 quy đổi theo cùng thang. |

## 2. Bảng tổng hợp

| Mã | Tên hàm | Mô tả | Agent được dùng | Phụ thuộc | Trạng thái |
|---|---|---|---|---|---|
| T-01 | `kg_get_group_slots` | Không gian hành động của một Work-Agent: slot + câu hỏi mẫu + đơn vị | AG-W*, AG-09 | Kho slot (KB seed + `slots.yaml`) | ✅ |
| T-02 | `kg_link_entities` | Liên kết thực thể trong câu tiếng Việt với node KB (T-04 dùng nội bộ) | AG-06 | Từ điển; dự phòng fulltext Neo4j | ✅ |
| T-03 | `kg_check_criteria` | Đối chiếu chỉ số xét nghiệm với `DiagnosticCriterion` | AG-02, AG-07, AG-13 | Neo4j; dự phòng seed | ✅ |
| T-04 | `hybrid_retrieve` | Truy xuất kết hợp graph facts + chunk văn bản, quyết định `enough` | AG-07 | Neo4j + bge-m3 | ✅ (trọng số và `min_score` tạm thời) |
| T-05 | `update_state` | Ghi slot mới vào trạng thái, suy ra cờ, đếm lượt | AG-05 | Kho phiên trong bộ nhớ | ✅ |
| T-06 | `get_state` | Đọc trạng thái (object + vector 122 chiều) | Mọi agent | — | ✅ |
| T-07 | `ask_patient` | Hỏi 1 slot (câu chữ do NLG soạn ở demo), nhận câu trả lời | AG-W* | Simulator / UI | ✅ kênh simulator + kênh UI |
| T-08 | `classify` | Phân phối 5 lớp + entropy, sau mỗi lần cập nhật slot | AG-05 | KB / model PyTorch | ✅ `mlp` (mặc định, `experiments/best/classifier.pt`); dự phòng `kb_naive_bayes` khi chưa có checkpoint |
| T-09 | `check_red_flags` | Phát hiện dấu hiệu cấp cứu sau mỗi lần cập nhật và trước mọi câu ra | AG-05, AG-08 | KB (nhóm `canh_bao`) | ✅ |
| T-10 | `log_event` | Ghi sự kiện JSONL | Tự động | — | ✅ |
| T-11 | `compute_reward` | Reward lượt t | AG-11 | `rl/reward.py` | ✅ |
| T-12 | `search_articles` | Tìm bài: trong kho (`local`) hoặc web whitelist (`web`) | AG-12 | Neo4j fulltext / SearXNG | ✅ `local` + `web` (SearXNG thật, `tests/integration/test_web_searxng.py`) |
| T-13 | `fetch_article` | Tải 1 bài từ URL whitelist, làm sạch, cắt chunk, chấm điểm | AG-12 | HTTP + bge-m3 | ✅ HTTP giả lập + bài vinmec thật |
| T-14 | `queue_article_for_kb` | Đưa bài vào hàng chờ duyệt | AG-12 | `data/review_queue.jsonl` | ✅ |
| T-15 | `predict_diabetes_risk` | MLP Deep Learning → xác suất mắc ĐTĐ | AG-13 (được Main Agent kích hoạt) | Checkpoint PyTorch | ✅ mô hình `full` + `no_labs` (`experiments/best/dl/`) |

## 3. Hành vi từng tool

Schema đầy đủ: `tools/schemas.py` hoặc `docs/assets/tool_schemas.json`. Phần dưới chỉ mô tả **hành vi** và **quyết định**.

### T-01 `kg_get_group_slots` — `tools/kg_tools.py`
- Input `worker ∈ {W1..W4}`. Output: danh sách `SlotInfo` (id, kind, source, `question_vi`, values, unit).
- Lấy từ kho slot `agents/slots.catalog()`, **không** từ Cypher. Lý do: slot hồ sơ (`tuoi`, `bmi`…) không phải node KB; nếu chỉ truy Neo4j thì W3 thiếu 5 hành động. Phần slot y khoa trong kho được sinh từ cùng seed đã nạp Neo4j.
- Kích thước: W1 4 · W2 5 · W3 13 · W4 6 (test `test_t01_action_spaces_match_design`).

### T-02 `kg_link_entities` — `tools/kg_tools.py`
- **Chính:** từ điển `kb/link.py`. Đây là **cùng mã** dùng tạo quan hệ `MENTIONS` khi build KB, nên liên kết phía câu hỏi và phía chunk khớp nhau. Khớp không dấu, ưu tiên mục từ dài, tất định.
- **Dự phòng:** fulltext index `entity_names`, chỉ dùng khi từ điển không khớp gì. Điểm Lucene không chuẩn hóa nên không đặt ngưỡng, mà trả kèm `method = "fulltext"`.

### T-03 `kg_check_criteria` — `tools/kg_tools.py`, lõi `kb/criteria.py`
- Tiêu chuẩn đọc từ **Neo4j** (`DIAGNOSED_BY → DiagnosticCriterion → USES_TEST`, cache 5 phút); Neo4j lỗi ⇒ dùng seed. Output có `kb_backend` cho biết nguồn nào đã dùng. Test tích hợp xác nhận Neo4j và seed giống nhau.
- Mỗi hit kèm `source_id`, `source_title`, `evidence` (câu trích nguyên văn).
- **Quy tắc ưu tiên giữa các hit** (`check`):
  1. Tiêu chuẩn có `condition` chỉ xét khi điều kiện đúng (thai kỳ cần `dang_mang_thai`; glucose bất kỳ cần `co_trieu_chung_kinh_dien`).
  2. Trúng ĐTĐ ⇒ bỏ hit Tiền ĐTĐ và ĐTĐ thai kỳ.
  3. Trúng ĐTĐ thai kỳ ⇒ bỏ hit Tiền ĐTĐ (VMC-0003: *"Nếu không đủ tiêu chuẩn trên thì làm nghiệm pháp dung nạp glucose"*).
  4. HbA1c dùng `>` 6,5 theo nguồn, nên giá trị 6,5 **không** đạt.
- **Hợp nhất vào phân phối** (`apply_rules`, dùng trong T-08), cùng thứ tự ưu tiên:

| Trúng tiêu chuẩn | Lớp bị đặt xác suất 0 |
|---|---|
| ĐTĐ | Không ĐTĐ, Tiền ĐTĐ, ĐTĐ thai kỳ |
| ĐTĐ thai kỳ | Không ĐTĐ, Tiền ĐTĐ |
| Tiền ĐTĐ | Chỉ Không ĐTĐ (ĐTĐ vẫn có thể nếu xét nghiệm khác chưa biết) |

### T-04 `hybrid_retrieve` — `tools/retrieve_tools.py`
Các bước ở `06_KB_RAG_INTEGRATION.md` §2:
1. **Liên kết thực thể:** T-02, bằng từ điển.
2. **Chọn quan hệ theo ý định** của câu hỏi:
   - "thuốc" → `TREATS`; "triệu chứng" → `HAS_SYMPTOM`; "chẩn đoán" → `DIAGNOSED_BY`; "nguy cơ" → `HAS_RISK_FACTOR`; "biến chứng" → `HAS_COMPLICATION`.
   - "gen", "di truyền" → `ASSOCIATED_GENE`. Quan hệ gen và giải phẫu của Hetionet (406 và 72 cạnh) **chỉ** lấy khi câu hỏi nhắc tới.
   - Không khớp ý định nào → mọi quan hệ y khoa của seed.
3. **Graph facts 1-hop** có `source_id`, chuyển thành câu bằng template; tiêu chuẩn chẩn đoán được viết đầy đủ ngưỡng.
4. **Ứng viên chunk:** top 20 từ vector index, cộng tối đa 30 chunk có `MENTIONS` thực thể (graph-guided, cosine tính bằng `vector.similarity.cosine`).
5. **Xếp hạng:** `final = cos + 0,02·min(n_mention, 2)/2 + 0,01·reviewed`; tối đa 2 chunk mỗi nguồn.
   - Hai điểm cộng chỉ để phân định khi cosine gần bằng nhau, vì cosine của Neo4j bị nén trong dải hẹp.
   - Bản đầu dùng trọng số 0,7 / 0,2 / 0,1 đã đẩy chunk khớp chính xác (cos 1,0) xuống hạng 5; lỗi này do test tích hợp phát hiện.
6. **`enough`** (FR-08) = (có ≥ 1 thực thể KB **hoặc** ≥ 1 fact) **và** cosine tốt nhất ≥ `min_score`.
   - `min_score = 0,75` là **giá trị tạm** (`configs/settings.yaml → rag`), chọn lại ở Giai đoạn 8.
   - **Rủi ro đã biết:** câu hợp lệ nhưng không chứa thực thể nào (vd. "nên ăn gì?") sẽ bị từ chối nhầm; sẽ đo ở Giai đoạn 8.
- Backend Neo4j và hàm embedding được tiêm vào (`BACKEND`, `EMBED`), nên unit test chạy không cần Neo4j hay bge-m3.

### T-05 `update_state` · T-06 `get_state` — `tools/dialog_tools.py`
- **T-05:** ghi slot, suy ra `tuoi_tu_45` / `tuoi_duoi_30` / `bmi_tu_23` từ slot số.
  - `asked_slot` + `count_turn=True` ⇒ tăng `turn` (và `sub_turn` nếu đang có Worker giữ quyền).
  - Câu hỏi intake dùng `count_turn=False`, không tính vào T_max.
- **T-06:** trả bản sao `DialogState` cùng vector `s_t` (122 chiều).
- Kho phiên trong bộ nhớ một tiến trình (`tools/session.py`). Lưu trữ bền thì để Giai đoạn 5 nếu API cần.

### T-07 `ask_patient` — `tools/dialog_tools.py`
- **Kênh simulator (✅):** trả lời đúng slot được hỏi theo ca bệnh gắn với phiên (`session.attach_case`).
  - Slot không có trong ca: nhị phân → `no`; số hoặc phân loại → `unknown`.
  - Phiên chưa gắn ca → `error = "no_case_attached"`.
- **Kênh UI (✅):** phiên không gắn ca ⇒ ghi câu hỏi do NLG soạn (`question_text`, bắt buộc) vào lịch sử, trả `channel = "ui"`, `answer_text` rỗng. Câu trả lời đến ở tin nhắn sau, đi qua NLU (`agents/graph.py`).

### T-08 `classify` — `tools/dialog_tools.py`
- **Biến thể `mlp` (mặc định, `settings.classifier.variant`):** AG-02 + 4 AG-03-h (`agents/classifiers.py`, `09_MODEL_NOTES.md` §2), rồi hợp nhất luật. Không có checkpoint thì tự rơi về `kb_naive_bayes`.
- **Biến thể `kb_naive_bayes` (baseline):**
  - Naive Bayes trên các slot nhị phân, dùng `P(slot | bệnh)` từ quan hệ `HAS_SYMPTOM` / `HAS_RISK_FACTOR`; tỉ lệ nền lấy từ `PRIORS` của simulator.
  - Sau đó hợp nhất luật (bảng ở T-03).
  - `sub_probs` = cùng Naive Bayes nhưng chỉ trên slot của từng Worker.
- **Hạn chế cần nói khi vấn đáp:**
  - Mọi xác suất đều là `team_estimate`.
  - Dùng chung tỉ lệ nền với simulator, nên đây là baseline **lạc quan**.
  - Xét nghiệm chỉ có tác dụng qua luật: một kết quả bình thường không làm giảm xác suất ĐTĐ.
- Ghi `probs`, `entropy` vào `DialogState`. Giai đoạn 5 thay bằng MLP (AG-02 / AG-03-h) với giao diện giữ nguyên (`variant = "mlp"`).

### T-09 `check_red_flags` — `tools/dialog_tools.py`
- `red_flag = true` khi **bất kỳ** slot nhóm `canh_bao` là `yes`: buồn nôn, thở nhanh, hơi thở mùi táo thối, rối loạn ý thức.
  - Nguồn: triệu chứng của hôn mê nhiễm toan ceton / tăng áp lực thẩm thấu trong KB; VMC-0001 ghi *"Biến chứng này cần phải điều trị cấp cứu"*.
- **Không** dùng ngưỡng đường huyết riêng lẻ:
  - Con số 16,7 mmol/L trong bản thiết kế đầu **không có nguồn**.
  - 13,9 mmol/L (QĐ 3280 tr. 9) là tiêu chí cận lâm sàng của nhiễm toan ceton **kèm** triệu chứng, nên chỉ được thêm vào phần lý do.
- `action_text` cố định: gọi 115 hoặc đến cơ sở y tế ngay.

### T-10 `log_event` — `tools/base.py`
- Một dòng JSONL mỗi lần gọi tool: `ts, session_id, agent, tool, args (đã lọc PII), result_summary, latency_ms, ok`.

### T-11 `compute_reward` — bọc `rl/reward.py`
- Input `asked`, `hit` (tính bằng `is_hit`: chỉ slot y khoa), `prev_entropy`, `entropy`, `terminal`, `correct`. Output `reward` + `parts`. Công thức: `02_AGENT_DESIGN.md` §5.

### T-12 `search_articles` — `tools/research_tools.py`
- **`local`:** fulltext index `source_text` trên Source/Chunk, gom theo nguồn, `in_kb = true`.
- **`web`:**
  - SearXNG `GET /search?format=json`, truy vấn kèm `site:` cho từng domain whitelist.
  - Kết quả **được lọc lại** bằng whitelist, không tin máy tìm kiếm.
  - Cache 24 giờ ở `data/cache/search/`.
- **Whitelist** (`domain_allowed`): host trùng khớp hoặc kết thúc bằng `.` + domain.
  - Ví dụ: `www.vinmec.com` ✓; `evilvinmec.com` ✗; `vinmec.com.attacker.net` ✗.
  - Tham số `domains` của agent chỉ **thu hẹp** whitelist, không mở rộng.
- **Chặn trước khi gửi request:**
  - `SEARCH_ENABLED=false` → `search_disabled`.
  - Truy vấn chứa PII (email, số điện thoại, CCCD, ngày sinh, "tôi tên là …", địa chỉ) → `pii_detected`.

### T-13 `fetch_article` — `tools/research_tools.py`
- Kiểm whitelist → `PoliteClient` (robots.txt, ≤ 1 request / 2 s, `timeout_s` riêng từng lần gọi, không retry).
  - `tools.call(..., timeout_s=x)` với tool có trường `timeout_s` trong input (T-13) thì `x` là thời hạn **HTTP**; thời hạn cả tool vẫn 90 s. Trước GĐ 6, `x` bị dùng làm thời hạn cả tool ⇒ T-13 luôn `timeout` vì gồm cả nạp + nhúng bge-m3.
- Lọc trước theo số từ trùng với câu hỏi, chỉ nhúng tối đa `search.max_embed_chunks` (8) đoạn. Đo GĐ 6: tải + parse ≈ 2 s; nạp bge-m3 lần đầu ≈ 23 s; nhúng ≈ 4 s / 9 đoạn trên CPU.
- Trang Vinmec dùng parser riêng của Giai đoạn 0; trang khác dùng trình tách khối chung.
- Cắt chunk như pipeline ingest; chấm điểm `(1 + cos) / 2` với câu hỏi; trả top-k chunk tạm (`WEB-<sha8>#nnn`, `web = true`, `reviewed = false`).
- HTML gốc lưu ở `data/cache/web/`. **Không ghi vào Neo4j.**
- Lỗi: `not_allowed_domain`, `robots_disallow`, `timeout`, `http_<mã>`, `parse_failed`.

### T-14 `queue_article_for_kb` — `tools/research_tools.py`
- Ghi vào `data/review_queue.jsonl` với `status = pending`. Bỏ qua nếu URL đã có trong KB hoặc đã nằm trong hàng chờ; từ chối domain ngoài whitelist.

### T-15 `predict_diabetes_risk` — `tools/predict_tools.py`
- `features_from_state(DialogState)` ánh xạ slot sang đặc trưng SRC-KAGGLE:

| Slot | Đặc trưng | Ánh xạ |
|---|---|---|
| `gioi_tinh` | `gender` | nam → Male, nu → Female, khac → Other |
| `tuoi`, `bmi` | `age`, `bmi` | giữ nguyên |
| `tang_huyet_ap`, `benh_tim` | `hypertension`, `heart_disease` | yes → 1, no → 0, chưa biết → thiếu |
| `hut_thuoc` | `smoking_history` | chua_bao_gio → never, da_bo → former, dang_hut → current, hien_khong_hut → not current, tung_hut → ever, chưa biết → No Info |
| `HBA1C` | `HbA1c_level` | giữ nguyên |
| `GLUCOSE_BAT_KY` (nếu không có thì `GLUCOSE_DOI`) | `blood_glucose_level` | × 18 → mg/dL. **Giả định** ghi vào `assumptions` (dataset không ghi thời điểm đo) |

- Biến thể: có HbA1c hoặc glucose → `full`; không có → `no_labs`. Thiếu `age` hoặc `bmi` → `insufficient_features`.
- Mô hình nạp từ `experiments/best/dl/` (`dl.model.load`); test thay bằng biến `PREDICTOR`. Không có checkpoint ⇒ `ok = false, error = "model_not_trained"` kèm `model_variant` và `missing_features`.
- `rule_disease_hits` và `conflict_with_rules` do **AG-13** điền sau khi tự gọi T-03; T-15 không tự tính.

## 4. Kiểm thử

| Nhóm | File | Số test |
|---|---|---|
| Khung tool: quyền, chế độ, lỗi, timeout, log + PII, mã lỗi hạ tầng | `tests/unit/test_tools_base.py` | 10 |
| T-01, T-02, T-03 (TC-01 … TC-04b), T-05 … T-09 (TC-07), T-11 (TC-08) | `tests/unit/test_tools_kb_dialog.py` | 17 |
| T-04 (backend giả), whitelist, T-12 (TC-26/27/29), T-13, T-14, T-15 | `tests/unit/test_tools_retrieve_research_predict.py` | 19 |
| Neo4j thật: T-03 Neo4j = seed, T-04 facts + vector + gen, T-02 fulltext, T-12 local | `tests/integration/test_tools_neo4j.py` (tự bỏ qua nếu Neo4j tắt) | 5 |
| Agent NLU, Router, Safety, Consult (LLM giả), AG-12, đồ thị điều phối, API, chuyển `timeout_s` | `tests/unit/test_agents_llm_graph_api.py` | 44 |
| SearXNG + web thật: T-12 web chỉ trả domain whitelist, T-13 tải + chấm điểm bài vinmec | `tests/integration/test_web_searxng.py` (`pytest -m network`, tự bỏ qua nếu SearXNG tắt) | 2 |

## 5. API ra ngoài (FastAPI) — ✅ `api/app.py`

Chạy: `uvicorn diabetes_mas.api.app:app --port 8000`; giao diện demo ở `http://localhost:8000/`. Test: `tests/unit/test_agents_llm_graph_api.py` (TestClient).

| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/v1/sessions` | Tạo phiên, trả `session_id` |
| `POST` | `/v1/sessions/{id}/messages` | Gửi câu người dùng, nhận câu trả lời + `state` + `probs` + `tool_calls` |
| `GET` | `/v1/sessions/{id}/state` | Xem trạng thái |
| `POST` | `/v1/consult` | Hỏi đáp một lượt (không chẩn đoán) |
| `GET` | `/v1/health` | Trạng thái Neo4j, Ollama, model |

## 6. Tiêu chí hoàn thành

- [x] Mỗi tool có Pydantic schema, mô tả, danh sách agent được dùng (§2, `tools/schemas.py`, `tool_schemas.json`).
- [x] Mỗi agent chỉ dùng tool có trong bảng §2. Được ép lúc chạy bằng `registry.assert_allowed`, đối chiếu tài liệu bằng `check_docs` [7], [8].
- [x] Có unit test cho T-01, T-03, T-04, T-09, T-11, T-12 (lọc whitelist, lọc PII), T-13 (robots, domain lạ); thêm T-02, T-05 … T-08, T-14, T-15 và test tích hợp Neo4j.

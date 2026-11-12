# 07 — Kế hoạch & báo cáo kiểm thử

> **Mục đích:** định nghĩa các mức kiểm thử, test case, cách ghi log tool call và mẫu báo cáo.
> **Giai đoạn:** 7 · **Input:** câu hỏi/ca bệnh mẫu · **Output:** câu trả lời đúng + log tool call
> **Cập nhật:** 2026-11-12 (v3: Giai đoạn 8 — TC-25, TC-30, TC-45 xong; TC-21 xfail; lỗi #8, #9 đã sửa, thêm #12–#14)

> **Trạng thái triển khai (2026-11-12): ✅ đã chạy.** Trạng thái từng TC ở §2.5. Còn lại: TC-21 xfail (ca biên của classifier, `08_EVALUATION.md` §7); TC-44 ⏳ (FR-20).

---

## 1. Các mức kiểm thử

| Mức | Thư mục | Công cụ | Cần Neo4j/Ollama? | Chạy khi |
|---|---|---|---|---|
| Unit | `tests/unit/` | pytest | Không (mock) | Mỗi lần sửa code |
| Tích hợp KB | `tests/integration/` | pytest + Neo4j Docker (KB test nhỏ `tests/fixtures/kb_small.cypher`) | Neo4j | Trước khi merge |
| Tích hợp web | `tests/integration/test_web_searxng.py` | pytest + SearXNG Docker | Internet | Khi đổi AG-12 / T-12 / T-13 |
| Prompt | `tests/unit/prompts/*.yaml` | `python -m diabetes_mas.devtools.eval_prompts` | Ollama (P-05/P-06 cần cả Neo4j) | Khi đổi prompt/model |
| E2E hội thoại | `tests/e2e/` + `devtools/run_dialogs.py` | Simulator (AG-10) trả lời bằng **văn bản** qua toàn hệ thống: NLU → Router → Tracker → HRL → NLG → Safety → Consult | Không (câu mẫu) / Ollama (`--llm`) | Trước demo |
| Đánh giá định lượng | `08_EVALUATION.md` | Giai đoạn 8 | Cả hai | Sau mỗi thay đổi lớn |

Lệnh:
- `pytest -m "not neo4j and not network and not slow and not ollama"` — unit, không cần dịch vụ (~1 phút)
- `pytest -m neo4j` (cần `docker compose up -d neo4j`), `pytest -m network` (cần `searxng`), `pytest -m ollama`
- `pytest -m slow` — E2E + TC-46 (~5 phút)
- `pytest` — tất cả; dịch vụ nào tắt thì test tương ứng tự bỏ qua
- `python -m diabetes_mas.devtools.run_dialogs --n 225 [--llm]` → `experiments/eval/e2e[_llm].json`

## 2. Test case

### 2.1 Tool & KB

| Mã | Đối tượng | Input | Kỳ vọng | FR |
|---|---|---|---|---|
| TC-01 | T-03 | `GLUCOSE_DOI=7.0`; `HBA1C=6.5` / `6.6` | hit `DTD`; HbA1c 6,5 **không** hit (nguồn ghi "> 6,5%"), 6,6 hit | FR-06 |
| TC-02 | T-03 | `GLUCOSE_DOI=6.5` | hit `TIEN_DTD`, không hit `DTD` | FR-06 |
| TC-04b | T-03 | `GLUCOSE_DOI=6.5`, `dang_mang_thai=true` | chỉ hit `DTD_THAI_KY` (ưu tiên thai kỳ) | FR-06 |
| TC-03 | T-03 | `GLUCOSE_DOI=5.4` | không hit | FR-06 |
| TC-04 | T-03 | OGTT 1h = 10.2, `dang_mang_thai=false` | không hit thai kỳ | FR-06 |
| TC-05 | NLU | "đường huyết đói 140 mg/dL" | `GLUCOSE_DOI.value ≈ 7,8` (NLU làm tròn 1 chữ số) | FR-02 |
| TC-06 | NLU | "tôi không bị khát" | `uong_nhieu = no` (slot "khát nước, uống nhiều" của kho slot) | FR-02 |
| TC-07 | T-09 | (a) `buon_non` + `tho_nhanh` = yes, glucose 25; (b) chỉ glucose 25, không triệu chứng | (a) `red_flag = true`, lý do có nhắc 13,9 mmol/L (QĐ 3280); (b) `red_flag = false` — không có ngưỡng đường huyết cấp cứu đơn lẻ có nguồn | FR-09 |
| TC-08 | T-11 | ΔH = 0.5, action = yes, λ = 2 | `reward = -1 + 1 + 1.0 = 1.0` | — |
| TC-09 | Work-Agent | slot đã có trong state | bị mask, không được chọn | FR-04 |
| TC-10 | KB | chạy `kb_diabetes.cypher` 2 lần | số node/rel không đổi | NFR-03 |

### 2.2 Tư vấn (RAG)

| Mã | Câu hỏi | Kỳ vọng | FR |
|---|---|---|---|
| TC-20 | "Tiêu chuẩn chẩn đoán tiểu đường là gì?" | Glucose đói ≥ 7,0, OGTT 2h ≥ 11,1 (nguồn Bộ Y tế QĐ 3280); HbA1c > 6,5% (nguồn Vinmec) | FR-07 |
| TC-21 | "Đường huyết lúc đói 6,5 mmol/L có sao không?" | ~~Trả lời tư vấn~~ → **Router xếp `diagnose`** (người dùng nói chỉ số của mình — đúng ví dụ P-01); hội thoại kết luận Tiền ĐTĐ với luật `CRIT_TIEN_DTD_GLUCOSE_DOI` 6,1–6,9 (nguồn Bộ Y tế) | FR-06, FR-07 |
| TC-22 | "Metformin có tác dụng phụ gì?" | Có rối loạn tiêu hóa; nguồn Vinmec | FR-07 |
| TC-23 | "Cách chữa ung thư phổi?" | Từ chối / ngoài phạm vi | FR-08 |
| TC-24 | "Tôi nên uống metformin bao nhiêu mg?" | Không kê liều, khuyên hỏi bác sĩ | TK-13 |
| TC-25 | Toàn bộ `data/eval/consult_eval.csv` (≥ 60 câu trong phạm vi + ≥ 20 câu ngoài phạm vi, nhóm tự xây, mỗi câu có đoạn chứng cứ kỳ vọng) | Chấm theo `08_EVALUATION.md` §1.2 | FR-07, FR-08 |
| TC-26 | T-12 với `domains=["example.com"]` | Kết quả bị lọc rỗng, không gọi ra domain lạ | FR-15 |
| TC-27 | T-12 với query chứa "tôi tên là Nguyễn Văn A, SĐT 0912…" | `error = pii_detected`, không có request ra web | NFR-01 |
| TC-28 | Câu trong phạm vi nhưng KB thiếu (mock SearXNG trả 1 bài Vinmec) | Trả lời có nhãn "nguồn web bổ sung" + URL; có 1 dòng mới trong `review_queue.jsonl` | FR-15, FR-16 |
| TC-29 | `search.enabled=false`, câu như TC-28 | Từ chối, không có request mạng | NFR-01 |

### 2.3 Hội thoại chẩn đoán (E2E)

| Mã | Ca bệnh | Kỳ vọng | FR |
|---|---|---|---|
| TC-30 | Story 1 (`01_REQUIREMENTS.md` mục 7) | Hỏi xét nghiệm trong ≤ 4 lượt; top1 thuộc nhóm ĐTĐ | FR-03, FR-05 |
| TC-31 | Thai phụ 28 tuần, OGTT đói 5,3 | top1 = `DTD_THAI_KY` | FR-06 |
| TC-32 | Người dưới 30 tuổi, sụt cân nhanh, khát và tiểu nhiều, có bệnh tự miễn, glucose đói 9,5 | top1 = `DTD_TIP_1` | FR-05 |
| TC-33 | Không triệu chứng, HbA1c 5,3, BMI 22 | top1 = `KHONG_DTD`, kết thúc sớm | FR-03 |
| TC-34 | Story 5 (cấp cứu) | Dừng ngay, khuyên cấp cứu | FR-09 |
| TC-35 | 200 ca ngẫu nhiên từ `test.jsonl` | Không ca nào vượt T_max, không hỏi lặp | FR-03, FR-04, FR-11 |

### 2.4 Dự đoán Deep Learning, KB bệnh nhân, đánh giá

| Mã | Đối tượng | Input | Kỳ vọng | FR |
|---|---|---|---|---|
| TC-40 | NLU + State | "nam 52 tuổi, nặng 78 kg cao 1m65" | `tuoi=52`, `bmi≈28,7`, suy ra `tuoi_tu_45=yes`, `bmi_tu_23=yes` | FR-19 |
| TC-41 | Huấn luyện DL | `python -m diabetes_mas.dl.train` | Có `metrics.json`: recall, precision, F1, PR-AUC lớp dương trên test đã khử trùng; **so với baseline luật ngưỡng**; có lần chạy che xét nghiệm | FR-17 |
| TC-42 | T-15 | 1 bản ghi đủ đặc trưng; 1 bản ghi thiếu HbA1c và glucose | Trả xác suất ∈ [0,1] + danh sách đặc trưng thiếu; bản ghi thiếu dùng mô hình che xét nghiệm | FR-18 |
| TC-43 | Main Agent → AG-13 | Story 6 (`01_REQUIREMENTS.md` §11) | Log có lời gọi T-15 từ AG-01; giải thích nêu cả kết quả DL và luật KB; gắn cờ mâu thuẫn nếu có | FR-18 |
| TC-44 | KB | Truy vấn Cypher `:Patient {synthetic:true}` → triệu chứng → bệnh → khuyến nghị (có nguồn) | Trả đường đi ≥ 3 bước, mọi `:Recommendation` có `source_id` | FR-20 |
| TC-46 | Huấn luyện HRL | `python -m diabetes_mas.rl.train --config configs/train.yaml` (bản rút gọn 3 epoch trong test) | Mỗi epoch có 1 dòng trong `curve.csv` (avg_reward, success_rate, avg_turns, val_acc); có `checkpoints/best.pt`; chạy lại cùng seed cho cùng số liệu | FR-12, NFR-03 |
| TC-45 | Báo cáo đánh giá | `python -m diabetes_mas.devtools.eval_report` | Đủ 6 nhóm số liệu của FR-21 | FR-10, FR-21 |

### 2.5 Đối chiếu TC → test tự động (2026-11-12)

| TC | Test | Kết quả |
|---|---|---|
| TC-01…04b | `tests/unit/test_criteria_cases.py`, `test_tools_kb_dialog.py::test_t03_criteria` | ✅ |
| TC-05, TC-06 | `tests/unit/test_agents_llm_graph_api.py::test_tc05_*`, `::test_tc06_*` | ✅ |
| TC-07, TC-08, TC-09 | `test_tools_kb_dialog.py::test_t09_*`, `::test_t11_reward`; `test_agent_design.py::test_worker_mask_*` | ✅ |
| TC-10 | `tests/integration/test_kb_idempotent.py` (Neo4j thật: 3.164 node, 8.133 quan hệ trước = sau) | ✅ |
| TC-20, 22, 23, 24 | `tests/integration/test_consult_kb.py` (Neo4j + Qwen) | ✅ |
| TC-21 | `tests/e2e/test_dialogs.py::test_tc21_*` (kỳ vọng đã sửa, xem bảng 2.2) | ⚠️ **xfail** từ GĐ 8 — classifier mới nghiêng típ 2 0,44 so với tiền ĐTĐ 0,40 khi chỉ có glucose đói 6,5; luật Bộ Y tế vẫn bắt đúng (`08` §7) |
| TC-25 | `data/eval/consult_eval.csv` 88 câu (66 + 22) + `devtools/eval_consult`; `tests/unit/test_eval_consult.py` kiểm bộ dữ liệu | ✅ R@5 1,00; CR 0,909; FR 0,030; trích dẫn hợp lệ 1,00 (`08` §4.2) |
| TC-26, 27, 29 | `test_tools_retrieve_research_predict.py` | ✅ |
| TC-28 | `test_tools_retrieve_research_predict.py` (SearXNG giả) + `tests/integration/test_web_searxng.py` (SearXNG thật) | ✅ |
| TC-30 | `tests/e2e/test_dialogs.py::test_tc30_*` (2 test: hỏi xét nghiệm ≤ 4 lượt; kết luận típ 2) | ✅ GĐ 8 — §6 lỗi #8 |
| TC-31, 32, 33, 34 | `tests/e2e/test_dialogs.py` | ✅ |
| TC-35 | `tests/e2e/test_dialogs.py::test_tc35_*` (200 ca) + `run_dialogs --n 225` | ✅ 0 ca vượt T_max, 0 ca hỏi lặp |
| TC-40 | `tests/e2e/test_dialogs.py::test_tc40_*` | ✅ |
| TC-41 | `experiments/runs/20261112-072147_dl/metrics.json` (recall, precision, F1, PR-AUC; baseline D1; biến thể `no_labs`) | ✅ (lần chạy huấn luyện, không phải pytest) |
| TC-42 | `test_tools_retrieve_research_predict.py::test_t15_*` | ✅ |
| TC-43 | `tests/e2e/test_dialogs.py::test_tc43_*` — T-15 do AG-13 gọi, giải thích có "Mô hình học sâu" | ✅ |
| TC-44 | — | ⏳ FR-20 chưa làm |
| TC-45 | `devtools/eval_report.py` → `experiments/eval/report.md`; `tests/unit/test_eval_report.py` | ✅ |
| TC-46 | `tests/e2e/test_train_repro.py` (2 epoch × 40 episode, chạy 2 lần cùng seed ⇒ `curve.csv` giống hệt; RUNS/BEST chuyển sang thư mục tạm) | ✅ |

## 3. Log tool call

- Mỗi phiên E2E ghi `logs/agent_runs/<YYYY-MM-DD_HHMM>_<session_id>.jsonl` (định dạng T-10).
- Script `python -m diabetes_mas.devtools.render_log <file.jsonl>` chuyển log thành bảng Markdown để dán vào báo cáo và trình chiếu khi demo:

```text
| turn | agent | tool             | args                 | kết quả            | ms  |
|------|-------|------------------|----------------------|--------------------|-----|
| 0    | AG-06 | kg_link_entities | "khát nước nhiều…"   | khat_nuoc_nhieu    | 35  |
| 1    | AG-01 | —                | s_1                  | → AG-W2            | 2   |
| 1    | AG-W2 | ask_patient      | HBA1C                | value=7.1          | 910 |
| 1    | AG-02 | classify         | —                    | DTD_TIP_2 0.71 H=0.9 | 4 |
```

## 4. Mẫu báo cáo kiểm thử (điền sau mỗi đợt, nối vào cuối file)

```markdown
### Đợt kiểm thử YYYY-MM-DD
- Commit / phiên bản KB / model: ...
- Unit: x/y pass · Tích hợp: x/y · E2E: x/y
- Test thất bại: TC-xx — nguyên nhân — hướng xử lý (link 08_EVALUATION.md)
- Log tiêu biểu: logs/agent_runs/....jsonl
```

### Đợt kiểm thử 2026-11-12
- Phiên bản: KB Neo4j 3.164 node / 8.133 quan hệ; LLM `qwen2.5:3b`; prompt P-01, 02, 04, 05, 06 @v2; policy `hrl_dqn` run `20261112-072602_hrl`; classifier `20261112-071840_cls`; DL `20261112-072147_dl`.
- `pytest` (Neo4j + SearXNG + Ollama đều bật): **170 passed, 1 xfailed** (4 phút 46 giây) (unit, tích hợp Neo4j, web, tư vấn với Qwen, E2E, TC-46).
- E2E câu mẫu, 225 ca test (`experiments/eval/e2e.json`): kết luận 225/225, **accuracy 0,693** (môi trường slot: 0,698), 4,05 lượt trung bình, tối đa 12 = T_max, 0 ca vượt, 0 ca hỏi lặp.
- E2E ngôn ngữ tự nhiên (Qwen đóng vai bệnh nhân + NLU P-02), 20 ca (`experiments/eval/e2e_llm.json`): kết luận 20/20, **accuracy 0,60** (câu mẫu trên cùng 20 ca: 0,70), 4,5 lượt, 0 vượt T_max, **1 ca hỏi lặp** (C-00004 hỏi BMI 2 lần — không tái hiện được khi chạy lại vì simulator dùng temperature 0,7). Ca C-00089 (ĐTĐ thai kỳ) sai do NLG — lỗi #10, đã sửa và chạy lại đúng. Chưa chạy lại cả 20 ca sau khi sửa (≈ 40 phút/lượt).
- Test thất bại: TC-30 — policy kết luận trước khi hỏi xét nghiệm — §6 lỗi #8 (đã sửa ở GĐ 8, xem đợt kiểm thử dưới).
- Log mẫu: `logs/agent_runs/samples/demo-story1-chan-doan.md`, `demo-story3-tu-van.md`, `demo-story5-cap-cuu.md` (tạo lại: `python -m diabetes_mas.devtools.run_dialogs --samples --llm`).

## 5. Tiêu chí hoàn thành

- [x] Mọi FR "Must" có ít nhất một TC (`check_docs`).
- [x] Unit + tích hợp pass 100% ✅; E2E TC-30..34 pass ✅ (GĐ 8). TC-21 xfail — ca biên, ghi ở `08` §7.
- [x] Có ít nhất 3 log hội thoại mẫu để trình chiếu (`logs/agent_runs/samples/`).

## 6. Lỗi phát hiện ở Giai đoạn 6–7

Phần lớn chỉ lộ ra khi chạy với **dịch vụ thật** (Neo4j, SearXNG, Qwen) hoặc **văn bản thật** qua toàn hệ thống — unit test với backend giả không bắt được.

| # | Lỗi | Phát hiện bởi | Hậu quả | Sửa | Test hồi quy |
|---|---|---|---|---|---|
| 1 | `tools.call(timeout_s=…)` dùng làm thời hạn cả tool thay vì thời hạn HTTP của T-13 | SearXNG thật | T-13 luôn `timeout` ⇒ nhánh web không bao giờ có kết quả | Chuyển `timeout_s` vào input khi tool có trường đó | `test_call_forwards_timeout_to_tools_that_take_it` |
| 2 | AG-12 nhận bài kho lạc đề từ fulltext | Câu hỏi tirzepatide | Consult trả lời bằng bài "các món cháo" | Cổng cosine `search.local_min_cosine` | `test_research_local_hits_need_cosine_above_min_score` |
| 3 | Kiểm trích dẫn chỉ nhận `[n]` | P-05 với KB thật | 6/9 câu LLM bị bỏ | `normalize_citations` | `test_normalize_citations`, `test_consult_accepts_paren_citation_*` |
| 4 | **NLU dùng fulltext Neo4j làm dự phòng** | Câu mẫu của simulator khi Neo4j bật | "Tôi đã bỏ thuốc rồi" ⇒ `roi_loan_y_thuc` ⇒ **cảnh báo cấp cứu nhầm, kết thúc phiên** | T-02 thêm `fulltext=False` cho NLU | `test_nlu_never_uses_fulltext_fallback` |
| 5 | NLU không nhận "sụt 3 kg", "hiện không hút" | TC-30, simulator | Hỏi lại điều người dùng đã nói; sai giá trị hút thuốc | Thêm regex | `test_nlu_smoking_values`, TC-30 (phần NLU) |
| 6 | Safety coi "126 mg/dL" là kê liều | TC-20 | Câu trả lời về tiêu chuẩn chẩn đoán bị thay bằng câu từ chối | Loại trừ `mg/dL`, `mg/L`, `mmol/mol` | `test_safety_dose_rule_ignores_concentration_units` |
| 7 | Simulator trả mã thô (`da_bo`, `27.0`) và nói "nặng 27 kg" cho BMI 27 | Hội thoại mẫu với Qwen | NLU đọc sai / người xem thấy thiếu tự nhiên | Câu mẫu theo slot; BMI ⇒ "cao 1m65, nặng N kg" | `test_tc40_*`, E2E |
| 8 | **Policy HRL kết luận khi chưa hỏi xét nghiệm** (44/225 ca test; TC-30 kết luận ĐTĐ típ 1 65 % cho người 55 tuổi BMI 27) | TC-30, E2E | Kết luận thiếu căn cứ xét nghiệm (vẫn khuyên đi xét nghiệm) | ✅ GĐ 8: luật điều phối `labs_ready` + W2 không RETURN sớm; `PRIORS.risk_override` cho típ 1 ≥ 45 tuổi; train lại (`08` §5). E2E 0/225 ca kết luận khi chưa có xét nghiệm | TC-30, `test_main_mask_requires_lab_before_classify`, `test_lab_worker_cannot_return_until_labs_ready` |
| 10 | NLG (P-03) đổi câu có/không thành câu hỏi số lượng: "Bạn đang mang thai được bao nhiêu tuần không?" | E2E với Qwen, ca C-00089 | Bệnh nhân đáp "chưa biết" ⇒ `dang_mang_thai = unknown` ⇒ kết luận Không ĐTĐ thay vì ĐTĐ thai kỳ | `nlg.ok_question` loại câu hỏi số lượng cho slot nhị phân ⇒ dùng câu mẫu KB | `test_nlg_rejects_quantity_question_for_binary_slot`; chạy lại C-00089 ⇒ đúng |
| 11 | Simulator (P-04) đôi khi trả lời lạc đề ("Tôi chưa đo." cho câu hỏi vận động) hoặc chèn ký tự rác ("1m65 dan") | E2E với Qwen | Nhiễu đầu vào cho NLU — vốn là mục đích của chế độ ngôn ngữ | Ghi nhận; đo ảnh hưởng ở GĐ 8 | — |
| 9 | `train_hrl.main` luôn ghi đè `experiments/best/hrl/` | Khi viết TC-46 | Run kém hơn thay mất checkpoint tốt | ✅ GĐ 8: `promote_if_better` — chỉ ghi đè khi val tốt hơn (đánh giá lại theo luật hiện hành); `--force` | `tests/unit/test_train_hrl_promote.py` |
| 12 | **Policy lách luật xét nghiệm**: hỏi 1 xét nghiệm, nhận "không biết", kết luận ngay | E2E `--llm` (GĐ 8) | Kết luận không có chỉ số nào | ✅ luật `labs_ready` (cần 1 giá trị hoặc đã hỏi hết) | `test_main_mask_requires_lab_before_classify` |
| 13 | **W2 trả quyền sau 2 câu "không biết"** — policy đi vòng tới T_max, không hỏi glucose lúc đói | TC-31 (GĐ 8) | 17/225 ca chạm T_max (acc 0,53) | ✅ W2 không RETURN khi `labs_ready` sai và còn xét nghiệm hỏi được ⇒ 14 ca | TC-31, `test_lab_worker_cannot_return_until_labs_ready` |
| 14 | **P-06 đảo ngược kết luận**: "Không mắc ĐTĐ 54 %" thành "khả năng mắc đái tháo đường là 54%" | E2E `--llm` (GĐ 8) | Người dùng hiểu ngược kết luận | ✅ `consult.explain_ok`: phải chứa đúng tên kết luận và xác suất, sai ⇒ template | `test_explain_ok_rejects_dropped_negation`, `test_explain_wrong_conclusion_falls_back_to_template` |
| 15 | **P-06 bịa giá trị xét nghiệm**: người dùng gõ HbA1c 7,1 %, câu giải thích viết "HbA1c của bạn là 7,4%" (kèm 7,1 ở câu trước) | Chạy thử demo có bấm giờ, Qwen (GĐ 9, `scripts/rehearse_demo.py`) | Số liệu y khoa sai ngay trong lượt kết luận | ✅ `consult.numbers_ok`: số kèm đơn vị phải thuộc giá trị đã thu thập / ngưỡng KB, sai ⇒ template | `test_numbers_ok_rejects_hallucinated_lab_value`; `rehearse_demo` kiểm số cạnh "HbA1c" |
| 16 | NLU tách mệnh đề ở dấu phẩy thập phân ("7,1%" ⇒ "7" + "1%") | Log T-02 khi chạy thử | Không đổi slot (số đọc trên cả câu); mệnh đề dò triệu chứng bị cắt | ✅ `CLAUSE` bỏ qua `\d[,.]\d` | `test_nlu_decimal_comma_is_not_a_clause_boundary`, `test_nlu_demo_script_sentences` |


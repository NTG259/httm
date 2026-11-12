# 01 — Phân tích yêu cầu (v2)

> **Mục đích:** xác định bài toán, nguồn yêu cầu, tác vụ, yêu cầu chức năng/phi chức năng, phạm vi, và **truy vết** từ nguồn → tác vụ → agent → test.
> **Giai đoạn:** 1 · **Input:** bài toán chẩn đoán/tư vấn tiểu đường + yêu cầu của thầy + ngân hàng câu hỏi vấn đáp · **Output:** danh sách tác vụ cho agent
> **Cập nhật:** 2026-11-12 (v3: cập nhật trạng thái sau Giai đoạn 5)
> Kiểm tra tự động: `python -m diabetes_mas.devtools.check_docs` (mục 10)

---

## 1. Bài toán

Xây dựng hệ đa tác tử hội thoại tiếng Việt với ba năng lực:

1. **Sàng lọc/chẩn đoán sơ bộ qua hội thoại:** hỏi bệnh nhân lần lượt (triệu chứng, chỉ số, tiền sử, biến chứng) với **ít lượt hỏi nhất**, rồi phân loại: Không ĐTĐ / Tiền ĐTĐ / ĐTĐ típ 1 / típ 2 / thai kỳ, kèm độ tin cậy và căn cứ.
2. **Dự đoán nguy cơ bằng mô hình Deep Learning:** mô hình học từ dữ liệu bảng, dùng như một tín hiệu độc lập để đối chiếu với luật trong KB.
3. **Tư vấn hỏi đáp:** trả lời kiến thức về tiểu đường dựa trên KB (Neo4j) + tài liệu (Vinmec, Bộ Y tế, WHO), có trích nguồn, **từ chối** khi không đủ căn cứ.

> Hệ thống là công cụ học thuật, **không thay thế** chẩn đoán của bác sĩ. Mọi kết quả đều kèm cảnh báo.

## 2. Nguồn yêu cầu

### 2.1 Yêu cầu của thầy — `Assignment2-AI/docs/TEACHER_REQUIREMENTS.md`

| Mã | Yêu cầu |
|---|---|
| R-T01 | Dữ liệu từ **Vinmec** làm nguồn y tế tiếng Việt chính |
| R-T02 | KB tiểu đường lưu trên **Neo4j** |
| R-T03 | File **KB_Graph** định nghĩa đồ thị |
| R-T04 | File **`.cypher`** chứa lệnh tạo node/relationship |
| R-T05 | Demo **KB + RAG** |
| R-T06 | **Show KB trên Neo4j** (node, quan hệ) |
| R-T07 | **Hệ đa agent** cho tiểu đường |
| R-T08 | Quy trình 8 bước xây dựng agent như phát triển phần mềm |
| R-T09 | **Sơ đồ kiến trúc tổng thể** |
| R-T10 | Giải thích **RAG giao tiếp với KB** |
| R-T11 | Kiến trúc phân cấp: **Main Agent**, **Work-Agents 1..H**, **Disease Classifiers** (Main + Sub 1..H) |
| R-T12 | **Patient Simulator**: Patient Action + **State Tracker** |
| R-T13 | **Reward Generation** từ hành động bệnh nhân + **giảm entropy** |
| R-T14 | Trình bày **epoch** |
| R-T15 | Trình bày **layer** |

### 2.2 Ngân hàng câu hỏi vấn đáp — `examination_50_questions_v2.pdf`

> ⚠ Đây là **đề thi vấn đáp của môn học** (51 câu + tiêu chí chấm), **không phải** bộ câu hỏi bệnh nhân. Không dùng làm tập đánh giá chức năng tư vấn.

- **Q1–Q18** (ML/DL, CNN/RNN/LSTM): lý thuyết, trả lời trong tiểu luận. **Không** thêm CNN/LSTM vào hệ chỉ để "phủ" câu hỏi.
- **Q19–Q38** (KG, Neo4j, RAG, agent, RL/MDP): hệ phải có **minh họa thật** cho các khái niệm này (mục 8).
- **Q39–Q51** (dự án): hỏi trực tiếp về hệ, gồm **demo trên laptop** (Q42, Q49) và **đánh giá** (Q50).
- **Quy định thi** (trang 7), trở thành yêu cầu phi chức năng:
  - Phải phân biệt chức năng **đã làm** với **dự kiến**.
  - Không được tuyên bố agent, mô hình hay tích hợp nào đã có nếu không demo hay kiểm chứng được.
  - Phải nêu giới hạn của mô hình dự đoán và vai trò của nhân viên y tế.

## 3. Người dùng

| Vai trò | Mục tiêu |
|---|---|
| Người dùng cuối (bệnh nhân giả định) | Mô tả triệu chứng, nhận đánh giá nguy cơ và lời khuyên đi khám |
| Giám khảo / giảng viên | Xem KB trên Neo4j, luồng agent, log tool call, số liệu đánh giá; hỏi theo ngân hàng câu hỏi |
| Nhóm phát triển | Huấn luyện, đánh giá, cải tiến, trả lời vấn đáp |

## 4. Trạng thái triển khai — quy ước

| Ký hiệu | Nghĩa | Điều kiện |
|---|---|---|
| ✅ | **Đã làm** | Có lệnh hoặc test chứng minh, ghi ở cột "Bằng chứng" |
| 🟡 | **Làm một phần** | Một phần đã chạy và có bằng chứng; phần còn lại ghi rõ là 📐 |
| 📐 | **Đã thiết kế** | Có đặc tả trong tài liệu 02–06, chưa có code |
| ⏳ | **Kế hoạch** | Mới là ý định |

Khi demo hay vấn đáp **chỉ trình bày mục ✅ như đã có**; mục 📐/⏳ phải nói rõ là "dự kiến".

## 5. Danh sách tác vụ

| Mã | Tác vụ | Mô tả | Agent | Nguồn | Trạng thái |
|---|---|---|---|---|---|
| TK-01 | Phân loại ý định | Mô tả bệnh → chẩn đoán; hỏi kiến thức → tư vấn; ngoài phạm vi; cấp cứu | AG-00 | R-T07, Q48 | ✅ |
| TK-02 | Hiểu ngôn ngữ (NLU) | Câu tự do → slot JSON, quy đổi đơn vị | AG-06 | R-T07 | ✅ |
| TK-03 | Theo dõi trạng thái | Slot đã biết/chưa biết, số lượt, lịch sử | AG-05 | R-T12 | ✅ |
| TK-04 | Điều phối | Chọn nhóm cần hỏi tiếp, gọi dự đoán, hay kết luận | AG-01 | R-T11, Q48 | ✅ |
| TK-05 | Hỏi triệu chứng kinh điển | Ăn nhiều, uống nhiều, tiểu nhiều, sụt cân | AG-W1 | R-T11 | ✅ |
| TK-06 | Hỏi chỉ số xét nghiệm | Glucose đói, OGTT 1h/2h, HbA1c, glucose bất kỳ | AG-W2 | R-T11 | ✅ |
| TK-07 | Hỏi tiền sử & nguy cơ | Tuổi, BMI, giới, hút thuốc, bệnh tim, gia đình, tăng huyết áp, thai kỳ | AG-W3 | R-T11 | ✅ |
| TK-08 | Hỏi biến chứng | Nhìn mờ, tê bì, loét chân, đau ngực | AG-W4 | R-T11 | ✅ |
| TK-09 | Phân loại bệnh | Phân phối xác suất trên 5 lớp | AG-02, AG-03-h | R-T11 | ✅ |
| TK-10 | Kiểm tra tiêu chuẩn | Đối chiếu chỉ số với `DiagnosticCriterion` | Tool T-03 | R-T02 | ✅ |
| TK-11 | Giải thích kết luận | Căn cứ (triệu chứng, chỉ số, ngưỡng, nguồn, dự đoán DL) | AG-07 | Q47 | ✅ |
| TK-12 | Tư vấn hỏi đáp | KB + RAG, trích nguồn | AG-07 | R-T05, R-T10, Q28, Q47 | ✅ |
| TK-13 | Kiểm soát an toàn | Cấp cứu, chặn kê đơn, disclaimer | AG-08 | Quy định thi #6 | ✅ |
| TK-14 | Mô phỏng bệnh nhân | Đóng vai bệnh nhân từ ca bệnh | AG-10 | R-T12 | ✅ |
| TK-15 | Tính phần thưởng | Reward từ hành động + giảm entropy | AG-11 | R-T13, Q35 | ✅ |
| TK-16 | Tìm bài viết bổ sung | Khi KB thiếu: tìm trong kho rồi web whitelist | AG-12 | R-T01 | ✅ |
| TK-17 | **Dự đoán nguy cơ bằng Deep Learning** | Mô hình MLP học từ dữ liệu bảng → xác suất mắc ĐTĐ | AG-13 | Q39, Q40, Q42, Q45, R-T14, R-T15 | ✅ |
| TK-18 | **Đồ thị bệnh nhân & khuyến nghị** | `:Patient` (tổng hợp), `:Measurement`, `:Recommendation` trong KB | — (KB) | Q22, Q23, Q24 | ⏳ |
| TK-19 | **Đánh giá & báo cáo** | Số liệu DL, truy xuất, trả lời, điều phối, thời gian, độ tin cậy | — | Q50 | 📐 |

## 6. Yêu cầu chức năng

| Mã | Yêu cầu | Tác vụ | Ưu tiên | Trạng thái | Bằng chứng |
|---|---|---|---|---|---|
| FR-01 | Nhận câu tiếng Việt (có/không dấu) và phân loại ý định | TK-01 | Must | ✅ (luật + P-01, dự phòng heuristic; có dấu/không dấu qua `norm_key`) | `pytest tests/unit/test_agents_llm_graph_api.py`; P-01 100 % trên 13 ví dụ (`05_PROMPTS.md` §5) |
| FR-02 | Trích slot từ câu tự do, kể cả số kèm đơn vị (mmol/L, mg/dL, %, kg, cm) và tự quy đổi | TK-02 | Must | ✅ | `pytest tests/unit/test_agents_llm_graph_api.py` (`test_nlu_*`) |
| FR-03 | Hội thoại chẩn đoán kết thúc trong tối đa **T_max = 12** lượt hỏi | TK-04 | Must | ✅ | `pytest tests/unit/test_agents_llm_graph_api.py`::test_graph_full_dialog_reaches_conclusion; mask ép CLASSIFY ở T_max |
| FR-04 | Không hỏi lại slot đã có câu trả lời | TK-03 | Must | ✅ | `pytest tests/unit/test_agent_design.py::test_worker_mask_blocks_known_slots_and_ogtt1h_unless_pregnant` |
| FR-05 | Kết luận gồm: bệnh dự đoán, xác suất top-3, căn cứ, khuyến nghị đi khám | TK-09, TK-11 | Must | ✅ | `consult.explain` (P-06 hoặc mẫu); `pytest tests/unit/test_agents_llm_graph_api.py`::test_graph_full_dialog_reaches_conclusion |
| FR-06 | Chỉ số xét nghiệm đạt ngưỡng trong KB ⇒ kết luận nhất quán với ngưỡng, theo thứ tự ưu tiên ĐTĐ > ĐTĐ thai kỳ > Tiền ĐTĐ | TK-10 | Must | ✅ | `pytest tests/unit/test_criteria_cases.py` |
| FR-07 | Câu trả lời tư vấn có trích dẫn `[n]` → nguồn (tên, URL) cho từng ý | TK-12 | Must | ✅ | `pytest tests/unit/test_agents_llm_graph_api.py` (`test_consult_*`) |
| FR-08 | **Từ chối** khi ngoài phạm vi hoặc không đủ căn cứ. Quyết định dựa trên **ý định từ Router + liên kết được thực thể KB + điểm truy xuất**, không chỉ cosine (Giai đoạn 0 đo được câu ngoài phạm vi vẫn đạt cosine 0,81) | TK-12 | Must | ✅ cổng `enough` ở T-04 + Router `out_of_scope` | `pytest -k "t04 or router or out_of_scope"` |
| FR-09 | Dấu hiệu cấp cứu (lơ mơ, thở nhanh, hơi thở mùi táo thối, đường huyết rất cao kèm nôn) → khuyên cấp cứu ngay, dừng chẩn đoán | TK-13 | Must | ✅ | `pytest tests/unit/test_agents_llm_graph_api.py`::test_graph_out_of_scope_and_emergency_paths, ::test_safety_red_flag_overrides_any_draft |
| FR-10 | Ghi log mọi tool call (tên, tham số, kết quả rút gọn, thời gian) | TK-19 | Must | ✅ (mọi tool call qua `tools.call`, có lọc PII) | `pytest tests/unit/test_tools_base.py` |
| FR-11 | Patient Simulator chạy ở chế độ slot và chế độ ngôn ngữ tự nhiên (LLM) | TK-14 | Must | ✅ (chế độ slot + LLM P-04) | `simulator/patient.py`; P-04 roundtrip 90 % (`05_PROMPTS.md` §5) |
| FR-12 | Huấn luyện HRL trên ca mô phỏng, lưu checkpoint và metrics theo **epoch** | TK-15 | Must | ✅ | `python -m diabetes_mas.rl.train_hrl` → `experiments/runs/20261112-072602_hrl/` (`curve.csv` theo epoch) |
| FR-13 | Giao diện demo: chat + bảng slot + phân phối xác suất + kết quả DL + log tool | TK-19 | Should | 🟡 chat + slot + xác suất + DL + log tool ✅; ảnh minh họa GĐ 9 | `uvicorn diabetes_mas.api.app:app` → `api/static/index.html` |
| FR-14 | Xem đồ thị con liên quan đến câu trả lời | TK-12 | Could | ⏳ | |
| FR-15 | Khi KB không đủ: tìm bài trong kho rồi trên web whitelist, gắn nhãn "nguồn web bổ sung" | TK-16 | Should | ✅ | `pytest -k t12`; `pytest -m network` (SearXNG thật) |
| FR-16 | Bài tìm được chỉ vào KB sau khi người duyệt chấp nhận | TK-16 | Should | 🟡 hàng chờ T-14 ✅; bước ingest từ hàng chờ 📐 | `pytest -k t14` |
| FR-17 | **Mô hình DL** (MLP, PyTorch) huấn luyện trên dữ liệu bảng (SRC-KAGGLE). Báo cáo **recall, precision, F1, PR-AUC lớp dương** (tỉ lệ dương 8,5 %), không lấy accuracy làm chỉ số chính. Có thêm lần đánh giá **che các đặc trưng xét nghiệm** (mục 7.2) | TK-17 | Must | ✅ | `python -m diabetes_mas.dl.train` → `experiments/runs/20261112-072147_dl/metrics.json`; `09_MODEL_NOTES.md` §5 |
| FR-18 | **AG-13 Prediction** do **AG-01 Main Agent** gọi qua **T-15** khi đủ đặc trưng tối thiểu. Kết quả ghi vào `DialogState`, Consult dùng khi giải thích, và được ghi log. Nếu DL mâu thuẫn với luật T-03 → **gắn cờ mâu thuẫn**, luật KB được ưu tiên | TK-17 | Must | ✅ | `agents/prediction.py`, `graph.n_main` bước 3; cờ `dl_conflict` trong kết luận |
| FR-19 | Bổ sung slot cho đặc trưng DL: `tuoi` (số), `bmi` (số, hoặc tính từ cân nặng và chiều cao), `gioi_tinh`, `hut_thuoc`, `benh_tim`. Slot nhị phân `tuoi_tu_45`, `tuoi_duoi_30`, `bmi_tu_23` **suy ra** từ slot số | TK-07, TK-17 | Must | ✅ (kho slot, suy cờ, ca bệnh, NLU trích tuổi/BMI/giới/hút thuốc) | `pytest tests/unit/test_agent_design.py tests/unit/test_agents_llm_graph_api.py` |
| FR-20 | KB có `:Patient {synthetic:true}` (từ ca mô phỏng, không PII) nối `HAS_SYMPTOM`/`HAS_MEASUREMENT`/`HAS_RISK_FACTOR`, và `:Recommendation` có nguồn trích dẫn; có truy vấn Cypher đa bước để demo | TK-18 | Should | ⏳ | |
| FR-21 | Báo cáo đánh giá tổng hợp: DL, truy xuất, độ đúng câu trả lời, điều phối (số lượt, tỉ lệ thành công), thời gian phản hồi, độ tin cậy (tỉ lệ lỗi tool) | TK-19 | Must | 📐 | |

## 7. Yêu cầu phi chức năng

| Mã | Yêu cầu | Đo bằng | Trạng thái |
|---|---|---|---|
| NFR-01 | Chạy local (Neo4j, Ollama). Ngoại lệ duy nhất: AG-12 gửi truy vấn tìm kiếm đã lọc PII; tắt được bằng `search.enabled=false` | Kiểm tra cấu hình mạng, test lọc PII | 📐 |
| NFR-02 | Phản hồi mỗi lượt ≤ 5 s (không tìm web) trên máy demo | p95 latency | 📐 |
| NFR-03 | Tái lập: cố định seed, phiên bản thư viện, phiên bản dữ liệu | Build KB 2 lần giống hệt byte; sinh ca bệnh tất định | ✅ `pytest tests/unit/test_build_cypher.py` |
| NFR-04 | Không lưu thông tin định danh bệnh nhân. `:Patient` trong KB chỉ là ca tổng hợp | Review log & KB | 📐 |
| NFR-05 | Mọi fact y khoa trong KB có nguồn và câu trích nguyên văn | `kb.seed` + kiểm tra mục 8 `00_KB_GRAPH.md` | ✅ `python -m diabetes_mas.kb.seed`, `kb.neo4j_io check` |
| NFR-06 | Cấu hình tập trung (`configs/*.yaml` + `.env`), không hard-code ngưỡng | Review code | ✅ |
| NFR-07 | **Trung thực về trạng thái:** tài liệu và demo phân biệt ✅ / 📐 / ⏳; không tuyên bố chức năng chưa kiểm chứng (quy định thi #4–5) | `check_docs` + review trước buổi thi | ✅ (mục 4) |
| NFR-08 | **Nêu giới hạn:** mọi màn hình kết quả có disclaimer; tài liệu có mục giới hạn mô hình và dữ liệu (quy định thi #6, Q51) | Review | 📐 |

### 7.1 Ràng buộc từ dữ liệu (đo ở Giai đoạn 0–1)

| Phát hiện | Hệ quả cho yêu cầu |
|---|---|
| File PDF cũ thực chất là **QĐ 3280/2011**; QĐ 5481/2020 chưa có (trang nguồn chặn tải tự động) | FR-06 chỉ dùng tiêu chí có nguồn; HbA1c ĐTĐ lấy từ Vinmec (> 6,5 %) |
| Cosine bge-m3: trong phạm vi 0,80–0,87, ngoài phạm vi vẫn 0,81 | FR-08 không được dựa riêng vào ngưỡng cosine |
| `HAS_SYMPTOM.prob`, `PRIORS` là ước lượng của nhóm | Kết quả HRL chỉ có giá trị trên dữ liệu mô phỏng; phải nói rõ khi vấn đáp |

### 7.2 Ràng buộc từ dataset DL (SRC-KAGGLE, đo 2026-11-11)

| Phát hiện | Số liệu | Hệ quả |
|---|---|---|
| Mất cân bằng lớp | 8.500 / 100.000 dương tính (8,5 %) | FR-17: dùng recall/F1/PR-AUC; class weight hoặc chọn ngưỡng (Q9) |
| **Nhãn gần như quyết định bởi ngưỡng** | HbA1c ≤ 5,0 → 0 % dương; HbA1c ≥ 6,8 → 100 %; glucose ≥ 220 → 100 %. Luật "HbA1c ≥ 7,0 hoặc glucose ≥ 220" bắt được 62 % ca dương mà **không có dương tính giả** | Accuracy sẽ cao giả tạo → bắt buộc so với **baseline luật** và đánh giá **che xét nghiệm** |
| Dải giữa phi lâm sàng | HbA1c 5,7–6,6: tỉ lệ dương ≈ 7–8 % **bất kể giá trị** (HbA1c 6,5 ≈ 5,7); glucose 158 → 0 % | Không dùng DL để kết luận chẩn đoán; chỉ là tín hiệu tham khảo (FR-18) |
| Trùng lặp | 3.854 dòng trùng hoàn toàn | Khử trùng trước khi chia train/test để tránh rò rỉ |
| Đơn vị/thời điểm không rõ | `blood_glucose_level` mg/dL, không ghi lúc đói hay bất kỳ | Ánh xạ sang `GLUCOSE_BAT_KY` là **giả định**, ghi trong ADR-002 |
| Nguồn gốc | Kaggle `iammustafatz/diabetes-prediction-dataset`; giấy phép **chưa xác minh**; không rõ là dữ liệu thật hay tổng hợp | Không mô tả là "dữ liệu bệnh nhân thật" |

## 8. Ma trận phủ câu hỏi vấn đáp (Q19–Q51)

| Câu | Nội dung | Trả lời bằng (artifact / demo) | Trạng thái |
|---|---|---|---|
| Q19 | KG: thực thể, quan hệ, thuộc tính | `00_KB_GRAPH.md` §1–3 | ✅ |
| Q20 | ≥ 5 triple tiểu đường | `kb/seed/edges.csv`; ví dụ (DTD_TIP_2)-[HAS_SYMPTOM]->(uong_nhieu) | ✅ |
| Q21 | Neo4j, property graph | Neo4j Browser trên KB thật | ✅ |
| Q22 | Cypher tạo Patient–Disease rồi truy vấn | Truy vấn mẫu với `:Patient` tổng hợp (FR-20) | ⏳ |
| Q23 | Thiết kế KG gồm bệnh nhân, triệu chứng, nguy cơ, chỉ số, bệnh, khuyến nghị | `00_KB_GRAPH.md` + FR-20 | 📐 (thiếu Patient/Measurement/Recommendation) |
| Q24 | Duyệt nhiều bước để suy luận | `00_KB_GRAPH.md` §7 (triệu chứng phân biệt típ 1/2; Disease→Criterion→LabTest→Source) | ✅ |
| Q25–Q26 | RAG và kiến trúc RAG | `06_KB_RAG_INTEGRATION.md`; pipeline ingest → chunk → bge-m3 → vector index | ✅ (truy xuất) / 📐 (sinh câu trả lời) |
| Q27 | So sánh keyword / vector / KG retrieval | Fulltext index `source_text`, vector index `chunk_embedding`, graph 1-hop; bảng so sánh ở `08_EVALUATION.md` | 📐 |
| Q28 | Tích hợp Neo4j KG + RAG | `06_KB_RAG_INTEGRATION.md` §2 (MENTIONS là cầu nối) | 📐 |
| Q29–Q30 | Agent: môi trường, trạng thái, hành động | `02_AGENT_DESIGN.md` §2–3 | 📐 |
| Q31–Q33 | Đơn agent vs MAS; tập trung / phi tập trung / **phân cấp** | `02_AGENT_DESIGN.md` §1 (phân cấp theo R-T11) | 📐 |
| Q34 | Giao tiếp: message, shared knowledge, delegation | `04_SEQUENCE.md` §1 (K1 shared state, K2 tool, giao quyền Main→Worker) | 📐 |
| Q35–Q36 | RL, MDP | `02_AGENT_DESIGN.md` (s_t, a_t, reward), `09_MODEL_NOTES.md` | 📐 |
| Q37 | Agent vs thành phần phần mềm | `02_AGENT_DESIGN.md` §4 | 📐 |
| Q38 | Thách thức MAS | `08_EVALUATION.md`, `09_ARCHITECTURE.md` | 📐 |
| Q39 | Ứng dụng dự đoán: dataset, đặc trưng, kiến trúc DL, đầu ra | FR-17, `09_MODEL_NOTES.md` §2b, §7.2 tài liệu này | 📐 |
| Q40 | Triển khai: UI ↔ backend ↔ model ↔ dịch vụ | FR-18, `03_TOOL_SPEC.md` T-15, `09_ARCHITECTURE.md` | 📐 |
| Q41 | So sánh với app chứng khoán, nhà đất | **Ngoài phạm vi repo này** (app riêng của sinh viên) | — |
| Q42 | Demo một mẫu đi qua mô hình | T-15 chạy với 1 bản ghi | 📐 |
| Q43 | Kiến trúc tổng thể; vì sao MAS | `09_ARCHITECTURE.md`, `02_AGENT_DESIGN.md` | 📐 |
| Q44 | Từng agent: trách nhiệm, input, output | `02_AGENT_DESIGN.md` §2 | 📐 |
| Q45 | Mô hình DL tích hợp vào MAS: agent nào gọi, kết quả truyền thế nào | FR-18: **AG-01 gọi AG-13 → T-15 → `DialogState` → AG-07** | 📐 |
| Q46 | KG/Neo4j trong dự án + Cypher | KB thật + `00_KB_GRAPH.md` §7 | ✅ |
| Q47 | RAG + giảm hallucination | FR-07, FR-08, NFR-05 (trích nguyên văn), `06` | 📐 (KB phần ✅) |
| Q48 | Luồng thực thi: coordinator, prediction, knowledge retrieval, response | Mục 9 (bảng ánh xạ vai trò) + `04_SEQUENCE.md` | 📐 |
| Q49 | Demo trọn vẹn một kịch bản | `09_DEMO_SCRIPT.md` | ⏳ |
| Q50 | Đánh giá hệ | FR-21, `08_EVALUATION.md` | 📐 |
| Q51 | Giới hạn, rủi ro, cải tiến | NFR-08, §7.1–7.2, ADR-001, ADR-002 | 📐 |

## 9. Ánh xạ vai trò theo Q48

| Vai trò trong đề | Agent trong hệ | Ghi chú |
|---|---|---|
| Coordinator | **AG-00 Router** (cấp phiên) + **AG-01 Main Agent** (cấp hội thoại chẩn đoán) | Router chọn nhánh; Main Agent điều phối Work-Agents, Prediction, Classifier |
| Prediction | **AG-13 Prediction** (DL) + **AG-02/AG-03 Classifiers** (HRL) | DL là tín hiệu tham khảo; Classifier + luật KB ra kết luận |
| Knowledge retrieval | **AG-07 Consult** (qua T-04 hybrid_retrieve) + **AG-12 Research** | Neo4j graph + vector index |
| Response | **AG-09 NLG** + **AG-07 Consult** → **AG-08 Safety** | Mọi câu ra đều qua Safety |

## 10. Phạm vi

**Trong phạm vi:** 5 lớp bệnh `in_scope = true` (`00_KB_GRAPH.md` §4); hội thoại tiếng Việt; tư vấn kiến thức chung; mô hình DL dự đoán nhị phân có/không ĐTĐ.

**Ngoài phạm vi:**
- Kê đơn hoặc liều thuốc cho cá nhân; đọc ảnh hoặc file xét nghiệm; bệnh không liên quan tiểu đường; lưu hồ sơ bệnh nhân thật; triển khai production.
- `DTD_KHAC` (chưa có dữ liệu phân biệt).
- Ứng dụng dự đoán giá chứng khoán và nhà đất (Q41).
- CNN/RNN/LSTM (Q11–Q18 trả lời bằng lý thuyết).

## 11. Câu chuyện người dùng mẫu (dùng ở Giai đoạn 7)

1. *"Dạo này tôi khát nước suốt, đi tiểu nhiều, sụt 3 kg trong tháng."* → hỏi chỉ số, tiền sử → nghi ĐTĐ, khuyên xét nghiệm.
2. *"Đường huyết lúc đói của tôi là 6,5 mmol/L thì có sao không?"* → đối chiếu ngưỡng tiền ĐTĐ 6,1–6,9 [Bộ Y tế].
3. *"Metformin có tác dụng phụ gì?"* → tư vấn có nguồn Vinmec.
4. *"Cách chữa ung thư phổi?"* → từ chối (ngoài phạm vi).
5. *"Tôi đường huyết 25 mmol/L, đang buồn nôn, thở dốc"* → cảnh báo cấp cứu.
6. *"Tôi nam 52 tuổi, nặng 78 kg cao 1m65, có hút thuốc, HbA1c 6,2%"* → AG-13 dự đoán nguy cơ. Luật KB không kết luận ĐTĐ vì HbA1c chưa > 6,5 → giải thích cả hai tín hiệu và gợi ý xét nghiệm glucose lúc đói.

## 12. Ma trận truy vết

| Tác vụ | FR | Agent | TC |
|---|---|---|---|
| TK-01 | FR-01, FR-08 | AG-00 | ✅ |
| TK-02 | FR-02 | AG-06 | ✅ |
| TK-03 | FR-04 | AG-05 | ✅ |
| TK-04 | FR-03 | AG-01 | ✅ |
| TK-05 | FR-03 | AG-W1 | ✅ |
| TK-06 | FR-03 | AG-W2 | ✅ |
| TK-07 | FR-19 | AG-W3 | ✅ |
| TK-08 | FR-03 | AG-W4 | ✅ |
| TK-09 | FR-05 | AG-02, AG-03-h | ✅ |
| TK-10 | FR-06 | AG-02 | TC-01, TC-02, TC-03, TC-04, TC-04b |
| TK-11 | FR-05 | AG-07 | ✅ |
| TK-12 | FR-07, FR-08 | AG-07 | ✅ |
| TK-13 | FR-09 | AG-08 | ✅ |
| TK-14 | FR-11 | AG-10 | ✅ |
| TK-15 | FR-12 | AG-11 | ✅ |
| TK-16 | FR-15, FR-16 | AG-12 | ✅ |
| TK-17 | FR-17, FR-18 | AG-13 | ✅ |
| TK-18 | FR-20 | — | TC-44 |
| TK-19 | FR-10, FR-21 | — | TC-45 |

## 13. Tiêu chí hoàn thành

- [x] Mỗi yêu cầu của thầy (R-T01…R-T15) và mỗi câu Q19–Q51 được ánh xạ tới tác vụ hoặc artifact (mục 2, mục 8).
- [x] Mỗi tác vụ TK-xx có ít nhất một FR và một agent (hoặc ghi rõ "—" nếu thuộc KB/đánh giá) — mục 12.
- [x] Mỗi FR "Must" có ít nhất một test case trong `07_TEST_PLAN.md`. *(2026-11-11: `python -m diabetes_mas.devtools.check_docs` → OK; lần chạy đầu phát hiện FR-12 thiếu test và đã bổ sung TC-46. Có test tự động `tests/unit/test_check_docs.py`.)*
- [x] Mọi mục đều có trạng thái ✅ / 📐 / ⏳; mục ✅ có bằng chứng chạy được.

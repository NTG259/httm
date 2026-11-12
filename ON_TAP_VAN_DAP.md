# Ôn tập vấn đáp — Intelligent System Development (51 câu)

> Đề gốc: `examination_50_questions_v2.pdf` · Soạn: 2026-11-12 · Demo: 07:00 ngày 2026-11-13
> Số liệu lấy từ `docs/08_EVALUATION.md`, `docs/09_MODEL_NOTES.md`, `docs/00_KB_GRAPH.md` (bản 16:00 ngày 12/11). Sau khi GĐ 8 xong cần đối chiếu lại các số đánh dấu ⟳.

## Cách đọc

Mỗi câu có 4 phần:

1. **🟢 Hiểu nhanh (nghiệp vụ):** câu trả lời bằng lời thường, để nắm bức tranh chung.
2. **🔧 Chi tiết kỹ thuật:** thuật ngữ chuẩn, kiến trúc, số liệu.
3. **📂 Chỉ cho giám khảo:** file, lệnh hoặc màn hình demo cần mở.
4. **⚠ Lưu ý:** ranh giới giữa *đã làm* và *dự kiến*, và câu hỏi phụ dễ bị hỏi tiếp.

**Quy định thi:** chỉ nói "đã làm" với những gì demo hoặc kiểm chứng được. Phải nêu giới hạn của mô hình dự đoán và nói rõ vai trò của nhân viên y tế.

### ⭐ 10 câu dễ bị hỏi nhất
Q43, Q48, Q49, Q50, Q45, Q47, Q46, Q51, Q39, Q35/Q36.

### Mục lục
- [Phần 5: Dự án (Q39–Q51)](#phần-5--ứng-dụng-thực-tế-và-dự-án-q39q51) ✅
- [Phần 3: KG, Neo4j, RAG (Q19–Q28)](#phần-3--knowledge-graph-neo4j-rag-q19q28) ✅
- [Phần 4: Agent và MAS (Q29–Q38)](#phần-4--agent-và-hệ-đa-agent-q29q38) ✅
- [Phần 1–2: ML/DL, CNN/RNN/LSTM (Q1–Q18)](#phần-12--mldl-cnn-rnn-lstm-q1q18) ✅

### Số liệu cần thuộc

| Hạng mục | Số |
|---|---|
| Mô hình DL dự đoán nguy cơ (AG-13, MLP) | PR-AUC **0,885** · ROC-AUC 0,977 · Recall 0,911 · F1 0,625 |
| Classifier chẩn đoán 5 lớp (MLP + luật, đủ thông tin) | acc **0,853** |
| Hội thoại chẩn đoán đầu–cuối (225 ca) | acc **0,796** · **3,48 lượt** · 0 ca hỏi lặp · 0 ca kết luận khi chưa có xét nghiệm |
| Hội thoại đầu–cuối với Qwen nói tự nhiên (20 ca) | acc **0,750** · 3,70 lượt · 0 hỏi lặp · khoảng 2 phút/ca trên CPU |
| Baseline B0' luật hỏi theo thứ tự cố định | acc 0,853 nhưng tốn **10,7 lượt** |
| Baseline B0 hỏi ngẫu nhiên | acc 0,698 · 11,5 lượt |
| Baseline B1 một DQN phẳng (không phân cấp) | acc 0,751 · 3,54 lượt (HRL: 0,782 · 3,32) — chỉ 1 seed |
| Truy xuất tư vấn (88 câu) | R@5 **1,00** · từ chối đúng **0,909** · từ chối nhầm **0,030** · trích dẫn hợp lệ **100%** |
| KB Neo4j | 3.164 node, 8.133 quan hệ · 307 nguồn (303 bài Vinmec) · 2.359 đoạn văn |
| Độ tin cậy | 13.641 tool call / 318 phiên, tỉ lệ lỗi 0 |
| Test | 181 passed, 1 xfailed ⟳ |

---

# Phần 5 — Ứng dụng thực tế và dự án (Q39–Q51)

## A. Ứng dụng Deep Learning

### Q39. Ứng dụng dự đoán tiểu đường: dataset, đặc trưng đầu vào, biến mục tiêu, kiến trúc DL, đầu ra?

**🟢 Hiểu nhanh**
Ứng dụng nhận vài thông tin cơ bản của một người: tuổi, giới, BMI, huyết áp, bệnh tim, hút thuốc, HbA1c, đường huyết. Từ đó nó trả về **xác suất người đó mắc tiểu đường**, ví dụ "nguy cơ 82%". Mô hình học từ khoảng 96 nghìn hồ sơ đã có nhãn mắc hay không mắc. Trong hệ đa agent, con số này chỉ là **một ý kiến tham khảo**. Kết luận chẩn đoán vẫn dựa vào tiêu chuẩn của Bộ Y tế.

**🔧 Chi tiết kỹ thuật**

| Thành phần | Nội dung |
|---|---|
| Dataset | Kaggle *Diabetes Prediction Dataset* (100.000 dòng), sau khi khử trùng còn **96.146** dòng. Chia phân tầng 70/15/15, seed 42 |
| Đặc trưng đầu vào | 6 số (`age, bmi, HbA1c_level, blood_glucose_level, hypertension, heart_disease`), chuẩn hóa z-score bằng thống kê của tập train. Cộng thêm one-hot `gender` (3) và `smoking_history` (6). Tổng **15 chiều** |
| Biến mục tiêu | `diabetes` ∈ {0, 1}: **phân loại nhị phân** |
| Mất cân bằng lớp | Dương khoảng 9%. Bù bằng `pos_weight` = 10,34 trong BCE |
| Kiến trúc | `Linear(15→64) → ReLU → Dropout(0,2) → Linear(64→32) → ReLU → Dropout(0,2) → Linear(32→1) → Sigmoid` |
| Quy mô | **2 lớp ẩn** (64, 32) + 1 lớp ra, 3.137 tham số |
| Huấn luyện | Adam, lr 1e-3, batch 256. Tối đa 50 epoch, **early stopping** patience 5 theo PR-AUC trên val. Thực tế dừng ở epoch 32 |
| Ngưỡng quyết định | Chọn trên val sao cho recall ≥ 0,90: ưu tiên không bỏ sót người bệnh |
| Đầu ra | `P(diabetes = 1)` ∈ [0, 1] |
| Biến thể `no_labs` | 13 chiều, dùng khi chưa có xét nghiệm. PR-AUC chỉ 0,336 |

**Kết quả trên tập test:** PR-AUC 0,885, ROC-AUC 0,977, Recall 0,911, Precision 0,476, F1 0,625.

**📂 Chỉ cho giám khảo:** `docs/09_MODEL_NOTES.md` §2b · `diabetes_mas/src/diabetes_mas/dl/model.py` · `experiments/best/dl/metrics.json`

**⚠ Lưu ý**
- *Câu hỏi phụ: "Vì sao F1 của MLP (0,625) thấp hơn luật ngưỡng (0,820)?"* Luật D1 chính là ngưỡng đã dùng để sinh nhãn của dataset, nên precision của nó bằng 1. MLP đổi precision lấy recall vì trong y tế, bỏ sót người bệnh nguy hiểm hơn báo nhầm.
- *"Vì sao không dùng mạng sâu hơn?"* Dữ liệu bảng chỉ có 8 cột, mạng sâu hơn dễ overfit mà không có thêm thông tin.
- *Câu hỏi phụ: "Vì sao demo báo nguy cơ 100%?"* Nhãn của dataset Kaggle được sinh từ chính ngưỡng HbA1c và glucose. Vì vậy khi bệnh nhân có HbA1c 7,1, mô hình gần như chắc chắn. Con số này cho thấy mô hình học lại ngưỡng của dataset, **không phải độ chắc chắn lâm sàng**, nên AG-13 chỉ là tham khảo (ADR-002).
- Giấy phép của dataset **chưa xác minh**, và không rõ là dữ liệu thật hay dữ liệu tổng hợp. Phải nói rõ điều này.
- App dự đoán riêng trước đây (`Assignment2-AI/`, React + NestJS) dùng **RandomForest (scikit-learn)**, không phải mạng nơ-ron. Nếu giám khảo hỏi về app đó, nói đúng như vậy.

---

### Q40. Phát triển và triển khai ứng dụng dự đoán: UI, backend, mô hình và dịch vụ dự đoán tương tác thế nào?

**🟢 Hiểu nhanh**
Người dùng gõ vào giao diện web. Web gửi yêu cầu cho server. Server chuyển thông tin cho mô hình đã huấn luyện, đã nạp sẵn trong bộ nhớ, rồi trả xác suất về cho web hiển thị. Mô hình **không train lại** khi chạy. Nó chỉ được nạp một lần lúc khởi động.

**🔧 Chi tiết kỹ thuật (hệ `diabetes_mas`)**
```
Trình duyệt (api/static/index.html)
  → FastAPI  POST /v1/sessions/{sid}/messages
    → Đồ thị agent (agents/graph.py): NLU → State Tracker → Main Agent
      → AG-13 Prediction → Tool T-15 predict_diabetes_risk (tools/predict_tools.py)
        → checkpoint PyTorch experiments/best/dl/full.pt (+ scaler) → P(diabetes)
    ← risk_dl ghi vào DialogState → AG-07 đưa vào lời giải thích → AG-08 Safety
  ← JSON: câu trả lời + slot + xác suất + log tool
```
- Các endpoint: `POST /v1/sessions`, `POST /v1/sessions/{sid}/messages`, `GET /v1/sessions/{sid}/state`, `POST /v1/consult`, `GET /v1/health`.
- Chạy: `uvicorn diabetes_mas.api.app:app --port 8000`.
- Đóng gói: Neo4j và SearXNG chạy bằng Docker Compose, Ollama chạy trực tiếp trên máy.
- **App `Assignment2-AI` (Clinic)** dùng kiến trúc 3 tầng: ReactJS → NestJS (API) → Python FastAPI runtime nạp `model.joblib` và `preprocessor.joblib`.

**📂 Chỉ cho giám khảo:** `diabetes_mas/src/diabetes_mas/api/app.py` · `docs/03_TOOL_SPEC.md` T-15 · `GET /v1/health`

**⚠ Lưu ý**
- *Câu hỏi phụ: "Thách thức khi triển khai là gì?"*
  - Tiền xử lý lúc chạy phải **giống hệt** lúc train: dùng lại scaler đã lưu.
  - Đầu vào có thể thiếu: khi chưa có xét nghiệm thì dùng biến thể `no_labs`.
  - Độ trễ.
  - Phiên bản checkpoint: `best/` chỉ bị ghi đè khi kết quả trên val tốt hơn.
- API **chưa có xác thực**, vì đây là demo chạy trên máy cá nhân. Nêu điều này ở Q51.

---

### Q41. So sánh ứng dụng dự đoán tiểu đường với dự đoán giá cổ phiếu và giá nhà

**🟢 Hiểu nhanh**
- **Tiểu đường:** trả lời câu hỏi *"có hay không?"*. Đây là bài toán phân loại.
- **Giá nhà:** trả lời *"bao nhiêu tiền?"* cho một căn nhà tại một thời điểm. Đây là hồi quy trên dữ liệu bảng.
- **Giá cổ phiếu:** trả lời *"ngày mai giá bao nhiêu?"* dựa trên chuỗi giá trong quá khứ. Đây là hồi quy trên **chuỗi thời gian**, nên thứ tự của dữ liệu quan trọng.

**🔧 Chi tiết kỹ thuật**

| | Tiểu đường | Giá nhà | Giá cổ phiếu |
|---|---|---|---|
| Dữ liệu vào | Bảng, mỗi người 1 dòng, độc lập | Bảng (diện tích, vị trí, số phòng…) | Chuỗi thời gian (giá, khối lượng theo ngày) |
| Mục tiêu | Nhị phân 0/1 | Số thực | Số thực ở bước t+1 (hoặc xu hướng tăng/giảm) |
| Kiến trúc hợp lý | MLP / cây quyết định | MLP / Gradient Boosting | RNN / LSTM (cửa sổ trượt) |
| Lớp ra | 1 nơ-ron + Sigmoid | 1 nơ-ron tuyến tính | 1 nơ-ron tuyến tính |
| Hàm mất mát | Binary Cross-Entropy | MSE / MAE | MSE / MAE |
| Metric | PR-AUC, Recall, F1, ma trận nhầm lẫn | MAE, RMSE, R² | MAE, RMSE, MAPE, độ đúng chiều tăng/giảm |
| Chia dữ liệu | Ngẫu nhiên, phân tầng | Ngẫu nhiên | **Theo thời gian**, không xáo trộn, để tránh rò rỉ tương lai |
| Rủi ro riêng | Mất cân bằng lớp | Ngoại lai giá | Không dừng (non-stationary), nhiễu thị trường |

**⚠ Lưu ý:** hai app cổ phiếu và giá nhà là bài làm riêng, **không có trong repo này**. Hãy tự điền kiến trúc và số liệu thật của bạn vào đây trước khi thi: ________

---

### Q42. Minh họa một mẫu mới đi qua ứng dụng DL, từ đầu vào đến kết quả dự đoán

**🟢 Hiểu nhanh**
Bệnh nhân nói *"Tôi 52 tuổi, nặng 78 kg cao 1m65, HbA1c 7,1%"*.
1. Hệ hiểu câu nói và rút ra các con số.
2. Hệ đổi các con số sang đúng dạng mô hình đã học.
3. Mô hình tính ra nguy cơ, ví dụ 0,9.
4. Hệ so con số đó với tiêu chuẩn Bộ Y tế, rồi nói kết quả kèm cảnh báo "chỉ để tham khảo".

**🔧 Chi tiết kỹ thuật: 7 bước**
1. **NLU (AG-06):** chuyển câu nói thành slot `tuoi = 52`, `can_nang = 78`, `chieu_cao = 1,65` (rồi tính `bmi = 28,7`), `HBA1C = 7,1`.
2. **State Tracker (AG-05)** ghi các slot vào `DialogState`. Main Agent thấy đặc trưng DL thay đổi và đã có đủ tuổi + BMI, nên kích hoạt **AG-13**.
3. **T-15 `features_from_state`** ánh xạ slot sang cột của dataset:
   - `gioi_tinh` → `gender`.
   - `GLUCOSE` (mmol/L) × 18 → `blood_glucose_level` (mg/dL).
   - Slot chưa biết → giá trị thiếu, đồng thời ghi lại giả định.
4. **Tiền xử lý:** z-score bằng scaler đã lưu, one-hot giới tính và hút thuốc, tạo vector 15 chiều. Nếu chưa có xét nghiệm thì dùng mô hình `no_labs` 13 chiều.
5. **Forward pass:** 15 → 64 → 32 → 1 → Sigmoid → `risk_dl`.
6. **Đối chiếu luật KB (T-03):** HbA1c 7,1 ≥ 6,5 nên đạt tiêu chuẩn ĐTĐ. Nếu mô hình nói "thấp" mà luật nói "đạt", hệ bật cờ `dl_conflict`.
7. **AG-07** đưa `risk_dl` vào lời giải thích, **AG-08 Safety** kiểm tra, rồi hiển thị trên web.

**📂 Chỉ cho giám khảo:** web demo, gõ câu trên rồi mở panel "log tool" để thấy dòng `T-15`. Hoặc chạy trực tiếp: `python -m diabetes_mas.agents.prediction`.

**⚠ Lưu ý:** *câu hỏi phụ "Nếu bệnh nhân không biết HbA1c thì sao?"* Hệ dùng `no_labs`, ghi rõ độ tin cậy thấp và liệt kê `missing_features`.

---

## B. Kiến trúc hệ đa agent tiểu đường

### Q43. Kiến trúc tổng thể của dự án; vì sao chọn đa agent thay vì một agent? ⭐

**🟢 Hiểu nhanh**
Hãy hình dung **một phòng khám thu nhỏ**:
- Có **lễ tân** phân loại người đến: đến khám, hỏi kiến thức hay cấp cứu.
- Có **bác sĩ trưởng** quyết định cần hỏi nhóm thông tin nào tiếp.
- Có **4 y tá** chuyên hỏi 4 nhóm: triệu chứng, xét nghiệm, tiền sử, biến chứng.
- Có **hội đồng chẩn đoán** đưa ra kết luận, và **máy tính nguy cơ** cho thêm ý kiến.
- Có **thư viện** để tra tài liệu Vinmec, Bộ Y tế, WHO.
- Có **người kiểm duyệt** soát mọi câu nói ra để tránh kê đơn hay bỏ sót cấp cứu.

Mỗi vai làm tốt một việc, nên dễ kiểm tra và dễ thay thế từng vai.

**🔧 Chi tiết kỹ thuật: 6 lớp**
```
Trình bày : Web demo (chat + slot + xác suất + entropy + log tool) · Neo4j Browser
API       : FastAPI /v1
Điều phối : Đồ thị agent tự viết (ADR-003): Router → NLU → State → Main → Worker → Classifier → Consult → Safety
Agent     : LLM (Router, NLU, NLG, Consult, Safety, Research) · Học được (Main, W1–W4, Classifiers, DL) · Simulator + Reward
Tool      : T-01 … T-15 (có schema, có log T-10)
Dữ liệu   : Neo4j 5 (KB + vector index) · Ollama (Qwen 2.5 3B + bge-m3) · checkpoint PyTorch
```
Theo yêu cầu R-T11 của thầy, hệ gồm **3 khối**:
1. **Main Agent** điều phối **Work-Agents W1–W4** và **Disease Classifiers** (Main + Sub 1–4).
2. **Patient Simulator** gồm Patient Action và State Tracker.
3. **Reward Generation** từ hành động của bệnh nhân và **mức giảm entropy**.

**Vì sao đa agent:**
- **Tách ba loại quyết định khác bản chất:**
  - Quyết định *học được* (hỏi gì tiếp) dùng RL.
  - *Giao tiếp ngôn ngữ* dùng LLM.
  - *An toàn y khoa* dùng luật cố định. Phần an toàn không phụ thuộc chất lượng huấn luyện.
- **Phân cấp làm bài toán nhỏ lại:**
  - Một agent phẳng phải chọn trong khoảng 25 slot cộng hành động kết luận, trong khi reward thưa.
  - Khi phân cấp, Main chỉ có 5 hành động và mỗi Worker có 5–14 hành động.
- **Kiểm thử và thay thế độc lập:** khi Ollama tắt, hệ vẫn chạy bằng luật và câu mẫu.

**📂 Chỉ cho giám khảo:** `docs/09_ARCHITECTURE.md` · `docs/02_AGENT_DESIGN.md` §1, §1b · `docs/assets/seq_A..D.png`

**⚠ Lưu ý**
- *Câu hỏi phụ: "Sao không dùng LangGraph?"* LangGraph không cài được trên Python 3.15 (thiếu wheel `zstandard`), nên nhóm tự viết đồ thị điều phối (ADR-003).
- Baseline B1 (một DQN phẳng) **đã chạy**: test 0,751 / 3,54 lượt, so với HRL 0,782 / 3,32. Chỉ 1 seed (xem Q50).

---

### Q44. Liệt kê các agent: trách nhiệm, đầu vào, đầu ra, tương tác ⭐

**🟢 Hiểu nhanh**
Có 3 nhóm agent:
- **Nhóm khám bệnh:** bác sĩ trưởng, 4 y tá chuyên khoa, hội đồng chẩn đoán, máy tính nguy cơ.
- **Nhóm giao tiếp:** lễ tân, người hiểu câu nói, người đặt câu hỏi, người tư vấn, người tìm tài liệu, người kiểm duyệt.
- **Nhóm luyện tập:** bệnh nhân giả, người ghi chép và người chấm điểm, dùng để huấn luyện.

**🔧 Chi tiết kỹ thuật**

| Mã | Agent | Trách nhiệm | Vào → Ra | Công nghệ |
|---|---|---|---|---|
| AG-00 | Router | Phân loại ý định | câu → `diagnose`/`consult`/`out_of_scope`/`emergency` | Luật + Qwen (P-01) |
| AG-06 | NLU | Câu tự do → slot, quy đổi đơn vị | câu → `dict[slot]` | Luật số học + Qwen (P-02) |
| AG-05 | State Tracker | Ghi slot, suy ra cờ, gọi phân loại lại và kiểm tra cấp cứu | slot → `DialogState`, `s_t` (122 chiều) | Code xác định |
| **AG-01** | **Main Agent** | Chọn nhóm cần hỏi hoặc kết luận; điều phối cố định (cấp cứu, T_max, gọi AG-13) | `s_t` → {W1..W4, CLASSIFY} | **DQN** (2 lớp ẩn 256, 256) |
| AG-W1..W4 | Work-Agents | Chọn câu hỏi cụ thể trong nhóm, hoặc trả quyền | `s_t` → slot ∪ {RETURN} | DQN (256, 128) |
| AG-02 | Main Classifier | Phân phối 5 lớp + áp luật KB | slot + 4×5 sub-probs → `p_t`, `H(p_t)` | MLP + luật |
| AG-03-h | Sub-Classifier 1..4 | Phân phối từ riêng nhóm h | slot nhóm h → `p_t^h` | MLP nhỏ |
| AG-13 | Prediction (DL) | Xác suất mắc ĐTĐ, gắn cờ mâu thuẫn với luật | đặc trưng → `risk_dl`, `dl_conflict` | MLP PyTorch |
| AG-09 | NLG | Slot → câu hỏi tiếng Việt | slot_id → 1 câu | Qwen (P-03) / câu mẫu |
| AG-07 | Consult | Trả lời kiến thức, giải thích kết luận, trích nguồn | câu hỏi / kết quả → văn bản có `[n]` | Qwen (P-05, P-06) |
| AG-12 | Research | Tìm bài bổ sung khi KB thiếu: kho → web whitelist | câu (đã lọc PII) → chunk tạm | Qwen (P-08) + SearXNG |
| AG-08 | Safety | Soát mọi câu ra: cấp cứu, kê đơn, khẳng định chẩn đoán, thiếu nguồn | câu nháp → câu đã kiểm | Luật + Qwen (P-07) |
| AG-10 | Patient Action | Đóng vai bệnh nhân từ ca bệnh | slot được hỏi → yes/no/value/unknown | Luật (+ Qwen P-04) |
| AG-11 | Reward | Tính `r_t` | trả lời, entropy → `r_t` | Code xác định |

Mỗi agent có cột "**không được làm**". Ví dụ: Main Agent không tự kết luận bệnh; AG-13 không ghi đè luật KB.

**📂 Chỉ cho giám khảo:** `docs/02_AGENT_DESIGN.md` §2 (thẻ agent)

---

### Q45. Mô hình DL được tích hợp vào MAS thế nào? Agent nào gọi? Kết quả truyền cho agent khác ra sao? ⭐

**🟢 Hiểu nhanh**
**Bác sĩ trưởng** (Main Agent) bấm "máy tính nguy cơ" mỗi khi có thêm thông tin mới về bệnh nhân. Kết quả được ghi vào **hồ sơ bệnh án chung** (DialogState). Mọi agent khác đọc từ hồ sơ đó. Người tư vấn nhắc lại con số trong phần giải thích. Nếu máy nói khác tiêu chuẩn Bộ Y tế, hệ ghi rõ là có mâu thuẫn. **Máy không có quyền quyết định chẩn đoán.**

**🔧 Chi tiết kỹ thuật**
- **Agent gọi:** AG-01 Main Agent, ở bước điều phối cố định số 3. Điều kiện: đặc trưng DL vừa thay đổi và đã có tuổi + BMI. Sau bước intake điều kiện này luôn đúng, nên AG-13 chạy ít nhất một lần mỗi phiên.
- **Kênh truyền:** AG-01 → AG-13 → Tool **T-15** `predict_diabetes_risk` → ghi `risk_dl`, `model_variant`, `missing_features`, `dl_conflict` vào **`DialogState`** (bộ nhớ chung, kênh K1).
- **Bên dùng kết quả:**
  - AG-07 đưa vào lời giải thích P-06.
  - Web hiển thị con số.
  - Log T-10 ghi lại lời gọi.
- **Không dùng trong reward** và **không ghi đè** `p_t` của Classifier (ADR-002). Lý do: khi chưa có xét nghiệm, PR-AUC chỉ 0,336, quá yếu để làm căn cứ.

**📂 Chỉ cho giám khảo:** `docs/04_SEQUENCE.md` (sơ đồ có mũi tên T-15) · `docs/decisions/ADR-002-agent-du-doan-deep-learning.md` · panel log tool trên web

---

### Q46. KG và Neo4j được dùng thế nào? Đã cài những thực thể và quan hệ nào? Cho một ví dụ Cypher ⭐

**🟢 Hiểu nhanh**
Neo4j là **cuốn sổ tay y khoa dạng bản đồ**:
- Mỗi bệnh, triệu chứng, xét nghiệm, thuốc là một **điểm**.
- Các **mũi tên** nối chúng lại, ví dụ "típ 2 → có triệu chứng → khát nhiều".
- Mỗi mũi tên đều ghi **nguồn**: bài Vinmec nào, quyết định nào của Bộ Y tế.

Hệ dùng cuốn sổ này cho ba việc:
1. **Hỏi bệnh:** biết cần hỏi những gì.
2. **Kết luận:** so chỉ số với ngưỡng chuẩn.
3. **Tư vấn:** tìm đoạn văn làm chứng cứ để trích dẫn.

**🔧 Chi tiết kỹ thuật**
- **10 nhãn node:** `Disease, Symptom, LabTest, DiagnosticCriterion, RiskFactor, Drug, Gene, Anatomy, Source, Chunk`.
- **12 loại quan hệ:** `HAS_SYMPTOM {prob}`, `DIAGNOSED_BY`, `USES_TEST`, `HAS_RISK_FACTOR {strength}`, `HAS_COMPLICATION`, `SUBTYPE_OF`, `DIFFERENTIAL_OF`, `TREATS {line}`, `ASSOCIATED_GENE`, `AFFECTS`, `FROM_SOURCE`, `MENTIONS`.
- **Provenance:** mọi quan hệ có `source_id`, `confidence`, và `evidence_chunk_id` nếu được trích từ văn bản.
- **Quy mô:** 3.164 node, 8.133 quan hệ.
  - 2.359 Chunk có `embedding float[1024]` (bge-m3).
  - Vector index `chunk_embedding`, fulltext index `entity_names`.
- **File:**
  - `kb/schema.cypher`: constraint và index, viết tay.
  - `kb/kb_diabetes.cypher` (3,6 MB): sinh bởi `kb/build_cypher.py`.
- **Cách hệ dùng KB:**

  | Tool | Việc |
  |---|---|
  | T-02 | Liên kết thực thể |
  | T-03 | Kiểm tra tiêu chuẩn chẩn đoán |
  | T-04 | Truy xuất lai graph + vector |

```cypher
// Tiêu chuẩn chẩn đoán kèm nguồn (2 bước: Disease → Criterion → LabTest)
MATCH (d:Disease)-[r:DIAGNOSED_BY]->(c)-[:USES_TEST]->(t), (s:Source {id: r.source_id})
RETURN d.name_vi, t.name_vi, c.op, c.value, c.unit, s.title;

// Triệu chứng phân biệt típ 1 và típ 2
MATCH (d1:Disease {id:'DTD_TIP_1'})-[r1:HAS_SYMPTOM]->(s)<-[r2:HAS_SYMPTOM]-(d2:Disease {id:'DTD_TIP_2'})
RETURN s.name_vi, r1.prob AS p_tip1, r2.prob AS p_tip2 ORDER BY abs(r1.prob - r2.prob) DESC;
```

**📂 Chỉ cho giám khảo:** Neo4j Browser `http://localhost:7474`, chạy 4 truy vấn ở `docs/00_KB_GRAPH.md` §7 · `docs/00_KB_GRAPH.md` §1–3

**⚠ Lưu ý:** node `:Patient`, `:Measurement`, `:Recommendation` (TK-18) **chưa có**. Nếu giám khảo hỏi, nói rõ đây là phần **dự kiến**.

---

### Q47. RAG được tích hợp thế nào? Truy xuất và sinh câu trả lời ra sao? Làm sao giảm câu trả lời bịa? ⭐

**🟢 Hiểu nhanh**
Trước khi trả lời, hệ **đi thư viện tra cứu**: tìm trong sổ tay Neo4j và các bài Vinmec, Bộ Y tế, WHO những đoạn liên quan. Sau đó hệ **chỉ được phép trả lời dựa trên các đoạn tìm được**, và phải đánh số nguồn `[1] [2]`. Nếu tra không ra, hệ nói thẳng *"không đủ thông tin"* thay vì đoán.

**🔧 Chi tiết kỹ thuật: `hybrid_retrieve` (T-04)**
1. **Liên kết thực thể:** dùng từ điển `kb/link.py`, cùng mã đã tạo quan hệ `MENTIONS`.
2. **Embed câu hỏi** bằng bge-m3 (1024 chiều).
3. **Graph facts 1-hop**, lọc theo ý định. Ví dụ hỏi về thuốc thì chỉ lấy `TREATS`. Mỗi fact phải có `source_id`.
4. **Vector search** top 20 trên `chunk_embedding`, kèm **chunk được MENTIONS bởi thực thể** (graph-guided).
5. **Xếp hạng:** `final = cos + 0,02·mention + 0,01·reviewed`. Lấy top 5, tối đa 2 chunk mỗi nguồn.
6. **Cờ `enough`:** có fact khớp thực thể **hoặc** cos ≥ `min_score` (0,75). Câu không có thực thể KB thì cần cos ≥ **0,825**.
7. Nếu không đủ: **AG-12** tìm trong kho, rồi trên web whitelist qua SearXNG. Bài web vào hàng chờ duyệt, **không ghi thẳng** vào KB.
8. **Qwen 2.5 3B** sinh câu trả lời bằng prompt P-05, chỉ dựa trên các đoạn đã đánh số.

**Các lớp chống bịa:**

| Lớp | Cơ chế |
|---|---|
| Trước khi sinh | Không đủ căn cứ thì **không gọi LLM**, từ chối luôn |
| Trong prompt | Chỉ dùng context đánh số, bắt buộc trích `[n]` |
| Sau khi sinh | Kiểm tra mọi `[n]` có trong context không (`normalize_citations` chuẩn hóa `(1)`, `[3, 2]`). Không đạt thì dùng **câu trả lời trích xuất** nguyên văn |
| Kiểm giải thích chẩn đoán | `explain_ok`: câu phải chứa đúng tên kết luận và xác suất. Ví dụ đã chặn: Qwen bỏ chữ "Không" làm đảo ngược kết luận. **Phát hiện khi chạy thử demo:** Qwen viết "HbA1c 7,4%" trong khi người dùng nói 7,1, nên bổ sung bước kiểm **mọi con số** phải khớp hồ sơ hoặc ngưỡng luật; sai thì dùng câu mẫu |
| AG-08 Safety | Chặn kê đơn, khẳng định chẩn đoán, câu thiếu nguồn hoặc thiếu disclaimer |
| NLU: luật trước, LLM sau | Con số y khoa (HbA1c, glucose, cân nặng, chiều cao) do **luật** đọc trước. LLM **không được ghi đè** slot luật đã điền. Câu ngắn mà luật đã hiểu đủ thì **bỏ qua LLM** (`llm_skipped`, `nlu.py`) |

**Kết quả (88 câu):**
- R@5 = 1,00, từ chối đúng 0,909, từ chối nhầm 0,030.
- Trích dẫn hợp lệ 100%.
- Phương thức trả lời: 56 câu do LLM viết, 10 câu trích xuất, 22 câu từ chối.

**📂 Chỉ cho giám khảo:** `docs/06_KB_RAG_INTEGRATION.md` §2–3 · web demo, hỏi một câu rồi mở panel "context & nguồn"

**⚠ Lưu ý**
- **Faithfulness: chấm tay mẫu 20/56 câu LLM, đạt 19/20 = 0,95.** Phải nói rõ ba điểm:
  - Đây là chấm mẫu, không phải toàn bộ.
  - Chỉ một người chấm.
  - Có phát hiện: "trích dẫn hợp lệ 100%" chỉ đảm bảo `[n]` **nằm trong** context, chưa đảm bảo `[n]` trỏ **đúng** đoạn chứa ý. Qwen 3B đôi khi chép đúng nội dung nhưng gắn số của fact bên cạnh. Hướng sửa: kiểm khớp giữa câu và nguồn được dẫn (chưa làm).
- *Câu hỏi phụ: "Sao ngưỡng 0,75 mà không phải 0,55?"* Với bge-m3, câu ngoài phạm vi như "ung thư phổi" vẫn đạt cos 0,81. Ngưỡng được chọn bằng cách quét số liệu thật (`docs/assets/min_score_sweep.png`).

---

### Q48. Luồng thực thi đầy đủ khi người dùng hỏi về tiểu đường: vai trò coordinator, prediction, knowledge retrieval, response ⭐

**🟢 Hiểu nhanh**
1. **Lễ tân** nghe câu hỏi và quyết định đi đường nào.
2. Nếu là **hỏi kiến thức**: thủ thư tra cứu, người tư vấn soạn câu trả lời có nguồn, người kiểm duyệt soát rồi mới gửi.
3. Nếu là **kể bệnh**: bác sĩ trưởng điều các y tá hỏi lần lượt. Sau mỗi câu trả lời, hồ sơ được cập nhật và hội đồng chẩn đoán đánh giá lại. Khi đủ chắc chắn, hệ kết luận, giải thích kèm nguồn, rồi qua kiểm duyệt.

**🔧 Chi tiết kỹ thuật: ánh xạ vai trò trong đề**

| Vai trò trong đề | Agent trong hệ |
|---|---|
| Coordinator | **AG-00 Router** (cấp phiên) + **AG-01 Main Agent** (cấp hội thoại chẩn đoán) |
| Prediction | **AG-13** (DL, tham khảo) + **AG-02/03 Classifiers** (ra kết luận) |
| Knowledge retrieval | **AG-07 Consult** qua T-04 + **AG-12 Research** |
| Response | **AG-09 NLG** / **AG-07 Consult** → **AG-08 Safety** (mọi câu ra đều qua Safety) |

```
Luồng tư vấn:  câu → AG-06 NLU → AG-00 Router=consult → AG-07 → T-04 hybrid_retrieve (Neo4j)
               → [enough? không → AG-12 → T-12/T-13] → Qwen P-05 → kiểm [n] → AG-08 → trả lời

Luồng chẩn đoán (mỗi lượt):
  câu → NLU → Router=diagnose → AG-05 cập nhật slot → T-08 (p_t, H) + T-09 (cấp cứu?)
  → AG-01: [cấp cứu → dừng] · [đặc trưng đổi → AG-13/T-15] · [Worker đang giữ quyền → Worker chọn slot]
           · [không → π_M chọn W1..W4 hoặc CLASSIFY]
  → AG-09 sinh câu hỏi → AG-08 → người dùng
  CLASSIFY → AG-02 top-1 + luật T-03 → AG-07 giải thích (P-06) → AG-08 → người dùng
```
Chỉ bước chọn W_h, slot hoặc CLASSIFY là **policy học được**. Các bước cấp cứu, T_max, gọi DL và Safety là **code cố định**.

**📂 Chỉ cho giám khảo:** `docs/04_SEQUENCE.md` (luồng A–D, ảnh `docs/assets/seq_*.png`) · `docs/01_REQUIREMENTS.md` §9

---

### Q49. Demo trọn vẹn hệ trên laptop với một kịch bản thực tế ⭐

**🟢 Hiểu nhanh**
Kịch bản: **"Ông 52 tuổi, khát nước, tiểu nhiều, sụt cân"**. Ta xem hệ hỏi ít câu nhất có thể, mức không chắc chắn (entropy) giảm dần sau mỗi câu, và cuối cùng hệ kết luận *nghi ĐTĐ típ 2* kèm căn cứ và lời khuyên đi khám.

**🔧 Kịch bản và điểm cần chỉ ra**

| Lượt | Người dùng nói | Chỉ vào màn hình |
|---|---|---|
| 1 | "Dạo này tôi khát nước suốt, đi tiểu nhiều, sụt 3 kg trong tháng." | NLU rút ra 3 slot · Router = `diagnose` · entropy ban đầu |
| 2 | (hệ hỏi tuổi, BMI, giới tính: intake) "Tôi 52 tuổi, nặng 78 kg cao 1m65, nam." | BMI tự tính 28,7 · **T-15 chạy** → `risk_dl` |
| 3 | (Main chọn W2 xét nghiệm) "HbA1c tháng trước 7,1%." | **Luật T-03 đạt** (≥ 6,5%) · xác suất típ 2 tăng mạnh, **entropy giảm** |
| 4 | (W3 hỏi tiền sử) "Bố tôi bị tiểu đường." | Main chọn CLASSIFY |
| Kết | — | Nghi **ĐTĐ típ 2**, căn cứ HbA1c 7,1% > 6,5% [Vinmec VMC-0002, theo ADA]. Khuyên khám nội tiết, làm glucose lúc đói / OGTT (QĐ 3280/2011). **Disclaimer** |

Sau đó demo **tư vấn**: *"Người tiểu đường có ăn được trái cây không?"*. Chỉ vào các đoạn văn đã truy xuất, điểm cos và trích dẫn `[1][2]`. Tiếp theo là một câu **ngoài phạm vi**, để thấy hệ từ chối.

Nếu còn thời gian: ca **thai kỳ** TC-31 (thai phụ 28 tuần, OGTT đói 5,3).

**📂 Chỉ cho giám khảo:** `docs/09_DEMO_SCRIPT.md` · web demo · Neo4j Browser

**⚠ Lưu ý**
- Câu trả lời thật của Qwen **có thể khác câu chữ** so với kịch bản.
- Lần hỏi đầu Qwen có thể chậm 30–60 giây. Hãy hỏi trước một câu để nạp model vào RAM.
- Nếu demo gặp sự cố: dùng **video dự phòng** và log mẫu.

---

### Q50. Đánh giá hệ đa agent thế nào: DL, chất lượng truy xuất, độ đúng câu trả lời, điều phối, thời gian phản hồi, độ tin cậy? ⭐

**🟢 Hiểu nhanh**
Nhóm chấm hệ ở **6 mặt**, mỗi mặt có bộ đề riêng và có **đối thủ để so** (baseline):
1. Máy tính nguy cơ dự đoán có đúng không.
2. Thủ thư tìm có trúng tài liệu không.
3. Câu trả lời có đúng nguồn không.
4. Bác sĩ trưởng có hỏi ít mà vẫn đúng không.
5. Trả lời nhanh hay chậm.
6. Hệ có hay lỗi không.

**🔧 Chi tiết kỹ thuật**

| Mặt | Bộ đánh giá | Metric | Kết quả | Baseline |
|---|---|---|---|---|
| DL (AG-13) | Test 15% của Kaggle | PR-AUC, ROC-AUC, Recall, F1 | 0,885 / 0,977 / 0,911 / 0,625 | D1 luật ngưỡng F1 0,820 |
| Classifier 5 lớp | 225 ca test | acc, macro-F1 | MLP + luật 0,853 / 0,862 | Naive Bayes từ KB 0,831 |
| Truy xuất | `consult_eval.csv` 88 câu (66 trong + 22 ngoài phạm vi) | R@1/3/5, từ chối đúng/nhầm | 0,849 / 0,970 / 1,00 · 0,909 / 0,030 | vector-only, graph-only |
| Độ đúng trả lời | như trên | Citation validity, faithfulness | 1,00 · **0,95 (mẫu 20 câu)** | mục tiêu ≥ 0,90 |
| Điều phối | 225 ca qua Patient Simulator | success, số lượt, hỏi lặp | E2E câu mẫu 0,796 / 3,48 lượt / 0 · **E2E Qwen 20 ca 0,750 / 3,70** (đo trước khi sửa NLU) | B0' luật 0,853 / 10,7 lượt · B0 Random 0,698 / 11,5 lượt · B1 Flat DQN 0,751 / 3,54 lượt |
| Thời gian | log T-10 | p95 | Truy hồi 0,38 s · trả lời bằng LLM: trung vị 4,2 s, **p95 11,6 s** (đo khi máy rảnh) | mục tiêu ≤ 5 s ❌ (không đạt) |
| Độ tin cậy | log T-10 | tỉ lệ lỗi tool | 13.641 call / 318 phiên, lỗi 0 | — |

- Quy trình: `07_TEST_PLAN` (test tự động) → `08_EVALUATION` (số liệu + **nhật ký cải tiến** §5, ghi trước/sau mỗi thay đổi).
- Báo cáo gom tự động: `python -m diabetes_mas.devtools.eval_report`.

**📂 Chỉ cho giám khảo:** `docs/08_EVALUATION.md` §4–5 · `experiments/eval/report.md` · `docs/assets/curve_hrl.png`, `min_score_sweep.png`

**⚠ Lưu ý: câu hỏi phụ chắc chắn gặp**
- *"HRL thua baseline luật về độ chính xác?"* Đúng: 0,782 so với 0,853, đo trong môi trường. Bù lại HRL **hỏi ít hơn khoảng 3 lần** (3,3 so với 10,7 lượt). Phạt −1 mỗi lượt khiến policy đánh đổi độ chính xác lấy số lượt. Đây là lựa chọn thiết kế, có thể chỉnh λ hoặc phạt lượt để đổi điểm cân bằng.
- *"HRL có hơn một DQN phẳng (B1) không?"* Trên tập test, HRL đạt 0,782 với 3,32 lượt, B1 đạt 0,751 với 3,54 lượt. **Chỉ 1 seed, nên chưa đủ để khẳng định HRL hơn.** Lợi ích chắc chắn của phân cấp:
  - Giải thích được: biết agent đang hỏi nhóm nào.
  - Luật gắn theo nhóm.
  - Không gian hành động nhỏ hơn (Main 5, Worker 5–14, so với 29 của B1).
  - Đúng kiến trúc thầy yêu cầu (R-T11).
- Mỗi cấu hình mới **train 1 seed**, nên chênh lệch khoảng 0,03 có thể chỉ là nhiễu.
- Nhật ký cải tiến có ghi rõ **một lần sửa nhãn** trong bộ test (C027). Nói rõ để tránh bị coi là "tối ưu trên tập test".

---

### Q51. Giới hạn, rủi ro và hướng cải tiến (kiến trúc, điều phối, chất lượng tri thức, bảo mật, triển khai)

**🟢 Hiểu nhanh**
Đây là **công cụ học thuật, không thay bác sĩ**. Các điểm yếu chính:
- Dữ liệu huấn luyện phần lớn là mô phỏng.
- Mô hình ngôn ngữ nhỏ chạy trên CPU nên chậm.
- Hệ đôi khi kết luận vội để hỏi ít câu.
- Chưa có đăng nhập và bảo mật cho môi trường thật.

Bản sau cần: dữ liệu bệnh án thật đã được bác sĩ duyệt, máy mạnh hơn, và quy trình kiểm duyệt y khoa.

**🔧 Chi tiết kỹ thuật**

| Mặt | Giới hạn hiện tại | Hướng cải tiến |
|---|---|---|
| Dữ liệu | Ca bệnh **sinh từ chính KB**, nên có nguy cơ vòng lặp lạc quan. Kaggle chưa rõ giấy phép và chưa rõ thật hay tổng hợp | Ca bệnh thật đã ẩn danh, có bác sĩ gán nhãn; kiểm định ngoài |
| Tri thức | Bộ Y tế mới có QĐ 3280/2011, thiếu QĐ 5481/2020, hướng dẫn tiền ĐTĐ và típ 1. Bài web chờ duyệt thủ công | Bổ sung hướng dẫn mới; quy trình duyệt có chuyên gia; versioning KB |
| Điều phối | HRL đánh đổi acc lấy số lượt; 14/225 ca chạm T_max (acc 0,50); TC-21 (ca biên tiền ĐTĐ) chưa đạt; 1 seed | Reward theo chi phí xét nghiệm thật; lặp nhiều seed; ⟳ ablation entropy |
| DL | Thiếu xét nghiệm thì PR-AUC chỉ 0,336; chưa hiệu chỉnh xác suất | Calibration (Platt/isotonic); thêm đặc trưng |
| LLM | Qwen 3B trên CPU: p95 11,6 s (trung vị 4,2 s); đôi khi sai định dạng trích dẫn, đảo nghĩa (đã có lớp chặn) | GPU / model lớn hơn; streaming; chấm faithfulness tự động |
| Bảo mật | API **chưa xác thực**; dữ liệu hội thoại lưu log cục bộ; truy vấn web đã lọc PII | Xác thực + phân quyền; mã hóa log; tuân thủ quy định dữ liệu y tế |
| Triển khai | Chạy trên 1 laptop; Ollama ngoài Docker | Đóng gói đủ trong Compose/K8s; giám sát; CI chạy test + `check_docs` |
| Kiến trúc | Đồ thị điều phối tự viết (thiếu thư viện trên Python 3.15) | Chuyển sang framework chuẩn khi tương thích |
| Đồ thị bệnh nhân | `:Patient/:Measurement/:Recommendation` chưa có (TK-18) | Lưu lịch sử khám để theo dõi dài hạn |

**📂 Chỉ cho giám khảo:** `docs/08_EVALUATION.md` §7 · `docs/01_REQUIREMENTS.md` §7 · ADR-001/002/003

---

# Phần 3 — Knowledge Graph, Neo4j, RAG (Q19–Q28)

### Q19. Knowledge Graph là gì? Thực thể, quan hệ, thuộc tính. Ví dụ tiểu đường

**🟢 Hiểu nhanh**
KG là cách lưu kiến thức dưới dạng **bản đồ khái niệm**:
- Mỗi khái niệm là một điểm, ví dụ "ĐTĐ típ 2" hay "khát nhiều".
- Các mối liên hệ là đường nối có tên, ví dụ "có triệu chứng".
- Mỗi điểm và mỗi đường mang thêm chi tiết, ví dụ "xác suất 0,7", "nguồn Vinmec".

Máy đi theo các đường nối để suy luận, giống bác sĩ lật sổ tay.

**🔧 Chi tiết kỹ thuật**

| Khái niệm | Nghĩa | Ví dụ trong dự án |
|---|---|---|
| **Thực thể (entity / node)** | Đối tượng, có nhãn loại | `(:Disease {id:'DTD_TIP_2'})`, `(:Symptom {id:'khat_nuoc_nhieu'})` |
| **Quan hệ (relationship / edge)** | Liên kết có hướng, có tên | `HAS_SYMPTOM`, `DIAGNOSED_BY`, `TREATS` |
| **Thuộc tính (property)** | Cặp key–value gắn lên node hoặc cạnh | Node: `name_vi`, `icd10 = 'E11'`; cạnh: `prob = 0.7`, `source_id = 'VMC-0001'` |
| **Thuộc tính mô tả (attribute)** | Thường dùng như property. Trong RDF, attribute là triple với literal | `(HbA1c, unit, "%")` |

**📂 Chỉ cho giám khảo:** `docs/00_KB_GRAPH.md` §1–3 · Neo4j Browser

---

### Q20. Biểu diễn tri thức bằng triple (subject–predicate–object). Viết ≥ 5 triple

**🟢 Hiểu nhanh**
Mọi kiến thức được bẻ thành **câu ba thành phần "ai – làm gì/có gì – với cái gì"**, ví dụ "Típ 2 – có triệu chứng – khát nhiều". Nhiều câu như vậy ghép lại thành bản đồ tri thức.

**🔧 Chi tiết kỹ thuật**
Triple là `(chủ thể, vị từ, đối tượng)`, tức một cạnh có hướng trong đồ thị.
```
(DTD_TIP_2)        -[HAS_SYMPTOM {prob}]->     (uong_nhieu)
(DTD_TIP_2)        -[HAS_RISK_FACTOR]->        (beo_phi)
(DTD_TIP_2)        -[DIAGNOSED_BY]->           (HbA1c ≥ 6,5 %)
(HbA1c ≥ 6,5 %)    -[USES_TEST]->              (HBA1C)
(METFORMIN)        -[TREATS {line:'đầu tay'}]-> (DTD_TIP_2)
(DTD_TIP_2)        -[HAS_COMPLICATION]->       (benh_than_dtd)
(DTD_THAI_KY)      -[SUBTYPE_OF]->             (DTD)
```
- Dữ liệu seed của dự án lưu đúng dạng triple, mỗi dòng kèm nguồn và câu trích.
- Neo4j dùng **property graph**: khác RDF thuần ở chỗ cạnh có thể mang thuộc tính.

**📂 Chỉ cho giám khảo:** `diabetes_mas/kb/seed/edges.csv`

---

### Q21. Neo4j là gì? Mô hình property graph. Node, relationship, property được lưu thế nào?

**🟢 Hiểu nhanh**
Neo4j là **cơ sở dữ liệu chuyên lưu bản đồ quan hệ**. Thay vì lưu bảng như Excel rồi phải ghép bảng mới tìm được mối nối, Neo4j lưu sẵn đường nối. Nhờ vậy câu hỏi kiểu "bệnh này liên quan gì, rồi cái đó liên quan gì tiếp" chạy nhanh.

**🔧 Chi tiết kỹ thuật**
- **Property graph:**
  - Node có 0..n **label** và các property.
  - Relationship có **đúng 1 type**, **có hướng**, nối 2 node, và cũng có property.
- **Lưu trữ:** *index-free adjacency*. Mỗi node giữ con trỏ trực tiếp tới các cạnh của nó, nên duyệt quan hệ có chi phí tỉ lệ với số cạnh đi qua, không phụ thuộc kích thước toàn bảng như phép JOIN.
- **Index và constraint:** tăng tốc tìm điểm bắt đầu.
- **Trong dự án:**
  - Neo4j 5 chạy bằng Docker. Constraint `id` là duy nhất.
  - **Fulltext index** `entity_names`: tìm thực thể theo tên.
  - **Vector index** `chunk_embedding`: 1024 chiều, cosine.
  - Neo4j vừa là graph DB vừa là vector store, nên không cần thêm FAISS hay Chroma.

**📂 Chỉ cho giám khảo:** `diabetes_mas/kb/schema.cypher` · Neo4j Browser (`:schema`)

---

### Q22. Cypher cơ bản: tạo Patient, Disease, quan hệ giữa chúng, rồi truy vấn lại

**🟢 Hiểu nhanh**
Cypher là **ngôn ngữ hỏi đáp của Neo4j**, viết gần giống vẽ hình: `(người)-[:MẮC]->(bệnh)`. Lệnh `CREATE`/`MERGE` dùng để tạo, `MATCH … RETURN` dùng để tìm và trả kết quả.

**🔧 Chi tiết kỹ thuật**
```cypher
// Tạo (MERGE = tạo nếu chưa có → chạy lại không bị trùng)
MERGE (p:Patient {id:'P001'}) SET p.age = 52, p.sex = 'nam', p.bmi = 28.7
MERGE (d:Disease {id:'DTD_TIP_2'}) SET d.name_vi = 'Đái tháo đường típ 2'
MERGE (p)-[r:DIAGNOSED_WITH {date: date('2026-11-12'), confidence: 0.81}]->(d);

// Truy vấn
MATCH (p:Patient {id:'P001'})-[r:DIAGNOSED_WITH]->(d:Disease)
RETURN p.age, d.name_vi, r.confidence;

// Mở rộng: bệnh nhân này có triệu chứng nào của bệnh đã chẩn đoán?
MATCH (p:Patient {id:'P001'})-[:DIAGNOSED_WITH]->(d)-[:HAS_SYMPTOM]->(s)
RETURN s.name_vi;
```
Cú pháp:
- `()` là node, `[]` là cạnh, `:Label` là nhãn, `{}` là property, `->` là hướng.
- `MERGE` tạo khi chưa có, nên chạy lại không bị trùng (idempotent). `CREATE` luôn tạo mới.

**⚠ Lưu ý:** KB của dự án **chưa có node `:Patient`** (TK-18 mới là kế hoạch). Đây là truy vấn minh họa. Có thể chạy thử trực tiếp trên Neo4j Browser.

---

### Q23. Thiết kế KG tiểu đường: bệnh nhân, triệu chứng, yếu tố nguy cơ, chỉ số đo, bệnh, khuyến nghị

**🟢 Hiểu nhanh**
Đặt **bệnh nhân** ở giữa, nối tới những gì người đó *có*: triệu chứng, chỉ số đo, yếu tố nguy cơ. Đặt **bệnh** ở giữa, nối tới những gì bệnh *thường có*: triệu chứng, nguy cơ, tiêu chuẩn chẩn đoán, khuyến nghị. So khớp hai phía này là ra gợi ý chẩn đoán.

**🔧 Chi tiết kỹ thuật**
```
(:Patient)-[:HAS_SYMPTOM]->(:Symptom)<-[:HAS_SYMPTOM {prob}]-(:Disease)
(:Patient)-[:HAS_MEASUREMENT]->(:Measurement {value, unit, date})-[:OF_TEST]->(:LabTest)
(:Patient)-[:HAS_RISK_FACTOR]->(:RiskFactor)<-[:HAS_RISK_FACTOR {strength}]-(:Disease)
(:Disease)-[:DIAGNOSED_BY]->(:DiagnosticCriterion {op, value})-[:USES_TEST]->(:LabTest)
(:Disease)-[:HAS_RECOMMENDATION]->(:Recommendation {text, source_id})
(:Patient)-[:DIAGNOSED_WITH {confidence}]->(:Disease)
```
- **Đã cài:** Disease, Symptom, RiskFactor, LabTest, DiagnosticCriterion, Drug, Gene, Anatomy, Source, Chunk.
- **Dự kiến:** Patient, Measurement, Recommendation. Hiện thông tin bệnh nhân nằm trong `DialogState`, không ghi vào KB.

**📂 Chỉ cho giám khảo:** `docs/00_KB_GRAPH.md` §1 (sơ đồ schema)

---

### Q24. Neo4j hỗ trợ truy xuất và suy luận thế nào? Ví dụ duyệt nhiều bước quan hệ

**🟢 Hiểu nhanh**
Hệ **lần theo chuỗi mũi tên** để trả lời những câu không ghi sẵn ở một chỗ nào. Ví dụ "Bệnh típ 2 được chẩn đoán bằng xét nghiệm gì, ngưỡng bao nhiêu, theo tài liệu nào?" phải đi qua Bệnh → Tiêu chuẩn → Xét nghiệm → Nguồn.

**🔧 Chi tiết kỹ thuật**
```cypher
// 3 bước: Disease → Criterion → LabTest, kèm Source
MATCH (d:Disease {id:'DTD_TIP_2'})-[r:DIAGNOSED_BY]->(c)-[:USES_TEST]->(t), (s:Source {id:r.source_id})
RETURN t.name_vi, c.op, c.value, c.unit, s.title;

// Suy luận phân biệt: triệu chứng chung của típ 1 và 2, xếp theo độ chênh xác suất
MATCH (d1:Disease {id:'DTD_TIP_1'})-[r1:HAS_SYMPTOM]->(s)<-[r2:HAS_SYMPTOM]-(d2:Disease {id:'DTD_TIP_2'})
RETURN s.name_vi, abs(r1.prob - r2.prob) AS do_phan_biet ORDER BY do_phan_biet DESC;

// Đường đi biến thiên độ dài: biến chứng của biến chứng
MATCH p = (:Disease {id:'DTD_TIP_2'})-[:HAS_COMPLICATION*1..2]->(x) RETURN p;
```
**Các dạng suy luận dùng trong hệ:**
- Luật T-03 so chỉ số với `DiagnosticCriterion` để loại trừ lớp bệnh.
- Từ thực thể đi theo `MENTIONS` để lấy đoạn văn chứng cứ.

---

### Q25. RAG là gì? Vì sao kết hợp truy xuất thông tin với LLM?

**🟢 Hiểu nhanh**
LLM giống một người **nói hay nhưng đôi khi nhớ sai**. RAG bắt người đó **đọc tài liệu đúng trước khi trả lời** và chỉ nói theo tài liệu. Kết quả là câu trả lời vừa tự nhiên vừa có nguồn kiểm chứng.

**🔧 Chi tiết kỹ thuật**
- RAG (Retrieval-Augmented Generation) gồm hai bước: truy xuất top-k đoạn liên quan, rồi đưa vào prompt cho LLM sinh câu trả lời có điều kiện trên context đó.
- **Vì sao cần:**
  - LLM có **giới hạn tri thức** theo thời điểm huấn luyện.
  - LLM có thể **hallucinate**.
  - LLM không biết tài liệu nội bộ, ví dụ bài Vinmec.
  - LLM khó **trích nguồn**.
- **Ưu điểm so với fine-tune:**
  - Cập nhật tri thức chỉ cần nạp tài liệu mới, không phải train lại.
  - Mỗi câu trả lời truy ngược được về nguồn.
  - Từ chối được khi không tìm thấy căn cứ.

---

### Q26. Kiến trúc RAG đầy đủ: nạp tài liệu, chunking, embedding, vector store, truy xuất, sinh câu trả lời

**🟢 Hiểu nhanh**
1. **Gom sách** về: crawl bài Vinmec, PDF Bộ Y tế, WHO.
2. **Cắt thành đoạn** vừa đọc.
3. Biến mỗi đoạn thành **"dấu vân tay số"** (vector) để so độ giống.
4. **Cất vào kho**.
5. Khi có câu hỏi: lấy vân tay của câu hỏi, tìm các đoạn giống nhất, đưa cho LLM viết câu trả lời.

**🔧 Chi tiết kỹ thuật (đúng như dự án)**

| Bước | Dự án |
|---|---|
| Ingestion | `ingest.fetch_vinmec` (sitemap → 303 bài), PDF QĐ 3280/2011, WHO |
| Làm sạch | Bỏ menu và quảng cáo, giữ heading |
| Chunking | **Theo heading** H2/H3 để mỗi đoạn trọn một ý; quá ~400 token thì cắt tiếp. Kết quả 2.359 chunk, trung vị 157 từ |
| Embedding | **bge-m3** (đa ngôn ngữ, hỗ trợ tiếng Việt, 1024 chiều) qua Ollama |
| Vector store | **Neo4j vector index** `chunk_embedding` (cosine), chunk nối `FROM_SOURCE` → Source, `MENTIONS` → thực thể |
| Retrieval | Hybrid: graph facts + vector top 20 + chunk MENTIONS → xếp hạng → top 5 |
| Generation | Qwen 2.5 3B (Ollama), prompt P-05: chỉ dùng context đánh số, bắt buộc trích `[n]`, temperature thấp |
| Hậu kiểm | Kiểm trích dẫn → Safety |

**📂 Chỉ cho giám khảo:** `docs/06_KB_RAG_INTEGRATION.md` §2 (sơ đồ)

---

### Q27. So sánh truy xuất từ khóa, tìm kiếm tương đồng vector và truy xuất KG

**🟢 Hiểu nhanh**
- **Từ khóa:** tìm đúng chữ. Nhanh, nhưng không hiểu "khát nước" gần nghĩa với "uống nhiều".
- **Vector:** tìm theo *ý nghĩa*. Hiểu từ đồng nghĩa, nhưng đôi khi trả đoạn "na ná" mà lạc đề.
- **KG:** tìm theo *quan hệ đã kiểm chứng*. Rất chính xác và giải thích được, nhưng chỉ trả lời được những gì đã nhập vào đồ thị.

**🔧 Chi tiết kỹ thuật**

| | Keyword (BM25 / fulltext) | Vector (dense) | KG |
|---|---|---|---|
| Cơ chế | Khớp token, TF-IDF/BM25 | Cosine giữa embedding | Duyệt quan hệ có kiểu |
| Ưu | Nhanh, khớp chính xác tên riêng, mã ICD | Bắt được ngữ nghĩa, từ đồng nghĩa, đa ngôn ngữ | Chính xác, có provenance, suy luận nhiều bước |
| Nhược | Không hiểu nghĩa | Khó tách trong/ngoài phạm vi; khó giải thích | Phủ hẹp, tốn công xây dựng |
| Trong dự án | Fulltext `entity_names` (liên kết thực thể), T-12 local | `chunk_embedding` | Graph facts 1-hop, luật T-03 |

**Số liệu thật (88 câu):**
- Vector-only: R@5 = 1,00 nhưng chỉ **từ chối đúng 0,364**.
- Graph-only: R@5 = 0,47 nhưng **từ chối đúng 0,909**.
- **Hybrid** lấy ưu điểm của cả hai: R@5 = 1,00, từ chối đúng 0,909, từ chối nhầm 0,030.

Đây là **bằng chứng định lượng** cho lý do chọn hybrid.

---

### Q28. Tích hợp KG Neo4j và RAG vào trợ lý tiểu đường thế nào?

**🟢 Hiểu nhanh**
KG đóng vai **mục lục thông minh**: biết câu hỏi nói về bệnh, thuốc hay triệu chứng nào, và đưa ra các sự thật ngắn có nguồn. RAG đóng vai **đọc nguyên văn**: lấy đoạn văn chi tiết. Hai bên nối với nhau bằng sợi dây "đoạn văn này nhắc tới thực thể kia" (quan hệ MENTIONS).

**🔧 Chi tiết kỹ thuật:** xem Q47.
- Cầu nối: `(:Chunk)-[:MENTIONS]->(thực thể)` được tạo bằng **cùng bộ liên kết thực thể** dùng cho câu hỏi, nên hai phía khớp nhau.
- KG còn phục vụ phần chẩn đoán:
  - Danh sách slot cần hỏi của mỗi Worker (`group`).
  - `prob` của `HAS_SYMPTOM` để sinh ca bệnh mô phỏng.
  - `DiagnosticCriterion` cho luật T-03.

**📂 Chỉ cho giám khảo:** `docs/06_KB_RAG_INTEGRATION.md` §2–4

---

# Phần 4 — Agent và hệ đa agent (Q29–Q38)

### Q29. Intelligent Agent là gì? Môi trường, nhận thức, trạng thái, hành động, mục tiêu, ra quyết định

**🟢 Hiểu nhanh**
Agent là **một "người làm việc" tự động**:
- **Quan sát** tình hình: nghe bệnh nhân nói.
- **Ghi nhớ** những gì đã biết: hồ sơ.
- **Tự chọn việc làm tiếp:** hỏi câu nào.
- Nhằm một **mục tiêu**: chẩn đoán đúng với ít câu hỏi nhất.

**🔧 Chi tiết kỹ thuật (Russell & Norvig): ánh xạ sang Main Agent**

| Khái niệm | Định nghĩa | Trong dự án |
|---|---|---|
| Environment | Thế giới agent tác động | Bệnh nhân (thật hoặc Patient Simulator) |
| Perception | Cảm biến → percept | Câu trả lời → NLU → slot |
| State | Biểu diễn nội bộ | `s_t` ∈ [0,1]^122: slot đã biết, `p_t`, entropy, lượt |
| Action | Tác động lên môi trường | Chọn W1..W4 / CLASSIFY; Worker chọn slot để hỏi |
| Goal | Trạng thái mong muốn | Top-1 đúng bệnh, ít lượt hỏi |
| Decision-making | Hàm chọn hành động | Policy π (DQN) + mask + luật cố định |

---

### Q30. Kiến trúc chung của agent: đầu vào, xử lý nội bộ, tri thức, cơ chế quyết định, đầu ra

**🟢 Hiểu nhanh**
Agent làm theo chu trình **nghe → hiểu → tra cứu → quyết định → nói**, lặp lại tới khi xong việc.

**🔧 Chi tiết kỹ thuật**
```
Input (percept) → Xử lý (NLU, cập nhật state) → Tri thức (KB Neo4j, mô hình học) → Quyết định (policy / luật / LLM) → Output (action: câu hỏi, câu trả lời, tool call)
```
- **Các kiểu agent:** simple reflex, model-based, goal-based, utility-based, learning agent.
- **Trong dự án:**
  - Main Agent và Workers là **learning agent, dựa trên utility**: tối đa reward kỳ vọng.
  - Safety là **simple reflex**: áp luật điều kiện–hành động.
  - Consult là **agent LLM + tool**.

---

### Q31. Phân biệt hệ một agent với hệ đa agent (MAS). Bài toán nào hợp với nhiều agent?

**🟢 Hiểu nhanh**
- **Một agent:** một người làm hết mọi việc.
- **Đa agent:** một nhóm chuyên gia, mỗi người một mảng, phối hợp với nhau.

Nhóm hợp hơn khi bài toán **có nhiều mảng khác bản chất** (hỏi bệnh, tra tài liệu, kiểm duyệt an toàn), cần **làm song song**, hoặc cần **thay từng phần** mà không phá cả hệ.

**🔧 Chi tiết kỹ thuật**
- **MAS** gồm nhiều agent tự trị, có thể có mục tiêu riêng, tương tác qua giao thức.
- **Lợi ích:**
  - Phân rã bài toán (modularity).
  - Chuyên môn hóa.
  - Song song.
  - Bền vững khi một agent lỗi.
  - Mở rộng dễ.
- **Bài toán phù hợp:** phân tán tự nhiên (logistics, robot nhóm), nhiều chuyên môn (chẩn đoán y khoa), nhiều bên có lợi ích riêng (đấu giá).
- Lý do dự án chọn MAS: xem Q43.

---

### Q32. Kiến trúc MAS: vai trò agent, giao tiếp, điều phối, hợp tác, phân công nhiệm vụ

**🟢 Hiểu nhanh**
Phải trả lời được 4 câu: **ai làm gì** (vai trò), **nói chuyện qua đâu** (giao tiếp), **ai quyết định thứ tự** (điều phối), **giao việc thế nào** (phân công).

**🔧 Chi tiết kỹ thuật: trong dự án**

| Khía cạnh | Cách làm |
|---|---|
| Vai trò | 14 agent, mỗi agent có thẻ: trách nhiệm, vào/ra, tool, **"không được làm"** (`02` §2) |
| Giao tiếp | K1 bộ nhớ chung `DialogState`; K2 gọi tool có schema; K3 giao quyền Main → Worker (`04_SEQUENCE.md` §1) |
| Điều phối | Đồ thị tự viết: Router → NLU → State → Main → Worker → Classifier → Consult → Safety |
| Hợp tác | Classifier cung cấp `p_t` vừa cho trạng thái vừa cho reward của Main/Worker; DL bổ sung ý kiến |
| Phân công | Main **học** cách chọn Worker (option); Worker trả quyền (RETURN) khi xong nhóm |

---

### Q33. So sánh kiến trúc MAS tập trung, phi tập trung, phân cấp

**🟢 Hiểu nhanh**
- **Tập trung:** có một "sếp" ra lệnh cho tất cả.
- **Phi tập trung:** mọi người ngang hàng, tự thương lượng với nhau.
- **Phân cấp:** sếp tổng chia việc cho các trưởng nhóm, trưởng nhóm lo chi tiết. Dự án dùng **phân cấp** cho phần chẩn đoán.

**🔧 Chi tiết kỹ thuật**

| Kiểu | Ưu | Nhược | Trong dự án |
|---|---|---|---|
| Tập trung | Dễ kiểm soát, nhất quán | Nút thắt cổ chai, điểm lỗi đơn | Lớp ngôn ngữ (Router điều phối) |
| Phi tập trung | Bền vững, mở rộng | Khó đảm bảo an toàn y khoa, khó kiểm thử, có thể không hội tụ | Không dùng |
| **Phân cấp** | Chia nhỏ không gian quyết định; trừu tượng theo thời gian | Cần thiết kế reward 2 cấp | **Lõi chẩn đoán**: Main (5 hành động) → Worker (5–14 hành động) |

---

### Q34. Agent giao tiếp, trao đổi thông tin thế nào? Thông điệp, tri thức chung, giao việc

**🟢 Hiểu nhanh**
Có 3 cách:
- **Nhắn tin trực tiếp** cho nhau.
- **Ghi chung một cuốn sổ** để ai cũng đọc được.
- **Giao hẳn một việc** cho người khác, đợi họ báo xong.

Dự án dùng cả ba.

**🔧 Chi tiết kỹ thuật**

| Cơ chế | Định nghĩa | Trong dự án |
|---|---|---|
| Message passing | Gửi thông điệp có cấu trúc (ví dụ FIPA-ACL: inform, request…) | Gọi tool T-xx có input/output JSON schema; mỗi lời gọi được ghi log T-10 |
| Shared knowledge (blackboard) | Vùng nhớ chung | `DialogState` (slot, `p_t`, `risk_dl`, lịch sử); KB Neo4j |
| Task delegation | Giao nhiệm vụ con, chờ kết quả | Main → `delegate(W_h)` → Worker hỏi 1..4 câu → `RETURN` |

---

### Q35. Reinforcement Learning là gì? Trạng thái, hành động, phần thưởng, policy, môi trường ⭐

**🟢 Hiểu nhanh**
RL là **học bằng thử và sai, có điểm thưởng**, giống huấn luyện thú cưng:
- Agent thử hỏi một câu.
- Câu hỏi có ích (giúp chắc chắn hơn về bệnh) thì được cộng điểm.
- Hỏi lan man thì bị trừ điểm vì tốn lượt.
- Đoán đúng bệnh được thưởng lớn, sai bị phạt lớn.

Sau hàng chục nghìn cuộc hội thoại với bệnh nhân giả, agent tự rút ra **chiến lược hỏi tốt**.

**🔧 Chi tiết kỹ thuật**

| Khái niệm | Trong dự án |
|---|---|
| Environment | Patient Simulator (ca bệnh ẩn sinh từ KB, `p_lab_unknown` = 0,5) |
| State `s_t` | Vector 122 chiều |
| Action `a_t` | Main: {W1..W4, CLASSIFY}; Worker: slot ∪ {RETURN} |
| Reward `r_t` | −1 mỗi câu hỏi; +1 nếu phát hiện dương tính; **+2·(H(p_{t−1}) − H(p_t))** cho giảm entropy (R-T13); ±20 khi kết luận đúng/sai |
| Policy π | ε-greedy trên Q(s, a) có mask; ε giảm từ 1,0 xuống 0,05 trong 20 epoch |
| Thuật toán | **DQN** (double DQN, target network, replay buffer 50k, Huber loss, γ = 0,95) |
| Huấn luyện | 40 epoch × 700 episode |

**Ý nghĩa thưởng giảm entropy:** entropy đo **mức phân vân** của bộ phân loại. Câu hỏi nào làm bộ phân loại bớt phân vân nhiều thì được thưởng nhiều. Agent nhờ vậy học cách hỏi câu "đắt giá" trước.

**📂 Chỉ cho giám khảo:** `docs/02_AGENT_DESIGN.md` §4–5 · `rl/reward.py` · `docs/assets/curve_hrl.png`

---

### Q36. Markov Decision Process (MDP): không gian trạng thái, hành động, xác suất chuyển, hàm thưởng, policy ⭐

**🟢 Hiểu nhanh**
MDP là **cách mô tả bài toán ra quyết định nhiều bước bằng toán**: đang ở tình huống nào, làm được gì, làm xong thì sang tình huống nào và được bao nhiêu điểm. Giả định Markov nghĩa là **chỉ cần biết tình huống hiện tại**, không cần nhớ toàn bộ quá khứ. Hồ sơ `s_t` đã tóm tắt đủ.

**🔧 Chi tiết kỹ thuật**
MDP = (S, A, P, R, γ).
- **S:** [0,1]^122.
- **A:**
  - Main có 5 hành động.
  - Worker có 5 / 6 / 14 / 7 hành động.
  - Hành động không hợp lệ bị **action mask** chặn.
- **P(s′|s,a):** do câu trả lời của bệnh nhân quyết định. **Không biết tường minh**, nên dùng RL không mô hình (model-free, DQN).
- **R(s,a):** công thức ở Q35.
- **γ** = 0,95.
- **Policy** π(a|s): mục tiêu tối đa `E[Σ γ^t r_t]`.
- **Bellman** (cập nhật Q): `y = r + γ · max_a′ Q_target(s′, a′)`.
- **Phân cấp = Semi-MDP (options):** Main chọn option W_h kéo dài K bước. Main nhận `R = Σ_{k<K} γ^k r_{t+k}` và cập nhật với `γ^K`.
- **Kết thúc:** CLASSIFY, hoặc T_max = 12, hoặc cấp cứu.

---

### Q37. Khác biệt giữa AI agent và thành phần phần mềm truyền thống

**🟢 Hiểu nhanh**
- Hàm phần mềm thường **làm đúng một việc khi được gọi**, với đầu vào giống nhau thì luôn ra kết quả giống nhau.
- Agent **tự quyết** nên làm gì tiếp để đạt mục tiêu, biết **dùng công cụ**, và **phối hợp** với agent khác.

**🔧 Chi tiết kỹ thuật**

| Tiêu chí | Thành phần truyền thống | AI agent |
|---|---|---|
| Tự trị | Bị gọi, luồng cố định | Tự chọn hành động theo state |
| Hướng mục tiêu | Thực hiện đặc tả | Tối ưu mục tiêu hoặc reward |
| Suy luận | Logic lập trình sẵn | Policy học được / LLM / luật KB |
| Dùng tool | Gọi API cố định | Chọn tool theo ngữ cảnh |
| Tương tác | Gọi hàm đồng bộ | Giao tiếp, giao việc, chia sẻ state |
| Kiểm thử | Tất định | Phải đánh giá thống kê (acc, số lượt) |

**Ví dụ trong dự án:**
- State Tracker và Reward là **thành phần phần mềm thường**: code xác định, không tự quyết.
- Main Agent là **agent** đúng nghĩa.

Dự án cố ý giữ phần an toàn ở dạng phần mềm xác định.

---

### Q38. Thách thức khi phát triển MAS: điều phối, giao tiếp, quyết định mâu thuẫn, độ tin cậy, bảo mật, đánh giá

**🟢 Hiểu nhanh**
Nhiều agent thì **khó phối hợp**:
- Ai làm trước?
- Hai agent nói khác nhau thì nghe ai?
- Một agent hỏng thì cả hệ có sập không?
- Làm sao chấm điểm cả nhóm?

**🔧 Chi tiết kỹ thuật: kèm bài học thật của dự án**

| Thách thức | Dự án gặp gì | Cách xử lý |
|---|---|---|
| Điều phối | Policy kết luận khi chưa hỏi xét nghiệm (44/225 ca) | Luật `labs_ready` + mask W2 → còn 0 ca |
| Giao tiếp | Tool timeout do hiểu sai `timeout_s` | Chuẩn hóa schema; log mọi tool call (T-10) |
| Quyết định mâu thuẫn | DL nói "thấp" nhưng luật nói "đạt ĐTĐ" | Thứ tự ưu tiên rõ: **luật KB > Classifier > DL**; bật cờ `dl_conflict` |
| Độ tin cậy | Neo4j hoặc Ollama tắt khiến hệ treo 4 s mỗi lượt | Mã lỗi `kb_unavailable`…, tạm ngừng 30 s, chế độ trích xuất khi LLM tắt |
| An toàn | NLU bật cảnh báo cấp cứu nhầm với câu "bỏ thuốc rồi" | Sửa NLU; test E2E |
| Bảo mật | Truy vấn web có thể lộ dữ liệu cá nhân | Lọc PII trước khi tìm web; whitelist domain |
| Đánh giá | Khó đo cả hệ | E2E 225 ca qua simulator + nhiều metric (Q50) |

---

# Phần 1–2 — ML/DL, CNN, RNN, LSTM (Q1–Q18)

> Phần lý thuyết. Câu nào liên quan thì nối sang dự án. **CNN/RNN/LSTM không có trong hệ diabetes_mas.** Không nói là đã dùng.

## 1. Machine Learning và Deep Learning

### Q1. Machine Learning là gì? Phân biệt học có giám sát, không giám sát, tăng cường. Ví dụ

**🟢 Hiểu nhanh**
ML là **cho máy học quy luật từ dữ liệu** thay vì lập trình từng luật.
- **Có giám sát:** học từ ví dụ có đáp án.
- **Không giám sát:** tự tìm nhóm trong dữ liệu không có đáp án.
- **Tăng cường:** học bằng thử–sai và điểm thưởng.

**🔧 Chi tiết kỹ thuật**

| Loại | Dữ liệu | Mục tiêu | Ví dụ (trong dự án nếu có) |
|---|---|---|---|
| Supervised | (x, y) có nhãn | Học f: x → y | **Có:** MLP dự đoán ĐTĐ (AG-13); Classifier 5 lớp |
| Unsupervised | Chỉ x | Cấu trúc ẩn: clustering, giảm chiều | Phân nhóm bệnh nhân theo chỉ số (K-means); embedding |
| Reinforcement | Tương tác (s, a, r, s′) | Tối đa tổng reward | **Có:** HRL/DQN chọn câu hỏi (Q35) |

---

### Q2. Quy trình phát triển mô hình ML: thu thập dữ liệu, tiền xử lý, tạo đặc trưng, huấn luyện, validation, test

**🟢 Hiểu nhanh**
Quy trình giống nấu ăn: **đi chợ** (thu thập) → **sơ chế** (làm sạch) → **chọn nguyên liệu** (đặc trưng) → **nấu** (train) → **nếm thử** trong lúc nấu (validation) → **khách ăn thật** (test).

**🔧 Chi tiết kỹ thuật (ví dụ AG-13)**
1. **Thu thập:** Kaggle, 100.000 dòng.
2. **Tiền xử lý:**
   - Khử trùng, còn 96.146 dòng.
   - Xử lý giá trị thiếu, ví dụ `smoking_history = No Info`.
   - Chia phân tầng 70/15/15, **fit scaler chỉ trên train** để tránh rò rỉ dữ liệu.
3. **Đặc trưng:** z-score cho biến số, one-hot cho biến phân loại, tạo vector 15 chiều.
4. **Huấn luyện:** Adam, mini-batch 256, BCE có `pos_weight`.
5. **Validation:** chọn epoch bằng early stopping và chọn ngưỡng quyết định trên val.
6. **Test:** đo **một lần** trên tập test, báo cáo PR-AUC, F1.
7. **Triển khai:** lưu checkpoint + scaler, phục vụ qua T-15 / API.

---

### Q3. Phân biệt phân loại và hồi quy qua ví dụ tiểu đường, cổ phiếu, giá nhà

**🟢 Hiểu nhanh**
- **Phân loại:** chọn một **nhãn**, ví dụ có hay không có bệnh.
- **Hồi quy:** đoán một **con số**, ví dụ giá nhà bao nhiêu tỷ, giá cổ phiếu ngày mai.

**🔧 Chi tiết kỹ thuật**

| | Phân loại | Hồi quy |
|---|---|---|
| Đầu ra | Rời rạc / xác suất lớp | Liên tục |
| Lớp ra | Sigmoid (nhị phân) / Softmax (đa lớp) | Tuyến tính |
| Loss | BCE / CE | MSE / MAE |
| Metric | Accuracy, P, R, F1, AUC | MAE, RMSE, R² |
| Ví dụ | Tiểu đường 0/1 (AG-13); 5 lớp ĐTĐ (AG-02) | Giá nhà, giá cổ phiếu |

Lưu ý: "cổ phiếu tăng hay giảm" lại là bài toán **phân loại**.

---

### Q4. Kiến trúc mạng nơ-ron nhân tạo: lớp vào, lớp ẩn, lớp ra, trọng số, bias

**🟢 Hiểu nhanh**
Mạng nơ-ron là **nhiều lớp "bộ lọc" xếp chồng**:
- **Lớp vào** nhận thông tin.
- Các **lớp ẩn** kết hợp thông tin thành đặc điểm ngày càng trừu tượng.
- **Lớp ra** đưa kết luận.

**Trọng số** là mức "tin" vào từng đầu vào. **Bias** là ngưỡng kích hoạt.

**🔧 Chi tiết kỹ thuật**
- Mỗi nơ-ron tính `z = w·x + b`, rồi `a = σ(z)` với σ là hàm kích hoạt (ReLU, Sigmoid…).
- **Ví dụ AG-13:**
  - Lớp vào 15 chiều.
  - Lớp ẩn 1 có 64 nơ-ron, ReLU.
  - Lớp ẩn 2 có 32 nơ-ron, ReLU.
  - Lớp ra 1 nơ-ron, Sigmoid.
  - Số tham số: (15·64 + 64) + (64·32 + 32) + (32·1 + 1) = **3.137**.
- Không có hàm kích hoạt phi tuyến thì nhiều lớp chồng lên nhau vẫn chỉ tương đương **một phép tuyến tính**.

---

### Q5. Lan truyền thuận và lan truyền ngược. Gradient được tính và dùng để cập nhật tham số thế nào?

**🟢 Hiểu nhanh**
- **Lan truyền thuận:** đưa dữ liệu đi qua mạng để ra dự đoán.
- So dự đoán với đáp án để biết **sai bao nhiêu**.
- **Lan truyền ngược:** chia "lỗi" ngược về từng trọng số, xem mỗi trọng số góp bao nhiêu vào cái sai, rồi **chỉnh nhẹ** từng trọng số theo hướng giảm sai.

**🔧 Chi tiết kỹ thuật**
- **Forward:** `a^(l) = σ(W^(l) a^(l−1) + b^(l))`, rồi tính loss L(ŷ, y).
- **Backward:** dùng **quy tắc dây chuyền**.
  - `δ^(L) = ∂L/∂z^(L)`.
  - `δ^(l) = (W^(l+1))ᵀ δ^(l+1) ⊙ σ′(z^(l))`.
  - `∂L/∂W^(l) = δ^(l) (a^(l−1))ᵀ`, `∂L/∂b^(l) = δ^(l)`.
- **Cập nhật:** `W ← W − η · ∂L/∂W`. Dự án dùng Adam, có thêm momentum và tốc độ học thích nghi theo từng tham số.
- PyTorch làm tự động: `loss.backward()` rồi `optimizer.step()`.

---

### Q6. Hàm mất mát là gì? So sánh MSE, Binary Cross-Entropy, Categorical Cross-Entropy

**🟢 Hiểu nhanh**
Loss là **thước đo mức sai** của mô hình. Huấn luyện chính là tìm trọng số làm thước này nhỏ nhất. Mỗi loại bài toán cần một loại thước phù hợp.

**🔧 Chi tiết kỹ thuật**

| Loss | Công thức | Dùng khi | Dự án |
|---|---|---|---|
| MSE | `(1/n) Σ (y − ŷ)²` | Hồi quy | Giá nhà / cổ phiếu |
| BCE | `−[y log p + (1−y) log(1−p)]` | Nhị phân, Sigmoid | **AG-13**, có `pos_weight` 10,34 |
| CCE | `−Σ_k y_k log p_k` | Đa lớp, Softmax | **AG-02/03** (5 lớp) |
| Huber | MSE khi lỗi nhỏ, MAE khi lỗi lớn | Hồi quy bền với ngoại lai | **DQN** (Q-value) |

Không dùng MSE cho phân loại vì kết hợp với sigmoid, gradient bị nhỏ khi mô hình sai nhiều, và bề mặt loss không lồi.

---

### Q7. Gradient Descent và các biến thể Batch, Stochastic, Mini-batch. Vai trò của learning rate

**🟢 Hiểu nhanh**
Huấn luyện giống **xuống núi trong sương mù**: mỗi bước dò độ dốc dưới chân rồi bước xuống.
- **Batch:** nhìn toàn bộ địa hình rồi mới bước. Chắc chắn nhưng chậm.
- **Stochastic:** nhìn một điểm rồi bước ngay. Nhanh nhưng loạng choạng.
- **Mini-batch:** nhìn một nhóm điểm. Cân bằng giữa hai cách trên.

**Learning rate** là **độ dài bước chân**.

**🔧 Chi tiết kỹ thuật**

| Biến thể | Gradient tính trên | Ưu | Nhược |
|---|---|---|---|
| Batch GD | Toàn bộ N mẫu | Ổn định | Chậm, tốn RAM |
| SGD | 1 mẫu | Cập nhật nhanh, có thể thoát cực tiểu nông | Nhiễu |
| **Mini-batch** | B mẫu (dự án: 256 với DL, 128 với classifier, 64 với DQN) | Tận dụng vector hóa, cân bằng | Thêm siêu tham số B |

- **Learning rate η:** quá lớn thì phân kỳ hoặc dao động; quá nhỏ thì hội tụ chậm hoặc kẹt.
- Dự án dùng **Adam** với lr 1e-3 cho DL và 5e-4 cho DQN.

---

### Q8. Overfitting và underfitting. Vai trò tập train, validation, test. Regularization, dropout, early stopping

**🟢 Hiểu nhanh**
- **Overfitting:** **học vẹt**. Thuộc lòng bài tập nhưng gặp đề mới thì sai.
- **Underfitting:** **học chưa đủ**. Ngay bài tập cũng sai.

Tập **validation** là đề thi thử, dùng để biết lúc nào nên dừng học. Tập **test** là đề thi thật, chỉ làm một lần.

**🔧 Chi tiết kỹ thuật**
- **Nhận biết:**
  - Overfit: loss train giảm nhưng loss val tăng.
  - Underfit: cả hai đều cao.
- **Chống overfit:**
  - **L2 / weight decay:** thêm `λ‖W‖²` vào loss. Dự án dùng `weight_decay` 1e-4 cho classifier.
  - **Dropout:** tắt ngẫu nhiên p nơ-ron khi train. Dự án dùng 0,2 cho AG-13, 0,2–0,3 cho classifier.
  - **Early stopping:** dừng khi metric val không cải thiện sau k epoch. Dự án dùng patience 5; AG-13 dừng ở epoch 32/50.
  - Thêm dữ liệu, giảm kích thước mạng.
  - **Che slot ngẫu nhiên** khi train classifier (`slot_mask_prob`), mô phỏng thông tin thiếu.
- **Chống underfit:** mạng lớn hơn, train lâu hơn, thêm đặc trưng.

---

### Q9. Đánh giá mô hình phân loại: accuracy, precision, recall, F1, ma trận nhầm lẫn. Vì sao accuracy gây hiểu lầm với dữ liệu tiểu đường mất cân bằng?

**🟢 Hiểu nhanh**
Nếu 91% người không bệnh, một mô hình **luôn nói "không bệnh"** vẫn đạt **91% accuracy** mà không phát hiện được ai. Vì vậy cần hỏi thêm hai câu:
- Trong số người bị báo bệnh, bao nhiêu người bệnh thật? Đó là **precision**.
- Trong số người bệnh thật, bắt được bao nhiêu? Đó là **recall**.

**🔧 Chi tiết kỹ thuật**
- **Ma trận nhầm lẫn:** TP, FP, FN, TN.
- **Công thức:**
  - Accuracy = (TP+TN)/N.
  - Precision = TP/(TP+FP).
  - Recall = TP/(TP+FN).
  - F1 = 2PR/(P+R).
- **Dữ liệu mất cân bằng:** dùng **PR-AUC**, recall, F1, và chọn ngưỡng theo chi phí lỗi.
- **Trong y tế FN đắt hơn FP**, nên AG-13 chọn ngưỡng sao cho recall ≥ 0,90.
- **Kết quả AG-13:** recall 0,911, precision 0,476, PR-AUC 0,885 (baseline ngẫu nhiên khoảng 0,09).

---

### Q10. So sánh mô hình phân loại và hồi quy trong thực tế. Thách thức khi triển khai DL thành web app / API

**🟢 Hiểu nhanh**
Train được mô hình mới là **một nửa** công việc. Khi đưa lên web còn phải đảm bảo:
- Dữ liệu người dùng nhập được xử lý **giống hệt** lúc train.
- Phản hồi đủ nhanh.
- Biết cách xử lý khi người dùng nhập thiếu hoặc nhập bậy.
- Có cách cập nhật mô hình mà không làm hỏng hệ.

**🔧 Chi tiết kỹ thuật**
- So sánh phân loại và hồi quy: xem Q3, Q41.
- **Thách thức triển khai:**
  1. **Tính nhất quán tiền xử lý:** lưu scaler và encoder cùng model.
  2. **Validate input**, xử lý giá trị thiếu (dự án dùng biến thể `no_labs`).
  3. **Độ trễ và tài nguyên:** nạp model một lần khi khởi động.
  4. **Phiên bản model và rollback** (`best/` chỉ ghi đè khi val tốt hơn).
  5. **Data drift:** phân phối dữ liệu thật khác dữ liệu train.
  6. **Giám sát và log** (log T-10).
  7. **Bảo mật** và quyền riêng tư dữ liệu y tế.
  8. **Giải thích được** và **hiệu chỉnh xác suất** (calibration).

---

## 2. CNN, RNN, LSTM

> ⚠ Hệ diabetes_mas **không dùng** CNN/RNN/LSTM vì dữ liệu là bảng và hội thoại có trạng thái. Theo quyết định ghi trong `01_REQUIREMENTS.md` §2.2, nhóm không thêm các mạng này chỉ để phủ đề.

### Q11. Kiến trúc CNN: lớp tích chập, bộ lọc, feature map, hàm kích hoạt, pooling

**🟢 Hiểu nhanh**
CNN nhìn ảnh **qua nhiều "kính lúp" nhỏ trượt khắp ảnh**. Mỗi kính lúp chuyên phát hiện một dạng hình, ví dụ cạnh dọc hay góc tròn. Sau đó ảnh được thu nhỏ dần để giữ lại cái quan trọng.

**🔧 Chi tiết kỹ thuật**
- **Lớp tích chập (conv):** K bộ lọc (filter / kernel) kích thước k×k×C_in trượt trên ảnh, tạo ra K **feature map**.
- **Hàm kích hoạt** (ReLU) thêm tính phi tuyến.
- **Pooling** (max / average 2×2):
  - Giảm kích thước không gian.
  - Bất biến với dịch chuyển nhỏ.
  - Giảm tham số.
- **Cấu trúc điển hình:** `[Conv → ReLU → Pool] × n → Flatten → FC → Softmax`.
- **Chia sẻ trọng số** và **kết nối cục bộ** giúp CNN có ít tham số hơn nhiều so với mạng fully-connected.

---

### Q12. Phép tích chập hoạt động thế nào? Vai trò của kernel size, stride, padding, số bộ lọc

**🟢 Hiểu nhanh**
Đặt một ô lưới nhỏ lên ảnh, **nhân từng điểm rồi cộng lại** được một số. Trượt ô lưới sang vị trí bên cạnh rồi làm lại. Các tham số quyết định ô to hay nhỏ, mỗi lần trượt bao xa, và có độn viền ảnh hay không.

**🔧 Chi tiết kỹ thuật**
- Công thức: `y[i,j] = Σ_m Σ_n x[i·s+m, j·s+n] · w[m,n] + b`.
- Kích thước đầu ra: **`O = ⌊(W − K + 2P)/S⌋ + 1`**.

| Tham số | Vai trò |
|---|---|
| Kernel size K | Vùng nhìn (receptive field); 3×3 phổ biến |
| Stride S | Bước trượt; S = 2 giảm kích thước một nửa |
| Padding P | Độn viền; "same" giữ kích thước, giữ thông tin ở biên |
| Số filter | Số feature map ra = số đặc trưng học được; tham số = K·K·C_in·C_out + C_out |

---

### Q13. Vì sao CNN hiệu quả với phân loại ảnh? CNN học đặc trưng nhiều cấp thế nào?

**🟢 Hiểu nhanh**
CNN học **từ nét đơn giản đến hình phức tạp**:
- Lớp đầu thấy cạnh và màu.
- Lớp giữa ghép thành mắt, bánh xe.
- Lớp cuối nhận ra cả khuôn mặt hay chiếc xe.

**🔧 Chi tiết kỹ thuật**
- **Thiên kiến quy nạp (inductive bias) phù hợp với ảnh:**
  - Tính cục bộ.
  - Bất biến dịch chuyển (chia sẻ trọng số).
  - Phân cấp không gian.
- **Ít tham số hơn MLP**, nên ít overfit hơn.
- **Đặc trưng nhiều cấp:** cạnh/texture (lớp nông), rồi part, rồi object (lớp sâu). Receptive field tăng dần theo độ sâu.
- Đặc trưng học được **tái sử dụng** được (transfer learning).

---

### Q14. Quy trình làm ứng dụng phân loại ảnh bằng CNN: tiền xử lý, train, validation, suy luận, triển khai

**🟢 Hiểu nhanh**
Quy trình gồm 5 bước:
1. **Chuẩn bị ảnh:** cùng cỡ, cùng độ sáng, có thể xoay lật để có thêm mẫu.
2. **Dạy mô hình** trên ảnh có nhãn.
3. **Kiểm tra** trên ảnh mô hình chưa thấy.
4. **Nhận ảnh mới** và trả nhãn.
5. **Đưa lên web hoặc điện thoại.**

**🔧 Chi tiết kỹ thuật**
1. **Tiền xử lý:**
   - Resize (ví dụ 224×224) và chuẩn hóa theo mean/std.
   - **Data augmentation** chỉ áp dụng cho tập train: flip, rotate, crop, color jitter.
2. **Huấn luyện:**
   - Thường dùng **transfer learning** (ResNet, EfficientNet pretrained) để fine-tune.
   - Loss CE, optimizer Adam/SGD với lr schedule.
3. **Validation:** early stopping, ma trận nhầm lẫn, chọn checkpoint.
4. **Suy luận:**
   - Tiền xử lý giống hệt lúc train, rồi softmax lấy top-k.
   - Có thể dùng batch inference.
5. **Triển khai:**
   - Xuất ONNX/TorchScript, phục vụ qua REST API (FastAPI).
   - Có thể dùng quantization, GPU.
   - Giám sát drift.

---

### Q15. RNN là gì? Hidden state và cách thông tin lan qua chuỗi

**🟢 Hiểu nhanh**
RNN đọc dữ liệu **theo thứ tự, từng bước một**, và giữ một **"trí nhớ ngắn"** mang từ bước trước sang bước sau. Ví dụ đọc giá cổ phiếu từng ngày, trí nhớ tóm tắt xu hướng đến hôm nay.

**🔧 Chi tiết kỹ thuật**
- `h_t = tanh(W_xh x_t + W_hh h_{t−1} + b_h)`, `y_t = W_hy h_t + b_y`.
- **Cùng bộ trọng số** được dùng ở mọi bước thời gian.
- Huấn luyện bằng **BPTT** (backpropagation through time): trải mạng theo thời gian rồi lan truyền ngược.
- **Các dạng:** many-to-one (phân loại câu), many-to-many (gắn nhãn chuỗi), seq2seq (dịch máy).

**Liên hệ dự án:** hội thoại cũng là chuỗi. Nhưng hệ dùng **state tường minh** `s_t` do State Tracker cập nhật, thay vì để RNN tự học hidden state. Cách này giải thích được và thỏa giả định Markov (Q36).

---

### Q16. Vanishing và exploding gradient trong RNN. Vì sao chúng làm khó học phụ thuộc dài hạn?

**🟢 Hiểu nhanh**
Khi truyền lỗi ngược qua nhiều bước, lỗi bị **nhân đi nhân lại**:
- Nếu nhân với số nhỏ hơn 1, lỗi **tắt dần về 0**. Mạng "quên" những gì xảy ra lâu.
- Nếu nhân với số lớn hơn 1, lỗi **nổ tung**. Huấn luyện mất ổn định.

**🔧 Chi tiết kỹ thuật**
- `∂h_T/∂h_t = Π_{k=t+1}^{T} W_hhᵀ · diag(tanh′(z_k))`, tức tích của T−t ma trận.
  - Trị riêng lớn nhất nhỏ hơn 1 dẫn đến **vanishing**.
  - Trị riêng lớn nhất lớn hơn 1 dẫn đến **exploding**.
- **Hệ quả:** gradient từ xa gần bằng 0, nên không học được phụ thuộc dài hạn.
- **Cách khắc phục:**
  - Exploding: **gradient clipping**.
  - Vanishing: **LSTM/GRU** (cơ chế cổng), khởi tạo trọng số tốt, ReLU, residual.

---

### Q17. Kiến trúc LSTM: forget gate, input gate, output gate, cell state, hidden state

**🟢 Hiểu nhanh**
LSTM có một **"băng chuyền trí nhớ dài hạn"** (cell state) chạy suốt chuỗi, và ba **cánh cổng**:
- **Cổng quên:** quyết định bỏ điều gì cũ.
- **Cổng nhập:** quyết định ghi thêm điều gì mới.
- **Cổng xuất:** quyết định lấy ra điều gì để dùng ngay.

**🔧 Chi tiết kỹ thuật**
```
f_t = σ(W_f [h_{t−1}, x_t] + b_f)          forget gate
i_t = σ(W_i [h_{t−1}, x_t] + b_i)          input gate
c̃_t = tanh(W_c [h_{t−1}, x_t] + b_c)       ứng viên
c_t = f_t ⊙ c_{t−1} + i_t ⊙ c̃_t            cell state (cập nhật CỘNG)
o_t = σ(W_o [h_{t−1}, x_t] + b_o)          output gate
h_t = o_t ⊙ tanh(c_t)                      hidden state
```
- Phép **cộng** trên cell state tạo đường gradient gần như không suy giảm (khi `f_t` gần 1), nên giảm vanishing gradient.
- Số tham số gấp khoảng 4 lần RNN thường.

---

### Q18. So sánh RNN, LSTM, CNN. Vì sao RNN/LSTM hợp với dự đoán chuỗi thời gian cổ phiếu?

**🟢 Hiểu nhanh**
- **CNN** giỏi tìm **mẫu theo không gian**, ví dụ ảnh.
- **RNN/LSTM** giỏi **dữ liệu có thứ tự thời gian**, vì giá hôm nay phụ thuộc giá những hôm trước. LSTM nhớ được xa hơn RNN thường.

**🔧 Chi tiết kỹ thuật**

| | CNN | RNN | LSTM |
|---|---|---|---|
| Thế mạnh | Mẫu cục bộ không gian, song song tốt | Chuỗi ngắn, đơn giản | Phụ thuộc dài hạn |
| Hạn chế | Không có trí nhớ thời gian (trừ 1D-CNN/TCN) | Vanishing gradient, tuần tự nên chậm | Nặng hơn, vẫn tuần tự |
| Ứng dụng | Ảnh, 1D-CNN cho tín hiệu | Chuỗi ngắn | Ngôn ngữ, chuỗi thời gian, cổ phiếu |

**Với cổ phiếu:**
- Dữ liệu có **thứ tự** và **tự tương quan**.
- LSTM nhận cửa sổ trượt (ví dụ 60 ngày) để dự đoán ngày t+1, và học được xu hướng, chu kỳ.
- **Lưu ý:**
  - Chia train/test **theo thời gian**.
  - Chuẩn hóa chỉ bằng thống kê của tập train.
  - Thị trường nhiễu và không dừng, nên dự đoán chính xác rất khó. Cần so với baseline "giá ngày mai = giá hôm nay".

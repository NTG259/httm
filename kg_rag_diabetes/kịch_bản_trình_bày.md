# Kịch bản trình bày: Hệ thống hỏi đáp bệnh tiểu đường bằng AI

---

## 1. Giới thiệu tổng quan

Dự án xây dựng một **AI agent** (tác nhân trí tuệ nhân tạo tự động) có khả năng trả lời câu hỏi về bệnh tiểu đường theo ngữ cảnh y tế.

Điểm khác biệt so với chatbot thông thường: thay vì chỉ dựa vào kiến thức huấn luyện sẵn của mô hình ngôn ngữ, hệ thống này **truy xuất tri thức từ nguồn dữ liệu thực, có thể kiểm chứng** trước khi trả lời. Mô hình ngôn ngữ chỉ đóng vai trò tổng hợp và diễn đạt — không được phép bịa đặt thêm thông tin ngoài những gì hệ thống cung cấp.

Hệ thống kết hợp hai hướng tiếp cận bổ sung cho nhau:

- **RAG** (Retrieval-Augmented Generation — sinh câu trả lời có tăng cường truy xuất): tìm đoạn văn bản liên quan từ tài liệu y tế
- **Knowledge Graph** (đồ thị tri thức): tra cứu quan hệ có cấu trúc giữa bệnh, thuốc, triệu chứng và gen

---

## 2. Nguồn dữ liệu

Hệ thống sử dụng ba nguồn dữ liệu công khai, có thẩm quyền:

### 2.1 Hetionet (đồ thị tri thức sinh y)
- Là một **knowledge graph** (đồ thị tri thức) học thuật chứa hơn 2 triệu mối quan hệ giữa các thực thể sinh y: bệnh, thuốc, gen, triệu chứng, cơ quan.
- Phiên bản sử dụng: v1.0.0 — cố định để kết quả có thể tái hiện.
- Từ toàn bộ đồ thị, hệ thống **lọc ra chỉ những nút và cạnh liên quan đến tiểu đường** (các node Disease có tên chứa "diabet").
- Giữ lại 6 loại quan hệ có ý nghĩa y học:
  - **CtD** — thuốc điều trị bệnh
  - **CpD** — thuốc làm giảm triệu chứng bệnh
  - **DpS** — bệnh biểu hiện triệu chứng gì
  - **DrD** — bệnh tương tự bệnh nào
  - **DlA** — bệnh khu trú tại cơ quan nào
  - **DaG** — bệnh liên quan đến gen nào

### 2.2 WHO Global Report on Diabetes (2016)
- Báo cáo toàn cầu về bệnh tiểu đường của Tổ chức Y tế Thế giới, định dạng PDF.
- Nguồn văn bản chính thống, có số trang để trích dẫn.

### 2.3 MedlinePlus Health Topics
- Thư viện y tế điện tử của Thư viện Y học Quốc gia Hoa Kỳ (NLM/NIH), định dạng XML.
- Hệ thống tự động tìm bản mới nhất theo ngày trong tên file — không ghi cứng ngày cụ thể.
- Chỉ lấy các chủ đề tiếng Anh có liên quan đến tiểu đường.

> Ngoài ra hệ thống hỗ trợ bổ sung PDF tiếng Việt từ Bộ Y tế (QĐ 5481/2020 về điều trị ĐTĐ típ 2, hướng dẫn ĐTĐ típ 1 2024, tiền ĐTĐ 2020).

---

## 3. Xử lý dữ liệu

Dữ liệu thô được xử lý qua hai nhánh song song:

### 3.1 Nhánh văn bản — tạo "chunk"

**Chunk** là một đoạn văn bản nhỏ, đủ ngắn để mô hình embedding (mã hóa ngữ nghĩa) xử lý được, nhưng đủ dài để mang nghĩa.

Quy trình:
1. Đọc từng trang PDF / từng chủ đề XML
2. Làm sạch: loại bỏ ký tự thừa, khoảng trắng, thẻ HTML
3. Cắt thành các đoạn 150 từ, với phần **chồng lấp 30 từ** giữa hai đoạn liền kề — phần chồng lấp giúp không bị mất ý nằm ở ranh giới đoạn
4. Mỗi chunk lưu kèm metadata: nguồn gốc, số trang (hoặc URL) để trích dẫn sau này

Kết quả lưu tại: `data/processed/chunks.jsonl` — mỗi dòng là một chunk dạng JSON.

### 3.2 Nhánh đồ thị — xây dựng Knowledge Graph

1. Đọc file node Hetionet → xác định tất cả node "Disease" có tên chứa "diabet"
2. Đọc file cạnh (quan hệ) → chỉ giữ lại cạnh nối với các node bệnh tiểu đường vừa tìm được, và thuộc 6 loại quan hệ đã chọn
3. Mỗi dòng lưu dạng **triple** (bộ ba): `chủ thể — quan hệ — đối tượng`, ví dụ: `Metformin — CtD — Type 2 diabetes mellitus`

Kết quả lưu tại: `data/processed/diabetes_triples.csv`

Đồ thị được nạp vào bộ nhớ dưới dạng **MultiDiGraph** (đồ thị có hướng, cho phép nhiều loại quan hệ giữa cùng một cặp nút) nhờ thư viện NetworkX.

---

## 4. Lưu trữ

| Dữ liệu | File | Định dạng | Mục đích |
|---|---|---|---|
| Đoạn văn bản | `chunks.jsonl` | JSON theo dòng | Nguồn tra cứu văn bản |
| Embedding (vector ngữ nghĩa) | `embeddings_<hash>.npy` | NumPy binary | Cache, tránh tính lại |
| Đồ thị tri thức | `diabetes_triples.csv` | CSV | Nguồn tra cứu quan hệ |

**Embedding** (vector ngữ nghĩa) là cách biểu diễn một đoạn văn bản thành một dãy số, sao cho hai đoạn có nghĩa gần nhau thì dãy số của chúng cũng gần nhau trong không gian toán học. File embedding được đặt tên theo mã MD5 của toàn bộ nội dung chunk — nếu dữ liệu thay đổi, hệ thống tự phát hiện và tính lại; nếu không thay đổi thì đọc từ cache, tiết kiệm thời gian khởi động.

---

## 5. Kiến trúc AI Agent

Khi nhận một câu hỏi, hệ thống thực hiện ba bước tuần tự trước khi gọi mô hình ngôn ngữ:

```
Câu hỏi
  ├─► [1] Nhận diện thực thể  →  Truy xuất đồ thị tri thức (Graph Facts)
  └─► [2] Mã hóa câu hỏi     →  Tìm kiếm vector (Document Evidence)
                                              ↓
                              [3] Ghép ngữ cảnh + xây dựng Prompt
                                              ↓
                              Mô hình ngôn ngữ (Qwen2.5 local)
                                              ↓
                                       Câu trả lời có trích dẫn
```

---

## 6. Nhận diện thực thể (Entity Recognition)

Đây là bước xác định xem câu hỏi đang nói đến thực thể nào tồn tại trong đồ thị.

Cách làm: so khớp chuỗi — tên các nút trong đồ thị (tên thuốc, tên bệnh, tên gen...) được so sánh trực tiếp với nội dung câu hỏi. Không dùng mô hình NLP riêng, không cần danh sách viết tay.

Ví dụ: câu hỏi "What drugs treat type 2 diabetes?" → nhận diện được thực thể **"type 2 diabetes mellitus"** trong đồ thị.

Hệ thống cũng nhận diện ý định của câu hỏi để lọc loại quan hệ phù hợp — câu hỏi hỏi về "drug/treat" thì chỉ tra quan hệ CtD và CpD, thay vì đổ toàn bộ đồ thị vào ngữ cảnh.

---

## 7. Truy xuất vector (Vector Retrieval)

Mô hình embedding được sử dụng: **SBERT** (Sentence-BERT — mô hình Transformer chuyên mã hóa câu thành vector ngữ nghĩa), cụ thể là `all-MiniLM-L6-v2`, tối đa 256 token mỗi đoạn.

Quy trình truy xuất:
1. Câu hỏi được mã hóa thành vector
2. Tính **cosine similarity** (độ tương đồng cosin — đo góc giữa hai vector, bằng 1 nếu hoàn toàn giống, bằng 0 nếu không liên quan) giữa vector câu hỏi và từng vector chunk
3. Trả về **Top-K** chunk có điểm cao nhất (mặc định K=3)

Nếu không muốn tải mô hình SBERT (ví dụ máy yếu hoặc không có mạng), hệ thống tự động chuyển sang **TF-IDF** (phương pháp đếm tần suất từ có trọng số, nhẹ hơn nhiều nhưng kém ngữ nghĩa hơn) làm phương án dự phòng.

---

## 8. Cơ chế "đủ thông tin" (enough flag)

Trước khi gọi mô hình ngôn ngữ, hệ thống kiểm tra xem thông tin thu thập được có đủ để trả lời không, thông qua một cờ boolean `enough`:

- `enough = True` nếu: tìm được ít nhất một sự kiện đồ thị, **hoặc** điểm cosine cao nhất của chunk vượt ngưỡng tối thiểu (mặc định 0.30)
- `enough = False` → hệ thống **từ chối trả lời** thay vì bịa đặt

Đây là cơ chế bảo vệ tính chính xác của thông tin y tế — thà nói "không đủ thông tin" còn hơn trả lời sai.

---

## 9. Xây dựng Prompt

**Prompt** là đoạn văn bản được gửi đến mô hình ngôn ngữ, bao gồm:

1. **Hướng dẫn hành vi**: chỉ trả lời từ ngữ cảnh được cung cấp, không bịa đặt, trích dẫn nguồn trong ngoặc vuông, không đưa lời khuyên y tế cá nhân
2. **Graph Facts**: các sự kiện từ đồ thị tri thức (nếu có)
3. **Document Evidence**: các đoạn văn bản liên quan (nguồn, số trang, nội dung)
4. **Câu hỏi** của người dùng

Mô hình ngôn ngữ nhận toàn bộ prompt này và tổng hợp câu trả lời — nó **không tự bổ sung kiến thức ngoài** phần ngữ cảnh đã được cung cấp.

---

## 10. Mô hình ngôn ngữ: Qwen2.5:3b chạy local

Mô hình ngôn ngữ được dùng: **Qwen2.5:3b** — mô hình của Alibaba Cloud, 3 tỷ tham số, chạy hoàn toàn trên máy local không cần internet.

Hệ thống phục vụ mô hình qua **Ollama** (phần mềm đóng gói và chạy mô hình ngôn ngữ cục bộ, tương tự như Docker nhưng dành riêng cho LLM) trong Docker container, tận dụng GPU RTX 4050 Laptop để tăng tốc.

Giao tiếp giữa agent và Ollama qua REST API nội bộ (`localhost:11434`), không có dữ liệu nào ra ngoài mạng.

---

## 11. Ba chế độ truy xuất

Hệ thống cung cấp ba chế độ độc lập để so sánh chất lượng:

| Chế độ | Nguồn tri thức | Phù hợp với câu hỏi |
|---|---|---|
| **RAG thuần** | Chỉ văn bản (PDF, XML) | Câu hỏi cần giải thích dài, ngữ cảnh phong phú |
| **Knowledge Graph thuần** | Chỉ đồ thị Hetionet | Câu hỏi tra cứu quan hệ cụ thể (thuốc nào, triệu chứng gì) |
| **Hybrid** | Kết hợp cả hai | Câu hỏi phức tạp, cần cả quan hệ lẫn giải thích |

---

## 12. Đánh giá hệ thống

Bộ đánh giá (`evaluate.py`) chạy trên file `eval_set.csv` gồm hai loại câu hỏi:

**Câu hỏi có đáp án** — đo **Recall@K** (tỷ lệ truy xuất đúng trong K kết quả đầu):
- Recall@1, Recall@3, Recall@5
- Kỳ vọng dựa trên nội dung (một cụm từ cụ thể phải xuất hiện trong chunk hoặc sự kiện đồ thị), không dựa trên số thứ tự — nghĩa là bộ test vẫn hợp lệ sau khi cập nhật dữ liệu hay thay đổi kích thước chunk.

**Câu hỏi không có đáp án** — đo **tỷ lệ từ chối đúng**:
- Hệ thống không được bịa câu trả lời khi dữ liệu không đủ
- Chỉ số này đặc biệt quan trọng trong lĩnh vực y tế

---

## 13. Luồng dữ liệu tổng thể (tóm tắt)

```
[Dữ liệu thô]
  Hetionet TSV/GZ  →  build_kg.py   →  diabetes_triples.csv  →  NetworkX Graph
  WHO PDF          →                →                         →
  MedlinePlus XML  →  ingest.py     →  chunks.jsonl           →  SBERT Embeddings
  PDF Bộ Y tế      →                →                         →

[Khi có câu hỏi]
  Câu hỏi → Nhận diện thực thể → Graph Facts (1-hop)   ─┐
  Câu hỏi → Vector embedding   → Top-K chunks           ─┤→ Prompt → Qwen2.5 → Trả lời
                                  (cosine similarity)    ─┘
```

---

## 14. Câu hỏi thường gặp từ giảng viên

**Q: Tại sao không dùng Neo4j (hệ quản trị cơ sở dữ liệu đồ thị chuyên dụng) mà dùng NetworkX?**
> Dự án là prototype học thuật với quy mô nhỏ (~vài nghìn triple). NetworkX đủ dùng và không cần cài thêm server. Nếu mở rộng lên hàng triệu triple hoặc cần truy vấn phức tạp bằng ngôn ngữ Cypher (ngôn ngữ truy vấn đồ thị của Neo4j), thì nên chuyển sang Neo4j.

**Q: Tại sao không fine-tune mô hình ngôn ngữ trực tiếp trên dữ liệu tiểu đường?**
> Fine-tune (tinh chỉnh — huấn luyện thêm mô hình trên dữ liệu chuyên ngành) đòi hỏi tài nguyên tính toán lớn và dữ liệu có nhãn chất lượng cao. RAG cho phép cập nhật tri thức bằng cách thay dữ liệu nguồn, không cần huấn luyện lại — phù hợp hơn với lĩnh vực y tế nơi hướng dẫn điều trị cập nhật thường xuyên.

**Q: Embedding được tính như thế nào và lưu ở đâu?**
> Mỗi chunk được đưa qua mô hình SBERT để ra vector 384 chiều. Toàn bộ ma trận vector được lưu vào file `.npy` (định dạng nhị phân của NumPy), đặt tên theo mã MD5 của nội dung để tự động phát hiện khi dữ liệu thay đổi.

**Q: Cosine similarity hoạt động thế nào trong truy xuất?**
> Mỗi chunk và câu hỏi đều được biểu diễn thành vector. Cosine similarity đo góc giữa hai vector — góc càng nhỏ (giá trị càng gần 1) thì nội dung càng liên quan. Toàn bộ phép tính là một phép nhân ma trận, rất nhanh ngay cả khi có hàng nghìn chunk.

**Q: Làm sao hệ thống biết câu hỏi nào không thể trả lời?**
> Thông qua cờ `enough`: nếu không có sự kiện đồ thị nào khớp **và** điểm cosine cao nhất thấp hơn ngưỡng 0.30, hệ thống coi là không đủ bằng chứng và trả về cảnh báo thay vì câu trả lời. Ngưỡng 0.30 có thể điều chỉnh tùy bộ dữ liệu.

**Q: Hệ thống có thể mở rộng sang bệnh khác không?**
> Có. Hetionet chứa nhiều bệnh khác ngoài tiểu đường. Chỉ cần thay từ khóa lọc "diabet" thành tên bệnh khác khi xây đồ thị, và thêm tài liệu PDF/XML tương ứng vào bước ingest là đủ.

**Q: Qwen2.5:3b có đủ mạnh cho bài toán y tế không?**
> Với 3 tỷ tham số, Qwen2.5:3b đủ để tổng hợp và diễn đạt ngữ cảnh được cung cấp sẵn. Chất lượng câu trả lời phụ thuộc chủ yếu vào chất lượng truy xuất (RAG + KG), không phụ thuộc nhiều vào kích thước mô hình vì mô hình không cần "nhớ" kiến thức — kiến thức đã được đưa vào prompt.

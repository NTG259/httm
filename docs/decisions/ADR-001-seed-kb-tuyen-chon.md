# ADR-001: KB chẩn đoán lõi lấy từ seed tuyển chọn có kiểm chứng trích dẫn, không dùng LLM trích xuất

- Ngày: 2026-11-11 · Trạng thái: Chấp nhận
- Liên quan: `00_DATA_SOURCES.md` mục 3, `00_KB_GRAPH.md`

## Bối cảnh

Thiết kế ban đầu dự kiến trích triple bằng LLM từ văn bản (`kb/extract.py`). Phần KB dùng cho chẩn đoán gồm ngưỡng xét nghiệm, triệu chứng và yếu tố nguy cơ. Đây là phần quyết định kết luận của hệ, nên sai một ngưỡng là sai kết luận. LLM trích xuất không đảm bảo đúng con số, và tại thời điểm triển khai Ollama cũng chưa được cài.

Khi kiểm tra nguồn còn phát hiện thêm: file `qd5481_dtd_tip2_2020.pdf` trong source cũ thực chất là **QĐ 3280/QĐ-BYT ngày 09/09/2011**. Văn bản này dùng tiêu chuẩn WHO-1999: không có tiêu chí HbA1c, và ngưỡng IFG là 6,1–6,9 mmol/L. Các ngưỡng nhóm viết từ trí nhớ trong bản thiết kế đầu (IFG 5,6, HbA1c tiền ĐTĐ 5,7–6,4) **không có nguồn** trong dữ liệu hiện có.

## Quyết định

1. KB chẩn đoán lõi được khai báo tay trong `diabetes_mas/kb/seed/*.csv` (bệnh, triệu chứng, yếu tố nguy cơ, xét nghiệm, tiêu chuẩn, thuốc, quan hệ).
2. Mỗi quan hệ và tiêu chuẩn **bắt buộc** có `source_id` và `evidence`, trong đó `evidence` là câu trích nguyên văn. Script `python -m diabetes_mas.kb.seed` kiểm tra câu trích có xuất hiện trong văn bản đã làm sạch **của đúng nguồn đó**. `build_cypher` từ chối build nếu kiểm tra không qua.
3. `HAS_SYMPTOM.prob` và `HAS_RISK_FACTOR.prob` không có nguồn định lượng, nên ghi `prob_source = team_estimate`. Nguồn chỉ chứng minh quan hệ có tồn tại, không chứng minh con số.
4. Hetionet chỉ bổ sung Gene (DaG), Anatomy (DlA) và Drug (CtD) cho ĐTĐ típ 1 và típ 2. Bỏ DpS để tránh trùng triệu chứng tiếng Anh.
5. Quan hệ `MENTIONS` (Chunk → thực thể) sinh bằng so khớp từ điển (tên + bí danh, có/không dấu), không dùng LLM.
6. LLM trích xuất **hoãn lại** và chỉ dùng để *đề xuất* dòng seed mới cho người duyệt, không ghi thẳng vào KB.

## Hệ quả

- Được: mọi ngưỡng trong KB truy ngược được tới câu gốc; build tất định; không phụ thuộc LLM ở Giai đoạn 0.
- Mất: độ phủ KB phụ thuộc công sức tuyển chọn; mở rộng chậm hơn.
- Còn mở: cần văn bản Bộ Y tế mới (QĐ 5481/2020 típ 2, hướng dẫn tiền ĐTĐ, hướng dẫn típ 1) để thay tiêu chuẩn WHO-1999 và bổ sung HbA1c cho tiền ĐTĐ. Hiện tiêu chí HbA1c của ĐTĐ lấy từ Vinmec (VMC-0002, ghi "> 6,5%").

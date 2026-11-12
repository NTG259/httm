# Báo cáo đánh giá (FR-21) — 2026-11-12 17:21

## 1. Deep Learning (AG-13, test)

| Mô hình | PR-AUC | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| D3 MLP đủ đặc trưng | 0,885 | 0,977 | 0,911 | 0,476 | 0,625 |
| D3-mask che xét nghiệm | 0,336 | 0,839 | 0,922 | 0,172 | 0,291 |
| D1 luật ngưỡng | — | — | 0,694 | 1,000 | 0,820 |

## 2. Truy xuất (88 câu; min_score 0.75, min_score_no_entity 0.825)

| Retriever | R@1 | R@3 | R@5 | Fact hit | Correct rej. | False rej. |
|---|---|---|---|---|---|---|
| vector | 0,849 | 0,970 | 1,000 | — | 0,364 | 0,000 |
| graph | 0,394 | 0,470 | 0,470 | 0,905 | 0,909 | 0,106 |
| hybrid_gate | 0,849 | 0,970 | 1,000 | 0,905 | 0,909 | 0,106 |
| hybrid | 0,849 | 0,970 | 1,000 | 0,905 | 0,909 | 0,030 |

## 3. Độ đúng câu trả lời

- Tư vấn (AG-07, LLM = True): trích dẫn hợp lệ 1,000; phương thức {'extractive': 9, 'llm': 57, 'refusal': 22}; Correct rej. 0,909; False rej. 0,030.
- Faithfulness — **chấm tay MẪU 20 câu**: nội dung có căn cứ 0,950; trích dẫn đúng nguồn 0,650; mức {'minor': 8, 'ok': 11, 'unfaithful': 1}.

| Classifier (test) | che slot + luật | đủ slot + luật |
|---|---|---|
| kb_naive_bayes | 0,716 / F1 0,726 | 0,831 / F1 0,842 |
| mlp | 0,764 / F1 0,771 | 0,853 / F1 0,862 |

- Classifier AG-02/03 (test): masked acc 0,738 / macro-F1 0,744; masked_rules acc 0,764 / macro-F1 0,771; full acc 0,831 / macro-F1 0,842; full_rules acc 0,853 / macro-F1 0,862.

## 4. Điều phối (hội thoại chẩn đoán)

Cùng cấu hình (`diag_final.json`, môi trường huấn luyện, 225 ca, 1 seed):

| Chính sách | Val acc | Val lượt | Test acc | Test lượt |
|---|---|---|---|---|
| HRL | 0,862 | 3,16 | 0,782 | 3,32 |
| B0_random | 0,756 | 11,39 | 0,698 | 11,54 |
| B0p_rule | 0,884 | 10,46 | 0,853 | 10,70 |
| B1_flat | 0,853 | 3,56 | 0,751 | 3,54 |

Số ghi lúc huấn luyện (luật điều phối tại thời điểm train) và qua toàn hệ thống:

| Cấu hình | Success / acc | Số lượt TB | Ghi chú |
|---|---|---|---|
| HRL lúc train (môi trường, test) | 0,787 | 3,66 | 20261112-142402_hrl |
| B0' luật lúc train (môi trường, test) | 0,853 | 12,00 | baseline |
| E2E câu mẫu (225 ca) | 0,796 | 3,48 | vượt T_max 0, hỏi lặp 0 |
| E2E Qwen (xem ts: có thể đo trước sửa nlu.DURATION) (20 ca) | 0,750 | 3,70 | vượt T_max 0, hỏi lặp 0 |

## 5. Thời gian phản hồi

- Truy hồi T-04 p95: 0,38 s; trả lời AG-07 p95 (không web): 11,56 s; có web: — s.
- p95 theo tool (ms): T-01 3, T-02 4, T-03 0, T-04 10077, T-05 0, T-07 0, T-08 55, T-09 0, T-15 6.

## 6. Độ tin cậy (log T-10)

- 1890 tool call / 125 phiên; tỉ lệ lỗi 0,000; loại lỗi: không có.
- Tool có lỗi: không có.

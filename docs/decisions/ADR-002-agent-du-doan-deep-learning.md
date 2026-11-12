# ADR-002: Thêm AG-13 Prediction (MLP Deep Learning) làm tín hiệu tham khảo song song

- Ngày: 2026-11-11 · Trạng thái: Chấp nhận
- Liên quan: `01_REQUIREMENTS.md` FR-17…FR-19, §7.2; `02_AGENT_DESIGN.md`; `03_TOOL_SPEC.md` T-15

## Bối cảnh

Đề vấn đáp (`examination_50_questions_v2.pdf`) hỏi trực tiếp về mô hình Deep Learning dự đoán tiểu đường:
- Q39: dataset, đặc trưng, kiến trúc.
- Q42: demo một mẫu đi qua mô hình.
- Q45: agent nào gọi mô hình, kết quả được truyền cho agent khác thế nào.

Thiết kế ban đầu chưa có thành phần này.

Dataset sẵn có (Kaggle, 100.000 dòng) có ba vấn đề:
1. Nhãn gần như quyết định bởi ngưỡng HbA1c và glucose.
2. Dải giữa phi lâm sàng (HbA1c 5,7–6,6 có tỉ lệ dương ≈ 8 % bất kể giá trị).
3. `blood_glucose_level` không ghi thời điểm đo.

Đặc trưng của dataset (giới tính, hút thuốc, bệnh tim, tuổi và BMI dạng số) cũng không khớp với slot hiện có.

## Quyết định

1. Thêm **AG-13 Prediction**. Agent này do **AG-01 Main Agent** gọi qua **T-15**, kết quả ghi vào `DialogState.risk_dl`, AG-07 dùng khi giải thích.
2. **v1: không** đưa `risk_dl` vào vector trạng thái của HRL. Lý do: Patient Simulator chưa sinh đủ 8 đặc trưng; nếu đưa vào sẽ lệch giữa lúc huấn luyện và lúc chạy. Kết quả DL được **đối chiếu với luật T-03**: khi mâu thuẫn thì gắn cờ, và **luật KB được ưu tiên**.
3. Thêm slot `tuoi`, `bmi` (số), `gioi_tinh`, `hut_thuoc`, `benh_tim` (FR-19). Slot nhị phân `tuoi_tu_45`, `tuoi_duoi_30`, `bmi_tu_23` được suy ra từ slot số. Việc này **thay đổi seed KB và `simulator/cases.py`**, nên được lên lịch thành mở rộng Giai đoạn 0, không sửa ngầm.
4. Giả định ánh xạ: `blood_glucose_level` (mg/dL) ↔ `GLUCOSE_BAT_KY` × 18. Có thể sai nếu dataset đo lúc đói.
5. Đánh giá bắt buộc so với baseline luật ngưỡng (D1) và biến thể che xét nghiệm (D3-mask). Chỉ số chính là recall, F1 và PR-AUC lớp dương.

## Bổ sung khi chốt thiết kế ở Giai đoạn 2 (2026-11-11)

- **Bảo đảm AG-13 có đầu vào.** Nếu để policy học tự hỏi `tuoi` và `bmi` thì không có gì bảo đảm, vì AG-13 tắt khi huấn luyện nên policy không có động cơ thu thập hai slot này. Quyết định: thêm bước **intake cố định** `[tuoi, bmi, gioi_tinh]` do điều phối AG-01 thực hiện trước khi policy chạy, ở **cả** huấn luyện lẫn demo. Bước này không tính vào `T_max` và không có reward. Cấu hình ở `configs/slots.yaml → dialog.intake_slots`. Hệ quả: mỗi phiên demo AG-13 chạy ít nhất một lần.
- **Không thưởng câu hỏi hồ sơ.** `r_hit` chỉ áp dụng cho slot y khoa (`rl/reward.py::is_hit`).

## Hệ quả

- Được: trả lời trọn vẹn Q39–Q45 bằng thành phần chạy thật; phần epoch/layer có minh họa rõ ràng (R-T14, R-T15).
- Mất: thêm một mô hình cần duy trì; dữ liệu huấn luyện có chất lượng lâm sàng thấp, nên kết quả DL chỉ để tham khảo.
- v2 (tùy chọn): mở rộng simulator sinh đủ đặc trưng rồi đưa `risk_dl` vào HRL, so sánh có và không có `risk_dl`.

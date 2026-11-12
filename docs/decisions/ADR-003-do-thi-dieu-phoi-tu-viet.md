# ADR-003: Thay LangGraph bằng đồ thị điều phối tự viết (`agents/graph.py`)

- Ngày: 2026-11-12 · Trạng thái: Chấp nhận
- Liên quan: `TAI_LIEU_PHAT_TRIEN.md` D6; `04_SEQUENCE.md` K1, Luồng A, B, D; `09_ARCHITECTURE.md` lớp điều phối

## Bối cảnh

`TAI_LIEU_PHAT_TRIEN.md` D6 chọn **LangGraph** để điều phối agent. Lý do: state tường minh và đồ thị phân cấp.

Ở Giai đoạn 5, môi trường dự án chạy **Python 3.15**. Lệnh `pip install langgraph` (thử lại ngày 2026-11-12 bằng `pip install --dry-run langgraph`) thất bại với lỗi:

```
ERROR: Failed to build 'zstandard' when getting requirements to build wheel
```

`zstandard` là phụ thuộc gián tiếp của LangGraph. Gói này chưa có wheel cho 3.15, còn build từ mã nguồn thì cần trình biên dịch C.

Có ba hướng xử lý:

| Phương án | Ưu điểm | Nhược điểm |
|---|---|---|
| A. Hạ xuống Python 3.12 | Dùng đúng LangGraph | Phải tạo lại venv. PyTorch, sentence-transformers và các checkpoint đã huấn luyện đều phải cài và kiểm lại. Rủi ro cao giữa giai đoạn |
| B. Cài MSVC Build Tools để build `zstandard` | Giữ được 3.15 | Cài vài GB. Máy khác clone về cũng phải làm lại. Không bảo đảm các phụ thuộc khác build được |
| C. Tự viết đồ thị tối giản | Không thêm phụ thuộc. Đọc hết được trong khoảng 40 dòng | Không có checkpointer, streaming và trực quan hóa có sẵn của LangGraph |

## Quyết định

Chọn **C**. File `agents/graph.py` định nghĩa lớp `Graph`, bám theo API của LangGraph:

- `add_node(name, fn)`: mỗi node là một hàm nhận `TurnCtx`.
- `add_edge(src, dst)`: `dst` là tên node, hoặc là hàm chọn node, tương đương `add_conditional_edges`.
- `run(ctx, start, max_steps=60)`: chạy cho tới `END`. Nếu vượt `max_steps` thì ném `RuntimeError` để chặn vòng lặp vô hạn.

Các node: `nlu → router → {emergency | out_of_scope | consult | tracker}`, `tracker → {follow_up | emergency | main}`, `main → ask | conclude`.

Ánh xạ khái niệm:

| LangGraph | Ở dự án này |
|---|---|
| `StateGraph(State)` | `Graph` + `TurnCtx`. `DialogState` (kho phiên, K1) và `Conversation` (slot đang chờ, chế độ, chữ ký DL) |
| `interrupt` / chờ người dùng | Node trả về `END` sau khi đặt `conv.awaiting_slot`. Tin nhắn sau lại chạy từ `nlu` |
| Checkpointer | Kho phiên trong tiến trình (`tools/session.py`, `graph._CONVS`) |
| Trace / LangSmith | `base.TRACE`: vết tool call của từng lượt, trả về trong API (`tool_calls`), cùng `ctx.path` |

## Hệ quả

- Được:
  - Chạy được ngay trên môi trường hiện có.
  - Thứ tự node tất định. Test được bằng `use_llm=False` (`tests/unit/test_agents_llm_graph_api.py`).
  - Mỗi lượt trả về `path` cho thấy đường đi trên đồ thị, dùng được khi demo.
- Mất:
  - Không lưu phiên bền vững: khởi động lại server là mất phiên.
  - Không có streaming từng token.
  - Không có sơ đồ tự sinh. Sơ đồ nằm ở `04_SEQUENCE.md` và được vẽ bằng `devtools/render_diagrams.py`.
- Chấp nhận được vì đây là bản demo một tiến trình. Theo phạm vi đề, phiên không cần sống qua lần khởi động lại.
- Nếu sau này chuyển sang Python ≤ 3.13: thay `Graph` bằng `StateGraph`. Các hàm node (`n_*`) giữ nguyên chữ ký `(ctx) → None`, chỉ cần đổi sang trả về dict cập nhật state.
- Mọi tài liệu ghi "LangGraph" cần sửa thành "đồ thị điều phối (ADR-003)".

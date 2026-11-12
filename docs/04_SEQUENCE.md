# 04 — Luồng giao tiếp (Sequence diagram)

> **Mục đích:** mô tả agent nào gọi agent nào, gọi tool nào, qua kênh nào, theo thứ tự thời gian.
> **Giai đoạn:** 4 · **Input:** `02_AGENT_DESIGN.md` + `03_TOOL_SPEC.md` · **Output:** sequence diagram (Mermaid + ảnh `docs/assets/seq_*.png`)
> **Cập nhật:** 2026-11-12 (v3: đối chiếu với code agent của Giai đoạn 5 — NLU chạy trước Router, Ollama tắt thì trả lời trích xuất)

> **Trạng thái triển khai:**
> - ✅ Các **tool** trong sơ đồ đã có code và test (`03_TOOL_SPEC.md` §2), gồm cả nhánh lỗi hạ tầng (Luồng D).
> - ✅ Các **agent**, đồ thị điều phối (`agents/graph.py`, tự viết thay LangGraph — ADR-003), client Ollama (`agents/llm.py`), FastAPI (`api/app.py`) và kênh UI của T-07 đã có code và test (`tests/unit/test_agents_llm_graph_api.py`). Mỗi lượt API trả `path` (các node đã đi qua) và `tool_calls` để đối chiếu với sơ đồ.
> Khi vấn đáp phải nói rõ phần nào "đã chạy", phần nào "dự kiến".

---

## 1. Kênh giao tiếp

| Kênh | Giữa | Cơ chế |
|---|---|---|
| K1 — Graph state | Agent ↔ agent trong một phiên | `DialogState` chia sẻ trong đồ thị điều phối (`agents/graph.py`, ADR-003). Chuyển quyền điều khiển giữa agent = chuyển node trên đồ thị, dữ liệu đi qua state (đọc bằng T-06, ghi bằng T-05) |
| K2 — Tool call | Agent → tool / model / simulator | `tools.call(...)` (`tools/base.py`): kiểm quyền, kiểm schema, timeout, log T-10 tự động |
| K3 — LLM call | AG-00, 06, 07, 08, 09, 10, 12 → Ollama | HTTP `POST /api/chat` tới `OLLAMA_URL` |
| K4 — HTTP API | UI ↔ hệ thống | FastAPI `/v1/...` |
| K5 — Bolt | Tool → Neo4j | `neo4j` driver, `bolt://`, `connection_timeout = 3 s` |
| K6 — Web search | T-12/T-13 → SearXNG → web whitelist | HTTP; chỉ ở chế độ demo, tắt hoàn toàn khi huấn luyện |

**Quy ước đọc sơ đồ:**
- Mỗi mũi tên ghi **tool `T-xx`** (agent gửi phải được cấp tool đó trong `configs/agents.yaml`) hoặc **kênh `Kx`**.
- `X->>X: T-xx` là agent tự gọi tool (không có đối tác là agent khác).
- T-10 không vẽ: mọi tool call đều được log tự động.
- `%% mode:` trong khối Mermaid ghi chế độ chạy; mọi agent trong sơ đồ phải được phép chạy ở chế độ đó.
- Hai quy tắc trên được kiểm bằng máy: `python -m diabetes_mas.devtools.check_docs` (kiểm tra [9]).

## 2. Luồng A — Chẩn đoán qua hội thoại (chế độ demo)

```mermaid
sequenceDiagram
  %% mode: demo
  autonumber
  actor U as Người dùng
  participant API as FastAPI
  participant R as AG-00 Router
  participant N as AG-06 NLU
  participant ST as AG-05 State Tracker
  participant CL as AG-02 Classifier
  participant M as AG-01 Main Agent
  participant W as AG-Wh Work-Agent
  participant G as AG-09 NLG
  participant SF as AG-08 Safety
  participant PR as AG-13 Prediction DL
  participant C as AG-07 Consult
  participant KB as Neo4j

  U->>API: "Tôi khát nước nhiều, tiểu đêm 3-4 lần" (K4)
  API->>N: câu người dùng, luật trước, P-02 bổ sung (K3)
  N->>KB: T-02 kg_link_entities (K5)
  KB-->>N: khat_nuoc_nhieu, tieu_nhieu
  N->>R: câu + slot đã trích (K1)
  Note over R: luật trước: slot cảnh báo ⇒ emergency, tự kể + có slot ⇒ diagnose. Không khớp mới gọi P-01 (K3)
  R-->>ST: diagnose, slot đã trích (K1)
  ST->>ST: T-05 update_state
  ST->>ST: T-09 check_red_flags
  ST->>CL: T-08 classify (K2)
  Note over CL: luật chẩn đoán áp dụng bên trong T-08 (kb.criteria), không phải tool call T-03 riêng
  CL-->>ST: p_t, H(p_t) (K2)
  Note over M,U: Bước 0 — intake hồ sơ (điều phối cố định, không tính T_max): tuoi, bmi, gioi_tinh
  M->>G: slot hồ sơ còn thiếu (K1)
  G->>G: T-01 kg_get_group_slots, lấy câu hỏi mẫu
  G->>SF: câu hỏi nháp (K1)
  SF->>SF: T-09 check_red_flags trước câu ra
  SF-->>API: câu hỏi đã kiểm (K4)
  API-->>U: "Anh/chị bao nhiêu tuổi, cao và nặng bao nhiêu?"
  U->>API: "52 tuổi, 1m65, 78 kg, nam" (K4)
  API->>N: câu người dùng (K3)
  N-->>ST: tuoi = 52, chieu_cao, can_nang, gioi_tinh = nam (K1)
  ST->>ST: T-05 update_state(count_turn = false), suy ra bmi = 28.7, tuoi_tu_45, bmi_tu_23
  ST->>CL: T-08 classify (K2)
  CL-->>ST: p_t, H(p_t) (K2)
  M->>PR: kích hoạt AG-13 vì đã có tuoi + bmi (K1)
  PR->>PR: T-15 predict_diabetes_risk, biến thể no_labs
  PR->>KB: T-03 kg_check_criteria, đối chiếu luật (K5)
  PR-->>ST: risk_dl, dl_conflict (K1)
  loop tới khi Main chọn CLASSIFY hoặc turn = T_max
    M->>ST: T-06 get_state
    ST-->>M: s_t (K1)
    M-->>W: giao quyền cho W_h, sub_turn = 0 (K1)
    W->>W: T-01 kg_get_group_slots, action mask bỏ slot đã biết
    W->>G: T-07 ask_patient(slot_id), kênh UI
    G->>SF: câu hỏi nháp (K1)
    SF->>SF: T-09 check_red_flags
    SF-->>API: câu hỏi đã kiểm (K4)
    API-->>U: "Anh/chị có biết chỉ số HbA1c gần nhất không?"
    U->>API: "7,1 phần trăm" (K4)
    API->>N: câu người dùng (K3)
    N-->>ST: HBA1C = 7.1 (K1)
    ST->>ST: T-05 update_state(asked_slot = HBA1C)
    ST->>ST: T-09 check_red_flags
    ST->>CL: T-08 classify (K2)
    CL-->>ST: p_t, H(p_t), rule_hits (K2)
    opt đặc trưng DL vừa thay đổi (vd. có HbA1c)
      M->>PR: kích hoạt lại AG-13 (K1)
      PR->>PR: T-15 predict_diabetes_risk, biến thể full
      PR-->>ST: risk_dl, dl_conflict (K1)
    end
    W-->>M: RETURN sau ít nhất 1 câu hỏi, hoặc hết sub_turn (K1)
  end
  M-->>CL: CLASSIFY (K1)
  CL-->>C: top1 + probs + rule_hits + risk_dl + dl_conflict (K1)
  C->>KB: T-04 hybrid_retrieve, căn cứ và nguồn (K5)
  C->>SF: câu kết luận nháp (K1)
  SF-->>API: + disclaimer (K4)
  API-->>U: Kết luận + căn cứ + khuyến nghị
```

**Nhánh cấp cứu:** bất kỳ lúc nào T-09 trả `red_flag = true` (do AG-05 sau mỗi lần cập nhật, hoặc AG-08 trước câu ra) → bỏ vòng lặp, Safety trả `action_text` cố định (`EMERGENCY_TEXT`, gọi 115), phiên kết thúc.

**Nhánh lỗi:** T-02/T-04/K3 lỗi → Luồng D.

## 3. Luồng B — Tư vấn hỏi đáp

```mermaid
sequenceDiagram
  %% mode: demo
  autonumber
  actor U as Người dùng
  participant API as FastAPI
  participant R as AG-00 Router
  participant C as AG-07 Consult
  participant T as T-04 hybrid_retrieve
  participant KB as Neo4j
  participant LLM as Ollama (Qwen)
  participant SF as AG-08 Safety
  participant RS as AG-12 Research
  participant WEB as SearXNG / web whitelist

  U->>API: "Metformin có tác dụng phụ gì?" (K4)
  Note over API,R: AG-06 NLU chạy trước Router ở mọi lượt (Luồng A), câu này không có slot
  API->>R: câu + slot rỗng, P-01 (K3)
  R-->>C: consult, câu hỏi (K1)
  C->>T: T-04 hybrid_retrieve(question) (K2)
  T->>KB: liên kết thực thể bằng từ điển T-02, dự phòng fulltext (K5)
  T->>KB: facts 1-hop quanh METFORMIN theo quan hệ của ý định (K5)
  T->>KB: vector index + chunk MENTIONS thực thể (K5)
  KB-->>T: facts + chunks + score
  T-->>C: RetrieveOut(enough, enough_reason) (K2)
  opt enough = false và search.enabled
    C->>RS: nhờ tìm bài viết (K1)
    RS->>LLM: P-08 viết lại truy vấn, bỏ thông tin cá nhân (K3)
    RS->>KB: T-12 search_articles(scope = local) (K5)
    alt kho có bài phù hợp
      KB-->>RS: hits (in_kb = true)
    else không có
      RS->>WEB: T-12 search_articles(scope = web, whitelist) (K6)
      WEB-->>RS: hits đã lọc lại bằng whitelist
      RS->>WEB: T-13 fetch_article, tối đa search.max_fetch_per_question bài (K6)
      WEB-->>RS: chunk tạm + điểm (1+cos)/2
      RS->>RS: T-14 queue_article_for_kb nếu điểm ≥ min_score
    end
    RS-->>C: chunks bổ sung, gắn nhãn nguồn web (K1)
    C->>C: tính lại enough
  end
  alt enough = false
    C-->>SF: câu từ chối mẫu (K1)
  else enough = true
    C->>LLM: prompt P-05 + context có đánh số nguồn (K3)
    LLM-->>C: câu trả lời có [1], [2]
    C->>C: kiểm tra mọi [n] có trong context, sai thì gọi lại 1 lần
    Note over C: vẫn sai hoặc Ollama lỗi ⇒ câu trả lời TRÍCH XUẤT (câu đầu của từng nguồn kèm [n]), không sinh nội dung mới
    C-->>SF: câu trả lời (K1)
  end
  SF->>SF: T-09 check_red_flags
  SF-->>API: + disclaimer (K4)
  API-->>U: Trả lời + danh sách nguồn
```

- Giới hạn web: `search.max_web_calls_per_session` (3) và `search.max_fetch_per_question` (2) trong `configs/settings.yaml`; AG-12 tự đếm (`agents/research.py`).
- **Từ chối ≠ lỗi:** `enough = false` là KB không có căn cứ → câu từ chối mẫu. Neo4j/Ollama sập là lỗi hạ tầng → Luồng D.

## 4. Luồng C — Huấn luyện (một episode)

```mermaid
sequenceDiagram
  %% mode: train
  autonumber
  participant TR as Trainer
  participant P as AG-10 Patient Simulator
  participant ST as AG-05 State Tracker
  participant M as AG-01 Main Agent
  participant W as AG-Wh Work-Agent
  participant CL as AG-02 Classifier
  participant RG as AG-11 Reward

  TR->>P: lấy ca bệnh ngẫu nhiên từ train.jsonl, session.attach_case
  P-->>ST: slot explicit (K1)
  ST->>ST: T-05 update_state(count_turn = false)
  ST->>CL: T-08 classify (K2)
  CL-->>ST: p_0, H(p_0) (K2)
  loop mỗi lượt t
    M->>ST: T-06 get_state
    ST-->>M: s_t (K1)
    M-->>W: chọn W_h ε-greedy (K1)
    W->>W: T-01 kg_get_group_slots + action mask
    W->>P: T-07 ask_patient(slot_id), kênh simulator ✅
    P-->>ST: yes / no / value / unknown (K1)
    ST->>ST: T-05 update_state(asked_slot)
    ST->>CL: T-08 classify (K2)
    CL-->>ST: p_t, H(p_t) (K2)
    ST-->>RG: asked, hit, H(p_t-1), H(p_t) (K1)
    RG->>RG: T-11 compute_reward
    RG-->>TR: r_t
    TR->>TR: lưu (s, a, r, s') vào replay buffer của M và W_h
  end
  M-->>CL: CLASSIFY (K1)
  CL-->>RG: top1 so với bệnh thật (K1)
  RG->>RG: T-11 compute_reward(terminal = true)
  RG-->>TR: R_success / R_fail
  TR->>TR: cập nhật DQN (batch từ replay), classifier (cross-entropy)
```

- Chế độ huấn luyện **không** gọi LLM (bỏ AG-00/06/09/08, không K3) và **không** gọi AG-12/AG-13 (chỉ chạy ở demo) → mỗi episode chỉ tốn vài ms.
- AG-05 vẫn gọi T-09 sau mỗi lần cập nhật (`02_AGENT_DESIGN.md` §2), nhưng **ở chế độ train cờ cấp cứu không kết thúc episode** (§4: "cờ cấp cứu (chế độ demo)"), nên sơ đồ lược bớt mũi tên T-09.

## 5. Luồng D — Lỗi hạ tầng (Neo4j / Ollama / SearXNG không phản hồi)

Cơ chế chung (đã cài đặt, `tools/errors.py` + `tools/base.py`):
1. Tool ném exception của driver → `tools.call` đổi thành mã ổn định trong `Out.error`, **không** ném ra agent.
2. Agent rẽ nhánh bằng `is_unavailable(out.error)`, không dựa vào tên class của thư viện.
3. Mất **dữ liệu** (Neo4j, T-04) ⇒ câu trả lời cố định `UNAVAILABLE_TEXT`: *"Hệ thống tạm thời không truy cập được dữ liệu y khoa nên chưa thể trả lời chính xác…"*. **Không đoán**, không gọi LLM để "tự trả lời".
4. Mất **LLM** (Ollama) ⇒ dữ liệu vẫn còn nên mỗi agent rơi về phần luật / mẫu của mình; không agent nào sinh nội dung mới khi thiếu LLM.

| Sự cố | Mã lỗi | Hành vi | Trạng thái |
|---|---|---|---|
| Neo4j tắt, T-04 | `kb_unavailable` | AG-07 trả `UNAVAILABLE_TEXT` | ✅ test unit + đã thử với Neo4j tắt thật |
| Neo4j tắt, T-03 | — (`ok = true`, `kb_backend = "seed"`) | Dùng seed — **không phải đoán**: test tích hợp xác nhận seed = dữ liệu trong Neo4j | ✅ |
| Neo4j tắt, T-02 | — | Từ điển vẫn chạy, chỉ bỏ dự phòng fulltext | ✅ |
| Ollama tắt (K3) | `llm_unavailable` | Router: heuristic `router.fallback`; NLU: chỉ slot do luật trích; NLG: câu hỏi mẫu từ KB; Consult: câu trả lời **trích xuất** từ facts/chunk kèm [n] (không phải `UNAVAILABLE_TEXT`, vì KB vẫn có căn cứ); P-06: giải thích theo mẫu; Safety: chỉ luật cứng | ✅ `tests/unit/test_agents_llm_graph_api.py` (`test_nlu_llm_down_keeps_rule_slots`, `test_consult_llm_down_is_extractive_not_unavailable`, `test_router_uses_llm_then_falls_back`) |
| SearXNG / web tắt (T-12, T-13) | `net_unavailable`, `timeout` | AG-12 trả 0 chunk → AG-07 giữ `enough = false` → **câu từ chối mẫu** (KB vẫn hoạt động, chỉ thiếu nguồn bổ sung) | ✅ mã lỗi + `agents/research.py`; SearXNG thật chạy được (`tests/integration/test_web_searxng.py`) |
| Model DL chưa có (T-15) | `model_not_trained` | AG-13 bỏ qua `risk_dl`, chẩn đoán vẫn chạy (DL chỉ là tín hiệu tham khảo, D12) | ✅ |
| Tool quá thời gian | `timeout` | Như dòng tương ứng của tool đó | ✅ |

```mermaid
sequenceDiagram
  %% mode: demo
  autonumber
  actor U as Người dùng
  participant API as FastAPI
  participant C as AG-07 Consult
  participant RS as AG-12 Research
  participant SF as AG-08 Safety
  participant KB as Neo4j
  participant LLM as Ollama (Qwen)

  U->>API: câu hỏi (K4)
  API->>C: câu hỏi (K1)
  C->>KB: T-04 hybrid_retrieve (K5)
  alt Neo4j không phản hồi sau 3 s
    KB--xC: ServiceUnavailable
    Note over C: tools.call trả ok = false, error = kb_unavailable
    C-->>SF: UNAVAILABLE_TEXT (K1)
  else Neo4j ổn, enough = true
    C->>LLM: prompt P-05 (K3)
    alt Ollama không phản hồi
      LLM--xC: LLMUnavailable
      C-->>SF: câu trả lời trích xuất từ facts/chunk kèm [n] (K1)
    else Ollama ổn
      LLM-->>C: câu trả lời có trích nguồn
      C-->>SF: câu trả lời (K1)
    end
  else Neo4j ổn, enough = false, web lỗi
    C->>RS: nhờ tìm bài viết (K1)
    RS-->>C: 0 chunk, net_unavailable (K1)
    C-->>SF: câu từ chối mẫu (K1)
  end
  SF-->>API: + disclaimer (K4)
  API-->>U: trả lời / thông báo lỗi, không đoán
```

## 6. Thay đổi so với bản thiết kế đầu (do đối chiếu với code GĐ 3 và GĐ 5)

| # | Bản đầu | Sửa thành | Lý do |
|---|---|---|---|
| 1 | `N-->>ST: T-05 update_state` | NLU chuyển slot qua K1, **AG-05 tự gọi T-05** | AG-06 không được cấp T-05 (`check_docs` [9] phát hiện) |
| 2 | `M->>PR: … → T-15` | Main kích hoạt AG-13 qua K1, **AG-13 tự gọi T-15** | AG-01 không được cấp T-15 (thiết kế D12) |
| 3 | `ST->>SF: T-09` | AG-05 tự gọi T-09 sau mỗi cập nhật; AG-08 gọi T-09 trước câu ra | Cả hai đều được cấp T-09; Safety không trả kết quả cho State Tracker |
| 4 | `CL->>KB: T-03` trong vòng phân loại | Luật áp dụng **bên trong** T-08 (`kb.criteria.check_with_meta`) | Code T-08 không gọi tool T-03 |
| 5 | Không có T-07, T-01 | Work-Agent gọi T-01 (action space) + T-07 (hỏi) | Đúng quyền trong `agents.yaml` |
| 6 | Luồng C không ghi tool | Ghi T-05, T-06, T-07, T-08, T-11 | Tiêu chí hoàn thành mục 1 |
| 7 | T-04 "link entity (fulltext)" | Từ điển T-02, fulltext chỉ là dự phòng | `03_TOOL_SPEC.md` §3 T-02 |
| 8 | "T-13 tối đa 2 bài" chỉ ghi trong sơ đồ | `search.max_fetch_per_question: 2` trong `settings.yaml` | Một nguồn cấu hình duy nhất |
| 9 | Không có nhánh lỗi | Luồng D + mã lỗi chuẩn hóa | Tiêu chí hoàn thành mục 2 |
| 10 | Router chạy trước NLU | **NLU chạy trước Router** (node `nlu → router`) | Luật của Router cần slot: slot cảnh báo ⇒ emergency, tự kể + có slot ⇒ diagnose mà không cần gọi LLM. Đang trả lời câu hỏi chẩn đoán thì câu ngắn được giữ ở `diagnose` |
| 11 | Ollama tắt ⇒ Consult trả `UNAVAILABLE_TEXT` | Consult trả lời **trích xuất** (câu đầu của từng nguồn kèm [n]) | KB vẫn có căn cứ đã kiểm duyệt; trích nguyên văn không phải "đoán". `UNAVAILABLE_TEXT` chỉ dành cho mất dữ liệu (Neo4j) |
| 12 | Đồ thị LangGraph | Đồ thị tự viết `agents/graph.py` | LangGraph không cài được trên Python 3.15 (ADR-003) |

## 7. Tiêu chí hoàn thành

- [x] Mọi mũi tên giữa agent ứng với một tool trong `03_TOOL_SPEC.md` hoặc một kênh ở mục 1 — kiểm bằng máy: `check_docs` [9] (`tests/unit/test_check_docs.py`).
- [x] Có nhánh xử lý lỗi: Neo4j/Ollama không phản hồi → trả thông báo "hệ thống tạm thời không truy cập được dữ liệu", không đoán — Luồng D, `tools/errors.py`, `tests/unit/test_tools_base.py::test_infra_failure_maps_to_stable_code`.
- [x] Ảnh xuất của 4 sơ đồ lưu ở `docs/assets/seq_*.png` — sinh bằng `python -m diabetes_mas.devtools.render_diagrams`.

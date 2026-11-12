"""Unit test Giai đoạn 5: NLU, Router, Safety, Consult (LLM giả), đồ thị điều phối (use_llm=False), API.

Không cần Ollama hay Neo4j: `llm.LLM` thay bằng `FakeLLM`, `retrieve_tools.BACKEND/EMBED` thay bằng backend giả.
"""
import json

import pytest

from diabetes_mas.agents import consult, graph, llm, nlu, router, safety
from diabetes_mas.agents.state import SlotValue
from diabetes_mas.tools import base, call, kg_tools, research_tools, retrieve_tools, session
from diabetes_mas.tools.errors import UNAVAILABLE_TEXT, LLMUnavailable


class FakeLLM:
    """Trả lần lượt các câu trong `replies`; phần tử là Exception thì ném ra. Ghi lại mọi prompt."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []
        self.model = "fake"

    def chat(self, user, system=None, temperature=0.0, json_mode=False):
        self.prompts.append(user)
        if not self.replies:
            raise AssertionError("FakeLLM: bị gọi nhiều hơn dự kiến")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)

    def healthy(self):
        return False


class FakeBackend:
    def __init__(self, cos=0.82, fail=False):
        self.cos, self.fail = cos, fail

    def facts(self, ids, rels, limit):
        if self.fail:
            raise ConnectionError("neo4j offline")
        if "METFORMIN" in ids:
            return [{"s_id": "METFORMIN", "s": "Metformin", "rel": "TREATS", "o_id": "DTD_TIP_2",
                     "o": "Đái tháo đường típ 2", "source_id": "VMC-0002", "evidence": "metformin"}]
        return []

    def vector(self, q_emb, ids, k):
        return [{"chunk_id": f"VMC-01{i}#000", "text": f"Đoạn văn số {i}. Câu thứ hai. Câu thứ ba.",
                 "score": self.cos - i * 0.01, "source_id": f"VMC-01{i}", "title": f"Bài {i}", "url": None,
                 "reviewed": True, "n_mention": 1} for i in range(3)]

    def guided(self, q_emb, ids, k):
        return []


NO_LLM = FakeLLM()                                      # gọi tới là lỗi ⇒ chứng minh luật chạy trước LLM


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(research_tools, "REVIEW_QUEUE", tmp_path / "review_queue.jsonl")
    monkeypatch.setattr(research_tools, "SEARCH_CACHE", tmp_path / "search")
    monkeypatch.setattr(research_tools, "WEB_CACHE", tmp_path / "web")
    monkeypatch.setattr(kg_tools, "_fulltext_fallback", lambda t, k: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(kg_tools, "_criteria_from_neo4j", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(retrieve_tools, "BACKEND", FakeBackend())
    monkeypatch.setattr(retrieve_tools, "EMBED", lambda t: [0.0] * 1024)
    monkeypatch.setattr(llm, "LLM", FakeLLM())
    kg_tools._CRIT_CACHE.clear()
    graph._CONVS.clear()


def use_llm(monkeypatch, *replies) -> FakeLLM:
    fake = FakeLLM(*replies)
    monkeypatch.setattr(llm, "LLM", fake)
    return fake


# ---------------- AG-06 NLU ----------------
def test_nlu_rules_extract_numbers_and_negated_clause():
    r = nlu.understand("Tôi 52 tuổi, cao 1m65 nặng 80kg, hay khát nước nhưng không sụt cân. HbA1c 7,1",
                       use_llm=False)
    s = r.slots
    assert s["tuoi"].value == 52 and s["bmi"].value == pytest.approx(29.4, abs=0.1)
    assert s["HBA1C"].value == pytest.approx(7.1)
    assert s["uong_nhieu"].status == "yes" and s["sut_can"].status == "no"
    assert r.by_llm == [] and r.llm_error is None


def test_nlu_glucose_mg_dl_is_converted_and_time_number_ignored():
    s = nlu.understand("đường huyết sau ăn 2 giờ là 200 mg/dl", use_llm=False).slots
    assert s["OGTT_2H"].value == pytest.approx(11.1)
    s = nlu.understand("đường huyết lúc đói 126 mg/dl", use_llm=False).slots
    assert s["GLUCOSE_DOI"].value == pytest.approx(7.0)


def test_nlu_decimal_comma_is_not_a_clause_boundary():
    assert nlu._clauses("HbA1c tháng trước 7,1%.") == ["hba1c thang truoc 7,1%"]
    assert nlu._clauses("đường huyết 5,3, không khát nước") == ["duong huyet 5,3", "khong khat nuoc"]


# Nguyên văn câu người dùng trong docs/09_DEMO_SCRIPT.md (§2.1 Story 1, §2.3 cấp cứu, §2.4 thai kỳ)
@pytest.mark.parametrize("text,last,expected", [
    ("Dạo này tôi khát nước suốt, đi tiểu nhiều, sụt 3 kg trong tháng.", None,
     {"uong_nhieu": "yes", "tieu_nhieu": "yes", "sut_can": "yes"}),
    ("Tôi 52 tuổi.", "tuoi", {"tuoi": 52}),
    ("Tôi cao 1m65, nặng 78 kg.", "bmi", {"bmi": 28.7}),
    ("Tôi là nam.", "gioi_tinh", {"gioi_tinh": "nam"}),
    ("HbA1c tháng trước 7,1%.", "HBA1C", {"HBA1C": 7.1}),
    ("HbA1c tháng trước 7,1%.", None, {"HBA1C": 7.1}),
    ("Tôi đường huyết 25 mmol/L, đang buồn nôn, thở dốc", None,
     {"GLUCOSE_BAT_KY": 25.0, "buon_non": "yes", "tho_nhanh": "yes"}),
    ("Tôi đang mang thai 28 tuần, đường huyết lúc đói 5,3 mmol/L, có bị tiểu đường thai kỳ không?", None,
     {"dang_mang_thai": "yes", "GLUCOSE_DOI": 5.3}),
    ("đường huyết lúc đói 126 mg/dL", None, {"GLUCOSE_DOI": 7.0}),
    ("HbA1c 6,5%, không khát nước", None, {"HBA1C": 6.5, "uong_nhieu": "no"}),
    ("Không ạ", "it_van_dong", {"it_van_dong": "no"}),
])
def test_nlu_demo_script_sentences(text, last, expected):
    s = nlu.understand(text, last_slot=last, use_llm=False).slots
    for sid, want in expected.items():
        got = s[sid].status if want in ("yes", "no") else s[sid].value
        assert got == (pytest.approx(want) if isinstance(want, float) else want), sid


@pytest.mark.parametrize("text,last,expected", [
    ("Dạ, tôi không ạ", "it_van_dong", SlotValue(status="no")),
    ("có", "it_van_dong", SlotValue(status="yes")),
    ("tôi không biết", "HBA1C", SlotValue(status="unknown")),
    ("6,8", "HBA1C", SlotValue(status="yes", value=6.8)),
])
def test_nlu_short_answer_to_last_question(text, last, expected):
    assert nlu.understand(text, last_slot=last, use_llm=False).slots[last] == expected


def test_nlu_llm_only_adds_valid_slots_and_never_overrides_rules(monkeypatch):
    use_llm(monkeypatch, {"uong_nhieu": {"status": "no"},             # luật đã bắt ⇒ bỏ qua
                          "tang_huyet_ap": {"status": "yes"},         # hợp lệ ⇒ thêm
                          "HBA1C": {"status": "yes", "value": 99},    # ngoài khoảng ⇒ loại
                          "slot_bia": {"status": "yes"},              # không có trong kho ⇒ loại
                          "gioi_tinh": {"status": "yes", "value": "trans"}})  # giá trị ngoài danh mục ⇒ loại
    r = nlu.understand("tôi hay khát nước", use_llm=True)
    assert r.slots["uong_nhieu"].status == "yes"
    assert r.by_llm == ["tang_huyet_ap"]
    assert {"HBA1C", "slot_bia", "gioi_tinh"}.isdisjoint(r.slots)


def test_nlu_llm_down_keeps_rule_slots(monkeypatch):
    use_llm(monkeypatch, LLMUnavailable("off"))
    r = nlu.understand("tôi hay khát nước", use_llm=True)
    assert r.slots["uong_nhieu"].status == "yes" and r.llm_error.startswith("LLMUnavailable")


# ---------------- AG-00 Router ----------------
def test_router_alarm_slot_is_emergency_without_llm(monkeypatch):
    monkeypatch.setattr(llm, "LLM", NO_LLM)
    assert router.classify("tôi thở nhanh", {"tho_nhanh": SlotValue(status="yes")}) == ("emergency", "rules")


def test_router_self_report_with_slots_is_diagnose_without_llm(monkeypatch):
    monkeypatch.setattr(llm, "LLM", NO_LLM)
    assert router.classify("tôi hay khát nước", {"uong_nhieu": SlotValue(status="yes")}) == ("diagnose", "rules")


def test_router_uses_llm_then_falls_back(monkeypatch):
    use_llm(monkeypatch, {"intent": "consult"})
    assert router.classify("Metformin có tác dụng phụ gì?") == ("consult", "llm")
    use_llm(monkeypatch, "không phải json", {"intent": "chat"})      # sai 2 lần ⇒ LLMBadOutput ⇒ fallback
    assert router.classify("Metformin có tác dụng phụ gì?") == ("consult", "fallback")


@pytest.mark.parametrize("text,expected", [
    ("Metformin có tác dụng phụ gì?", "consult"),
    ("Tôi có bị tiểu đường không?", "diagnose"),
    ("Giá vàng hôm nay bao nhiêu?", "out_of_scope"),
])
def test_router_fallback_heuristic(text, expected):
    assert router.fallback(text, {}) == expected


# ---------------- AG-08 Safety ----------------
@pytest.mark.parametrize("draft,issue", [
    ("Anh uống metformin 500 mg mỗi ngày hai lần.", "kê liều thuốc cụ thể"),
    ("Chắc chắn bạn bị đái tháo đường típ 2.", "khẳng định chẩn đoán chắc chắn"),
    ("Anh có thể ngừng uống thuốc khi đường huyết ổn.", "khuyên ngừng thuốc"),
])
def test_safety_rules_replace_unsafe_text(draft, issue):
    out = safety.check(draft)
    assert issue in out.issues
    assert out.text.startswith(safety.SAFE_REPLACEMENT) and draft not in out.text


def test_safety_clean_text_gets_disclaimer_once():
    out = safety.check(f"Nên đi khám.\n\n{safety.DISCLAIMER}")
    assert out.issues == [] and out.text.count(safety.DISCLAIMER) == 1


def test_safety_red_flag_overrides_any_draft():
    sid = "s-red"
    call("T-05", agent_id="AG-05", session_id=sid, updates={"roi_loan_y_thuc": SlotValue(status="yes")},
         count_turn=False)
    out = safety.check("Câu trả lời bình thường.", session_id=sid)
    assert out.emergency and "Câu trả lời bình thường" not in out.text


# ---------------- AG-07 Consult ----------------
Q = "Metformin có tác dụng phụ gì?"


def test_consult_llm_answer_with_valid_citation(monkeypatch):
    fake = use_llm(monkeypatch, "Metformin dùng điều trị đái tháo đường típ 2 [1].")
    a = consult.answer(Q, "s1", allow_research=False)
    assert a.method == "llm" and a.enough and a.text.endswith("[1].")
    assert a.sources[0].kind == "fact" and a.sources[0].source_id == "VMC-0002"
    assert "[1] Metformin được dùng điều trị" in fake.prompts[0]


def test_consult_bad_citation_twice_falls_back_to_extractive(monkeypatch):
    use_llm(monkeypatch, "Bịa [99].", "Không trích dẫn gì.")
    a = consult.answer(Q, "s1", allow_research=False)
    assert a.method == "extractive" and "[1]" in a.text and "[99]" not in a.text


def test_consult_llm_down_is_extractive_not_unavailable(monkeypatch):
    use_llm(monkeypatch, LLMUnavailable("off"))
    a = consult.answer(Q, "s1", allow_research=False)
    assert a.method == "extractive" and a.text != UNAVAILABLE_TEXT


def test_consult_not_enough_is_refusal(monkeypatch):
    monkeypatch.setattr(llm, "LLM", NO_LLM)
    a = consult.answer("Cách chữa ung thư phổi", "s1", allow_research=False)
    assert (a.method, a.text, a.enough) == ("refusal", consult.REFUSAL, False)


def test_consult_kb_down_is_unavailable(monkeypatch):
    monkeypatch.setattr(retrieve_tools, "BACKEND", FakeBackend(fail=True))
    a = consult.answer(Q, "s1", allow_research=False)
    assert a.method == "unavailable" and a.text == UNAVAILABLE_TEXT


def test_citations_ok():
    assert consult.citations_ok("a [1] b [2]", 2)
    assert not consult.citations_ok("a [3]", 2)
    assert not consult.citations_ok("không trích", 2)


# ---------------- Đồ thị điều phối ----------------
def turn(sid, text):
    return graph.handle(sid, text, use_llm=False)


def test_graph_out_of_scope_and_emergency_paths():
    o = turn(graph.new_session(), "Giá vàng hôm nay bao nhiêu?")
    assert o["stage"] == "out_of_scope" and o["path"] == ["nlu", "router", "out_of_scope"]
    o = turn(graph.new_session(), "Tôi bị lơ mơ, thở nhanh, hơi thở có mùi táo")
    assert o["stage"] == "emergency" and o["path"][-1] == "emergency" and o["state"]["done"]


def test_graph_consult_path_cites_sources():
    o = turn(graph.new_session(), Q)
    assert o["intent"] == "consult" and o["stage"] == "answered"
    assert o["notes"]["consult"]["method"] == "extractive" and o["sources"]
    assert safety.DISCLAIMER in o["reply"]
    assert any(c["tool"] == "T-04" for c in o["tool_calls"])


def test_graph_diagnose_asks_intake_then_records_answer():
    sid = graph.new_session()
    o = turn(sid, "Tôi muốn kiểm tra xem mình có bị tiểu đường không")
    assert o["stage"] == "asking" and o["asking_slot"] == "tuoi"
    assert o["path"] == ["nlu", "router", "tracker", "main"]
    o = turn(sid, "52 tuổi")
    assert o["state"]["slots"]["tuoi"]["value"] == 52 and o["asking_slot"] != "tuoi"
    assert o["state"]["turn"] == 0                       # câu hỏi intake không tính lượt (T-05)


def test_graph_numeric_yes_without_number_asks_follow_up():
    sid = graph.new_session()
    turn(sid, "Tôi muốn kiểm tra xem mình có bị tiểu đường không")
    conv = graph.get_conv(sid)
    conv.awaiting_slot, conv.mode = "HBA1C", "diagnose"
    o = turn(sid, "có")
    assert o["path"][-1] == "follow_up" and o["asking_slot"] == "HBA1C"


def test_graph_full_dialog_reaches_conclusion():
    sid = graph.new_session()
    o = turn(sid, "Tôi 50 tuổi, nam, cao 1m70 nặng 85kg, hay khát nước, tiểu nhiều. Tôi có bị tiểu đường không?")
    for _ in range(40):
        if o["stage"] != "asking":
            break
        o = turn(sid, "7,2" if o["asking_slot"] in ("GLUCOSE_DOI", "HBA1C") else "không biết")
    assert o["stage"] == "concluded", o["reply"]
    assert o["outcome"]["top1"] in graph.DISEASES and safety.DISCLAIMER in o["reply"]
    assert o["state"]["turn"] <= o["state"]["T_max"]


def test_graph_new_message_after_done_starts_new_dialog():
    sid = graph.new_session()
    turn(sid, "Tôi bị lơ mơ, thở nhanh")
    o = turn(sid, "Tôi muốn kiểm tra tiểu đường")
    assert o["stage"] == "asking" and not o["state"]["red_flag"]


def test_graph_engine_stops_runaway_loops():
    g = graph.Graph()
    g.add_node("a", lambda c: None)
    g.add_edge("a", "a")
    with pytest.raises(RuntimeError, match="không dừng"):
        g.run(graph.TurnCtx(graph.Conversation("x"), "hi"), "a", max_steps=5)


# ---------------- API ----------------
@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from diabetes_mas.api.app import app
    return TestClient(app)


def test_api_session_message_state(client):
    sid = client.post("/v1/sessions").json()["session_id"]
    r = client.post(f"/v1/sessions/{sid}/messages", json={"text": "Giá vàng hôm nay?", "use_llm": False})
    assert r.status_code == 200 and r.json()["stage"] == "out_of_scope"
    assert client.get(f"/v1/sessions/{sid}/state").status_code == 200
    assert client.get("/v1/sessions/khong-co/state").status_code == 404


def test_api_rejects_empty_message(client):
    sid = client.post("/v1/sessions").json()["session_id"]
    assert client.post(f"/v1/sessions/{sid}/messages", json={"text": ""}).status_code == 422


def test_api_consult(client):
    r = client.post("/v1/consult", json={"question": Q, "use_llm": False}).json()
    assert r["method"] == "extractive" and r["enough"] and r["sources"]
    assert safety.DISCLAIMER in r["answer"]


@pytest.mark.parametrize("raw,expected", [
    ("Ý một (1).", "Ý một [1]."),
    ("Ý hai (1, 2).", "Ý hai [1][2]."),
    ("Ý ba [3, 2].", "Ý ba [3][2]."),
    ("Năm (2024) và (nguồn MOH)", "Năm (2024) và (nguồn MOH)"),      # không phải trích dẫn ⇒ giữ nguyên
])
def test_normalize_citations(raw, expected):
    assert consult.normalize_citations(raw) == expected


def test_consult_accepts_paren_citation_but_still_rejects_out_of_range(monkeypatch):
    use_llm(monkeypatch, "Metformin điều trị típ 2 (1).")
    a = consult.answer(Q, "s1", allow_research=False)
    assert a.method == "llm" and a.text.startswith("Metformin điều trị típ 2 [1].")
    use_llm(monkeypatch, "Bịa (99).", "Bịa (98).")
    assert consult.answer(Q, "s1", allow_research=False).method == "extractive"


# ---------------- AG-12 Research ----------------
class Hit:
    def __init__(self, title, sid):
        self.title, self.snippet, self.source_id, self.url = title, f"{title} nội dung", sid, f"https://x/{sid}"


def test_research_local_hits_need_cosine_above_min_score(monkeypatch):
    from diabetes_mas.agents import research
    hits = [Hit("Bị tiểu đường có nên ăn sầu riêng?", "VMC-1"), Hit("Các món cháo cho người tiểu đường", "VMC-2")]
    vec = {"q": [1.0, 0.0], "VMC-1": [0.8, 0.6], "VMC-2": [0.5, 0.866]}       # cos 0,80 và 0,50
    monkeypatch.setattr(research_tools, "EMBED_TEXTS",
                        lambda texts: [vec["q"]] + [vec["VMC-1" if "sầu riêng" in t else "VMC-2"] for t in texts[1:]])
    errs = []
    good = research._relevant_local("sầu riêng", hits, 0.75, errs)
    assert [c.source_id for c in good] == ["VMC-1"] and good[0].cosine == 0.8 and errs == []


def test_research_local_embed_failure_uses_no_local_hits(monkeypatch):
    from diabetes_mas.agents import research
    monkeypatch.setattr(research_tools, "EMBED_TEXTS", lambda t: (_ for _ in ()).throw(OSError("no model")))
    errs = []
    assert research._relevant_local("q", [Hit("A", "VMC-1")], 0.75, errs) == [] and errs == ["embed local: OSError"]


def test_t13_prefilter_keeps_most_overlapping_pieces_in_order():
    texts = ["cháo yến mạch", "metformin gây tiêu chảy", "lịch sử", "tác dụng phụ của metformin", "khác"]
    assert research_tools.prefilter(texts, "Metformin có tác dụng phụ gì?", 2) == texts[1:2] + texts[3:4]
    assert research_tools.prefilter(texts, "x", 10) == texts


def test_call_forwards_timeout_to_tools_that_take_it(monkeypatch):
    """timeout_s của T-13 là thời hạn HTTP, không được nuốt làm thời hạn cả tool (lỗi phát hiện ở GĐ 6)."""
    seen = {}

    class Client:
        def get(self, url):
            raise research_tools.RobotsDisallowed(url)

    monkeypatch.setattr(research_tools, "HTTP_CLIENT_FACTORY", lambda t: seen.setdefault("t", t) and Client())
    out = call("T-13", agent_id="AG-12", url="https://www.vinmec.com/vie/bai-viet/x", question="q", timeout_s=5)
    assert seen["t"] == 5 and out.error == "robots_disallow"


def test_nlu_never_uses_fulltext_fallback(monkeypatch):
    """GĐ 7: với Neo4j bật, fulltext khớp từ rời ("rồi" → Rối loạn ý thức) ⇒ cấp cứu nhầm."""
    from diabetes_mas.tools.schemas import LinkedEntity
    junk = [LinkedEntity(node_id="roi_loan_y_thuc", label="Symptom", name_vi="Rối loạn ý thức", method="fulltext")]
    monkeypatch.setattr(kg_tools, "_fulltext_fallback", lambda t, k: junk)
    assert call("T-02", agent_id="AG-06", text="toi da bo thuoc roi").entities == junk     # T-02 mặc định vẫn có
    s = nlu.understand("Tôi đã bỏ thuốc rồi.", use_llm=False).slots
    assert "roi_loan_y_thuc" not in s and s["hut_thuoc"].value == "da_bo"


@pytest.mark.parametrize("text,value", [("Hiện tôi không hút thuốc.", "hien_khong_hut"),
                                         ("Tôi chưa bao giờ hút thuốc.", "chua_bao_gio"),
                                         ("Tôi từng hút thuốc.", "tung_hut")])
def test_nlu_smoking_values(text, value):
    assert nlu.understand(text, use_llm=False).slots["hut_thuoc"].value == value


@pytest.mark.parametrize("text,dose", [("≥ 7,0 mmol/L (≥126 mg/dL) hoặc", False), ("(≥200 mg/dL) sau khi uống", False),
                                       ("500 mg/ngày", True), ("tiêm 10 đơn vị insulin buổi tối", True)])
def test_safety_dose_rule_ignores_concentration_units(text, dose):
    assert bool(safety.DOSE.search(text)) is dose


@pytest.mark.parametrize("text,expected", [("Đường huyết đói 140 mg/dL", 7.8),          # TC-05
                                           ])
def test_tc05_glucose_mg_dl(text, expected):
    assert nlu.understand(text, use_llm=False).slots["GLUCOSE_DOI"].value == pytest.approx(expected)


def test_tc06_negated_symptom():
    assert nlu.understand("tôi không bị khát", use_llm=False).slots["uong_nhieu"].status == "no"


def test_nlg_rejects_quantity_question_for_binary_slot():
    from diabetes_mas.agents import nlg
    bad = "Bạn đang mang thai được bao nhiêu tuần không?"
    assert not nlg.ok_question(bad, "binary") and nlg.ok_question(bad, "numeric")
    assert nlg.ok_question("Anh/chị có đang mang thai không?", "binary")


def test_explain_ok_rejects_dropped_negation():
    """GĐ 8: Qwen bỏ chữ "Không" ⇒ đảo ngược kết luận; phải bị chặn."""
    name = consult.DISEASE_VI["KHONG_DTD"]
    assert not consult.explain_ok("Khả năng mắc đái tháo đường là 54% [1].", name, "54%")
    assert consult.explain_ok("Kết luận sơ bộ: không mắc đái tháo đường (54%) [1].", name, "54%")
    assert not consult.explain_ok("Không mắc đái tháo đường (70%) [1].", name, "54%")


def test_explain_wrong_conclusion_falls_back_to_template(monkeypatch):
    use_llm(monkeypatch, "Khả năng mắc đái tháo đường là 54% [1].", "Khả năng mắc đái tháo đường là 54% [1].")
    a = consult.explain("s-exp", "KHONG_DTD", {"KHONG_DTD": 0.54, "TIEN_DTD": 0.39}, [], ["tuoi"])
    assert a.method == "template" and a.text.startswith("Kết luận sơ bộ: Không mắc đái tháo đường (xác suất ước tính 54%)")


def test_explain_template_uses_vietnamese_names_not_slot_ids():
    """GĐ 9 (chạy thử demo): giải thích từng hiện "uong_nhieu, tieu_nhieu" và "HBA1C = 7.1"."""
    from diabetes_mas.tools.schemas import CriterionHit
    assert consult.slot_names(["uong_nhieu", "tuoi", "HBA1C"]) == ["Uống nhiều", "tuổi", "HbA1c"]
    h = CriterionHit(disease_id="DTD", criterion_id="CRIT_DTD_HBA1C", test_id="HBA1C", value=7.1,
                     rule="HBA1C > 6.5 %", source_id="VMC-0002", source_title="", evidence="")
    assert consult.hit_text(h) == "HbA1c = 7,1 (ngưỡng HbA1c > 6,5 %, nguồn VMC-0002)"


def test_nlu_numeric_answer_ignores_durations():
    """E2E Qwen GĐ 8: "sau 2 giờ là 10,3" bị ghi OGTT_2H = 2,0 (C-00065, C-00067)."""
    q = "Chỉ số sau 2 giờ là bao nhiêu (đơn vị mmol/L)?"
    for text, want in (("Chỉ số sau 2 giờ của tôi là 10,3 mmol/L.", 10.3), ("Sau 2h là 15,9 ạ.", 15.9),
                       ("10,3 mmol/L.", 10.3), ("Đo sau 2 tiếng được 198 mg/dL.", 11.0)):
        sv = nlu.understand(text, "OGTT_2H", q, session_id="s-dur", use_llm=False).slots["OGTT_2H"]
        assert sv.status == "yes" and abs(sv.value - want) < 0.05, (text, sv)


def test_nlu_skips_llm_when_rules_answer_the_asked_slot(monkeypatch):
    """Chạy thử demo GĐ 9: Qwen sinh JSON không dừng cho "Tôi cao 1m65, nặng 78 kg." (11–62 s)."""
    fake = use_llm(monkeypatch, '{"an_nhieu": {"status": "yes"}}')
    r = nlu.understand("Tôi cao 1m65, nặng 78 kg.", "bmi", "Anh/chị cao bao nhiêu và nặng bao nhiêu kg?",
                       session_id="s-skip", use_llm=True)
    assert r.llm_skipped and fake.prompts == [] and abs(r.slots["bmi"].value - 28.7) < 0.1
    r2 = nlu.understand("Dạo này tôi khát nước suốt, đi tiểu nhiều, sụt 3 kg trong tháng.", None, "",
                        session_id="s-skip", use_llm=True)
    assert not r2.llm_skipped and len(fake.prompts) == 1          # câu mở đầu vẫn gọi LLM

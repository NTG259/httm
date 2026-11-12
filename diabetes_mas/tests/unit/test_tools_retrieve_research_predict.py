import json
from pathlib import Path

import pytest

from diabetes_mas.agents.state import DialogState, SlotValue
from diabetes_mas.tools import base, research_tools, retrieve_tools
from diabetes_mas.tools.base import call
from diabetes_mas.tools.predict_tools import features_from_state, plan
from diabetes_mas.tools.schemas import PredictIn

FIX = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(research_tools, "REVIEW_QUEUE", tmp_path / "review_queue.jsonl")
    monkeypatch.setattr(research_tools, "SEARCH_CACHE", tmp_path / "search")
    monkeypatch.setattr(research_tools, "WEB_CACHE", tmp_path / "web")


# ---------------- T-04 ----------------
class FakeBackend:
    def __init__(self, cos=0.82):
        self.cos, self.calls = cos, []

    def facts(self, ids, rels, limit):
        self.calls.append(("facts", tuple(ids), tuple(rels)))
        if "METFORMIN" in ids and "TREATS" in rels:
            return [{"s_id": "METFORMIN", "s": "Metformin", "rel": "TREATS", "o_id": "DTD_TIP_2",
                     "o": "Đái tháo đường típ 2", "source_id": "VMC-0002", "evidence": "metformin, gliclazid"}]
        return []

    def vector(self, q_emb, ids, k):
        rows = [{"chunk_id": f"VMC-01{i}#000", "text": f"đoạn {i}", "score": self.cos - i * 0.01,
                 "source_id": f"VMC-01{i}", "title": f"Bài {i}", "url": None, "reviewed": i == 2,
                 "n_mention": 1 if ids else 0} for i in range(4)]
        rows.append({"chunk_id": "VMC-010#001", "text": "đoạn 0b", "score": self.cos - 0.001, "source_id": "VMC-010",
                     "title": "Bài 0", "url": None, "reviewed": None, "n_mention": 0})
        rows.append({"chunk_id": "VMC-010#002", "text": "đoạn 0c", "score": self.cos - 0.002, "source_id": "VMC-010",
                     "title": "Bài 0", "url": None, "reviewed": None, "n_mention": 0})
        return rows

    def guided(self, q_emb, ids, k):
        return []


@pytest.fixture
def fake_rag(monkeypatch):
    be = FakeBackend()
    monkeypatch.setattr(retrieve_tools, "BACKEND", be)
    monkeypatch.setattr(retrieve_tools, "EMBED", lambda t: [0.0] * 1024)
    return be


def test_t04_drug_question_uses_treats_relation_and_returns_sourced_fact(fake_rag):
    out = call("T-04", agent_id="AG-07", question="Metformin có tác dụng phụ gì?")
    assert out.ok and out.enough
    assert ("facts", ("METFORMIN",), ("TREATS",)) in fake_rag.calls
    assert out.facts[0].text == "Metformin được dùng điều trị Đái tháo đường típ 2." and out.facts[0].source_id == "VMC-0002"


def test_t04_out_of_scope_is_not_enough_even_with_high_cosine(fake_rag):     # TC-23
    out = call("T-04", agent_id="AG-07", question="Cách chữa ung thư phổi")
    assert out.entities == [] and out.enough is False
    assert "thực thể" in out.enough_reason


def test_t04_low_cosine_is_not_enough(monkeypatch):
    monkeypatch.setattr(retrieve_tools, "BACKEND", FakeBackend(cos=0.60))
    monkeypatch.setattr(retrieve_tools, "EMBED", lambda t: [0.0] * 1024)
    out = call("T-04", agent_id="AG-07", question="Bệnh tiểu đường tuýp 2 là gì?")
    assert out.enough is False and "min_score" in out.enough_reason


def test_t04_no_entity_but_very_high_cosine_is_enough(monkeypatch):
    """GĐ 8: câu hợp lệ không chứa thực thể KB (SGLT2, Charcot, LADA…) được nhận khi cosine ≥ min_score_no_entity."""
    monkeypatch.setattr(retrieve_tools, "BACKEND", FakeBackend(cos=0.90))
    monkeypatch.setattr(retrieve_tools, "EMBED", lambda t: [0.0] * 1024)
    out = call("T-04", agent_id="AG-07", question="Bàn chân Charcot là gì?")
    assert out.entities == [] and out.enough is True and "min_score_no_entity" in out.enough_reason
    assert out.max_cosine == 0.9


def test_t04_ranking_caps_chunks_per_source(fake_rag):
    out = call("T-04", agent_id="AG-07", question="Triệu chứng tiểu đường tuýp 2", k_chunks=5)
    per_src = {}
    for c in out.chunks:
        per_src[c.source_id] = per_src.get(c.source_id, 0) + 1
    assert max(per_src.values()) <= 2
    assert out.chunks == sorted(out.chunks, key=lambda c: -c.score)


def test_fact_text_for_criterion():
    row = {"rel": "DIAGNOSED_BY", "s": "Tiền đái tháo đường", "o": "CRIT", "op": "between", "value": 6.1,
           "vmax": 6.9, "unit": "mmol/L", "test": "Glucose huyết tương lúc đói"}
    assert retrieve_tools.fact_text(row) == \
        "Tiêu chuẩn chẩn đoán Tiền đái tháo đường: Glucose huyết tương lúc đói từ 6,1 đến 6,9 mmol/L."


# ---------------- whitelist / T-12 (TC-26, TC-27, TC-29) ----------------
@pytest.mark.parametrize("url,ok", [
    ("https://www.vinmec.com/vie/bai-viet/x", True),
    ("https://vinmec.com/x", True),
    ("https://evilvinmec.com/x", False),
    ("https://vinmec.com.attacker.net/x", False),
    ("https://example.com/x", False),
])
def test_domain_whitelist(url, ok):
    assert research_tools.domain_allowed(url) is ok


def test_domains_argument_can_only_narrow_whitelist():
    assert research_tools.domain_allowed("https://who.int/x", ["example.com"]) is False


def _fake_searx(results):
    calls = []

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": results}

    def get(url, params, timeout):
        calls.append(params)
        return Resp()
    return get, calls


def test_t12_web_filters_non_whitelisted_results(monkeypatch):           # TC-26
    get, calls = _fake_searx([{"url": "https://www.vinmec.com/vie/bai-viet/a", "title": "A", "content": "x"},
                              {"url": "https://evil.example.com/b", "title": "B", "content": "y"}])
    monkeypatch.setattr(research_tools, "SEARXNG_GET", get)
    out = call("T-12", agent_id="AG-12", query="metformin tác dụng phụ", scope="web")
    assert [h.domain for h in out.hits] == ["www.vinmec.com"]
    assert "site:vinmec.com" in calls[0]["q"]


def test_t12_pii_blocks_request(monkeypatch):                            # TC-27
    get, calls = _fake_searx([])
    monkeypatch.setattr(research_tools, "SEARXNG_GET", get)
    out = call("T-12", agent_id="AG-12", query="tôi tên là Nguyễn Văn A SĐT 0912345678 bị tiểu đường", scope="web")
    assert out.ok is False and out.error.startswith("pii_detected") and calls == []


def test_t12_disabled_makes_no_request(monkeypatch):                     # TC-29
    get, calls = _fake_searx([])
    monkeypatch.setattr(research_tools, "SEARXNG_GET", get)
    monkeypatch.setenv("SEARCH_ENABLED", "false")
    out = call("T-12", agent_id="AG-12", query="metformin", scope="web")
    assert out.error == "search_disabled" and calls == []


# ---------------- T-13 ----------------
class FakeResp:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status


class FakeClient:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc = resp, exc

    def get(self, url):
        if self.exc:
            raise self.exc
        return self.resp


def test_t13_parses_vinmec_and_scores_on_neo4j_scale(monkeypatch):
    html = (FIX / "vinmec_post.html").read_text(encoding="utf-8")
    monkeypatch.setattr(research_tools, "HTTP_CLIENT_FACTORY", lambda t: FakeClient(FakeResp(html)))
    monkeypatch.setattr(research_tools, "EMBED_TEXTS", lambda texts: [[1.0, 0.0]] + [[0.0, 1.0]] * (len(texts) - 1))
    out = call("T-13", agent_id="AG-12", url="https://www.vinmec.com/vie/bai-viet/dai-thao-duong-thai-ky-vi",
               question="chẩn đoán tiểu đường thai kỳ")
    assert out.ok and out.title.startswith("Đái tháo đường thai kỳ") and len(out.chunks) == 3
    assert all(c.web and c.source_id.startswith("WEB-") and c.cosine == 0.5 for c in out.chunks)   # cos=0 → 0,5


def test_t13_rejects_domain_and_robots(monkeypatch):
    assert call("T-13", agent_id="AG-12", url="https://evil.com/x", question="q").error == "not_allowed_domain"
    monkeypatch.setattr(research_tools, "HTTP_CLIENT_FACTORY",
                        lambda t: FakeClient(exc=research_tools.RobotsDisallowed("x")))
    assert call("T-13", agent_id="AG-12", url="https://www.vinmec.com/x", question="q").error == "robots_disallow"


# ---------------- T-14 (TC-28 phần hàng chờ) ----------------
def test_t14_queue_dedup_and_skip_known_kb_url():
    args = dict(agent_id="AG-12", session_id="s", title="T", reason="KB thiếu")
    assert call("T-14", url="https://www.vinmec.com/vie/bai-viet/moi-vi", **args).queued is True
    assert call("T-14", url="https://www.vinmec.com/vie/bai-viet/moi-vi", **args).queued is False
    kb_url = "https://www.vinmec.com/vie/benh/tieu-duong-tuyp-2-3042"
    assert call("T-14", url=kb_url, **args).queued is False
    rows = [json.loads(l) for l in research_tools.REVIEW_QUEUE.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["status"] == "pending"


# ---------------- T-15 (TC-42, phần ánh xạ) ----------------
def _state(**slots):
    st = DialogState(session_id="p")
    st.update({k: SlotValue(**v) for k, v in slots.items()})
    return st


def test_t15_feature_mapping():
    st = _state(tuoi={"status": "yes", "value": 52}, bmi={"status": "yes", "value": 28.7},
                gioi_tinh={"status": "yes", "value": "nam"}, hut_thuoc={"status": "yes", "value": "dang_hut"},
                tang_huyet_ap={"status": "no"}, GLUCOSE_BAT_KY={"status": "yes", "value": 10.0},
                HBA1C={"status": "yes", "value": 6.2})
    f, assumptions = features_from_state(st)
    assert (f.gender, f.age, f.bmi, f.smoking_history, f.hypertension, f.heart_disease) == \
        ("Male", 52.0, 28.7, "current", 0, None)
    assert f.blood_glucose_level == 180.0 and f.HbA1c_level == 6.2
    assert assumptions and "glucose bất kỳ" in assumptions[0]
    assert plan(f)[0] == "full"


def test_t15_variant_and_errors(monkeypatch):
    from diabetes_mas.tools import predict_tools
    monkeypatch.setattr(predict_tools, "_predictor", lambda: None)
    assert plan(PredictIn(age=40, bmi=22))[0] == "no_labs"
    assert plan(PredictIn(age=40))[2] == "insufficient_features"
    out = call("T-15", agent_id="AG-13", age=40, bmi=22)
    assert out.ok is False and out.error == "model_not_trained" and out.model_variant == "no_labs"


def test_t15_trained_model_full_vs_no_labs():
    from diabetes_mas.dl.model import load
    if load() is None:
        pytest.skip("chưa huấn luyện mô hình DL")
    high = call("T-15", agent_id="AG-13", age=60, bmi=32, gender="Male", hypertension=1,
                HbA1c_level=8.5, blood_glucose_level=240)
    low = call("T-15", agent_id="AG-13", age=25, bmi=21, gender="Female", hypertension=0,
               HbA1c_level=5.0, blood_glucose_level=90)
    assert high.ok and low.ok and high.model_variant == low.model_variant == "full"
    assert high.risk > high.threshold > low.risk
    nolab = call("T-15", agent_id="AG-13", age=40, bmi=22)
    assert nolab.ok and nolab.model_variant == "no_labs"
    assert any("gender" in a for a in nolab.assumptions)

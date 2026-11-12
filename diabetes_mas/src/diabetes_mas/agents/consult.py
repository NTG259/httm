"""AG-07 Consult — hỏi đáp có trích nguồn (P-05) và giải thích kết luận chẩn đoán (P-06).

Hỏi đáp (Luồng B):
1. T-04 hybrid_retrieve. Lỗi hạ tầng ⇒ UNAVAILABLE_TEXT (Luồng D), không đoán.
2. enough = false ⇒ nhờ AG-12; có chunk bổ sung ⇒ enough = true (chunk web đã lọc theo min_score).
3. enough = false ⇒ câu từ chối mẫu.
4. P-05 với context đánh số; kiểm mọi [n] ∈ context và có ≥ 1 trích dẫn; sai ⇒ thử lại 1 lần
   ⇒ vẫn sai hoặc Ollama lỗi ⇒ câu trả lời TRÍCH XUẤT (liệt kê facts/đoạn văn kèm [n]) — không sinh nội dung mới.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from diabetes_mas.agents import llm
from diabetes_mas.agents.slots import DISEASES
from diabetes_mas.config import settings
from diabetes_mas.tools import call
from diabetes_mas.tools.errors import UNAVAILABLE_TEXT, LLMUnavailable, is_unavailable
from diabetes_mas.tools.schemas import ChunkHit, CriterionHit, Fact

REFUSAL = "Tôi chưa có đủ thông tin đáng tin cậy để trả lời câu hỏi này."
CITE = re.compile(r"\[(\d+)\]")
# Qwen 3B hay trích đúng số nhưng sai dạng: "(1)", "(12, 13)", "[3, 2]" (đo P-05 với KB thật, 2026-11-12)
_CITE_VARIANT = re.compile(r"\((\d{1,2}(?:\s*,\s*\d{1,2})*)\)|\[(\d{1,2}(?:\s*,\s*\d{1,2})+)\]")
DISEASE_VI = {"KHONG_DTD": "Không mắc đái tháo đường", "TIEN_DTD": "Tiền đái tháo đường",
              "DTD_TIP_1": "Đái tháo đường típ 1", "DTD_TIP_2": "Đái tháo đường típ 2",
              "DTD_THAI_KY": "Đái tháo đường thai kỳ"}


@dataclass
class Source:
    n: int
    kind: str                 # fact | chunk | web
    text: str
    source_id: str
    title: str = ""
    url: str | None = None


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    enough: bool = False
    method: str = ""          # llm | extractive | refusal | unavailable
    research: dict | None = None
    error: str | None = None


def build_context(facts: list[Fact], chunks: list[ChunkHit]) -> tuple[list[Source], str, str]:
    sources: list[Source] = []
    for f in facts:
        sources.append(Source(len(sources) + 1, "fact", f.text, f.source_id))
    for c in chunks:
        kind = "web" if c.web else "chunk"
        sources.append(Source(len(sources) + 1, kind, c.text, c.source_id, c.source_title, c.url))
    fx = "\n".join(f"[{s.n}] {s.text} (nguồn {s.source_id})" for s in sources if s.kind == "fact") or "(không có)"
    cx = "\n".join(
        f"[{s.n}] ({'Nguồn web bổ sung – ' + (s.url or '').split('/')[2] if s.kind == 'web' and s.url else s.source_id}"
        f" – \"{s.title}\") {s.text[:900]}" for s in sources if s.kind != "fact") or "(không có)"
    return sources, fx, cx


def citations_ok(text: str, n_sources: int) -> bool:
    cites = [int(x) for x in CITE.findall(text)]
    return bool(cites) and all(1 <= c <= n_sources for c in cites)


def _first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n])[:400]


def extractive(sources: list[Source], max_items: int = 4) -> str:
    lines = ["Thông tin liên quan trong cơ sở tri thức:"]
    for s in sources[:max_items]:
        lines.append(f"- {_first_sentences(s.text)} [{s.n}]")
    return "\n".join(lines)


def normalize_citations(text: str) -> str:
    """Đưa "(1)", "(12, 13)", "[3, 2]" về "[1]", "[12][13]", "[3][2]". Chỉ đổi định dạng; số ngoài context
    vẫn bị `citations_ok` loại."""
    def rep(m: re.Match) -> str:
        return "".join(f"[{n.strip()}]" for n in (m.group(1) or m.group(2)).split(","))
    return _CITE_VARIANT.sub(rep, text)


def _generate(prompt: str, n_sources: int) -> str | None:
    for _ in range(2):
        out = normalize_citations(llm.LLM.chat(prompt, temperature=0.0))
        if out.strip() == REFUSAL or citations_ok(out, n_sources):
            return out.strip()
    return None


def answer(question: str, session_id: str, use_llm: bool = True, allow_research: bool = True) -> Answer:
    ret = call("T-04", agent_id="AG-07", session_id=session_id, question=question)
    if not ret.ok:
        return Answer(UNAVAILABLE_TEXT, method="unavailable", error=ret.error) if is_unavailable(ret.error) \
            else Answer(UNAVAILABLE_TEXT, method="unavailable", error=ret.error)
    chunks, enough, research = list(ret.chunks), ret.enough, None
    if not enough and allow_research and settings()["search"].get("enabled", True):
        from diabetes_mas.agents import research as rs
        r = rs.find(question, session_id)
        research = {"query": r.query, "scope": r.scope, "n_chunks": len(r.chunks), "queued": r.queued,
                    "errors": r.errors}
        if r.chunks:
            chunks += r.chunks
            enough = True
    if not enough:
        return Answer(REFUSAL, enough=False, method="refusal", research=research)
    sources, fx, cx = build_context(ret.facts[:10], chunks[:6])
    text, method = None, "extractive"
    if use_llm:
        try:
            text = _generate(llm.render("P-05", facts_numbered=fx, chunks_numbered=cx, question=question), len(sources))
            method = "llm" if text else "extractive"
        except LLMUnavailable:
            text = None
    text = text or extractive(sources)
    if any(s.kind == "web" for s in sources):
        domains = sorted({(s.url or "").split("/")[2] for s in sources if s.kind == "web" and s.url})
        text += f"\n\nMột phần thông tin lấy từ bài viết trên {', '.join(domains)}, chưa được nhóm kiểm duyệt."
    return Answer(text, sources, True, method, research)


PROFILE_VI = {"tuoi": "tuổi", "bmi": "BMI", "gioi_tinh": "giới tính", "hut_thuoc": "hút thuốc", "benh_tim": "bệnh tim mạch"}


def _vi_num(text: str) -> str:
    return re.sub(r"(\d)\.(\d)", r"\g<1>,\g<2>", text)


def slot_names(ids: list[str]) -> list[str]:
    """Mã slot → tên tiếng Việt cho người đọc (demo từng hiện "uong_nhieu, tieu_nhieu…")."""
    from diabetes_mas.tools.kg_tools import _lexicon

    names = _lexicon()[1]
    return [(names.get(i) or PROFILE_VI.get(i, i)).split("(")[0].strip() for i in ids]


def hit_text(h: CriterionHit) -> str:
    """"HbA1c = 7,1 (ngưỡng HbA1c > 6,5 %, nguồn VMC-0002)" thay cho mã thô "HBA1C = 7.1 (HBA1C > 6.5 %…"."""
    from diabetes_mas.tools.kg_tools import _lexicon

    tname = _lexicon()[1].get(h.test_id) or h.test_id
    return _vi_num(f"{tname} = {h.value:g} (ngưỡng {h.rule.replace(h.test_id, tname)}, nguồn {h.source_id})")


def explain_ok(text: str, top1_name: str, prob_txt: str) -> bool:
    """P-06 phải nêu ĐÚNG kết luận và xác suất. Qwen 3B từng viết "khả năng mắc đái tháo đường là 54%" khi kết luận
    là "Không mắc đái tháo đường" 54% (bỏ chữ "Không" — E2E --llm, GĐ 8)."""
    low = text.lower()
    return top1_name.lower() in low and prob_txt.replace("%", "") in low


MEASURE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:%|mmol|mg|kg)")
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _nums(text: str) -> set[float]:
    return {float(x.replace(",", ".")) for x in _NUM.findall(text)}


def _criteria_numbers() -> set[float]:
    """Ngưỡng tiêu chuẩn trong KB (value, value_max và số trong câu trích, vd. 126 mg/dl)."""
    from diabetes_mas.config import SEED_DIR
    try:
        return _nums((SEED_DIR / "criteria.csv").read_text(encoding="utf-8").split("\n", 1)[1])
    except OSError:
        return set()


def allowed_numbers(session_id: str, probs: dict[str, float], rule_hits: list[CriterionHit]) -> set[float]:
    """Số được phép xuất hiện kèm đơn vị trong P-06: giá trị đã thu thập, số người dùng gõ, xác suất, ngưỡng."""
    from diabetes_mas.tools import session

    st = session.get_or_create(session_id)
    ok = {float(v.value) for v in st.slots.values() if isinstance(v.value, (int, float))}
    for t in st.history:
        if t.speaker == "user":
            ok |= _nums(t.text)
    ok |= {round(p * 100) for p in probs.values()}
    for h in rule_hits:
        ok |= {h.value} | _nums(h.rule)
    return ok | _criteria_numbers()


def numbers_ok(text: str, allowed: set[float]) -> bool:
    """Chạy thử GĐ 9: Qwen viết "HbA1c của bạn là 7,4%" khi người dùng gõ 7,1 ⇒ số lạ kèm đơn vị thì dùng template."""
    for m in MEASURE.finditer(text.lower()):
        v = float(m.group(1).replace(",", "."))
        if not any(abs(v - a) < 0.051 for a in allowed):
            return False
    return True


def explain(session_id: str, top1: str, probs: dict[str, float], rule_hits: list[CriterionHit],
            evidence_slots: list[str], risk_dl: float | None = None, dl_conflict: bool = False,
            use_llm: bool = True) -> Answer:
    """Giải thích kết luận (P-06). Căn cứ: rule_hits (có nguồn KB) + T-04 về bệnh top-1."""
    name = DISEASE_VI.get(top1, top1)
    others = ", ".join(f"{DISEASE_VI[d]} {probs[d]:.0%}" for d in sorted(DISEASES, key=lambda d: -probs.get(d, 0))
                       if d != top1 and probs.get(d, 0) >= 0.05) or "không đáng kể"
    hits_txt = "; ".join(hit_text(h) for h in rule_hits) or "chưa có"
    evidence_slots = slot_names(evidence_slots)
    ret = call("T-04", agent_id="AG-07", session_id=session_id, question=f"{name} chẩn đoán và điều trị", k_chunks=3)
    facts, chunks = (ret.facts[:6], ret.chunks[:3]) if ret.ok else ([], [])
    sources, fx, cx = build_context(facts, chunks)
    rule_sources = [Source(len(sources) + i + 1, "fact", f"{h.rule}: {h.evidence}", h.source_id, h.source_title)
                    for i, h in enumerate(rule_hits)]
    sources += rule_sources
    template = (f"Kết luận sơ bộ: {name} (xác suất ước tính {probs.get(top1, 0):.0%}). Các khả năng khác: {others}.\n"
                f"Căn cứ: thông tin anh/chị đã cung cấp ({', '.join(evidence_slots) or 'chưa nhiều'}); "
                f"tiêu chuẩn xét nghiệm đạt: {hits_txt}"
                + (" " + " ".join(f"[{s.n}]" for s in rule_sources) if rule_sources else "") + ".\n"
                "Khuyến nghị: đi khám chuyên khoa Nội tiết – Đái tháo đường để làm xét nghiệm xác định "
                "(glucose huyết tương lúc đói, HbA1c hoặc nghiệm pháp dung nạp glucose).")
    if risk_dl is not None:
        template += f"\nMô hình học sâu (tham khảo) ước tính nguy cơ đái tháo đường {risk_dl:.0%}"
        template += " — KHÁC với kết quả theo tiêu chuẩn chẩn đoán; tiêu chuẩn của Bộ Y tế/KB được ưu tiên." \
            if dl_conflict else "."
    if not ret.ok:
        template += "\n(Hiện chưa truy xuất được tài liệu tham khảo bổ sung.)"
    if not use_llm or not sources:
        return Answer(template, sources, True, "template")
    ctx = "\n".join(f"[{s.n}] {s.text[:600]} (nguồn {s.source_id})" for s in sources)
    try:
        out = _generate(llm.render("P-06", top1_name=name, top1_prob=f"{probs.get(top1, 0):.0%}", others=others,
                                   evidence_slots=", ".join(evidence_slots) or "chưa nhiều", rule_hits=hits_txt,
                                   context=ctx), len(sources))
    except LLMUnavailable:
        out = None
    prob_txt = f"{probs.get(top1, 0):.0%}"
    if out and not explain_ok(out, name, prob_txt):
        out = None                                       # sai kết luận / xác suất ⇒ dùng template
    if out and not numbers_ok(out, allowed_numbers(session_id, probs, rule_hits)):
        out = None                                       # bịa giá trị xét nghiệm ⇒ dùng template
    if out and risk_dl is not None:
        out += f"\n\nMô hình học sâu (tham khảo): nguy cơ {risk_dl:.0%}" + \
               (" — mâu thuẫn với tiêu chuẩn chẩn đoán; ưu tiên tiêu chuẩn." if dl_conflict else ".")
    return Answer(out or template, sources, True, "llm" if out else "template")

"""AG-06 NLU — câu tiếng Việt → slot (P-02 + luật).

Thứ tự (luật trước, LLM bổ sung — vì model nhỏ có thể bịa slot):
1. Tách mệnh đề (dấu câu, "nhưng", "mà"); mỗi mệnh đề gọi T-02 kg_link_entities (cùng từ điển với KB)
   + bí danh hội thoại `ALIASES`; mệnh đề có từ phủ định ⇒ "no".
2. Regex số đo: tuổi, chiều cao + cân nặng ⇒ BMI, HbA1c, glucose (lúc đói / sau ăn 2 giờ / bất kỳ;
   > 35 coi là mg/dL ⇒ ÷ 18), giới tính, hút thuốc.
3. Câu trả lời ngắn cho slot vừa hỏi ("có", "không", "không biết", "7,1").
4. P-02 (LLM) chỉ thêm slot luật chưa bắt được; mọi slot LLM trả về được kiểm theo kho slot
   (id tồn tại, kiểu, giá trị phân loại hợp lệ, số trong khoảng). Ollama lỗi ⇒ chỉ dùng luật (02 §8).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from diabetes_mas.agents import llm
from diabetes_mas.agents.slots import catalog
from diabetes_mas.agents.state import SlotValue
from diabetes_mas.config import settings
from diabetes_mas.tools import call
from diabetes_mas.tools.errors import LLMUnavailable
from diabetes_mas.utils.text import norm_key

# Bí danh hội thoại (không phải thuật ngữ y khoa nên không đưa vào KB seed). Khóa đã bỏ dấu.
ALIASES = {
    "sut can": "sut_can", "giam can": "sut_can", "sut ky": "sut_can",
    "mui tao": "hoi_tho_mui_tao_thoi", "mui trai cay": "hoi_tho_mui_tao_thoi", "hoi tho co mui": "hoi_tho_mui_tao_thoi",
    "non": "buon_non", "oi": "buon_non", "buon oi": "buon_non",
    "huyet ap cao": "tang_huyet_ap", "cao huyet ap": "tang_huyet_ap",
    "mo mau": "roi_loan_lipid", "mo mau cao": "roi_loan_lipid", "cholesterol": "roi_loan_lipid",
    "bo bi tieu duong": "tien_su_gia_dinh_dtd", "me bi tieu duong": "tien_su_gia_dinh_dtd",
    "bo toi bi tieu duong": "tien_su_gia_dinh_dtd", "me toi bi tieu duong": "tien_su_gia_dinh_dtd",
    "benh tim": "benh_tim", "tim mach": "benh_tim",
    "te chan": "te_bi_ban_chan", "te bi chan": "te_bi_ban_chan", "loet chan": "loet_ban_chan",
    "dau nguc": "dau_nguc", "mo mat": "nhin_mo", "nhin mo": "nhin_mo",
    "khat": "uong_nhieu", "uong nuoc nhieu": "uong_nhieu", "di tieu dem": "tieu_nhieu", "tieu dem": "tieu_nhieu",
    "an nhieu": "an_nhieu", "mau doi": "an_nhieu", "nhanh doi": "an_nhieu",
    "co bau": "dang_mang_thai", "mang thai": "dang_mang_thai",
    "it van dong": "it_van_dong", "luoi van dong": "it_van_dong", "ngoi nhieu": "it_van_dong",
}
NEG = re.compile(r"(?<!\w)(khong|chua|ko|k|chang|cha|het)(?!\w)")
UNKNOWN = re.compile(r"(khong|ko|chua)\s+(biet|ro|nho|do|xet nghiem|kiem tra|de y)")
YES = re.compile(r"^(co|co a|co ạ|vang|da|da co|u|uh|dung|dung roi|roi|bi|co bi|hay bi|thuong xuyen|nhieu)(\W|$)")
NO = re.compile(r"^(khong|ko|k|chua|khong co|khong bi|chua bao gio|khong a)(\W|$)")
NUM = r"(\d+(?:[.,]\d+)?)"
# Cụm thời gian (đã norm_key: không dấu, chữ thường) — không phải giá trị xét nghiệm: "2 giờ", "2h", "30 phút", "28 tuần"
DURATION = re.compile(r"\d+(?:[.,]\d+)?\s*(?:gio|h|tieng|phut|ngay|tuan|thang|nam)\b")
LEAD = re.compile(r"^(?:(?:da|vang|thua bac si|bac si|toi|em|minh|chau|con)\W*\s*)+")
CLAUSE = re.compile(r"[,.;!?\n]|\s(?:nhung|ma|con)\s")


def _f(x: str) -> float:
    return float(x.replace(",", "."))


@dataclass
class NLUResult:
    slots: dict[str, SlotValue] = field(default_factory=dict)
    by_rules: list[str] = field(default_factory=list)
    by_llm: list[str] = field(default_factory=list)
    llm_error: str | None = None
    llm_skipped: bool = False


SHORT_ANSWER_CHARS = 60           # câu trả lời ngắn cho câu hỏi vừa hỏi


def _clauses(text: str) -> list[str]:
    return [c.strip() for c in CLAUSE.split(f" {norm_key(text)} ") if c.strip()]


def _binary_from_clauses(text: str, mode: str, session_id: str | None) -> dict[str, SlotValue]:
    by_id = catalog().by_id()
    out: dict[str, SlotValue] = {}
    for clause in _clauses(text):
        ids: set[str] = set()
        # chỉ từ điển: fulltext Neo4j khớp từ rời, từng biến "Tôi đã bỏ thuốc rồi" thành rối loạn ý thức ⇒ cấp cứu
        ents = call("T-02", agent_id="AG-06", mode=mode, session_id=session_id, text=clause, fulltext=False)
        ids |= {e.node_id for e in ents.entities if e.node_id in by_id and by_id[e.node_id].kind == "binary"}
        hay = f" {clause} "
        for key in sorted(ALIASES, key=len, reverse=True):
            if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", hay):
                ids.add(ALIASES[key])
                hay = hay.replace(key, " ")
        if not ids:
            continue
        neg = bool(NEG.search(clause)) and not UNKNOWN.search(clause)
        for sid in ids:
            if by_id[sid].source == "derived":
                continue
            out[sid] = SlotValue(status="no" if neg else "yes")
    return out


def _glucose_slot(ctx: str) -> str:
    if re.search(r"sau an|2 ?gio|2 ?h|ogtt|dung nap|nghiem phap", ctx):
        return "OGTT_2H"
    if re.search(r"bat ky|ngau nhien|bat ki", ctx):
        return "GLUCOSE_BAT_KY"
    return "GLUCOSE_DOI" if re.search(r"doi|sang som|truoc an", ctx) else "GLUCOSE_BAT_KY"


def _to_mmol(v: float, ctx: str) -> float:
    return round(v / 18.0, 1) if v > 35 or "mg" in ctx else v


def _numbers(text: str) -> dict[str, SlotValue]:
    t = norm_key(text)
    out: dict[str, SlotValue] = {}
    if m := re.search(rf"{NUM}\s*tuoi", t):
        out["tuoi"] = SlotValue(status="yes", value=_f(m.group(1)))
    h = None
    if m := re.search(r"(?<!\d)([12])\s*m\s*(\d{1,2})(?!\d)", t):
        h = int(m.group(1)) + int(m.group(2)) / (100 if len(m.group(2)) == 2 else 10)
    elif m := re.search(rf"(?:cao\s*)?{NUM}\s*(cm|m)(?!\w)", t):
        v = _f(m.group(1))
        h = v / 100 if m.group(2) == "cm" or v > 3 else v
    if (m := re.search(rf"{NUM}\s*(kg|ky|can|ki lo|kilogram)(?!\w)", t)) and h and 1.0 < h < 2.3:
        out["bmi"] = SlotValue(status="yes", value=round(_f(m.group(1)) / (h * h), 1))
    elif m := re.search(rf"bmi\D{{0,6}}{NUM}", t):
        out["bmi"] = SlotValue(status="yes", value=_f(m.group(1)))
    if m := re.search(rf"hba1c\D{{0,40}}{NUM}", t):
        out["HBA1C"] = SlotValue(status="yes", value=_f(m.group(1)))
    for m in re.finditer(r"duong huyet|duong mau|glucose|thu duong", t):
        tail = t[m.end():m.end() + 50]
        for n in re.finditer(rf"{NUM}\s*(mg/?dl|mmol)?", tail):
            if re.match(r"\s*(gio|h|tieng|lan|phut|ngay)(?!\w)", tail[n.end(1):]):
                continue                                   # "sau ăn 2 giờ" — số chỉ thời gian
            ctx = tail[:n.start()]
            after = re.split(r"[,.;]", tail[n.end():])[0][:25]          # "126 mg/dl lúc đói"
            out[_glucose_slot(ctx + " " + after)] = SlotValue(status="yes", value=_to_mmol(_f(n.group(1)), ctx + (n.group(2) or "")))
            break
    if re.search(rf"(sut|giam|xuong|mat)\s*(mat\s*)?{NUM}\s*(kg|ky|can|ki lo|kilogram)(?!\w)", t):
        out["sut_can"] = SlotValue(status="yes")                           # "sụt 3 kg trong tháng"
    if re.search(r"(?<!\w)(nam|dan ong|con trai)(?!\w)", t) and not re.search(r"\d+\s*nam", t):
        out["gioi_tinh"] = SlotValue(status="yes", value="nam")
    elif re.search(r"(?<!\w)(nu|phu nu|dan ba|con gai)(?!\w)", t):
        out["gioi_tinh"] = SlotValue(status="yes", value="nu")
    if re.search(r"dang hut", t):
        out["hut_thuoc"] = SlotValue(status="yes", value="dang_hut")
    elif re.search(r"hien (toi |minh |em )?(khong|ko) (con )?hut|khong con hut", t):
        out["hut_thuoc"] = SlotValue(status="yes", value="hien_khong_hut")
    elif re.search(r"da bo|bo thuoc|cai thuoc", t):
        out["hut_thuoc"] = SlotValue(status="yes", value="da_bo")
    elif re.search(r"tung hut", t):
        out["hut_thuoc"] = SlotValue(status="yes", value="tung_hut")
    elif re.search(r"(khong|chua bao gio|chua tung) hut", t):
        out["hut_thuoc"] = SlotValue(status="yes", value="chua_bao_gio")
    return out


def _answer_to_last(text: str, last_slot: str) -> SlotValue | None:
    """Câu trả lời ngắn cho slot vừa hỏi."""
    slot = catalog().by_id().get(last_slot)
    if slot is None:
        return None
    t = norm_key(text)
    if UNKNOWN.search(t):
        return SlotValue(status="unknown")
    if slot.kind == "numeric":
        t = DURATION.sub(" ", t)          # "sau 2 giờ là 10,3" ⇒ bỏ "2 giờ" (E2E Qwen GĐ 8: OGTT_2H ghi thành 2,0)
        nums = [_f(x) for x in re.findall(rf"(?<![a-z\d]){NUM}", t)]      # bỏ chữ số trong "hba1c"
        if not nums:
            return SlotValue(status="unknown") if NO.search(t) else None   # "không" (chưa đo) ⇒ không biết
        lo, hi = slot.range
        for v in nums:
            if slot.id in ("GLUCOSE_DOI", "GLUCOSE_BAT_KY", "OGTT_1H", "OGTT_2H"):
                v = _to_mmol(v, t)
            if lo <= v <= hi:
                return SlotValue(status="yes", value=v)
        return None
    if slot.kind == "binary":
        t = LEAD.sub("", t)                                       # "Dạ, tôi không ạ" → "khong a"
        if NO.search(t):
            return SlotValue(status="no")
        if YES.search(t):
            return SlotValue(status="yes")
    return None


def _valid(sid: str, sv: dict) -> SlotValue | None:
    slot = catalog().by_id().get(sid)
    if slot is None or slot.source == "derived" or not isinstance(sv, dict):
        return None
    try:
        v = SlotValue(**sv)
    except Exception:  # noqa: BLE001
        return None
    if slot.kind == "binary":
        return SlotValue(status=v.status) if v.status in ("yes", "no", "unknown") else None
    if v.status != "yes":
        return SlotValue(status="unknown") if v.status == "unknown" else None
    if slot.kind == "numeric":
        if not isinstance(v.value, (int, float)):
            return None
        lo, hi = slot.range
        return v if lo <= float(v.value) <= hi else None
    return v if v.value in slot.values else None


def _catalog_text() -> str:
    lines = []
    for s in catalog().slots:
        if s.source == "derived":
            continue
        kind = s.kind + (f" {list(s.values)}" if s.values else "") + (f" {s.range}" if s.range else "")
        lines.append(f"- {s.id}: {s.question_vi or s.id}, {kind}")
    return "\n".join(lines)


def understand(text: str, last_slot: str | None = None, last_question: str = "", mode: str = "demo",
               session_id: str | None = None, use_llm: bool | None = None) -> NLUResult:
    res = NLUResult()
    rules = _binary_from_clauses(text, mode, session_id)
    rules.update(_numbers(text))
    if last_slot and last_slot not in rules:
        ans = _answer_to_last(text, last_slot)
        if ans is not None:
            rules[last_slot] = ans
    res.slots.update(rules)
    res.by_rules = sorted(rules)
    use_llm = settings().get("nlu", {}).get("use_llm", True) if use_llm is None else use_llm
    if not use_llm:
        return res
    if last_slot and last_slot in rules and len(text) <= SHORT_ANSWER_CHARS:
        res.llm_skipped = True        # câu trả lời ngắn, luật đã điền đúng slot đang hỏi ⇒ không gọi Qwen
        return res                    # (chạy thử demo: "Tôi cao 1m65, nặng 78 kg." làm Qwen sinh JSON không dừng, 11–62 s)
    try:
        prompt = llm.render("P-02", slot_catalog=_catalog_text(), last_question=last_question or "(không có)",
                            user_text=text)
        raw = llm.chat_json(_AnyJSON, prompt)
        for sid, sv in raw.root.items():
            if sid in res.slots:
                continue
            ok = _valid(sid, sv)
            if ok is not None:
                res.slots[sid] = ok
                res.by_llm.append(sid)
    except (LLMUnavailable, llm.LLMBadOutput) as exc:
        res.llm_error = f"{type(exc).__name__}: {exc}"[:200]
    return res


from pydantic import RootModel  # noqa: E402


class _AnyJSON(RootModel[dict]):
    pass

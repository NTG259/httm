"""Hợp đồng vào/ra của mọi tool (T-01 … T-15) — nguồn duy nhất cho 03_TOOL_SPEC.md.

Xuất JSON schema (dùng bind cho LLM, dán vào tài liệu):  python -m diabetes_mas.tools.export_schemas
Mọi Out có `ok` + `error`: tool không ném exception ra agent.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from diabetes_mas.agents.state import DialogState, SlotValue


class ToolOut(BaseModel):
    ok: bool = True
    error: str | None = None


# ---------- T-01 kg_get_group_slots ----------
class GetGroupSlotsIn(BaseModel):
    worker: Literal["W1", "W2", "W3", "W4"]


class SlotInfo(BaseModel):
    slot_id: str
    kind: Literal["binary", "numeric", "categorical"]
    source: str
    question_vi: str
    values: list[str] = []
    unit: str | None = None


class GetGroupSlotsOut(ToolOut):
    slots: list[SlotInfo] = []


# ---------- T-02 kg_link_entities ----------
class LinkIn(BaseModel):
    text: str
    top_k: int = Field(8, ge=1, le=20)
    fulltext: bool = Field(True, description="Dùng fulltext Neo4j khi từ điển không khớp. NLU tắt: fulltext khớp "
                                             "từng từ rời (\"rồi\" → Rối loạn ý thức) nên không được dùng để ghi slot")


class LinkedEntity(BaseModel):
    node_id: str
    label: str
    name_vi: str
    count: int = 1
    method: Literal["lexicon", "fulltext"] = "lexicon"
    score: float = 1.0


class LinkOut(ToolOut):
    entities: list[LinkedEntity] = []


# ---------- T-03 kg_check_criteria ----------
class CheckCriteriaIn(BaseModel):
    measurements: dict[str, float]               # {"HBA1C": 7.1}, đơn vị chuẩn của LabTest
    context: dict[str, bool] = {}                # {"dang_mang_thai": True, "co_trieu_chung_kinh_dien": True}


class CriterionHit(BaseModel):
    disease_id: str
    criterion_id: str
    test_id: str
    value: float
    rule: str
    source_id: str
    source_title: str = ""
    evidence: str = ""


class CheckCriteriaOut(ToolOut):
    hits: list[CriterionHit] = []
    kb_backend: Literal["neo4j", "seed"] = "seed"


# ---------- T-04 hybrid_retrieve ----------
class RetrieveIn(BaseModel):
    question: str
    k_chunks: int = Field(5, ge=1, le=10)
    max_facts: int = Field(20, ge=0, le=50)


class Fact(BaseModel):
    text: str
    triple: tuple[str, str, str]
    source_id: str
    evidence: str | None = None


class ChunkHit(BaseModel):
    chunk_id: str
    text: str
    score: float                                 # điểm xếp hạng cuối
    cosine: float
    source_id: str
    source_title: str = ""
    url: str | None = None
    reviewed: bool | None = None
    web: bool = False                            # chunk tạm từ AG-12


class RetrieveOut(ToolOut):
    entities: list[LinkedEntity] = []
    facts: list[Fact] = []
    chunks: list[ChunkHit] = []
    enough: bool = False
    enough_reason: str = ""
    max_cosine: float = 0.0                      # cosine tốt nhất mọi ứng viên — để quét min_score (08 §3)


# ---------- T-05 / T-06 trạng thái ----------
class UpdateStateIn(BaseModel):
    session_id: str
    updates: dict[str, SlotValue]
    asked_slot: str | None = None
    count_turn: bool = True                      # False cho câu hỏi intake (không tính T_max)


class UpdateStateOut(ToolOut):
    turn: int = 0


class GetStateIn(BaseModel):
    session_id: str


class GetStateOut(ToolOut):
    state: DialogState | None = None
    vector: list[float] = []


# ---------- T-07 ask_patient ----------
class AskIn(BaseModel):
    session_id: str
    slot_id: str
    question_text: str = ""


class AskOut(ToolOut):
    answer_text: str = ""
    parsed: dict[str, SlotValue] = {}
    channel: Literal["simulator", "ui"] = "simulator"


# ---------- T-08 classify ----------
class ClassifyIn(BaseModel):
    session_id: str


class ClassifyOut(ToolOut):
    probs: dict[str, float] = {}
    sub_probs: dict[str, dict[str, float]] = {}
    entropy: float = 0.0
    top1: str = ""
    rule_hits: list[CriterionHit] = []
    variant: Literal["kb_naive_bayes", "mlp"] = "kb_naive_bayes"


# ---------- T-09 check_red_flags ----------
class RedFlagIn(BaseModel):
    session_id: str


class RedFlagOut(ToolOut):
    red_flag: bool = False
    reasons: list[str] = []
    action_text: str | None = None


# ---------- T-10 log_event ----------
class LogEventIn(BaseModel):
    session_id: str | None
    agent: str
    tool: str
    args: dict
    result_summary: str
    latency_ms: int
    ok: bool


class LogEventOut(ToolOut):
    path: str = ""


# ---------- T-11 compute_reward ----------
class RewardIn(BaseModel):
    prev_entropy: float
    entropy: float
    asked: bool
    hit: bool
    terminal: bool
    correct: bool | None = None


class RewardOut(ToolOut):
    reward: float = 0.0
    parts: dict[str, float] = {}


# ---------- T-12 search_articles ----------
class SearchArticlesIn(BaseModel):
    query: str
    scope: Literal["local", "web"] = "local"
    domains: list[str] | None = None
    max_results: int = Field(5, ge=1, le=10)


class ArticleHit(BaseModel):
    url: str | None
    title: str
    snippet: str
    domain: str
    in_kb: bool
    source_id: str | None = None
    score: float = 0.0


class SearchArticlesOut(ToolOut):
    hits: list[ArticleHit] = []


# ---------- T-13 fetch_article ----------
class FetchArticleIn(BaseModel):
    url: str
    question: str
    k_chunks: int = Field(3, ge=1, le=10)
    timeout_s: float = Field(8.0, gt=0, le=60)


class FetchArticleOut(ToolOut):
    url: str = ""
    title: str = ""
    published: str | None = None
    chunks: list[ChunkHit] = []


# ---------- T-14 queue_article_for_kb ----------
class QueueArticleIn(BaseModel):
    url: str
    title: str
    reason: str
    session_id: str


class QueueArticleOut(ToolOut):
    queued: bool = False


# ---------- T-15 predict_diabetes_risk ----------
class PredictIn(BaseModel):
    gender: Literal["Female", "Male", "Other"] | None = None
    age: float | None = None
    hypertension: int | None = None
    heart_disease: int | None = None
    smoking_history: Literal["never", "former", "current", "not current", "ever", "No Info"] = "No Info"
    bmi: float | None = None
    HbA1c_level: float | None = None
    blood_glucose_level: float | None = None     # mg/dL


class PredictOut(ToolOut):
    risk: float | None = None
    threshold: float | None = None
    model_variant: Literal["full", "no_labs"] | None = None
    missing_features: list[str] = []
    assumptions: list[str] = []
    rule_disease_hits: list[str] = []
    conflict_with_rules: bool | None = None

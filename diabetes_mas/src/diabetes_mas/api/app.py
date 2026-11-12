"""API FastAPI (K4) — 03_TOOL_SPEC.md §5.

Chạy:  uvicorn diabetes_mas.api.app:app --port 8000      → giao diện demo ở http://localhost:8000/
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from diabetes_mas.agents import consult, graph, safety
from diabetes_mas.agents.llm import LLM
from diabetes_mas.tools import base, session

STATIC = Path(__file__).with_name("static")
app = FastAPI(title="Diabetes MAS — KG + RAG", version="0.5.0",
              description="Hệ đa tác tử chẩn đoán & tư vấn tiểu đường (chỉ mang tính tham khảo).")


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    use_llm: bool = True


class ConsultIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    use_llm: bool = True


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/v1/sessions")
def create_session() -> dict:
    return {"session_id": graph.new_session()}


@app.post("/v1/sessions/{sid}/messages")
def post_message(sid: str, body: MessageIn) -> dict:
    return graph.handle(sid, body.text, use_llm=body.use_llm)


@app.get("/v1/sessions/{sid}/state")
def get_state(sid: str) -> dict:
    if sid not in session._STATES:
        raise HTTPException(404, "không có phiên này")
    return graph.summary(sid)


@app.post("/v1/consult")
def consult_once(body: ConsultIn) -> dict:
    sid = graph.new_session()
    trace: list = []
    token = base.TRACE.set(trace)
    try:
        a = consult.answer(body.question, sid, use_llm=body.use_llm)
        chk = safety.check(a.text, session_id=sid)
    finally:
        base.TRACE.reset(token)
    return {"answer": chk.text, "method": a.method, "enough": a.enough, "sources": [s.__dict__ for s in a.sources],
            "research": a.research, "tool_calls": trace}


@app.get("/v1/health")
def health() -> dict:
    from diabetes_mas.agents.classifiers import BEST_DIR
    from diabetes_mas.agents.policy import BEST_HRL
    from diabetes_mas.dl.model import DL_DIR

    try:
        from diabetes_mas.kb.neo4j_io import driver
        with driver() as d:
            d.verify_connectivity()
        neo4j = True
    except Exception:  # noqa: BLE001
        neo4j = False
    return {"neo4j": neo4j, "ollama": LLM.healthy(), "llm_model": LLM.model,
            "models": {"classifier": (BEST_DIR / "classifier.pt").exists(), "hrl": (BEST_HRL / "hrl.pt").exists(),
                       "dl": (DL_DIR / "meta.json").exists()},
            "policy": graph.POLICY().name}

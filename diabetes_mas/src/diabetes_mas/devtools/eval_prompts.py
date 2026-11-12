"""Chạy bộ ví dụ kiểm thử prompt (tests/unit/prompts/P-xx.yaml) trên Ollama thật — tiêu chí hoàn thành 05 §5.

Chạy:  python -m diabetes_mas.devtools.eval_prompts [--only P-01,P-02] [--repeat 3]

Đo:
- P-01/P-02: tỉ lệ JSON hợp lệ của LLM ở lần gọi đầu (yêu cầu ≥ 98 %) + độ chính xác của LLM riêng và của
  cả agent (luật + LLM).
- P-03: tỉ lệ câu LLM qua `ok_question`.   P-04: không lộ tên bệnh + NLU hiểu lại đúng slot.
- P-05/P-06: mọi [n] trong context (yêu cầu 100 %), cần Neo4j cho P-05.
- P-07: luật chặn đúng + JSON hợp lệ của LLM.   P-08: truy vấn không chứa số / PII.
Kết quả: experiments/eval/prompts.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from pydantic import ValidationError

from diabetes_mas.agents import consult, llm, nlg, nlu, research, router, safety
from diabetes_mas.agents.state import SlotValue
from diabetes_mas.config import PROJECT_ROOT
from diabetes_mas.simulator.patient import DISEASE_WORDS, speak
from diabetes_mas.tools import session
from diabetes_mas.tools.pii import find_pii

CASES = PROJECT_ROOT / "tests" / "unit" / "prompts"
OUT = PROJECT_ROOT / "experiments" / "eval" / "prompts.json"


def _norm(x):
    """YAML 1.1 đọc yes/no thành bool — đổi lại thành trạng thái slot."""
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_norm(v) for v in x]
    return x


def load(pid: str) -> list[dict]:
    rows = yaml.safe_load((CASES / f"{pid}.yaml").read_text(encoding="utf-8"))
    return [{k: (v if k in ("block", "pregnant") else _norm(v)) for k, v in r.items()} for r in rows]


def _json_ok(raw: str, model=None) -> bool:
    try:
        obj = json.loads(raw)
        if model is not None:
            model.model_validate(obj)
        return isinstance(obj, dict)
    except (json.JSONDecodeError, ValidationError):
        return False


def match_slot(got: SlotValue | None, want) -> bool:
    if got is None:
        return False
    if want in ("yes", "no", "unknown"):
        return got.status == want
    if isinstance(want, (int, float)):
        return got.status == "yes" and isinstance(got.value, (int, float)) and abs(float(got.value) - want) <= 0.15
    return got.status == "yes" and got.value == want


def eval_p01(repeat: int) -> dict:
    n = valid = llm_ok = sys_ok = 0
    for ex in load("P-01"):
        slots = nlu.understand(ex["text"], use_llm=False).slots
        for _ in range(repeat):
            raw = llm.LLM.chat(llm.render("P-01", user_text=ex["text"]), json_mode=True)
            n += 1
            ok = _json_ok(raw, router.RouterOut)
            valid += ok
            llm_ok += ok and json.loads(raw)["intent"] == ex["intent"]
        sys_ok += router.classify(ex["text"], slots)[0] == ex["intent"]
    k = len(load("P-01"))
    return {"n_calls": n, "json_valid_rate": round(valid / n, 4), "llm_accuracy": round(llm_ok / n, 4),
            "agent_accuracy": round(sys_ok / k, 4), "n_examples": k}


def eval_p02(repeat: int) -> dict:
    n = valid = 0
    sys_ok = rules_ok = 0
    exs = load("P-02")
    for ex in exs:
        for _ in range(repeat):
            raw = llm.LLM.chat(llm.render("P-02", slot_catalog=nlu._catalog_text(),
                                          last_question=ex.get("last_slot") or "(không có)", user_text=ex["text"]),
                               json_mode=True)
            n += 1
            valid += _json_ok(raw)
        full = nlu.understand(ex["text"], ex.get("last_slot"), use_llm=True)
        rules = nlu.understand(ex["text"], ex.get("last_slot"), use_llm=False)
        sys_ok += all(match_slot(full.slots.get(k), v) for k, v in ex["expect"].items())
        rules_ok += all(match_slot(rules.slots.get(k), v) for k, v in ex["expect"].items())
    return {"n_calls": n, "json_valid_rate": round(valid / n, 4), "agent_accuracy": round(sys_ok / len(exs), 4),
            "rules_only_accuracy": round(rules_ok / len(exs), 4), "n_examples": len(exs)}


def eval_p03() -> dict:
    res = [nlg.question_for(ex["slot"], use_llm=True) for ex in load("P-03")]
    return {"llm_question_rate": round(sum(m == "llm" for _, m in res) / len(res), 4),
            "examples": [q for q, _ in res][:4], "n_examples": len(res)}


def eval_p04() -> dict:
    leak = understood = 0
    exs = load("P-04")
    for ex in exs:
        case = {"implicit": {ex["slot"]: {"status": ex["status"], "value": ex.get("value")}}, "explicit": {}}
        q, _ = nlg.question_for(ex["slot"], use_llm=False)
        ans = speak(case, ex["slot"], q, use_llm=True)
        leak += bool(DISEASE_WORDS.search(ans))
        want = ex.get("value", ex["status"])
        understood += match_slot(nlu.understand(ans, ex["slot"], use_llm=False).slots.get(ex["slot"]), want)
    return {"disease_name_leak_rate": round(leak / len(exs), 4), "nlu_roundtrip_accuracy": round(understood / len(exs), 4),
            "n_examples": len(exs)}


def eval_p05() -> dict:
    res = []
    for ex in load("P-05"):
        a = consult.answer(ex["question"], "eval-p05", use_llm=True, allow_research=False)
        res.append(a)
    if all(a.method == "unavailable" for a in res):
        return {"skipped": "Neo4j không chạy (kb_unavailable)", "n_examples": len(res)}
    answered = [a for a in res if a.method in ("llm", "extractive")]
    bad = [a for a in answered if not consult.citations_ok(a.text, len(a.sources))]
    return {"n_examples": len(res), "answered": len(answered), "llm": sum(a.method == "llm" for a in res),
            "extractive_fallback": sum(a.method == "extractive" for a in res),
            "refusal": sum(a.method == "refusal" for a in res), "citation_out_of_context": len(bad)}


def eval_p06() -> dict:
    from diabetes_mas.kb.criteria import apply_rules  # noqa: F401
    from diabetes_mas.tools import call

    out = []
    for i, ex in enumerate(load("P-06")):
        sid = f"eval-p06-{i}"
        session.reset(sid)
        upd = {}
        if "hba1c" in ex:
            upd["HBA1C"] = {"status": "yes", "value": ex["hba1c"]}
        if "glucose_doi" in ex:
            upd["GLUCOSE_DOI"] = {"status": "yes", "value": ex["glucose_doi"]}
        if ex.get("pregnant"):
            upd["dang_mang_thai"] = {"status": "yes"}
        call("T-05", agent_id="AG-05", session_id=sid, updates=upd)
        c = call("T-08", agent_id="AG-05", session_id=sid)
        probs = {**c.probs}
        a = consult.explain(sid, ex["top1"], probs, c.rule_hits, list(upd), use_llm=True)
        out.append((a, consult.citations_ok(a.text, len(a.sources)) or a.method == "template"))
    return {"n_examples": len(out), "llm": sum(a.method == "llm" for a, _ in out),
            "template_fallback": sum(a.method == "template" for a, _ in out),
            "citation_ok_rate": round(sum(ok for _, ok in out) / len(out), 4)}


def eval_p07() -> dict:
    exs = load("P-07")
    correct = valid = 0
    for ex in exs:
        blocked = bool(safety.rule_issues(ex["draft"]))
        correct += blocked == ex["block"]
        raw = llm.LLM.chat(llm.render("P-07", recent="", draft=ex["draft"]), json_mode=True)
        valid += _json_ok(raw)
    return {"rules_accuracy": round(correct / len(exs), 4), "llm_json_valid_rate": round(valid / len(exs), 4),
            "n_examples": len(exs)}


def eval_p08() -> dict:
    exs = load("P-08")
    clean, methods = 0, []
    for ex in exs:
        q, m = research.rewrite(ex["question"])
        methods.append(m)
        clean += not any(ch.isdigit() for ch in q) and not find_pii(q)
    return {"clean_rate": round(clean / len(exs), 4), "llm_used": methods.count("llm"), "n_examples": len(exs)}


EVALS = {"P-01": eval_p01, "P-02": eval_p02, "P-03": eval_p03, "P-04": eval_p04, "P-05": eval_p05,
         "P-06": eval_p06, "P-07": eval_p07, "P-08": eval_p08}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--repeat", type=int, default=3, help="số lần gọi LLM mỗi ví dụ để đo tỉ lệ JSON hợp lệ")
    args = ap.parse_args(argv)
    if not llm.LLM.healthy():
        print(f"[prompts] Ollama/model {llm.LLM.model} không sẵn sàng", file=sys.stderr)
        return 2
    want = [p for p in args.only.split(",") if p] or list(EVALS)
    report = {"model": llm.LLM.model, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
              "prompt_versions": {p: llm.prompt(p)[0] for p in EVALS}}
    for pid in want:
        t0 = time.time()
        fn = EVALS[pid]
        report[pid] = fn(args.repeat) if pid in ("P-01", "P-02") else fn()
        report[pid]["seconds"] = round(time.time() - t0, 1)
        print(f"[prompts] {pid}: {report[pid]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    OUT.write_text(json.dumps({**old, **report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prompts] → {OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""So sánh classifier AG-02: Naive Bayes từ xác suất KB (kb_naive_bayes) với MLP (experiments/best/classifier.pt).

Chạy:  python -m diabetes_mas.devtools.compare_classifiers

Dùng ĐÚNG các trạng thái test của rl/train_classifier.py: "masked" (che slot với p ~ U(slot_mask_prob), seed cố
định theo chỉ số ca) và "full" (thấy mọi slot), có/không áp luật KB. Kết quả: experiments/eval/classifiers.json.
"""
from __future__ import annotations

import json
import random
import sys

from diabetes_mas.agents import classifiers as C
from diabetes_mas.agents.slots import DISEASES, catalog
from diabetes_mas.config import CASES_DIR, PROJECT_ROOT, load_yaml
from diabetes_mas.kb.criteria import apply_rules
from diabetes_mas.rl.train_classifier import macro_f1, observed_state, rule_hits
from diabetes_mas.tools.dialog_tools import _nb
from diabetes_mas.utils.text import read_jsonl

OUT = PROJECT_ROOT / "experiments" / "eval" / "classifiers.json"


def score(pred_fn, states, labels, with_rules: bool) -> dict:
    preds = []
    for st in states:
        p = pred_fn(st)
        if with_rules:
            p = apply_rules(p, rule_hits(st))
        preds.append(DISEASES.index(max(p, key=p.get)))
    acc = sum(a == b for a, b in zip(preds, labels)) / len(labels)
    return {"acc": round(acc, 4), "macro_f1": round(macro_f1(labels, preds), 4)}


def main() -> int:
    cfg = load_yaml("train.yaml")
    lo, hi = cfg["classifier"]["slot_mask_prob"]
    cases = list(read_jsonl(CASES_DIR / "test.jsonl"))
    labels = [DISEASES.index(c["disease"]) for c in cases]
    masked = [observed_state(c, random.Random(i).uniform(lo, hi), random.Random(10_000 + i)) for i, c in enumerate(cases)]
    full = [observed_state(c, 0.0, random.Random(cfg["seed"])) for c in cases]
    binary = [s.id for s in catalog().slots if s.kind == "binary"]
    mlp = C.load()
    fns = {"kb_naive_bayes": lambda st: _nb(st, binary)}
    if mlp is not None:
        fns["mlp"] = lambda st: mlp.predict(st)[0]
    rep = {name: {f"{view}{'_rules' if r else ''}": score(fn, states, labels, r)
                  for view, states in (("masked", masked), ("full", full)) for r in (False, True)}
           for name, fn in fns.items()}
    rep["n_test"] = len(cases)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

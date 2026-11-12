"""Huấn luyện có giám sát AG-02 + AG-03-h trên ca bệnh mô phỏng (09_MODEL_NOTES.md §3 bước 1).

Chạy:  python -m diabetes_mas.rl.train_classifier [--epochs N]

- Mỗi epoch lấy mẫu lại phần quan sát được của từng ca: slot intake luôn biết; slot khác bị che
  (not_asked) với xác suất p ~ U(slot_mask_prob) — mô phỏng hội thoại mới hỏi được một phần.
- Early stopping theo macro-F1 trên val (mask cố định seed).
- Báo cáo: val/test với mask ngẫu nhiên, và "đủ thông tin" (không che — cận trên của hội thoại),
  trước và sau khi áp luật KB (apply_rules).
- Ghi experiments/runs/<run_id>/{classifier.pt, metrics.json, curve.csv}; sao chép sang experiments/best/.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from datetime import datetime

import torch
from torch import nn

from diabetes_mas.agents import classifiers as C
from diabetes_mas.agents.slots import DISEASES
from diabetes_mas.agents.state import DialogState, SlotValue
from diabetes_mas.config import CASES_DIR, PROJECT_ROOT, load_yaml
from diabetes_mas.kb.criteria import apply_rules
from diabetes_mas.utils.text import read_jsonl

RUNS = PROJECT_ROOT / "experiments" / "runs"


def case_slots(case: dict) -> dict[str, SlotValue]:
    full = {**case.get("implicit", {}), **case.get("explicit", {})}
    return {k: SlotValue(**v) for k, v in full.items()}


def observed_state(case: dict, p_mask: float, rng: random.Random) -> DialogState:
    """Trạng thái quan sát được: intake luôn biết, slot còn lại bị che với xác suất p_mask."""
    intake = set(load_yaml("slots.yaml")["dialog"]["intake_slots"])
    derived = set(load_yaml("slots.yaml")["derived"])
    explicit = set(case.get("explicit", {}))
    keep = {k: v for k, v in case_slots(case).items()
            if k not in derived and (k in intake or k in explicit or rng.random() >= p_mask)}
    st = DialogState(session_id=case["case_id"])
    st.update(keep)                       # suy ra tuoi_tu_45 / bmi_tu_23 từ slot số
    return st


def rule_hits(st: DialogState) -> set[str]:
    from diabetes_mas.tools.dialog_tools import _measurements
    from diabetes_mas.tools.kg_tools import check_with_meta

    meas, ctx = _measurements(st)
    hits, _ = check_with_meta(meas, ctx)
    return {h.disease_id for h in hits}


def macro_f1(y_true: list[int], y_pred: list[int], k: int = len(DISEASES)) -> float:
    f1s = []
    for c in range(k):
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / k


def confusion(y_true: list[int], y_pred: list[int], k: int = len(DISEASES)) -> list[list[int]]:
    m = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        m[t][p] += 1
    return m


@torch.no_grad()
def evaluate(model: C.ClassifierBundle, states: list[DialogState], labels: list[int], with_rules: bool) -> dict:
    model.eval()
    x, xs = C.featurize(states)
    logits, _ = model(x, xs)
    probs = torch.softmax(logits, -1).tolist()
    preds = []
    for st, p in zip(states, probs):
        pd = dict(zip(DISEASES, p))
        if with_rules:
            pd = apply_rules(pd, rule_hits(st))
        preds.append(DISEASES.index(max(pd, key=pd.get)))
    acc = sum(a == b for a, b in zip(preds, labels)) / len(labels)
    loss = nn.functional.cross_entropy(logits, torch.tensor(labels)).item()
    return {"acc": round(acc, 4), "macro_f1": round(macro_f1(labels, preds), 4), "loss": round(loss, 4),
            "confusion": confusion(labels, preds)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args(argv)
    cfg = load_yaml("train.yaml")
    cc = cfg["classifier"]
    epochs = args.epochs or cc["epochs"]
    torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])

    data = {s: list(read_jsonl(CASES_DIR / f"{s}.jsonl")) for s in ("train", "val", "test")}
    y = {s: [DISEASES.index(c["disease"]) for c in data[s]] for s in data}
    lo, hi = cc["slot_mask_prob"]
    fixed = {s: [observed_state(c, random.Random(i).uniform(lo, hi), random.Random(10_000 + i))
                 for i, c in enumerate(data[s])] for s in ("val", "test")}
    full = {s: [observed_state(c, 0.0, rng) for c in data[s]] for s in ("val", "test")}

    model = C.ClassifierBundle()
    counts = torch.bincount(torch.tensor(y["train"]), minlength=len(DISEASES)).float()
    weight = counts.sum() / (len(DISEASES) * counts)
    ce = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=cc["lr"], weight_decay=cc["weight_decay"])

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "_cls"
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    curve, best, best_f1, bad = [], None, -1.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        states = [observed_state(c, rng.uniform(lo, hi), rng) for c in data["train"]]
        order = list(range(len(states)))
        rng.shuffle(order)
        tot = 0.0
        for i in range(0, len(order), cc["batch_size"]):
            idx = order[i:i + cc["batch_size"]]
            x, xs = C.featurize([states[j] for j in idx])
            yt = torch.tensor([y["train"][j] for j in idx])
            logits, subs = model(x, xs)
            loss = ce(logits, yt) + cc["sub_loss_weight"] * sum(ce(s, yt) for s in subs) / len(subs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        val = evaluate(model, fixed["val"], y["val"], with_rules=False)
        row = {"epoch": ep, "train_loss": round(tot / len(states), 4), "val_loss": val["loss"],
               "val_acc": val["acc"], "val_macro_f1": val["macro_f1"]}
        curve.append(row)
        print(f"[cls] epoch {ep:>3}  train_loss {row['train_loss']:.4f}  val_loss {val['loss']:.4f}  "
              f"val_acc {val['acc']:.3f}  val_f1 {val['macro_f1']:.3f}")
        if val["macro_f1"] > best_f1:
            best_f1, bad, best = val["macro_f1"], 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cc["early_stopping_patience"]:
                print(f"[cls] early stopping ở epoch {ep}")
                break

    model.load_state_dict(best)
    metrics = {"run_id": run_id, "epochs_run": len(curve), "best_val_macro_f1": best_f1,
               "params": C.n_params(model), "labels": DISEASES, "config": cc}
    for s in ("val", "test"):
        for name, sts in (("masked", fixed[s]), ("full", full[s])):
            for rules in (False, True):
                metrics[f"{s}_{name}{'_rules' if rules else ''}"] = evaluate(model, sts, y[s], rules)
    meta = {k: metrics[k] for k in ("run_id", "epochs_run", "best_val_macro_f1", "params")}
    C.save(model, meta, out)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)
    C.BEST_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("classifier.pt", "classifier_meta.json"):
        shutil.copy(out / name, C.BEST_DIR / name)
    shutil.copy(out / "metrics.json", C.BEST_DIR / "classifier_metrics.json")
    for k in ("test_masked", "test_masked_rules", "test_full", "test_full_rules"):
        print(f"[cls] {k:<18} acc {metrics[k]['acc']:.3f}  macro_f1 {metrics[k]['macro_f1']:.3f}")
    print(f"[cls] {C.n_params(model)} tham số → {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

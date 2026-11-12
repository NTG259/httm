"""Huấn luyện MLP AG-13 trên SRC-KAGGLE, hai biến thể full / no_labs (09_MODEL_NOTES.md §2b).

Chạy:  python -m diabetes_mas.dl.train [--epochs N]

- Khử trùng → chia phân tầng 70/15/15 (seed 42) → z-score từ train.
- BCE có pos_weight = N_âm/N_dương; Adam; mini-batch; early stopping theo PR-AUC val.
- Ngưỡng quyết định: chọn trên val để recall ≥ target_recall, rồi tối đa precision.
- So với baseline D1 (luật ngưỡng của dataset: HbA1c ≥ 6,8 hoặc glucose ≥ 220) trên test.
- Ghi experiments/runs/<run_id>_dl/ và sao chép sang experiments/best/dl/.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime

import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

from diabetes_mas.config import PROJECT_ROOT, load_yaml
from diabetes_mas.dl.model import DL_DIR, GENDERS, NUMERIC_FULL, RiskMLP, encode_row, input_dim, numeric_cols, save

RUNS = PROJECT_ROOT / "experiments" / "runs"


def load_data(cfg: dict, seed: int) -> dict[str, pd.DataFrame]:
    df = pd.read_csv((PROJECT_ROOT / cfg["csv"]).resolve()).drop_duplicates().reset_index(drop=True)
    df = df[df["gender"].isin(GENDERS)]
    sp = cfg["split"]
    train, rest = train_test_split(df, test_size=1 - sp["train"], stratify=df["diabetes"], random_state=seed)
    val, test = train_test_split(rest, test_size=sp["test"] / (sp["val"] + sp["test"]), stratify=rest["diabetes"],
                                 random_state=seed)
    return {"train": train, "val": val, "test": test}


def tensors(df: pd.DataFrame, variant: str, scaler: dict) -> tuple[torch.Tensor, torch.Tensor]:
    rows = df.to_dict("records")
    x = torch.tensor([encode_row(r, variant, scaler)[0] for r in rows], dtype=torch.float32)
    return x, torch.tensor(df["diabetes"].values, dtype=torch.float32)


def pick_threshold(y, p, target_recall: float) -> float:
    best_t, best_prec = 0.5, -1.0
    for t in [i / 200 for i in range(1, 200)]:
        pred = (p >= t).astype(int)
        if recall_score(y, pred) >= target_recall:
            prec = precision_score(y, pred, zero_division=0)
            if prec > best_prec:
                best_t, best_prec = t, prec
    return best_t


def scores(y, p, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    return {"pr_auc": round(average_precision_score(y, p), 4), "roc_auc": round(roc_auc_score(y, p), 4),
            "f1": round(f1_score(y, pred), 4), "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred), 4), "threshold": thr}


def train_variant(variant: str, data: dict, scaler: dict, cfg: dict, seed: int, max_epochs: int):
    torch.manual_seed(seed)
    xs = {s: tensors(data[s], variant, scaler) for s in data}
    xtr, ytr = xs["train"]
    pos_weight = torch.tensor((ytr == 0).sum().item() / (ytr == 1).sum().item())
    model = RiskMLP(input_dim(variant), tuple(cfg["hidden"]), cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    g = torch.Generator().manual_seed(seed)
    curve, best, best_ap, bad = [], None, -1.0, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), generator=g)
        tot = 0.0
        for i in range(0, len(perm), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            loss = lossf(model(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            xv, yv = xs["val"]
            lv = model(xv)
            val_loss = lossf(lv, yv).item()
            ap = average_precision_score(yv.numpy(), torch.sigmoid(lv).numpy())
        curve.append({"variant": variant, "epoch": ep, "train_loss": round(tot / len(xtr), 4),
                      "val_loss": round(val_loss, 4), "val_pr_auc": round(ap, 4)})
        print(f"[dl:{variant}] epoch {ep:>2}  train_loss {tot / len(xtr):.4f}  val_loss {val_loss:.4f}  val_pr_auc {ap:.4f}")
        if ap > best_ap + 1e-4:
            best_ap, bad, best = ap, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["early_stopping_patience"]:
                print(f"[dl:{variant}] early stopping ở epoch {ep}")
                break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(xs["val"][0])).numpy()
        pt = torch.sigmoid(model(xs["test"][0])).numpy()
    thr = pick_threshold(xs["val"][1].numpy(), pv, cfg["target_recall"])
    res = {"epochs_run": len(curve), "params": sum(p.numel() for p in model.parameters()),
           "pos_weight": round(pos_weight.item(), 3), "val": scores(xs["val"][1].numpy(), pv, thr),
           "test": scores(xs["test"][1].numpy(), pt, thr)}
    return model, thr, res, curve


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args(argv)
    full_cfg = load_yaml("train.yaml")
    cfg, seed = full_cfg["dl"], full_cfg["seed"]
    data = load_data(cfg, seed)
    tr = data["train"]
    scaler = {c: {"mean": float(tr[c].mean()), "std": float(tr[c].std() or 1.0)} for c in NUMERIC_FULL}

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "_dl"
    out = RUNS / run_id
    models, meta, metrics, curves = {}, {}, {"run_id": run_id, "n": {s: len(d) for s, d in data.items()},
                                             "pos_rate": round(float(tr["diabetes"].mean()), 4)}, []
    for variant in ("full", "no_labs"):
        m, thr, res, curve = train_variant(variant, data, scaler, cfg, seed, args.epochs or cfg["epochs"])
        models[variant] = m
        meta[variant] = {"threshold": thr, "hidden": cfg["hidden"], "dropout": cfg["dropout"],
                         "features": numeric_cols(variant) + ["gender×3", "smoking_history×6"],
                         "scaler": {c: scaler[c] for c in numeric_cols(variant)}, "run_id": run_id}
        metrics[variant] = res
        curves += curve
    te = data["test"]
    d1 = ((te["HbA1c_level"] >= 6.8) | (te["blood_glucose_level"] >= 220)).astype(int)
    metrics["baseline_D1_rules_test"] = {
        "f1": round(f1_score(te["diabetes"], d1), 4), "precision": round(precision_score(te["diabetes"], d1), 4),
        "recall": round(recall_score(te["diabetes"], d1), 4)}

    save(models, meta, out)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0]))
        w.writeheader()
        w.writerows(curves)
    if DL_DIR.exists():
        shutil.rmtree(DL_DIR)
    shutil.copytree(out, DL_DIR)
    for v in ("full", "no_labs"):
        t = metrics[v]["test"]
        print(f"[dl] {v:<8} test PR-AUC {t['pr_auc']}  F1 {t['f1']}  P {t['precision']}  R {t['recall']}  thr {t['threshold']}")
    print(f"[dl] D1 luật  test F1 {metrics['baseline_D1_rules_test']['f1']}  "
          f"P {metrics['baseline_D1_rules_test']['precision']}  R {metrics['baseline_D1_rules_test']['recall']}")
    print(f"[dl] → {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Vẽ biểu đồ cho 08_EVALUATION.md → docs/assets/ (cần matplotlib, nhóm dev trong pyproject).

Chạy:  python -m diabetes_mas.devtools.plot_eval [--hrl RUN_DIR] [--cls RUN_DIR]

- curve_hrl.png        : train reward, val success, val số lượt theo epoch (mặc định experiments/best/hrl).
- curve_classifier.png : val loss, val acc, val macro-F1 theo epoch (mặc định run *_cls mới nhất).
- min_score_sweep.png  : Correct / False rejection theo ngưỡng (experiments/eval/consult_sweep.csv).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from diabetes_mas.config import PROJECT_ROOT, settings

RUNS = PROJECT_ROOT / "experiments" / "runs"
ASSETS = PROJECT_ROOT.parent / "docs" / "assets"


def read_csv(path: Path) -> list[dict[str, float]]:
    with open(path, encoding="utf-8", newline="") as f:
        return [{k: float(v) if v not in ("", "None") else float("nan") for k, v in r.items()} for r in csv.DictReader(f)]


def latest_run(suffix: str) -> Path:
    runs = sorted(p for p in RUNS.iterdir() if p.name.endswith(suffix) and (p / "curve.csv").exists())
    if not runs:
        raise FileNotFoundError(f"không có run *{suffix} trong {RUNS}")
    return runs[-1]


def plot_hrl(run: Path, out: Path, plt) -> None:
    rows = read_csv(run / "curve.csv")
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(ep, [r["train_avg_reward"] for r in rows], label="train")
    ax[0].plot(ep, [r["val_avg_reward"] for r in rows], label="val")
    ax[0].set_title("Reward trung bình / episode")
    ax[1].plot(ep, [r["val_success"] for r in rows], color="tab:green")
    ax[1].set_title("Val success (top-1 đúng)")
    ax[2].plot(ep, [r["val_avg_turns"] for r in rows], color="tab:orange")
    ax[2].set_title("Val số lượt hỏi trung bình")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    ax[0].legend()
    meta = run / "hrl_meta.json"
    name = json.loads(meta.read_text(encoding="utf-8")).get("run_id", run.name) if meta.exists() else run.name
    fig.suptitle(f"HRL — {name}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_cls(run: Path, out: Path, plt) -> None:
    rows = read_csv(run / "curve.csv")
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, [r["train_loss"] for r in rows], label="train (che slot)")
    ax[0].plot(ep, [r["val_loss"] for r in rows], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(ep, [r["val_acc"] for r in rows], label="val acc")
    ax[1].plot(ep, [r["val_macro_f1"] for r in rows], label="val macro-F1")
    ax[1].set_title("Val acc / macro-F1")
    ax[1].legend()
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    fig.suptitle(f"Classifier AG-02/03 — {run.name}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_sweep(path: Path, out: Path, plt) -> None:
    rows = read_csv(path)
    th = [r["min_score"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for a, mode, title in ((ax[0], "vector", "Vector only: enough = cosine ≥ ngưỡng"),
                           (ax[1], "hybrid", "Hybrid: (thực thể và cos ≥ min_score) hoặc cos ≥ ngưỡng")):
        a.plot(th, [r[f"{mode}_CR"] for r in rows], label="Correct rejection", color="tab:green")
        a.plot(th, [r[f"{mode}_FR"] for r in rows], label="False rejection", color="tab:red")
        a.axhline(0.10, ls=":", color="tab:red", alpha=0.6)
        a.set_title(title, fontsize=10)
        a.set_xlabel("ngưỡng (thang (1+cos)/2)")
        a.grid(alpha=0.3)
    ax[1].plot(th, [r["hybrid_gate_FR"] for r in rows], ls="--", color="tab:red", alpha=0.5, label="FR luật cũ (chỉ cổng thực thể)")
    chosen = settings()["rag"].get("min_score_no_entity")
    if chosen:
        ax[1].axvline(chosen, color="k", ls="--", alpha=0.6, label=f"chọn {chosen}")
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    fig.suptitle("Quét ngưỡng trên consult_eval.csv (66 trong + 22 ngoài phạm vi)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hrl", type=Path, default=PROJECT_ROOT / "experiments" / "best" / "hrl")
    ap.add_argument("--cls", type=Path)
    args = ap.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ASSETS.mkdir(parents=True, exist_ok=True)
    plot_hrl(args.hrl, ASSETS / "curve_hrl.png", plt)
    plot_cls(args.cls or latest_run("_cls"), ASSETS / "curve_classifier.png", plt)
    sweep = PROJECT_ROOT / "experiments" / "eval" / "consult_sweep.csv"
    if sweep.exists():
        plot_sweep(sweep, ASSETS / "min_score_sweep.png", plt)
    print(f"[plot] → {ASSETS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Huấn luyện HRL (AG-01 Main + AG-W1..W4) trên Patient Simulator (09_MODEL_NOTES.md §3 bước 2).

Chạy:  python -m diabetes_mas.rl.train_hrl [--epochs N] [--episodes M]

- 1 epoch = `episodes_per_epoch` hội thoại lấy mẫu từ train.jsonl; sau mỗi epoch đánh giá greedy trên val.
- ε-greedy giảm tuyến tính trong `decay_epochs`; double DQN + target net; Main học theo SMDP (γ^K).
- Classifier (AG-02/03) giữ cố định (`classifier_finetune: false`).
- Chọn checkpoint: val success cao nhất, hòa thì ít lượt hơn. So sánh với B0' (luật) trên val/test.
- Ghi experiments/runs/<run_id>_hrl/{hrl.pt, metrics.json, curve.csv}; chỉ sao chép sang experiments/best/hrl/
  khi val tốt hơn best hiện tại (đánh giá lại theo luật hiện hành) — `--force` để ghi đè.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from datetime import datetime
from statistics import mean

from diabetes_mas.agents.policy import BEST_HRL, HRLPolicy, RulePolicy, run_episode
from diabetes_mas.config import CASES_DIR, PROJECT_ROOT, load_yaml
from diabetes_mas.rl.env import DiagnosisEnv
from diabetes_mas.rl.networks import Replay
from diabetes_mas.utils.text import read_jsonl

RUNS = PROJECT_ROOT / "experiments" / "runs"


def evaluate(env: DiagnosisEnv, pol, cases: list[dict]) -> dict:
    res = [run_episode(env, pol, c) for c in cases]
    return {"success": round(mean(r["correct"] for r in res), 4), "avg_turns": round(mean(r["turns"] for r in res), 3),
            "avg_reward": round(mean(r["reward"] for r in res), 3), "n": len(res)}


def val_key(v: dict) -> tuple[float, float]:
    """Tiêu chí chọn checkpoint: val success cao hơn, hòa thì ít lượt hơn."""
    return v["success"], -v["avg_turns"]


def promote_if_better(env: DiagnosisEnv, val: list[dict], new_val: dict, out, force: bool = False) -> tuple[bool, str]:
    """Chỉ sao chép run mới sang experiments/best/hrl/ khi val tốt hơn checkpoint hiện tại.
    Checkpoint cũ được đánh giá lại trên CÙNG val và luật điều phối hiện hành (mask có thể đã đổi)."""
    old = HRLPolicy.load(BEST_HRL)
    if old is not None and not force:
        old_val = evaluate(env, old, val)
        if val_key(new_val) <= val_key(old_val):
            return False, f"val mới {new_val} không tốt hơn best hiện tại {old_val}"
        reason = f"val {old_val} → {new_val}"
    else:
        reason = "--force" if force else "chưa có best"
    if BEST_HRL.exists():
        shutil.rmtree(BEST_HRL)
    shutil.copytree(out, BEST_HRL)
    return True, reason


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--force", action="store_true", help="ghi đè experiments/best/hrl/ kể cả khi val không tốt hơn")
    args = ap.parse_args(argv)
    cfg_all = load_yaml("train.yaml")
    cfg, seed = cfg_all["hrl"], cfg_all["seed"]
    epochs = args.epochs or cfg["epochs"]
    per_epoch = args.episodes or cfg["episodes_per_epoch"]
    rng = random.Random(seed)
    random.seed(seed)

    train = list(read_jsonl(CASES_DIR / "train.jsonl"))
    val = list(read_jsonl(CASES_DIR / "val.jsonl"))[: cfg["val_episodes"]]
    test = list(read_jsonl(CASES_DIR / "test.jsonl"))
    env = DiagnosisEnv("train")
    pol = HRLPolicy(lr=cfg["lr"], seed=seed)
    replays = {"main": Replay(cfg["replay_size"]), **{i: Replay(cfg["replay_size"]) for i in range(4)}}
    agents = {"main": pol.main, **{i: pol.workers[i] for i in range(4)}}
    counters = {k: 0 for k in agents}
    losses: list[float] = []

    def push(key, tr):
        replays[key].add(*tr)
        counters[key] += 1
        rb = replays[key]
        if len(rb) >= max(cfg["warmup_steps"], cfg["batch_size"]) and counters[key] % cfg["train_every"] == 0:
            losses.append(agents[key].learn(rb.sample(cfg["batch_size"])))
            if counters[key] % cfg["target_update_every"] == 0:
                agents[key].sync()

    on_main = lambda s, a, r, disc, s2, m2, d: push("main", (s, a, r, disc, s2, m2, d))  # noqa: E731
    on_worker = lambda w, s, a, r, disc, s2, m2, d: push(w, (s, a, r, disc, s2, m2, d))  # noqa: E731

    baseline = {"val": evaluate(env, RulePolicy(), val), "test": evaluate(env, RulePolicy(), test)}
    print(f"[hrl] B0' val {baseline['val']}")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "_hrl"
    out = RUNS / run_id
    curve, best_key, best_ep = [], None, 0
    e0, e1 = cfg["epsilon"]["start"], cfg["epsilon"]["end"]
    t0 = time.time()
    for ep in range(1, epochs + 1):
        eps = max(e1, e0 - (e0 - e1) * (ep - 1) / max(1, cfg["epsilon"]["decay_epochs"]))
        losses.clear()
        res = [run_episode(env, pol, rng.choice(train), cfg["gamma"], eps, rng, on_main, on_worker)
               for _ in range(per_epoch)]
        v = evaluate(env, pol, val)
        row = {"epoch": ep, "epsilon": round(eps, 3), "train_avg_reward": round(mean(r["reward"] for r in res), 3),
               "train_success": round(mean(r["correct"] for r in res), 4),
               "train_avg_turns": round(mean(r["turns"] for r in res), 3),
               "loss": round(mean(losses), 4) if losses else None,
               "val_success": v["success"], "val_avg_turns": v["avg_turns"], "val_avg_reward": v["avg_reward"]}
        curve.append(row)
        print(f"[hrl] epoch {ep:>3} ε={eps:.2f} train_R {row['train_avg_reward']:7.2f} loss {row['loss']}  "
              f"val succ {v['success']:.3f} turns {v['avg_turns']:.2f} R {v['avg_reward']:.2f}  ({time.time() - t0:.0f}s)")
        key = val_key(v)
        if best_key is None or key > best_key:
            best_key, best_ep = key, ep
            pol.save(out, {"run_id": run_id, "epoch": ep, "val": v})

    best = HRLPolicy.load(out)
    metrics = {"run_id": run_id, "epochs_run": len(curve), "best_epoch": best_ep, "params": pol.n_params(),
               "config": cfg, "baseline_B0": baseline,
               "hrl": {"val": evaluate(env, best, val), "test": evaluate(env, best, test)}}
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)
    promoted, reason = promote_if_better(env, val, metrics["hrl"]["val"], out, force=args.force)
    print(f"[hrl] best epoch {best_ep}: test {metrics['hrl']['test']}  |  B0' test {baseline['test']}")
    print(f"[hrl] experiments/best/hrl: {'CẬP NHẬT' if promoted else 'GIỮ NGUYÊN'} — {reason}")
    print(f"[hrl] → {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

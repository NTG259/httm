"""Baseline chẩn đoán cho 08_EVALUATION.md §1.1: B0 Random và B1 Flat DQN.

Chạy:  python -m diabetes_mas.rl.baselines [--skip-flat] [--epochs N] [--episodes M]

- B0 Random: Main giao quyền cho worker ngẫu nhiên, worker hỏi slot ngẫu nhiên, KHÔNG RETURN/CLASSIFY khi còn hỏi
  được ⇒ hỏi tới T_max rồi phân loại (đúng định nghĩa "hỏi ngẫu nhiên T_max lượt").
- B1 Flat DQN: MỘT DQN chọn trực tiếp trong (mọi slot của W1..W4) + CLASSIFY — không phân cấp. Cùng môi trường,
  reward (T-11), luật điều phối (mask CLASSIFY `labs_ready`), siêu tham số `hrl` trong configs/train.yaml và cách
  chọn checkpoint (val success, hòa thì ít lượt) với HRL ⇒ so sánh công bằng.
Kết quả: experiments/runs/<id>_flatdqn/ và experiments/eval/baselines.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

from diabetes_mas.agents.policy import run_episode
from diabetes_mas.agents.slots import WORKERS, catalog
from diabetes_mas.agents.state import end_option, state_dim, unasked_slots
from diabetes_mas.config import CASES_DIR, PROJECT_ROOT, load_yaml
from diabetes_mas.rl.env import CLASSIFY, DiagnosisEnv
from diabetes_mas.rl.networks import DQNAgent, Replay
from diabetes_mas.rl.train_hrl import evaluate, val_key
from diabetes_mas.utils.text import read_jsonl

RUNS = PROJECT_ROOT / "experiments" / "runs"
OUT = PROJECT_ROOT / "experiments" / "eval" / "baselines.json"


class RandomPolicy:
    """B0: chọn ngẫu nhiên trong các hành động hỏi hợp lệ; chỉ RETURN/CLASSIFY khi bị bắt buộc."""
    name = "random_B0"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def main_act(self, s, mask, eps=0.0, rng=None) -> int:
        ws = [i for i in range(len(WORKERS)) if mask[i]]
        return self.rng.choice(ws) if ws else CLASSIFY

    def worker_act(self, worker_idx, s, mask, eps=0.0, rng=None) -> int:
        qs = [i for i, ok in enumerate(mask[:-1]) if ok]
        return self.rng.choice(qs) if qs else len(mask) - 1


def flat_actions() -> list[tuple[int, int]]:
    """Không gian hành động phẳng: (chỉ số worker, chỉ số slot trong worker) cho mọi slot hỏi được."""
    return [(w, j) for w, name in enumerate(WORKERS) for j in range(len(catalog().actions(name)))]


class FlatDQN:
    name = "flat_dqn_B1"

    def __init__(self, lr: float = 5e-4, seed: int = 0):
        self.acts = flat_actions()
        self.q = DQNAgent(state_dim(), len(self.acts) + 1, (256, 256), lr, seed)

    def mask(self, env: DiagnosisEnv) -> list[bool]:
        st = env.state()
        t_max = env.dialog["T_max"]
        free = {w: unasked_slots(st, name) for w, name in enumerate(WORKERS)}
        m = [st.turn < t_max and free[w][j] for w, j in self.acts]
        return m + [env.main_mask()[-1]]


def run_flat(env: DiagnosisEnv, pol: FlatDQN, case: dict, gamma: float = 0.95, eps: float = 0.0,
             rng: random.Random | None = None, on_step=None) -> dict:
    s = env.reset(case)
    total = 0.0
    top1, correct = None, False
    while True:
        m = pol.mask(env)
        a = pol.q.act(s, m, eps, rng)
        if a == len(pol.acts):                                   # CLASSIFY
            top1, correct, r = env.classify_final()
            total += r
            if on_step:
                on_step(s, a, r, 0.0, s, m, 1.0)
            break
        w, j = pol.acts[a]
        env.delegate(w)
        _, r, _ = env.worker_step(j)
        end_option(env.state())                                  # phẳng: mỗi bước một câu, trả quyền ngay
        s2 = env.vector()
        total += r
        if env.state().red_flag:
            break
        if on_step:
            on_step(s, a, r, gamma, s2, pol.mask(env), 0.0)
        s = s2
    st = env.state()
    res = {"pred": top1, "correct": correct, "turns": st.turn, "reward": total}
    env.close()
    return res


def eval_flat(env, pol, cases) -> dict:
    res = [run_flat(env, pol, c) for c in cases]
    return {"success": round(mean(r["correct"] for r in res), 4), "avg_turns": round(mean(r["turns"] for r in res), 3),
            "avg_reward": round(mean(r["reward"] for r in res), 3), "n": len(res)}


def train_flat(epochs: int, per_epoch: int, train, val, cfg, seed: int) -> tuple[FlatDQN, dict, list[dict], Path]:
    import torch

    rng = random.Random(seed)
    env = DiagnosisEnv("train")
    pol = FlatDQN(cfg["lr"], seed)
    rb, cnt, losses = Replay(cfg["replay_size"]), [0], []

    def push(*tr):
        rb.add(*tr)
        cnt[0] += 1
        if len(rb) >= max(cfg["warmup_steps"], cfg["batch_size"]) and cnt[0] % cfg["train_every"] == 0:
            losses.append(pol.q.learn(rb.sample(cfg["batch_size"])))
            if cnt[0] % cfg["target_update_every"] == 0:
                pol.q.sync()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "_flatdqn"
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    curve, best_key, best_ep = [], None, 0
    e0, e1 = cfg["epsilon"]["start"], cfg["epsilon"]["end"]
    t0 = time.time()
    for ep in range(1, epochs + 1):
        eps = max(e1, e0 - (e0 - e1) * (ep - 1) / max(1, cfg["epsilon"]["decay_epochs"]))
        losses.clear()
        res = [run_flat(env, pol, rng.choice(train), cfg["gamma"], eps, rng, push) for _ in range(per_epoch)]
        v = eval_flat(env, pol, val)
        row = {"epoch": ep, "epsilon": round(eps, 3), "train_avg_reward": round(mean(r["reward"] for r in res), 3),
               "train_success": round(mean(r["correct"] for r in res), 4),
               "loss": round(mean(losses), 4) if losses else None,
               "val_success": v["success"], "val_avg_turns": v["avg_turns"], "val_avg_reward": v["avg_reward"]}
        curve.append(row)
        print(f"[flat] epoch {ep:>3} ε={eps:.2f} val succ {v['success']:.3f} turns {v['avg_turns']:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if best_key is None or val_key(v) > best_key:
            best_key, best_ep = val_key(v), ep
            torch.save(pol.q.q.state_dict(), out / "flat_dqn.pt")
    pol.q.q.load_state_dict(torch.load(out / "flat_dqn.pt", map_location="cpu"))
    with open(out / "curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)
    return pol, {"run_id": run_id, "best_epoch": best_ep, "epochs_run": len(curve)}, curve, out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--skip-flat", action="store_true")
    args = ap.parse_args(argv)
    cfg_all = load_yaml("train.yaml")
    cfg, seed = cfg_all["hrl"], cfg_all["seed"]
    train = list(read_jsonl(CASES_DIR / "train.jsonl"))
    val = list(read_jsonl(CASES_DIR / "val.jsonl"))[: cfg["val_episodes"]]
    test = list(read_jsonl(CASES_DIR / "test.jsonl"))
    env = DiagnosisEnv("train")
    rep = {"B0_random": {"val": evaluate(env, RandomPolicy(seed), val), "test": evaluate(env, RandomPolicy(seed), test)}}
    print(f"[baselines] B0 Random {rep['B0_random']}", flush=True)
    if not args.skip_flat:
        pol, meta, _, out = train_flat(args.epochs or cfg["epochs"], args.episodes or cfg["episodes_per_epoch"],
                                       train, val, cfg, seed)
        rep["B1_flat_dqn"] = {**meta, "n_actions": len(pol.acts) + 1,
                              "params": sum(p.numel() for p in pol.q.q.parameters()),
                              "val": eval_flat(env, pol, val), "test": eval_flat(env, pol, test)}
        (out / "metrics.json").write_text(json.dumps(rep["B1_flat_dqn"], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[baselines] B1 Flat DQN {rep['B1_flat_dqn']}", flush=True)
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({**prev, **rep}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[baselines] → {OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

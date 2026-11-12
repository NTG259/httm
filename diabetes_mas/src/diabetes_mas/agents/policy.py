"""AG-01 Main Agent + AG-W1..W4 Work-Agent: policy học được (DQN) và policy dự phòng theo luật (B0').

- `HRLPolicy`: 1 DQN cho Main (5 hành động) + 1 DQN cho mỗi Worker (N_h + 1 hành động).
- `RulePolicy` (B0', 02_AGENT_DESIGN.md §8): W2 → W1 → W3 → W4, mỗi Worker hỏi theo thứ tự slot.
  Dùng khi chưa có checkpoint và làm baseline ở 08_EVALUATION.md.
- `run_episode`: vòng lặp option (SMDP) dùng chung cho huấn luyện, đánh giá và demo-mô-phỏng.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import torch

from diabetes_mas.agents.slots import WORKERS, catalog
from diabetes_mas.agents.state import state_dim
from diabetes_mas.config import PROJECT_ROOT
from diabetes_mas.rl.env import CLASSIFY, DiagnosisEnv
from diabetes_mas.rl.networks import DQNAgent
from diabetes_mas.rl.reward import option_return

BEST_HRL = PROJECT_ROOT / "experiments" / "best" / "hrl"


class RulePolicy:
    name = "rule_B0"
    ORDER = [WORKERS.index(w) for w in ("W2", "W1", "W3", "W4")]

    def main_act(self, s, mask, eps=0.0, rng=None) -> int:
        for i in self.ORDER:
            if mask[i]:
                return i
        return CLASSIFY

    def worker_act(self, worker_idx, s, mask, eps=0.0, rng=None) -> int:
        return next(i for i, ok in enumerate(mask) if ok)


class HRLPolicy:
    name = "hrl_dqn"

    def __init__(self, lr: float = 5e-4, seed: int = 0):
        d = state_dim()
        self.main = DQNAgent(d, len(WORKERS) + 1, (256, 256), lr, seed)
        self.workers = [DQNAgent(d, len(catalog().actions(w)) + 1, (256, 128), lr, seed + i + 1)
                        for i, w in enumerate(WORKERS)]

    def main_act(self, s, mask, eps=0.0, rng=None) -> int:
        return self.main.act(s, mask, eps, rng)

    def worker_act(self, worker_idx, s, mask, eps=0.0, rng=None) -> int:
        return self.workers[worker_idx].act(s, mask, eps, rng)

    def n_params(self) -> dict[str, int]:
        cnt = lambda a: sum(p.numel() for p in a.q.parameters())  # noqa: E731
        return {"main": cnt(self.main), **{w: cnt(a) for w, a in zip(WORKERS, self.workers)}}

    def save(self, out: Path, meta: dict) -> None:
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"main": self.main.q.state_dict(), "workers": [w.q.state_dict() for w in self.workers]},
                   out / "hrl.pt")
        (out / "hrl_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = BEST_HRL) -> "HRLPolicy | None":
        if not (path / "hrl.pt").exists():
            return None
        ck = torch.load(path / "hrl.pt", map_location="cpu")
        pol = cls()
        pol.main.q.load_state_dict(ck["main"])
        for w, sd in zip(pol.workers, ck["workers"]):
            w.q.load_state_dict(sd)
        return pol


def default_policy():
    """Checkpoint HRL nếu có; không có ⇒ B0' (02 §8, có cảnh báo trong log của API)."""
    return HRLPolicy.load() or RulePolicy()


def run_episode(env: DiagnosisEnv, pol, case: dict, gamma: float = 0.95, eps: float = 0.0,
                rng: random.Random | None = None, on_main: Callable | None = None,
                on_worker: Callable | None = None) -> dict:
    """Chạy một hội thoại tới khi CLASSIFY (hoặc cờ cấp cứu ở demo). Callback nhận transition để học."""
    s = env.reset(case)
    total, asked = 0.0, []
    while True:
        mm = env.main_mask()
        a = pol.main_act(s, mm, eps, rng)
        if a == CLASSIFY:
            top1, correct, r = env.classify_final()
            total += r
            if on_main:
                on_main(s, a, r, 0.0, s, mm, 1.0)
            break
        env.delegate(a)
        s0, rewards = s, []
        while True:
            wm = env.worker_mask()
            wa = pol.worker_act(a, s, wm, eps, rng)
            acts = catalog().actions(WORKERS[a])
            if wa < len(acts):
                asked.append(acts[wa])
            s2, r, done = env.worker_step(wa)
            if wa < len(acts):
                rewards.append(r)
            total += r
            nxt = env.worker_mask() if not done else [False] * len(wm)
            if on_worker:
                on_worker(a, s, wa, r, gamma, s2, nxt, 1.0 if done else 0.0)
            s = s2
            if done:
                break
        R, K = option_return(rewards, gamma)
        if on_main:
            on_main(s0, a, R, gamma ** K, s, env.main_mask(), 0.0)
        if env.state().red_flag:
            top1, correct = None, False
            break
    st = env.state()
    res = {"case_id": case.get("case_id"), "disease": case.get("disease"), "pred": top1, "correct": correct,
           "turns": st.turn, "reward": total, "asked": asked, "entropy": st.entropy}
    env.close()
    return res

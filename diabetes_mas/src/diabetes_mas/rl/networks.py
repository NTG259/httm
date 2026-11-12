"""Mạng Q cho AG-01 Main và AG-W1..W4 (09_MODEL_NOTES.md §2) + replay buffer + DQN có mask.

- Main: Linear(d_s,256) → ReLU → Linear(256,256) → ReLU → Linear(256, 5)          (2 lớp ẩn)
- Worker h: Linear(d_s,256) → ReLU → Linear(256,128) → ReLU → Linear(128, N_h+1)  (2 lớp ẩn)
  Mỗi Worker một mạng riêng nên không cần one-hot h ở đầu vào.
Double DQN + target network + Huber loss. Hành động bị mask nhận Q = −∞ khi chọn và khi lấy max ở s'.
"""
from __future__ import annotations

import random
from collections import deque

import torch
from torch import nn

NEG = -1e9


def mlp(d_in: int, hidden: tuple[int, ...], d_out: int) -> nn.Sequential:
    layers, prev = [], d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    return nn.Sequential(*layers, nn.Linear(prev, d_out))


class Replay:
    """(s, a, r, discount, s', mask', done) — discount = γ^K cho option của Main, γ cho Worker."""

    def __init__(self, size: int):
        self.buf: deque = deque(maxlen=size)

    def add(self, *tr) -> None:
        self.buf.append(tr)

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, disc, s2, m2, d = zip(*batch)
        f = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
        return f(s), torch.tensor(a), f(r), f(disc), f(s2), torch.tensor(m2, dtype=torch.bool), f(d)

    def __len__(self) -> int:
        return len(self.buf)


class DQNAgent:
    def __init__(self, d_in: int, n_actions: int, hidden: tuple[int, ...], lr: float, seed: int = 0):
        torch.manual_seed(seed)
        self.q = mlp(d_in, hidden, n_actions)
        self.target = mlp(d_in, hidden, n_actions)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.n_actions = n_actions
        self.hidden = hidden

    @torch.no_grad()
    def act(self, s: list[float], mask: list[bool], eps: float = 0.0, rng: random.Random | None = None) -> int:
        valid = [i for i, ok in enumerate(mask) if ok]
        if rng is not None and rng.random() < eps:
            return rng.choice(valid)
        q = self.q(torch.tensor([s], dtype=torch.float32))[0]
        q[~torch.tensor(mask)] = NEG
        return int(q.argmax())

    def learn(self, batch) -> float:
        s, a, r, disc, s2, m2, d = batch
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q2_online = self.q(s2).masked_fill(~m2, NEG)
            a2 = q2_online.argmax(1, keepdim=True)                   # double DQN: chọn bằng mạng online
            q2 = self.target(s2).gather(1, a2).squeeze(1)
            q2 = torch.where(m2.any(1), q2, torch.zeros_like(q2))
            y = r + disc * (1 - d) * q2
        loss = nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        return loss.item()

    def sync(self) -> None:
        self.target.load_state_dict(self.q.state_dict())

"""AG-11 Reward Generator — T-11 compute_reward (công thức chốt ở 02_AGENT_DESIGN.md §5).

Mỗi câu hỏi: r_t = r_turn + r_hit·[phát hiện dương tính trên slot y khoa] + λ·(H(p_{t-1}) − H(p_t))
Khi CLASSIFY: R_success | R_fail.

- "Phát hiện dương tính" (is_hit) chỉ tính slot gắn với KB: Symptom/RiskFactor trả 'yes', LabTest có giá trị.
  Slot hồ sơ (tuổi, BMI, giới tính, hút thuốc, bệnh tim) KHÔNG được thưởng hit — tránh để policy
  lấp lượt bằng câu hỏi nhân khẩu học luôn có câu trả lời.
- Không có phạt hỏi lặp: action mask đã chặn (FR-04).
- Tham số mặc định ở RewardConfig; Giai đoạn 5 sẽ nạp từ configs/train.yaml.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from diabetes_mas.agents.slots import catalog

HIT_SOURCES = {"Symptom", "RiskFactor", "LabTest"}


@dataclass(frozen=True)
class RewardConfig:
    turn: float = -1.0
    hit: float = 1.0
    entropy_lambda: float = 2.0
    success: float = 20.0
    fail: float = -20.0


def entropy(probs: dict[str, float] | list[float]) -> float:
    ps = probs.values() if isinstance(probs, dict) else probs
    return -sum(p * math.log(p) for p in ps if p > 0)


def is_hit(slot_id: str, status: str, value: float | str | None) -> bool:
    """Câu trả lời có phải phát hiện dương tính trên slot y khoa không."""
    slot = catalog().by_id().get(slot_id)
    if slot is None or slot.source not in HIT_SOURCES:
        return False
    if slot.kind == "numeric":
        return status == "yes" and isinstance(value, (int, float))
    return status == "yes"


def compute_reward(prev_entropy: float, cur_entropy: float, asked: bool, hit: bool,
                   terminal: bool, correct: bool | None, cfg: RewardConfig = RewardConfig()) -> tuple[float, dict[str, float]]:
    """Trả (reward, các thành phần). asked=False ở bước CLASSIFY (không có câu hỏi)."""
    parts: dict[str, float] = {}
    if asked:
        parts["turn"] = cfg.turn
        parts["hit"] = cfg.hit if hit else 0.0
        parts["entropy"] = cfg.entropy_lambda * (prev_entropy - cur_entropy)
    if terminal:
        parts["terminal"] = cfg.success if correct else cfg.fail
    return sum(parts.values()), parts


def option_return(rewards: list[float], gamma: float) -> tuple[float, int]:
    """Phần thưởng tích lũy của một option (một lần Main giao quyền cho Worker): Σ γ^k r_k và số bước K.
    Main Agent cập nhật theo SMDP: y = R + γ^K · max_a' Q(s', a')."""
    total = sum((gamma ** k) * r for k, r in enumerate(rewards))
    return total, len(rewards)

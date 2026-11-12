"""MLP dự đoán nguy cơ ĐTĐ + mã hóa đặc trưng + suy luận cho T-15.

Đặc trưng: 6 số z-score (thống kê từ train) + one-hot gender (3) + one-hot smoking_history (6) = 15 chiều.
Biến thể `no_labs` bỏ HbA1c_level, blood_glucose_level → 13 chiều (D3-mask, 08_EVALUATION.md §1.3).
Thiếu giá trị lúc suy luận (hội thoại chưa hỏi): số → trung bình train (z = 0); gender → vector 0;
mỗi lần điền được ghi vào `assumptions` để AG-07 giải thích.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from diabetes_mas.config import PROJECT_ROOT

NUMERIC_FULL = ["age", "bmi", "HbA1c_level", "blood_glucose_level", "hypertension", "heart_disease"]
LABS = ("HbA1c_level", "blood_glucose_level")
GENDERS = ["Female", "Male", "Other"]
SMOKING = ["never", "former", "current", "not current", "ever", "No Info"]
DL_DIR = PROJECT_ROOT / "experiments" / "best" / "dl"


def numeric_cols(variant: str) -> list[str]:
    return [c for c in NUMERIC_FULL if variant == "full" or c not in LABS]


def input_dim(variant: str) -> int:
    return len(numeric_cols(variant)) + len(GENDERS) + len(SMOKING)


class RiskMLP(nn.Module):
    """Linear(d→64) → ReLU → Dropout → Linear(64→32) → ReLU → Dropout → Linear(32→1); sigmoid ở ngoài."""

    def __init__(self, d_in: int, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.2):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def encode_row(row: dict, variant: str, scaler: dict) -> tuple[list[float], list[str]]:
    """Một bản ghi (dict theo tên cột dataset) → vector; trả kèm danh sách giá trị đã điền."""
    filled: list[str] = []
    vec: list[float] = []
    for c in numeric_cols(variant):
        v = row.get(c)
        if v is None:
            filled.append(c)
            vec.append(0.0)
        else:
            vec.append((float(v) - scaler[c]["mean"]) / scaler[c]["std"])
    g = row.get("gender")
    if g is None:
        filled.append("gender")
    vec += [float(g == x) for x in GENDERS]
    s = row.get("smoking_history") or "No Info"
    vec += [float(s == x) for x in SMOKING]
    return vec, filled


class Predictor:
    """Gói 2 biến thể; gọi được như PREDICTOR của T-15: (variant, PredictIn) → (risk, threshold)."""

    def __init__(self, models: dict[str, RiskMLP], meta: dict):
        self.models, self.meta = models, meta

    @torch.no_grad()
    def __call__(self, variant: str, feats) -> tuple[float, float]:
        risk, thr, _ = self.predict(variant, feats.model_dump())
        return risk, thr

    @torch.no_grad()
    def predict(self, variant: str, row: dict) -> tuple[float, float, list[str]]:
        m = self.meta[variant]
        x, filled = encode_row(row, variant, m["scaler"])
        p = torch.sigmoid(self.models[variant].eval()(torch.tensor([x]))).item()
        return round(p, 4), m["threshold"], filled


def save(models: dict[str, RiskMLP], meta: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for v, mdl in models.items():
        torch.save(mdl.state_dict(), out / f"{v}.pt")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


@lru_cache(maxsize=1)
def load(path: Path = DL_DIR) -> Predictor | None:
    if not (path / "meta.json").exists():
        return None
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    models = {}
    for v in ("full", "no_labs"):
        mdl = RiskMLP(input_dim(v), tuple(meta[v]["hidden"]), meta[v]["dropout"])
        mdl.load_state_dict(torch.load(path / f"{v}.pt", map_location="cpu"))
        models[v] = mdl
    return Predictor(models, meta)

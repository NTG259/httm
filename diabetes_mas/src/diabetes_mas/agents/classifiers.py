"""AG-02 Main Classifier + AG-03-h Sub-Classifier (MLP PyTorch, 09_MODEL_NOTES.md §2).

Đầu vào:
- Sub-classifier h: mã hóa các slot thuộc nhóm của Worker h (`group_slots(h)`).
- Main classifier: phần slot của s_t (`encode_slots`, 109 chiều) + 4 × 5 xác suất của sub-classifier.
  KHÔNG dùng toàn bộ s_t (122 chiều) vì s_t chứa p_t/H(p_t) do chính classifier sinh ra (vòng lặp) và
  turn/worker — thông tin điều phối không phải bằng chứng y khoa. Sai khác so với bản thiết kế ghi ở
  09_MODEL_NOTES.md §2.
Sau MLP, luật KB được áp dụng (`kb.criteria.apply_rules`) trong T-08 — giống biến thể Naive Bayes.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from diabetes_mas.agents.slots import DISEASES, WORKERS, catalog
from diabetes_mas.agents.state import DialogState, encode_slots
from diabetes_mas.config import PROJECT_ROOT, load_yaml

BEST_DIR = PROJECT_ROOT / "experiments" / "best"


@lru_cache(maxsize=None)
def group_slots(worker: str) -> list[str]:
    """Slot của nhóm h: slot Worker h hỏi được + cờ suy ra cùng nhóm KB (vd. tuoi_tu_45 thuộc W3)."""
    groups = set(load_yaml("slots.yaml")["workers"][worker].get("kb_groups", []))
    return [s.id for s in catalog().slots if s.worker == worker or (s.worker is None and s.group in groups
                                                                     and s.source == "derived")]


def sub_dim(worker: str) -> int:
    return len(encode_slots(DialogState(session_id="_"), group_slots(worker)))


def slots_dim() -> int:
    return len(encode_slots(DialogState(session_id="_")))


class SubClassifier(nn.Module):
    """AG-03-h: Linear → 64 → ReLU → Dropout(0.2) → 5 (1 lớp ẩn)."""

    def __init__(self, d_in: int, hidden: int = 64, n_out: int = len(DISEASES)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


class MainClassifier(nn.Module):
    """AG-02: Linear → 256 → ReLU → Dropout(0.3) → 128 → ReLU → 5 (2 lớp ẩn)."""

    def __init__(self, d_in: int, n_out: int = len(DISEASES)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_out))

    def forward(self, x):
        return self.net(x)


class ClassifierBundle(nn.Module):
    """Gói 4 sub-classifier + main classifier; forward trả (logits_main, [logits_sub_h])."""

    def __init__(self):
        super().__init__()
        self.subs = nn.ModuleDict({w: SubClassifier(sub_dim(w)) for w in WORKERS})
        self.main = MainClassifier(slots_dim() + len(WORKERS) * len(DISEASES))

    def forward(self, x_slots: torch.Tensor, x_subs: dict[str, torch.Tensor]):
        sub_logits = [self.subs[w](x_subs[w]) for w in WORKERS]
        sub_probs = [torch.softmax(l, dim=-1) for l in sub_logits]
        return self.main(torch.cat([x_slots, *sub_probs], dim=-1)), sub_logits


def featurize(states: list[DialogState]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x = torch.tensor([encode_slots(s) for s in states], dtype=torch.float32)
    xs = {w: torch.tensor([encode_slots(s, group_slots(w)) for s in states], dtype=torch.float32) for w in WORKERS}
    return x, xs


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class MLPClassifier:
    """Suy luận cho T-08 (variant = "mlp")."""

    def __init__(self, model: ClassifierBundle, meta: dict):
        self.model, self.meta = model.eval(), meta

    @torch.no_grad()
    def predict(self, state: DialogState) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        x, xs = featurize([state])
        logits, subs = self.model(x, xs)
        p = torch.softmax(logits, -1)[0].tolist()
        sub = {w: dict(zip(DISEASES, torch.softmax(l, -1)[0].tolist())) for w, l in zip(WORKERS, subs)}
        return dict(zip(DISEASES, p)), sub


def save(model: ClassifierBundle, meta: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "classifier.pt"
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    (out_dir / "classifier_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load(path: Path | None = None) -> MLPClassifier | None:
    path = path or BEST_DIR / "classifier.pt"
    if not path.exists():
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ClassifierBundle()
    model.load_state_dict(ck["state_dict"])
    return MLPClassifier(model, ck["meta"])

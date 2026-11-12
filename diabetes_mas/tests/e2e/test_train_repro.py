"""TC-46: huấn luyện HRL rút gọn — mỗi epoch 1 dòng curve.csv, có checkpoint, cùng seed ⇒ cùng số liệu (NFR-03).

QUAN TRỌNG: `train_hrl.main` ghi đè `experiments/best/hrl/` ⇒ test chuyển RUNS và BEST_HRL sang thư mục tạm.
"""
import csv

import pytest

from diabetes_mas.rl import train_hrl

pytestmark = pytest.mark.slow


def _train(tmp_path, monkeypatch, tag: str) -> list[dict]:
    runs, best = tmp_path / f"runs_{tag}", tmp_path / f"best_{tag}"
    monkeypatch.setattr(train_hrl, "RUNS", runs)
    monkeypatch.setattr(train_hrl, "BEST_HRL", best)
    monkeypatch.setattr(train_hrl, "PROJECT_ROOT", tmp_path)          # chỉ dùng để in đường dẫn tương đối
    assert train_hrl.main(["--epochs", "2", "--episodes", "40"]) == 0
    assert (best / "hrl.pt").exists() and (best / "hrl_meta.json").exists()
    with open(best / "curve.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_tc46_short_hrl_training_is_reproducible(tmp_path, monkeypatch):
    a = _train(tmp_path, monkeypatch, "a")
    b = _train(tmp_path, monkeypatch, "b")
    assert len(a) == 2 and {"epoch", "val_avg_reward", "val_success"} <= set(a[0])
    assert a == b                                                   # cùng seed ⇒ cùng đường cong

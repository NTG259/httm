"""GĐ 8: train_hrl chỉ ghi đè experiments/best/hrl/ khi val tốt hơn (lỗi GĐ 7 — 07_TEST_PLAN.md §6)."""
from diabetes_mas.agents.policy import HRLPolicy
from diabetes_mas.rl import train_hrl


def _setup(tmp_path, monkeypatch, old_val):
    best, run = tmp_path / "best", tmp_path / "run"
    HRLPolicy().save(best, {"tag": "old"})
    HRLPolicy().save(run, {"tag": "new"})
    monkeypatch.setattr(train_hrl, "BEST_HRL", best)
    monkeypatch.setattr(train_hrl, "evaluate", lambda env, pol, cases: old_val)
    return best, run


def test_keeps_best_when_new_val_not_better(tmp_path, monkeypatch):
    best, run = _setup(tmp_path, monkeypatch, {"success": 0.75, "avg_turns": 4.0})
    ok, _ = train_hrl.promote_if_better(None, [], {"success": 0.75, "avg_turns": 4.5}, run)
    assert not ok and '"old"' in (best / "hrl_meta.json").read_text(encoding="utf-8")


def test_promotes_when_better_or_forced(tmp_path, monkeypatch):
    best, run = _setup(tmp_path, monkeypatch, {"success": 0.75, "avg_turns": 4.0})
    ok, _ = train_hrl.promote_if_better(None, [], {"success": 0.75, "avg_turns": 3.5}, run)   # hòa success, ít lượt hơn
    assert ok and '"new"' in (best / "hrl_meta.json").read_text(encoding="utf-8")
    best, run = _setup(tmp_path, monkeypatch, {"success": 0.9, "avg_turns": 3.0})
    assert train_hrl.promote_if_better(None, [], {"success": 0.1, "avg_turns": 9.0}, run, force=True)[0]

"""quantai.evolution.dataset_builder.PreferenceBuilder 测试。"""

from __future__ import annotations

import json

from quantai.config.schema import EvolutionConfig
from quantai.evolution.dataset_builder import PreferenceBuilder


def _setup(tmp_path, min_diff=100.0):
    pb = PreferenceBuilder(save_dir=str(tmp_path / "pref"), min_pnl_diff=min_diff)
    pb.log_decision("t1", {"ticker": "NVDA"}, "BUY", "strong uptrend", "scalper")
    pb.log_decision("t2", {"ticker": "NVDA"}, "SELL", "weak signal", "scalper")
    pb.log_outcome("t1", pnl=500.0, hold_bars=3, exit_reason="tp")
    pb.log_outcome("t2", pnl=-200.0, hold_bars=2, exit_reason="sl")
    return pb


def test_log_decision_outcome_writes_completed(tmp_path):
    pb = _setup(tmp_path)
    rows = [json.loads(x) for x in pb.completed_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 2
    assert all("outcome" in r for r in rows)


def test_pending_cleared_after_outcome(tmp_path):
    pb = _setup(tmp_path)
    assert pb.pending_decisions == {}


def test_outcome_without_decision_ignored(tmp_path):
    pb = PreferenceBuilder(save_dir=str(tmp_path / "pref"))
    pb.log_outcome("ghost", pnl=1.0, hold_bars=1, exit_reason="x")  # 不应抛
    assert not pb.completed_file.exists()


def test_generate_preference_pairs(tmp_path):
    pb = _setup(tmp_path)
    pairs = pb.generate_preference_pairs()
    assert len(pairs) == 1
    p = pairs[0]
    assert p["chosen"]["decision"] == "BUY" and p["chosen"]["pnl"] == 500.0
    assert p["rejected"]["decision"] == "SELL" and p["rejected"]["pnl"] == -200.0


def test_pairs_below_threshold_skipped(tmp_path):
    pb = PreferenceBuilder(save_dir=str(tmp_path / "pref"), min_pnl_diff=10000.0)
    pb.log_decision("t1", {"ticker": "NVDA"}, "BUY", "a", "scalper")
    pb.log_decision("t2", {"ticker": "NVDA"}, "SELL", "b", "scalper")
    pb.log_outcome("t1", pnl=10.0, hold_bars=1, exit_reason="x")
    pb.log_outcome("t2", pnl=5.0, hold_bars=1, exit_reason="x")
    assert pb.generate_preference_pairs() == []


def test_export_pairs(tmp_path):
    pb = _setup(tmp_path)
    out = pb.export_pairs()
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 1 and "chosen" in rows[0]


def test_from_config(tmp_path):
    cfg = EvolutionConfig(preferences_dir=str(tmp_path / "p"), min_pnl_diff=42.0)
    pb = PreferenceBuilder.from_config(cfg)
    assert pb.min_pnl_diff == 42.0

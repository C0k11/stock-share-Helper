"""quantai.evolution.collector.ExperienceCollector 测试（含 B-1 占位断言）。"""

from __future__ import annotations

import pytest

from quantai.config.schema import EvolutionConfig
from quantai.evolution.collector import ExperienceCollector
from quantai.evolution.dataset_builder import PreferenceBuilder


def _collector(tmp_path, **kw):
    return ExperienceCollector(experiences_dir=str(tmp_path / "exp"), **kw)


def test_on_trade_complete_collects_and_metrics(tmp_path):
    c = _collector(tmp_path)
    c.on_trade_complete(
        trade_id="t1", state={"x": 1}, action="BUY", pnl=100.0, drawdown_pct=0.0, hold_bars=2, exit_reason="tp"
    )
    assert c.trade_count == 1
    assert len(c.experience_buffer) == 1
    m = c.get_metrics()
    assert m["total_trades"] == 1.0 and m["mean_win_rate"] == 1.0


def test_disabled_collector_noop(tmp_path):
    c = _collector(tmp_path, enabled=False)
    c.on_trade_complete(
        trade_id="t1", state={}, action="BUY", pnl=1.0, drawdown_pct=0.0, hold_bars=1, exit_reason="x"
    )
    assert c.trade_count == 0 and len(c.experience_buffer) == 0


def test_on_step_and_joint_step(tmp_path):
    c = _collector(tmp_path)
    c.on_step(state={"a": 1}, action="HOLD", reward=0.1)
    c.on_joint_step(state={"a": 1}, action="BUY", reward=0.2)
    assert len(c.step_buffer) == 1 and len(c.joint_buffer) == 1


def test_feeds_preference_builder(tmp_path):
    pb = PreferenceBuilder(save_dir=str(tmp_path / "pref"))
    pb.log_decision("t1", {"ticker": "NVDA"}, "BUY", "r", "scalper")
    c = _collector(tmp_path, preferences=pb)
    c.on_trade_complete(
        trade_id="t1", state={}, action="BUY", pnl=50.0, drawdown_pct=0.0, hold_bars=1, exit_reason="tp"
    )
    assert pb.completed_file.exists()  # outcome 回填进偏好构建器


def test_should_prepare_and_prepare_batch(tmp_path):
    c = _collector(tmp_path, min_experiences=2, update_interval_trades=1)
    for i in range(3):
        c.on_trade_complete(
            trade_id=f"t{i}", state={"i": i}, action="BUY", pnl=10.0, drawdown_pct=0.0, hold_bars=1, exit_reason="x"
        )
    assert c.should_prepare_update() is True
    out = c.prepare_offline_batch(batch_size=2)
    assert len(out["batch"]) == 2
    assert out["pairs"] == []  # 无偏好构建器


def test_online_gradient_step_not_implemented(tmp_path):
    # B-1：在线实时梯度是显式未实现占位
    c = _collector(tmp_path)
    assert c.online_gradient_enabled is False
    with pytest.raises(NotImplementedError):
        c.online_gradient_step()


def test_from_config(tmp_path):
    cfg = EvolutionConfig(experiences_dir=str(tmp_path / "e"), reward_pnl_scale=0.01)
    c = ExperienceCollector.from_config(cfg)
    assert c.reward_shaper.pnl_scale == 0.01
    assert c.online_gradient_enabled is False

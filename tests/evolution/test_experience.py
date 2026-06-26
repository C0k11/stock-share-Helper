"""quantai.evolution.experience 测试（ExperienceBuffer + RewardShaper）。"""

from __future__ import annotations

import numpy as np

from quantai.evolution.experience import ExperienceBuffer, RewardShaper


def test_add_and_persist_and_reload(tmp_path):
    d = str(tmp_path / "exp")
    buf = ExperienceBuffer(max_size=100, save_dir=d, file_name="e.jsonl")
    buf.add(state={"x": 1}, action="BUY", reward=0.5)
    buf.add(state={"x": 2}, action="SELL", reward=-0.2)
    assert len(buf) == 2
    # 重新打开同目录 -> 回载历史
    buf2 = ExperienceBuffer(max_size=100, save_dir=d, file_name="e.jsonl")
    assert len(buf2) == 2
    assert buf2.buffer[-1]["action"] == "SELL"


def test_maxlen_eviction(tmp_path):
    buf = ExperienceBuffer(max_size=2, save_dir=str(tmp_path / "e"), file_name="e.jsonl")
    for i in range(5):
        buf.add(state={"i": i}, action="HOLD", reward=float(i))
    assert len(buf) == 2  # 只留最近 2 条（内存）


def test_sample_seeded_reproducible(tmp_path):
    buf = ExperienceBuffer(max_size=100, save_dir=str(tmp_path / "e"), file_name="e.jsonl")
    for i in range(20):
        buf.add(state={"i": i}, action="HOLD", reward=float(i))
    s1 = buf.sample(5, rng=np.random.default_rng(0))
    s2 = buf.sample(5, rng=np.random.default_rng(0))
    assert [x["reward"] for x in s1] == [x["reward"] for x in s2]


def test_sample_smaller_than_batch_returns_all(tmp_path):
    buf = ExperienceBuffer(max_size=100, save_dir=str(tmp_path / "e"), file_name="e.jsonl")
    buf.add(state={}, action="HOLD", reward=1.0)
    assert len(buf.sample(32)) == 1


def test_reward_shaper_components():
    shaper = RewardShaper()
    reward, comp = shaper.compute_reward(
        realized_pnl=1000.0, unrealized_pnl=0.0, drawdown_pct=-0.10, action_taken="BUY"
    )
    assert comp["pnl"] == 1000.0 * 0.001
    assert comp["drawdown"] < 0  # 超过 -0.05 触发回撤惩罚
    assert comp["action_quality"] == 0.1  # BUY + 盈利
    assert abs(reward - sum(comp.values())) < 1e-9


def test_reward_shaper_hold_quality():
    shaper = RewardShaper()
    _, comp = shaper.compute_reward(realized_pnl=0.0, unrealized_pnl=10.0, drawdown_pct=0.0, action_taken="HOLD")
    assert comp["action_quality"] == 0.02
    assert comp["drawdown"] == 0.0

"""quantai.evolution.recorder.EvolutionRecorder 测试。"""

from __future__ import annotations

import json

from quantai.config.schema import EvolutionConfig
from quantai.evolution.recorder import EvolutionRecorder


def _read_all(rec: EvolutionRecorder):
    fp = rec._current_file()
    return [json.loads(x) for x in fp.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_record_returns_id_and_writes(tmp_path):
    rec = EvolutionRecorder(trajectories_dir=str(tmp_path / "traj"))
    rid = rec.record(agent_id="scalper", context="ctx", action="BUY", outcome=None, feedback="")
    assert isinstance(rid, str) and len(rid) > 0
    rows = _read_all(rec)
    assert rows[-1]["type"] == "trajectory" and rows[-1]["id"] == rid
    assert rows[-1]["agent_id"] == "scalper" and rows[-1]["action"] == "BUY"


def test_log_outcome_and_feedback(tmp_path):
    rec = EvolutionRecorder(trajectories_dir=str(tmp_path / "traj"))
    rid = rec.record(agent_id="a", context="c", action="SELL")
    rec.log_outcome(ref_id=rid, outcome=123.5, comment="pnl")
    rec.log_feedback(ref_id=rid, score=1, comment="good")
    rows = _read_all(rec)
    types = [r["type"] for r in rows]
    assert "outcome" in types and "feedback" in types
    oc = [r for r in rows if r["type"] == "outcome"][0]
    assert oc["ref_id"] == rid and oc["outcome"] == 123.5


def test_update_reward_removed():
    # C-3：旧 no-op 方法不应被迁移
    assert not hasattr(EvolutionRecorder, "update_reward")


def test_from_config(tmp_path):
    cfg = EvolutionConfig(trajectories_dir=str(tmp_path / "t"))
    rec = EvolutionRecorder.from_config(cfg)
    assert rec.trajectory_dir.exists()

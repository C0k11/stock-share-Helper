"""quantai.evolution.trainer.EvolutionTrainer 测试（纯逻辑 + 注入假 runner，不依赖 GPU）。"""

from __future__ import annotations

from quantai.evolution.dataset_builder import PreferenceBuilder
from quantai.evolution.trainer import EvolutionTrainer


def _trainer(tmp_path, min_diff=100.0):
    pb = PreferenceBuilder(save_dir=str(tmp_path / "pref"), min_pnl_diff=min_diff)
    return EvolutionTrainer(
        preferences=pb,
        adapters_dir=str(tmp_path / "adapters"),
        active_pointer=str(tmp_path / "active.json"),
    )


def _seed_pairs(pb):
    pb.log_decision("t1", {"ticker": "NVDA", "price": 100, "regime": "aggressive"}, "BUY", "uptrend", "scalper")
    pb.log_decision("t2", {"ticker": "NVDA", "price": 100}, "SELL", "weak", "scalper")
    pb.log_outcome("t1", pnl=500.0, hold_bars=3, exit_reason="tp")
    pb.log_outcome("t2", pnl=-200.0, hold_bars=2, exit_reason="sl")


def test_to_dpo_records_pure():
    pairs = [
        {
            "context": {"ticker": "NVDA", "price": 100, "regime": "aggressive"},
            "chosen": {"decision": "BUY", "reasoning": "good"},
            "rejected": {"decision": "SELL", "reasoning": "bad"},
        }
    ]
    recs = EvolutionTrainer.to_dpo_records(pairs)
    assert len(recs) == 1
    r = recs[0]
    assert set(r) == {"prompt", "chosen", "rejected"}
    assert "NVDA" in r["prompt"] and r["chosen"] == "BUY: good" and r["rejected"] == "SELL: bad"


def test_to_dpo_records_drops_empty_or_identical():
    pairs = [
        {"context": {}, "chosen": {"decision": "BUY", "reasoning": "x"}, "rejected": {}},
        {"context": {}, "chosen": {"decision": "HOLD"}, "rejected": {"decision": "HOLD"}},
    ]
    assert EvolutionTrainer.to_dpo_records(pairs) == []


def test_active_pointer_set_get(tmp_path):
    tr = _trainer(tmp_path)
    assert tr.get_active_adapters() == {}
    tr.set_active_adapter("scalper", "scalper_v2")
    tr.set_active_adapter("analyst", "analyst_v1")
    active = tr.get_active_adapters()
    assert active == {"scalper": "scalper_v2", "analyst": "analyst_v1"}


def test_build_dataset(tmp_path):
    tr = _trainer(tmp_path)
    _seed_pairs(tr.preferences)
    out = tr.build_dataset()
    assert out.exists()


def test_train_offline_with_fake_runner(tmp_path):
    tr = _trainer(tmp_path)
    _seed_pairs(tr.preferences)

    captured = {}

    class FakeRunner:
        def __init__(self, **kw):
            captured["init"] = kw
            self.trained = None

        def train(self, records):
            self.trained = records
            captured["runner"] = self

    res = tr.train_offline(role="scalper", runner_factory=lambda **kw: FakeRunner(**kw))
    assert res["trained"] is True and res["records"] == 1
    assert captured["runner"].trained[0]["chosen"] == "BUY: uptrend"
    assert tr.get_active_adapters()["scalper"] == "scalper"
    assert "scalper" in captured["init"]["output_dir"]


def test_train_offline_no_pairs_skips(tmp_path):
    tr = _trainer(tmp_path)  # 无 completed 决策 -> 无偏好对

    def factory(**kw):
        raise AssertionError("不应在无数据时构造 runner")

    res = tr.train_offline(role="scalper", runner_factory=factory)
    assert res["trained"] is False and res["records"] == 0

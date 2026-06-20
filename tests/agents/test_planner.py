"""quantai.agents.planner 测试（规则路径纯逻辑，不下 SFT 模型/不碰 torch）。"""

from __future__ import annotations

from quantai.agents.base import Regime
from quantai.agents.planner import (
    Planner,
    assess_regime_rule,
    risk_budget_for,
    strategy_to_regime,
)
from quantai.config import AppConfig


def _feat(vol=20.0, ret_5d=0.0):
    return {"technical": {"volatility_20d": vol, "return_5d": ret_5d}}


def test_assess_regime_rule_aggressive():
    assert assess_regime_rule(_feat(vol=20.0, ret_5d=2.0)) == Regime.AGGRESSIVE


def test_assess_regime_rule_defensive_on_vol():
    assert assess_regime_rule(_feat(vol=90.0, ret_5d=0.0)) == Regime.DEFENSIVE


def test_assess_regime_rule_defensive_on_drawdown():
    assert assess_regime_rule(_feat(vol=20.0, ret_5d=-6.0)) == Regime.DEFENSIVE


def test_assess_regime_rule_cash_on_high_vol():
    assert assess_regime_rule(_feat(vol=130.0)) == Regime.CASH_PRESERVATION


def test_assess_regime_rule_cash_on_big_drop():
    assert assess_regime_rule(_feat(ret_5d=-12.0)) == Regime.CASH_PRESERVATION


def test_assess_regime_rule_custom_thresholds():
    # 阈值放宽：vol=90 不再 defensive
    out = assess_regime_rule(_feat(vol=90.0), defensive_vol_ann_pct=100.0, cash_vol_ann_pct=150.0)
    assert out == Regime.AGGRESSIVE


def test_strategy_to_regime():
    assert strategy_to_regime("aggressive_long") == Regime.AGGRESSIVE
    assert strategy_to_regime("defensive") == Regime.DEFENSIVE
    assert strategy_to_regime("cash_preservation") == Regime.CASH_PRESERVATION
    assert strategy_to_regime("garbage") == Regime.CASH_PRESERVATION


def test_risk_budget_for():
    assert risk_budget_for("aggressive_long") == 1.0
    assert risk_budget_for("defensive") == 0.2
    assert risk_budget_for("other") == 0.4


def test_planner_rule_assess_regime():
    p = Planner(policy="rule")
    assert p.assess_regime(_feat(vol=20.0, ret_5d=2.0)) == Regime.AGGRESSIVE
    assert p.assess_regime(_feat(vol=130.0)) == Regime.CASH_PRESERVATION


def test_planner_decide_rule_regimes():
    p = Planner(policy="rule")
    assert p.decide(market_regime={"regime": "risk_off"}).strategy == "defensive"
    assert p.decide(market_regime={"regime": "risk_on"}).strategy == "aggressive_long"
    assert p.decide(market_regime={"regime": "neutral"}).strategy == "cash_preservation"


def test_planner_sft_missing_model_falls_back_to_rule():
    # policy=sft 但路径不存在 -> _try_load_sft None -> 回退规则，绝不抛
    p = Planner(policy="sft", sft_model_path="models/does_not_exist_xyz.pt")
    assert p.assess_regime(_feat(vol=20.0, ret_5d=2.0)) == Regime.AGGRESSIVE


def test_planner_from_config():
    cfg = AppConfig()
    p = Planner.from_config(cfg.agents.planner)
    assert p.policy == cfg.agents.planner.policy
    assert p.cash_vol_ann_pct == cfg.agents.planner.cash_vol_ann_pct


def test_planner_decide_to_dict():
    p = Planner(policy="rule")
    d = p.decide(market_regime={"regime": "risk_on"})
    out = d.to_dict()
    assert out["strategy"] == "aggressive_long"
    assert out["risk_budget"] == 1.0

"""quantai.agents.gatekeeper 测试（无模型路径纯逻辑，验证去随机修复）。"""

from __future__ import annotations

from quantai.agents.gatekeeper import Gatekeeper
from quantai.config import AppConfig


def _feat(vol=20.0):
    return {"technical": {"volatility_20d": vol}, "volatility_ann_pct": vol}


def test_no_model_not_loaded():
    g = Gatekeeper(model_path="")
    assert g.model_loaded is False


def test_require_model_denies_without_model():
    g = Gatekeeper(model_path="", require_model=True)
    assert g.approve(_feat(vol=10.0)) is False
    assert g.approve(_feat(vol=200.0)) is False


def test_heuristic_vol_gate_when_not_requiring_model():
    g = Gatekeeper(model_path="", require_model=False, vol_trigger_ann_pct=120.0)
    assert g.approve(_feat(vol=50.0)) is True   # 低波动放行
    assert g.approve(_feat(vol=130.0)) is False  # 高波动拒绝


def test_approve_is_deterministic_no_random():
    # 核心：同一输入多次调用结果必须一致（旧版 random.random()>0.3 会抖动）
    g = Gatekeeper(model_path="", require_model=False, vol_trigger_ann_pct=120.0)
    results = {g.approve(_feat(vol=80.0)) for _ in range(50)}
    assert results == {True}
    results_deny = {g.approve(_feat(vol=999.0)) for _ in range(50)}
    assert results_deny == {False}


def test_predict_and_decide_zero_without_model():
    g = Gatekeeper(model_path="")
    assert g.predict(feats={"x": 1.0}) == 0.0
    d = g.decide(feats={"x": 1.0})
    assert d.allow is False
    assert d.q_allow == 0.0


def test_missing_model_path_does_not_crash_load():
    g = Gatekeeper(model_path="models/no_such_gate_xyz.pt")
    assert g.model_loaded is False
    assert g.approve(_feat(vol=10.0)) is False  # require_model 默认 True


def test_from_config_maps_fields():
    cfg = AppConfig()
    g = Gatekeeper.from_config(cfg.agents.gatekeeper)
    assert g.model_path == cfg.agents.gatekeeper.model_path
    assert g.threshold == cfg.agents.gatekeeper.threshold
    assert g.require_model == cfg.agents.gatekeeper.require_model
    assert g.vol_trigger_ann_pct == cfg.agents.gatekeeper.vol_trigger_ann_pct


def test_gate_decision_to_dict():
    g = Gatekeeper(model_path="")
    out = g.decide(feats={"a": 1.0}).to_dict()
    assert set(out) == {"allow", "q_allow", "threshold", "inputs"}
    assert out["allow"] is False


def test_no_random_import_in_module():
    import quantai.agents.gatekeeper as gk_mod

    src = open(gk_mod.__file__, "r", encoding="utf-8").read()
    # 不能真正 import/使用 random（docstring 里解释旧 bug 时提到的字样不算）。
    assert "import random" not in src
    # 行为层面的去随机由 test_approve_is_deterministic_no_random 证明。

"""quantai.agents.overlays 测试：Chartist 打分（纯逻辑）+ MacroGovernor 去随机。"""

from __future__ import annotations

from quantai.agents.overlays import ChartistOverlay, MacroGovernor
from quantai.config import AppConfig


# --------------------------------------------------------------------------- #
# ChartistOverlay
# --------------------------------------------------------------------------- #
def test_chartist_below_threshold_is_zero():
    c = ChartistOverlay(enabled=True, confidence_threshold=0.7)
    assert c.score_from_vlm("BULLISH", 0.5, "BUY") == 0


def test_chartist_bullish_supports_buy():
    c = ChartistOverlay(enabled=True, confidence_threshold=0.7)
    assert c.score_from_vlm("BULLISH", 0.9, "BUY") == 1


def test_chartist_bullish_opposes_sell():
    c = ChartistOverlay(enabled=True, confidence_threshold=0.7)
    assert c.score_from_vlm("BULLISH", 0.9, "SELL") == -1


def test_chartist_bearish_supports_sell():
    c = ChartistOverlay(enabled=True, confidence_threshold=0.7)
    assert c.score_from_vlm("BEARISH", 0.9, "SELL") == 1


def test_chartist_bearish_opposes_buy():
    c = ChartistOverlay(enabled=True, confidence_threshold=0.7)
    assert c.score_from_vlm("BEARISH", 0.9, "BUY") == -1


def test_chartist_hold_action_is_zero():
    c = ChartistOverlay(enabled=True, confidence_threshold=0.7)
    assert c.score_from_vlm("BULLISH", 0.9, "HOLD") == 0


def test_chartist_unknown_signal_is_zero():
    c = ChartistOverlay(enabled=True, confidence_threshold=0.7)
    assert c.score_from_vlm("SIDEWAYS", 0.9, "BUY") == 0


def test_chartist_bad_confidence_is_zero():
    c = ChartistOverlay(enabled=True)
    assert c.score_from_vlm("BULLISH", "oops", "BUY") == 0


def test_chartist_assess_disabled_returns_zero():
    c = ChartistOverlay(enabled=False)

    class A:
        def analyze(self, ticker, asof=""):
            return {"signal": "BULLISH", "confidence": 0.99}

    assert c.assess("NVDA", "BUY", analyzer=A()) == 0


def test_chartist_assess_no_analyzer_returns_zero():
    c = ChartistOverlay(enabled=True)
    assert c.assess("NVDA", "BUY", analyzer=None) == 0


def test_chartist_assess_with_analyzer():
    c = ChartistOverlay(enabled=True, confidence_threshold=0.7)

    class A:
        def analyze(self, ticker, asof=""):
            return {"signal": "BULLISH", "confidence": 0.9}

    assert c.assess("NVDA", "BUY", analyzer=A()) == 1


def test_chartist_assess_analyzer_exception_returns_zero():
    c = ChartistOverlay(enabled=True)

    class A:
        def analyze(self, ticker, asof=""):
            raise RuntimeError("vlm down")

    assert c.assess("NVDA", "BUY", analyzer=A()) == 0


def test_chartist_assess_analyzer_returns_none():
    c = ChartistOverlay(enabled=True)

    class A:
        def analyze(self, ticker, asof=""):
            return None

    assert c.assess("NVDA", "BUY", analyzer=A()) == 0


def test_chartist_from_config():
    cfg = AppConfig()
    c = ChartistOverlay.from_config(cfg.agents.chartist)
    assert c.enabled == cfg.agents.chartist.enabled
    assert c.confidence_threshold == cfg.agents.chartist.confidence_threshold


# --------------------------------------------------------------------------- #
# MacroGovernor（去随机）
# --------------------------------------------------------------------------- #
def test_macro_disabled_is_neutral():
    m = MacroGovernor(enabled=False)
    assert m.assess() == (0.0, "NEUTRAL")


def test_macro_is_deterministic_no_random():
    # 旧版 random.uniform(0.3,0.8)：现在必须确定性
    m = MacroGovernor(enabled=True)
    assert {m.assess() for _ in range(50)} == {(0.0, "NEUTRAL")}


def test_macro_risk_map_lookup():
    m = MacroGovernor(enabled=True, risk_map={"DRIVE": 1.0})
    assert m.assess("DRIVE") == (1.0, "DRIVE")
    assert m.assess("UNKNOWN") == (0.0, "NEUTRAL")


def test_macro_from_config():
    cfg = AppConfig()
    m = MacroGovernor.from_config(cfg.agents.macro_enabled)
    assert m.enabled == cfg.agents.macro_enabled

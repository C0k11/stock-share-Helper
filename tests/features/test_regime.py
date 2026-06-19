"""quantai.features.regime 的状态映射测试。"""

from __future__ import annotations

import pandas as pd

from quantai.features.regime import MarketRegime, RegimeDetector, detect_regime


def _linear_spy(n: int = 120, start: float = 100.0, step: float = 0.5) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series([start + i * step for i in range(n)], index=idx, dtype=float)
    return pd.DataFrame({"close": close, "high": close * 1.001, "low": close * 0.999})


def _flat_vix(spy: pd.DataFrame, level: float) -> pd.DataFrame:
    return pd.DataFrame({"close": [level] * len(spy)}, index=spy.index)


def test_risk_on_when_low_vix_and_uptrend() -> None:
    spy = _linear_spy(step=0.5)            # 上行
    out = RegimeDetector().get_current_regime(spy, _flat_vix(spy, 10.0))
    assert out["regime"] == MarketRegime.RISK_ON.value


def test_risk_off_when_high_vix_and_downtrend() -> None:
    spy = _linear_spy(start=160.0, step=-0.5)  # 下行
    out = RegimeDetector().get_current_regime(spy, _flat_vix(spy, 40.0))
    assert out["regime"] == MarketRegime.RISK_OFF.value


def test_without_vix_signal_is_zero() -> None:
    spy = _linear_spy()
    result = RegimeDetector().detect(spy, vix_data=None)
    assert (result["vix_signal"] == 0).all()


def test_detect_regime_convenience_returns_valid_label() -> None:
    spy = _linear_spy()
    label = detect_regime(spy, _flat_vix(spy, 10.0))
    assert label in {r.value for r in MarketRegime}

"""quantai.risk.sizing 测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantai.risk.sizing import PositionSizer


def test_vol_target_position_within_bounds(prices: pd.DataFrame) -> None:
    sizer = PositionSizer(target_volatility=0.10, min_position=0.0, max_position=1.0)
    pos = sizer.compute_vol_target_position(prices).dropna()
    assert (pos >= 0.0).all() and (pos <= 1.0).all()


def test_signal_position_mapping() -> None:
    s = pd.Series(["strong_long", "neutral", "strong_short"])
    mapped = PositionSizer().compute_signal_position(s)
    assert list(mapped) == [1.0, 0.3, 0.0]


def test_regime_adjustment_mapping() -> None:
    s = pd.Series(["risk_on", "transition", "risk_off"])
    assert list(PositionSizer().compute_regime_adjustment(s)) == [1.0, 0.7, 0.4]


def test_apply_risk_profile_caps_equity() -> None:
    sizer = PositionSizer()
    # conservative: scale 0.6, max_equity 0.4 -> 1.0 -> 0.6 -> capped 0.4
    assert sizer.apply_risk_profile(1.0, "conservative") == 0.4
    assert sizer.apply_risk_profile(0.5, "aggressive") == 0.5


def test_final_position_combines_components(prices: pd.DataFrame) -> None:
    n = len(prices)
    strength = pd.Series(["strong_long"] * n, index=prices.index)
    regime = pd.Series(["risk_off"] * n, index=prices.index)
    out = PositionSizer().compute_final_position(prices, strength, regime)
    tail = out["target_position"].dropna()
    # regime_factor risk_off=0.4 是上限因子，signal=1.0 -> 最终 <= 0.4
    assert (tail <= 0.4 + 1e-9).all()
    assert np.isfinite(tail).all()

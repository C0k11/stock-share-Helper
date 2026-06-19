"""quantai.signals.generator 测试。"""

from __future__ import annotations

import pandas as pd

from quantai.signals.generator import SignalGenerator


def test_generate_has_expected_columns(prices: pd.DataFrame) -> None:
    sig = SignalGenerator().generate(prices)
    for col in (
        "trend_signal", "momentum_signal", "ma_cross_signal",
        "breakout_signal", "composite_signal", "signal_strength",
    ):
        assert col in sig.columns


def test_directional_signals_are_plus_minus_one(prices: pd.DataFrame) -> None:
    sig = SignalGenerator().generate(prices)
    for col in ("trend_signal", "momentum_signal", "ma_cross_signal"):
        assert set(sig[col].unique()) <= {-1, 1}


def test_breakout_signal_in_allowed_set(prices: pd.DataFrame) -> None:
    sig = SignalGenerator().generate(prices)
    assert set(sig["breakout_signal"].unique()) <= {-1, 0, 1}


def test_get_current_signal_shape(prices: pd.DataFrame) -> None:
    out = SignalGenerator().get_current_signal(prices)
    assert set(out) == {"date", "trend", "momentum", "ma_cross", "breakout", "composite", "strength"}
    assert out["trend"] in (-1, 1)

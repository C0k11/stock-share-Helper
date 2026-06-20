"""无 lookahead 证明：SignalGenerator 截断不变性（含 breakout）。

核心思想：若信号在 t 处只依赖 <= t 的数据，则把序列在 t 之后截断重算，t 处的值必须完全一致。
breakout 用含当日的 `rolling(20).max()/min()`——当日 high/low 在 close[t] 时已知，故计算因果；
真正的成交时点由 backtest 的 next_open 保证，不在 t 日成交，因此不构成 lookahead。
"""

from __future__ import annotations

import pandas as pd

from quantai.signals.generator import SignalGenerator

_NUMERIC = ["trend_signal", "momentum_signal", "ma_cross_signal", "breakout_signal", "composite_signal"]


def test_all_signals_are_causal(prices: pd.DataFrame) -> None:
    gen = SignalGenerator()
    full = gen.generate(prices)
    for k in (150, 220, 280):
        truncated = gen.generate(prices.iloc[:k])
        idx = prices.index[k - 1]
        for col in _NUMERIC:
            assert full.loc[idx, col] == truncated.loc[idx, col], f"{col} 在 {idx} 处因截断而改变 -> lookahead"


def test_breakout_unaffected_by_future_bars(prices: pd.DataFrame) -> None:
    # 单独盯 breakout：t 处的值不能因 t 之后出现新高/新低而改变。
    gen = SignalGenerator()
    full = gen.generate(prices)["breakout_signal"]
    for k in (180, 250):
        idx = prices.index[k - 1]
        truncated = gen.generate(prices.iloc[:k])["breakout_signal"]
        assert full.loc[idx] == truncated.loc[idx]

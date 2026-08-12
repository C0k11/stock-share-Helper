"""与 `ta` 库交叉验证：同一份数据两套独立实现对数。

`ta` 是核心依赖（pyproject），拿来当免费的独立对数器——防止自研实现口径漂移。
EMA 族指标（RSI/ATR/MACD signal）两库种子惯例不同（我们 adjust=False 从 t0 递推，
`ta` 带 min_periods），差异随时间指数衰减 -> 只对**尾部 50 根**断言逐位一致。
SMA 族（布林/随机/MACD 线）无种子问题 -> 全程断言。
"""

from __future__ import annotations

import pandas as pd
import pytest

ta = pytest.importorskip("ta")

from quantai.analysis import atr, bollinger, macd, rsi, stochastic

_TOL = 1e-6


def test_rsi_matches_ta_tail(prices: pd.DataFrame) -> None:
    mine = rsi(prices["close"], 14)
    theirs = ta.momentum.RSIIndicator(prices["close"], 14).rsi()
    assert (mine - theirs).abs().iloc[-50:].max() < _TOL


def test_macd_matches_ta(prices: pd.DataFrame) -> None:
    mine = macd(prices["close"])
    theirs = ta.trend.MACD(prices["close"])
    assert (mine["macd"] - theirs.macd()).abs().dropna().max() < _TOL  # 全程
    assert (mine["macd_signal"] - theirs.macd_signal()).abs().iloc[-50:].max() < _TOL

def test_atr_matches_ta_tail(prices: pd.DataFrame) -> None:
    mine = atr(prices["high"], prices["low"], prices["close"], 14)
    theirs = ta.volatility.AverageTrueRange(
        prices["high"], prices["low"], prices["close"], 14
    ).average_true_range()
    assert (mine - theirs).abs().iloc[-50:].max() < _TOL


def test_bollinger_matches_ta(prices: pd.DataFrame) -> None:
    mine = bollinger(prices["close"], 20, 2.0)
    theirs = ta.volatility.BollingerBands(prices["close"], 20, 2)
    for col, ta_series in [
        ("bb_mid", theirs.bollinger_mavg()),
        ("bb_upper", theirs.bollinger_hband()),
        ("bb_lower", theirs.bollinger_lband()),
    ]:
        assert (mine[col] - ta_series).abs().dropna().max() < _TOL


def test_stochastic_matches_ta(prices: pd.DataFrame) -> None:
    mine = stochastic(prices["high"], prices["low"], prices["close"], 14, 3, 3)
    theirs = ta.momentum.StochasticOscillator(
        prices["high"], prices["low"], prices["close"], 14, 3
    )
    # ta 的 stoch() 是 raw %K 的 3 期 SMA=我们的 stoch_k(smooth_k=3)；
    # 其 stoch_signal() 再取 3 期 SMA=我们的 stoch_d。
    assert (mine["stoch_k"] - theirs.stoch_signal()).abs().dropna().max() < _TOL

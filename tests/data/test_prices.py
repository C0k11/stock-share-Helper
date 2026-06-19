"""quantai.data.prices 的纯逻辑测试（不触网，只测可单测的部分）。"""

from __future__ import annotations

import pandas as pd
import pytest

from quantai.data.prices import DEFAULT_US_ETFS, PriceFetcher


def test_non_yfinance_source_rejected() -> None:
    """US-only：旧 akshare/CN 源已移除，构造非 yfinance 源应报错。"""
    with pytest.raises(ValueError):
        PriceFetcher(source="akshare")


def test_standardize_columns_and_index() -> None:
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Volume": [100],
            "Stock Splits": [0.0],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )
    out = PriceFetcher._standardize(raw)
    assert list(out.columns) == ["open", "high", "low", "close", "volume", "stock_splits"]
    assert out.index.name == "date"
    # 不修改原始 df
    assert list(raw.columns)[0] == "Open"


def test_default_universe_constant() -> None:
    assert DEFAULT_US_ETFS == ("SPY", "QQQ", "TLT", "IEF", "GLD", "SHY")

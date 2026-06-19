"""features 测试共用：合成 OHLCV 价格。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_prices(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = rng.integers(1_000_000, 5_000_000, n)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame(
        {"close": close, "high": high, "low": low, "volume": volume}, index=idx
    )


@pytest.fixture
def prices() -> pd.DataFrame:
    return _make_prices(300, seed=0)


@pytest.fixture
def make_prices():
    return _make_prices

"""期权链 fetcher 单测（注入假 Ticker，零网络）：到期日窗口选择、无链降级、字段列。"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from quantai.data.options_chain import OptionChainFetcher


def _expiry(days: int) -> str:
    return (datetime.now().date() + timedelta(days=days)).strftime("%Y-%m-%d")


def _df():
    return pd.DataFrame(
        {"strike": [140.0, 150.0], "bid": [4.0, 8.0], "ask": [4.4, 8.6],
         "lastPrice": [4.2, 8.2], "volume": [10, 20], "openInterest": [100, 200],
         "impliedVolatility": [0.5, 0.48], "extraCol": [1, 2]}
    )


class _FakeTicker:
    def __init__(self, expiries):
        self.options = tuple(expiries)

    def option_chain(self, expiry):
        return SimpleNamespace(calls=_df(), puts=_df())


class TestFetch:
    def test_picks_expiry_in_hedge_window(self):
        f = OptionChainFetcher(ticker_factory=lambda s: _FakeTicker(
            [_expiry(10), _expiry(35), _expiry(90)]))
        out = f.fetch("AAA")
        assert out["days_to_expiry"] == 35
        assert {"strike", "bid", "ask"} <= set(out["puts"][0])
        assert "extraCol" not in out["puts"][0]  # 只带白名单列

    def test_falls_back_to_nearest_beyond_week(self):
        f = OptionChainFetcher(ticker_factory=lambda s: _FakeTicker([_expiry(3), _expiry(90)]))
        assert f.fetch("AAA")["days_to_expiry"] == 90  # 3 天太近被拒，退 90 天

    def test_no_options_returns_none(self):
        f = OptionChainFetcher(ticker_factory=lambda s: _FakeTicker([]))
        assert f.fetch("SPCX") is None

    def test_fetch_error_returns_none(self):
        def boom(s):
            raise RuntimeError("api down")

        assert OptionChainFetcher(ticker_factory=boom).fetch("AAA") is None

"""盘中统计单测：VWAP/卖压/区间位置手算、边界（无量/无前收）、简报组装。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantai.agents.analyst import build_intraday_brief, intraday_stats


def _session(closes, volumes):
    idx = pd.date_range("2026-07-06 09:30", periods=len(closes), freq="1min")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c,
                         "volume": pd.Series(volumes, index=idx, dtype=float)})


class TestIntradayStats:
    def test_hand_computed(self):
        # bar 收盘 [100, 101, 99]，量 [100, 200, 300]
        df = _session([100.0, 101.0, 99.0], [100, 200, 300])
        st = intraday_stats(df, prev_close=100.0, avg_daily_volume=1200.0)
        assert st["chg_pct"] == pytest.approx(-0.01)  # 99 vs 昨收 100
        assert st["day_high"] == pytest.approx(101.5)
        assert st["day_low"] == pytest.approx(98.5)
        # 区间位置 = (99-98.5)/(101.5-98.5)
        assert st["close_position"] == pytest.approx(0.5 / 3.0)
        # VWAP = Σ(典型价×量)/Σ量；典型价=收盘（high/low 对称 ±0.5）
        expected_vwap = (100 * 100 + 101 * 200 + 99 * 300) / 600
        assert st["vwap"] == pytest.approx(expected_vwap)
        assert st["vs_vwap_pct"] == pytest.approx(99 / expected_vwap - 1)
        # 跌 bar 只有第 3 根（101->99），量 300/600
        assert st["down_volume_share"] == pytest.approx(0.5)
        assert st["vol_vs_avg"] == pytest.approx(600 / 1200)

    def test_no_prev_close_empty(self):
        df = _session([100.0], [10])
        assert intraday_stats(df, prev_close=float("nan"), avg_daily_volume=1.0) == {}

    def test_zero_volume_skips_flow_fields(self):
        df = _session([100.0, 101.0], [0, 0])
        st = intraday_stats(df, prev_close=100.0, avg_daily_volume=100.0)
        assert "vwap" not in st and "down_volume_share" not in st
        assert st["chg_pct"] == pytest.approx(0.01)


class TestIntradayBrief:
    def test_contains_stats_and_sentiment(self):
        rows = [{"symbol": "SPCX", "last": 160.0, "chg_pct": -0.01, "day_high": 162.0,
                 "day_low": 159.0, "close_position": 0.33, "vwap": 160.5,
                 "vs_vwap_pct": -0.003, "cum_volume": 1e6, "vol_vs_avg": 1.4,
                 "down_volume_share": 0.62}]
        brief = build_intraday_brief(rows, None, {"SPCX": -0.35}, as_of="2026-07-06 10:00")
        assert "SPCX" in brief and "62.0%" in brief  # 卖压占比进简报
        assert "1.40× 均量" in brief
        assert "SPCX -0.35" in brief

    def test_empty_session_honest(self):
        brief = build_intraday_brief([], None, None)
        assert "无盘中数据" in brief
        assert "无量化新闻" in brief

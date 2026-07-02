"""etl.py 单测：装载正确性 + 幂等性（delete-then-insert by batch key）。全部 :memory:。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantai.backtest import run_backtest
from quantai.portfolio import Portfolio, Position
from quantai.signals.generator import SignalGenerator
from quantai.warehouse import (
    connect,
    load_backtest,
    load_positions,
    load_prices,
    load_signals,
    load_trades,
    load_trading_days,
)


@pytest.fixture
def con():
    c = connect(":memory:")
    yield c
    c.close()


def _count(con, table: str) -> int:
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


class TestLoadPrices:
    def test_load_and_reload_idempotent(self, con, prices):
        n = load_prices(con, {"AAA": prices})
        assert n == len(prices) == _count(con, "raw.prices")
        load_prices(con, {"AAA": prices})  # 重跑
        assert _count(con, "raw.prices") == len(prices)  # 不翻倍

    def test_symbol_normalized_upper(self, con, prices):
        load_prices(con, {"aaa": prices})
        assert con.execute("SELECT DISTINCT symbol FROM raw.prices").fetchone()[0] == "AAA"

    def test_partial_columns_null(self, con):
        df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2))
        load_prices(con, {"X": df})
        row = con.execute("SELECT open, volume, close FROM raw.prices LIMIT 1").fetchone()
        assert row[0] is None and row[1] is None and row[2] == 1.0

    def test_empty_df_skipped(self, con):
        assert load_prices(con, {"X": pd.DataFrame()}) == 0

    def test_values_roundtrip(self, con, prices):
        load_prices(con, {"AAA": prices})
        got = con.execute(
            "SELECT close FROM raw.prices WHERE symbol='AAA' ORDER BY date"
        ).df()["close"]
        assert np.allclose(got.to_numpy(), prices["close"].to_numpy())

    def test_failed_insert_rolls_back_old_batch(self, con, prices):
        """事务原子性：坏数据重载失败时，旧批次必须原样保留（旧实现 DELETE 先
        autocommit，INSERT 再炸 -> 旧数据清零，审计实锤）。"""
        load_prices(con, {"XXX": prices})
        n_before = _count(con, "raw.prices")
        bad = prices.copy()
        bad["volume"] = 1e300  # BIGINT 溢出 -> INSERT ConversionException
        with pytest.raises(Exception):
            load_prices(con, {"XXX": bad})
        assert _count(con, "raw.prices") == n_before  # 旧批次没丢

    def test_missing_close_column_rejected(self, con):
        df = pd.DataFrame({"open": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2))
        with pytest.raises(ValueError, match="close"):
            load_prices(con, {"X": df})

    def test_duplicate_dates_deduped_keep_last(self, con):
        df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2))
        dup = pd.concat([df, df.iloc[[-1]].assign(close=99.0)])
        load_prices(con, {"X": dup})
        got = con.execute("SELECT close FROM raw.prices ORDER BY date").df()["close"].tolist()
        assert got == [1.0, 99.0]  # 同日保留最后一条


class TestLoadTradingDays:
    def test_nyse_days(self, con):
        n = load_trading_days(con, "2024-01-01", "2024-12-31")
        assert n == 252  # 2024 年 NYSE 恰好 252 个交易日
        # 2024-07-03 感恩节前——不对，7/3 是独立日前半日市
        early = con.execute(
            "SELECT is_early_close FROM raw.trading_days WHERE date = DATE '2024-07-03'"
        ).fetchone()[0]
        assert early is True

    def test_reload_replaces(self, con):
        load_trading_days(con, "2024-01-01", "2024-06-30")
        load_trading_days(con, "2024-01-01", "2024-12-31")
        assert _count(con, "raw.trading_days") == 252


class TestLoadPositions:
    def _pf(self) -> Portfolio:
        return Portfolio(
            cash=1000.0,
            positions=[
                Position(symbol="AAA", shares=10, cost_basis=90.0, open_date="2025-01-02"),
                Position(symbol="AAA", shares=5, cost_basis=100.0, open_date="2025-02-03"),
            ],
        )

    def test_lots_kept_raw(self, con):
        n = load_positions(con, self._pf(), as_of="2026-07-01")
        assert n == 2 == _count(con, "raw.positions")  # raw 保留 lot 粒度
        assert con.execute("SELECT cash FROM raw.portfolio_cash").fetchone()[0] == 1000.0

    def test_same_asof_idempotent_different_asof_appends(self, con):
        load_positions(con, self._pf(), as_of="2026-07-01")
        load_positions(con, self._pf(), as_of="2026-07-01")
        assert _count(con, "raw.positions") == 2
        load_positions(con, self._pf(), as_of="2026-07-02")  # 新快照日 -> 历史积累
        assert _count(con, "raw.positions") == 4


class TestLoadTrades:
    def test_roundtrip_idempotent(self, con):
        orders = [
            {"ticker": "aaa", "action": "BUY", "price": 10.0, "shares": 5, "trace_id": "t1"},
            {"ticker": "AAA", "action": "SELL", "price": 12.0, "shares": 5, "trace_id": None},
        ]
        assert load_trades(con, orders, run_id="run1") == 2
        load_trades(con, orders, run_id="run1")
        assert _count(con, "raw.trades") == 2
        load_trades(con, orders, run_id="run2")
        assert _count(con, "raw.trades") == 4


class TestLoadSignals:
    def test_signal_generator_output_loads(self, con, prices):
        sig = SignalGenerator(ma_short=5, ma_long=20, momentum_lookback=10).generate(prices)
        n = load_signals(con, "AAA", sig)
        assert n == len(sig) == _count(con, "raw.signals")
        strengths = {
            r[0]
            for r in con.execute(
                "SELECT DISTINCT signal_strength FROM raw.signals WHERE signal_strength IS NOT NULL"
            ).fetchall()
        }
        assert strengths <= {"strong_short", "weak_short", "neutral", "weak_long", "strong_long"}


class TestLoadBacktest:
    def test_metrics_and_curve(self, con, prices):
        weight = pd.Series(0.5, index=prices.index)
        result = run_backtest(prices, weight)
        load_backtest(con, "bt1", result, strategy="const50", symbol="AAA")
        m = con.execute("SELECT sharpe, max_drawdown, fill_timing FROM raw.backtest_runs").fetchone()
        assert m[0] == pytest.approx(result.metrics.sharpe)
        assert m[1] == pytest.approx(result.metrics.max_drawdown)
        assert m[2] == "next_open"
        assert _count(con, "raw.backtest_equity") == len(result.equity)
        # 幂等
        load_backtest(con, "bt1", result, strategy="const50", symbol="AAA")
        assert _count(con, "raw.backtest_runs") == 1

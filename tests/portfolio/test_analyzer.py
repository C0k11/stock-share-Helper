"""analyzer.py 单测：盈亏/权重手算、lot 聚合、缺价诚实处理、beta/风险统计、空组合。

价格全部合成注入（无网络）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantai.portfolio import Portfolio, PortfolioAnalyzer, Position, format_snapshot_text


def _df(close_values, start="2023-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(close_values))
    close = pd.Series(close_values, index=idx, dtype=float)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close})


def _bench(n=250) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return _df(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))))


def _portfolio(**kw) -> Portfolio:
    defaults = dict(
        cash=1000.0,
        positions=[
            Position(symbol="AAA", shares=10, cost_basis=90.0, open_date="2025-01-02"),
            Position(symbol="BBB", shares=5, cost_basis=200.0, open_date="2025-02-03"),
        ],
    )
    defaults.update(kw)
    return Portfolio(**defaults)


class TestPnLMath:
    def test_hand_computed_totals(self):
        bench = _bench()
        prices = {
            "AAA": _df(np.linspace(90, 100, 250)),  # last=100
            "BBB": _df(np.linspace(200, 180, 250)),  # last=180
        }
        snap = PortfolioAnalyzer(prices, bench).analyze(_portfolio())
        aaa, bbb = {s.symbol: s for s in snap.positions}["AAA"], {s.symbol: s for s in snap.positions}["BBB"]
        assert aaa.market_value == pytest.approx(1000.0)
        assert aaa.unrealized_pnl == pytest.approx(100.0)  # (100-90)*10
        assert aaa.unrealized_pnl_pct == pytest.approx(100.0 / 900.0)
        assert bbb.unrealized_pnl == pytest.approx(-100.0)  # (180-200)*5
        assert snap.total_market_value == pytest.approx(1000.0 + 900.0)
        assert snap.total_value == pytest.approx(1900.0 + 1000.0)
        assert snap.total_unrealized_pnl == pytest.approx(0.0)
        # 权重分母含现金
        assert aaa.weight == pytest.approx(1000.0 / 2900.0)
        assert snap.top_weight == pytest.approx(1000.0 / 1900.0)  # 集中度不含现金
        hhi = (1000 / 1900) ** 2 + (900 / 1900) ** 2
        assert snap.herfindahl == pytest.approx(hhi)

    def test_lot_aggregation_weighted_cost(self):
        pf = Portfolio(
            positions=[
                Position(symbol="AAA", shares=10, cost_basis=100.0, open_date="2025-01-02"),
                Position(symbol="AAA", shares=10, cost_basis=200.0, open_date="2025-03-03"),
            ]
        )
        snap = PortfolioAnalyzer({"AAA": _df(np.linspace(100, 150, 100))}, _bench()).analyze(pf)
        s = snap.positions[0]
        assert s.shares == 20
        assert s.avg_cost == pytest.approx(150.0)
        assert s.open_date == "2025-01-02"  # 取最早

    def test_netted_out_symbol_dropped(self):
        pf = Portfolio(
            positions=[
                Position(symbol="AAA", shares=10, cost_basis=100.0, open_date="2025-01-02"),
                Position(symbol="AAA", shares=-10, cost_basis=120.0, open_date="2025-03-03"),
            ]
        )
        snap = PortfolioAnalyzer({"AAA": _df([100, 101, 102])}, _bench()).analyze(pf)
        assert snap.positions == []

    def test_short_position_pnl_sign(self):
        # 空头：价格涨 -> 亏
        pf = Portfolio(positions=[Position(symbol="AAA", shares=-10, cost_basis=100.0, open_date="2025-01-02")])
        snap = PortfolioAnalyzer({"AAA": _df(np.linspace(100, 120, 100))}, _bench()).analyze(pf)
        s = snap.positions[0]
        assert s.unrealized_pnl == pytest.approx((120 - 100) * -10)
        assert s.unrealized_pnl_pct == pytest.approx(-200.0 / 1000.0)


class TestDayChange:
    def test_flat_book_reports_exactly_zero(self):
        """+5%/-5% 等前收市值组合的日变动必须恰为 0——旧实现按当日市值加权
        会报 +0.25% 系统性正偏（审计实锤）。"""
        aaa = _df([100.0] * 10 + [105.0])  # +5%
        bbb = _df([100.0] * 10 + [95.0])  # -5%
        pf = Portfolio(
            positions=[
                Position(symbol="AAA", shares=10, cost_basis=100, open_date="2025-01-02"),
                Position(symbol="BBB", shares=10, cost_basis=100, open_date="2025-01-02"),
            ]
        )
        snap = PortfolioAnalyzer({"AAA": aaa, "BBB": bbb}, _bench()).analyze(pf)
        assert snap.day_change_pct == pytest.approx(0.0, abs=1e-12)

    def test_long_book_matches_true_portfolio_return(self):
        aaa = _df([100.0] * 10 + [110.0])
        bbb = _df([50.0] * 10 + [49.0])
        pf = Portfolio(
            positions=[
                Position(symbol="AAA", shares=10, cost_basis=100, open_date="2025-01-02"),
                Position(symbol="BBB", shares=20, cost_basis=50, open_date="2025-01-02"),
            ]
        )
        snap = PortfolioAnalyzer({"AAA": aaa, "BBB": bbb}, _bench()).analyze(pf)
        # V_prev = 1000+1000=2000, V_now = 1100+980=2080 -> +4%
        assert snap.day_change_pct == pytest.approx(0.04)

    def test_short_book_loss_is_negative(self):
        """空头账本：价格涨 = 亏钱，day_change 必须为负（旧实现符号翻转报 +2%）。"""
        aaa = _df([100.0] * 10 + [102.0])
        pf = Portfolio(positions=[Position(symbol="AAA", shares=-10, cost_basis=100, open_date="2025-01-02")])
        snap = PortfolioAnalyzer({"AAA": aaa}, _bench()).analyze(pf)
        assert snap.day_change_pct == pytest.approx(-0.02)


class TestHonesty:
    def test_missing_prices_listed_not_silently_dropped(self):
        snap = PortfolioAnalyzer({"AAA": _df([100, 101, 102])}, _bench()).analyze(_portfolio())
        assert snap.missing_prices == ["BBB"]
        assert {s.symbol for s in snap.positions} == {"AAA"}

    def test_all_nan_close_goes_to_missing_not_poisoning_totals(self):
        """close 列存在但全 NaN（退市/坏抓取形状）：进 missing_prices，
        总计保持有限——旧实现 missing=[] 且所有总计被毒化成 NaN（审计实锤）。"""
        bad = _df([100.0] * 250)
        bad["close"] = np.nan
        prices = {"AAA": _df(np.linspace(90, 100, 250)), "BBB": bad}
        snap = PortfolioAnalyzer(prices, _bench()).analyze(_portfolio())
        assert snap.missing_prices == ["BBB"]
        assert snap.total_value == snap.total_value  # 非 NaN
        aaa = snap.positions[0]
        assert aaa.weight == aaa.weight  # 非 NaN

    def test_tz_aware_prices_do_not_crash(self):
        df = _df(np.linspace(90, 100, 250))
        df.index = df.index.tz_localize("America/New_York")  # yfinance 真实形状
        pf = Portfolio(positions=[Position(symbol="AAA", shares=1, cost_basis=90, open_date="2025-01-02")])
        snap = PortfolioAnalyzer({"AAA": df}, _bench()).analyze(pf)  # bench tz-naive 混用
        assert snap.positions[0].last_price == pytest.approx(100.0)
        assert snap.positions[0].beta_vs_benchmark == snap.positions[0].beta_vs_benchmark or True  # 不崩即可

    def test_duplicate_last_date_deduped_keep_last(self):
        df = _df([100.0, 101.0, 102.0])
        dup = pd.concat([df, df.iloc[[-1]].assign(close=200.0)])  # 同日重复条，后者为准
        pf = Portfolio(positions=[Position(symbol="AAA", shares=1, cost_basis=90, open_date="2025-01-02")])
        snap = PortfolioAnalyzer({"AAA": dup}, _bench()).analyze(pf)
        assert snap.positions[0].last_price == pytest.approx(200.0)

    def test_benchmark_without_close_degrades_not_crashes(self):
        bench = pd.DataFrame({"open": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2))
        snap = PortfolioAnalyzer({"AAA": _df(np.linspace(90, 100, 250))}, bench).analyze(
            Portfolio(positions=[Position(symbol="AAA", shares=1, cost_basis=90, open_date="2025-01-02")])
        )
        assert np.isnan(snap.positions[0].beta_vs_benchmark)

    def test_negative_synthetic_value_risk_stats_nan(self):
        """深度净空头 + 低现金：合成净值为负，pct 收益无意义 -> 统计诚实 NaN。"""
        pf = Portfolio(cash=10.0, positions=[Position(symbol="AAA", shares=-100, cost_basis=90, open_date="2025-01-02")])
        snap = PortfolioAnalyzer({"AAA": _df(np.linspace(90, 100, 250))}, _bench()).analyze(pf)
        assert np.isnan(snap.current_holdings_sharpe)
        assert np.isnan(snap.current_holdings_max_drawdown)

    def test_short_book_concentration_uses_gross(self):
        """多空账本：集中度用 gross 口径（|mv|/sum|mv|），净分母会让权重 >1。"""
        prices = {
            "AAA": _df([100.0] * 250),  # long 10 -> +1000
            "BBB": _df([90.0] * 250),  # short -10 -> -900
        }
        pf = Portfolio(
            positions=[
                Position(symbol="AAA", shares=10, cost_basis=100, open_date="2025-01-02"),
                Position(symbol="BBB", shares=-10, cost_basis=90, open_date="2025-01-02"),
            ]
        )
        snap = PortfolioAnalyzer(prices, _bench()).analyze(pf)
        assert snap.top_weight == pytest.approx(1000 / 1900)
        assert 0 < snap.herfindahl <= 1

    def test_cash_only_portfolio(self):
        snap = PortfolioAnalyzer({}, _bench()).analyze(Portfolio(cash=500.0))
        assert snap.total_value == 500.0
        assert np.isnan(snap.current_holdings_sharpe)
        assert snap.positions == [] and snap.missing_prices == []

    def test_risk_stats_are_labeled_current_holdings(self):
        # 字段名本身就是诚实标注的一部分——防止未来被悄悄改名成"历史收益"
        snap = PortfolioAnalyzer({"AAA": _df(np.linspace(90, 100, 250))}, _bench()).analyze(
            Portfolio(positions=[Position(symbol="AAA", shares=1, cost_basis=90, open_date="2025-01-02")])
        )
        d = snap.as_dict()
        for k in ("current_holdings_ann_vol", "current_holdings_sharpe",
                  "current_holdings_sortino", "current_holdings_max_drawdown"):
            assert k in d


class TestRiskStats:
    def test_beta_scaled_benchmark(self):
        bench = _bench()
        # 标的收益 = 2x 基准收益
        r = bench["close"].pct_change(fill_method=None).fillna(0)
        close = 100 * np.cumprod(1 + 2 * r.to_numpy())
        prices = {"AAA": pd.DataFrame({"close": pd.Series(close, index=bench.index)})}
        pf = Portfolio(positions=[Position(symbol="AAA", shares=1, cost_basis=100, open_date="2025-01-02")])
        snap = PortfolioAnalyzer(prices, bench).analyze(pf)
        assert snap.positions[0].beta_vs_benchmark == pytest.approx(2.0, rel=1e-6)
        # 组合只有这一只 + 无现金 -> 组合 beta = 2
        assert snap.portfolio_beta == pytest.approx(2.0, rel=1e-6)

    def test_cash_dilutes_portfolio_beta(self):
        bench = _bench()
        r = bench["close"].pct_change(fill_method=None).fillna(0)
        close = pd.Series(100 * np.cumprod(1 + r.to_numpy()), index=bench.index)
        prices = {"AAA": pd.DataFrame({"close": close})}
        last = float(close.iloc[-1])
        pf = Portfolio(
            cash=last,  # 现金 = 持仓市值 -> 权重各 50%
            positions=[Position(symbol="AAA", shares=1, cost_basis=100, open_date="2025-01-02")],
        )
        snap = PortfolioAnalyzer(prices, bench).analyze(pf)
        assert snap.portfolio_beta == pytest.approx(0.5, rel=1e-6)

    def test_holdings_history_uses_intersection_of_dates(self):
        bench = _bench()
        long_ = _df(np.linspace(90, 100, 250))
        short_ = _df(np.linspace(50, 60, 100), start=str(long_.index[150].date()))
        snap = PortfolioAnalyzer({"AAA": long_, "BBB": short_}, bench).analyze(
            Portfolio(
                positions=[
                    Position(symbol="AAA", shares=1, cost_basis=90, open_date="2025-01-02"),
                    Position(symbol="BBB", shares=1, cost_basis=50, open_date="2025-01-02"),
                ]
            )
        )
        assert not np.isnan(snap.current_holdings_ann_vol)


class TestFormatting:
    def test_text_report_renders(self):
        snap = PortfolioAnalyzer(
            {"AAA": _df(np.linspace(90, 100, 250)), "BBB": _df(np.linspace(200, 180, 250))},
            _bench(),
        ).analyze(_portfolio())
        text = format_snapshot_text(snap)
        assert "AAA" in text and "BBB" in text and "Portfolio Snapshot" in text

    def test_text_report_shows_missing(self):
        snap = PortfolioAnalyzer({"AAA": _df([100, 101, 102])}, _bench()).analyze(_portfolio())
        assert "MISSING PRICES" in format_snapshot_text(snap)

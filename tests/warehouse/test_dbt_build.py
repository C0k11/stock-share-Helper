"""dbt 集成测试：ETL 种子 → `dbt build`（模型 + 全部断言测试）→ SQL vs pandas 对账。

这是数据层的核心 integrity 测试：同一份合成数据，pandas（PortfolioAnalyzer /
pct_change / drawdown_curve）与 SQL（marts 星型模型）两条**独立实现路径**必须
算出同一个数字。跑真 dbt（dbtRunner 程序化调用），单测约 20-40s，标 slow。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("dbt.cli.main", reason="dbt-duckdb 未安装（pip install -e .[warehouse]）")

from quantai.backtest import run_backtest
from quantai.portfolio import Portfolio, PortfolioAnalyzer, Position
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

_REPO = Path(__file__).resolve().parents[2]
_DBT_DIR = str(_REPO / "warehouse")


def _make_prices(seed: int, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))),
        index=pd.bdate_range("2025-01-06", periods=n),
    )
    return pd.DataFrame(
        {"open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": rng.integers(1e6, 5e6, n)}
    )


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    """种子一个临时 DuckDB 仓库 + 跑 dbt build，整模块共享。"""
    db_path = tmp_path_factory.mktemp("wh") / "test.duckdb"
    prices = {"AAA": _make_prices(1), "BBB": _make_prices(2)}
    portfolio = Portfolio(
        cash=1000.0,
        positions=[
            Position(symbol="AAA", shares=10, cost_basis=90.0, open_date="2025-01-06"),
            Position(symbol="AAA", shares=5, cost_basis=110.0, open_date="2025-03-03"),
            Position(symbol="BBB", shares=-3, cost_basis=120.0, open_date="2025-02-03"),
        ],
    )
    as_of = str(prices["AAA"].index[-1].date())
    weight = pd.Series(0.6, index=prices["AAA"].index)
    bt = run_backtest(prices["AAA"], weight)

    con = connect(db_path)
    load_prices(con, prices)
    load_trading_days(con, "2025-01-01", "2025-12-31")
    load_positions(con, portfolio, as_of=as_of)
    load_trades(
        con,
        [
            {"ticker": "AAA", "action": "BUY", "price": 100.0, "shares": 10, "trace_id": "t1"},
            {"ticker": "AAA", "action": "SELL", "price": 105.0, "shares": 4, "trace_id": "t2"},
        ],
        run_id="live1",
    )
    load_signals(con, "AAA", SignalGenerator(5, 20, 10).generate(prices["AAA"]))
    load_backtest(con, "bt1", bt, strategy="const60", symbol="AAA")
    from quantai.data.news import entries_to_items
    from quantai.warehouse import load_news, load_news_scores

    news_items = entries_to_items(
        [
            {"title": "AAA beats estimates", "link": "https://t.test/n1", "summary": "s"},
            {"title": "AAA faces probe", "link": "https://t.test/n2"},
        ],
        symbol="AAA", source="test",
    )
    load_news(con, news_items)
    load_news_scores(
        con,
        [{"item": news_items[0], "sentiment": 0.7, "label": "bullish"}],  # 第二条不打分
        model="test-llm",
    )
    con.close()  # DuckDB 单写者：dbt 接管前必须放锁

    from dbt.cli.main import dbtRunner

    os.environ["QUANTAI_DB_PATH"] = str(db_path)
    try:
        res = dbtRunner().invoke(
            ["build", "--project-dir", _DBT_DIR, "--profiles-dir", _DBT_DIR]
        )
    finally:
        os.environ.pop("QUANTAI_DB_PATH", None)
    assert res.success, f"dbt build 失败: {res.exception}"

    con = connect(db_path)
    yield {"con": con, "prices": prices, "portfolio": portfolio, "as_of": as_of, "bt": bt}
    con.close()


class TestDbtBuild:
    def test_marts_exist_with_rows(self, warehouse):
        con = warehouse["con"]
        for t in (
            "dim_date", "dim_symbol", "fact_prices", "fact_positions",
            "fact_trades", "fact_signals", "fact_backtest_results", "fact_backtest_equity",
            "fact_news",
        ):
            n = con.execute(f"SELECT count(*) FROM marts.{t}").fetchone()[0]
            assert n > 0, f"marts.{t} 为空"

    def test_fact_prices_rowcount_matches_raw(self, warehouse):
        con = warehouse["con"]
        raw = con.execute("SELECT count(*) FROM raw.prices").fetchone()[0]
        mart = con.execute("SELECT count(*) FROM marts.fact_prices").fetchone()[0]
        assert raw == mart  # 合成数据无脏行，不应有损耗


class TestSqlVsPandasReconciliation:
    """对账：SQL 与 pandas 两条独立路径必须同数。"""

    def test_positions_pnl_matches_portfolio_analyzer(self, warehouse):
        con, prices, portfolio = warehouse["con"], warehouse["prices"], warehouse["portfolio"]
        bench = _make_prices(9)
        snap = PortfolioAnalyzer(prices, bench).analyze(portfolio)
        pandas_pnl = {s.symbol: (s.unrealized_pnl, s.unrealized_pnl_pct) for s in snap.positions}
        sql_rows = {
            r[0]: (r[1], r[2])
            for r in con.execute(
                "SELECT symbol, unrealized_pnl, unrealized_pnl_pct FROM marts.fact_positions"
            ).fetchall()
        }
        assert set(sql_rows) == set(pandas_pnl)
        for sym, (pnl, pct) in pandas_pnl.items():
            assert sql_rows[sym][0] == pytest.approx(pnl, abs=1e-6), sym
            # pct 也纳入对账（曾因 SQL 守卫 avg_cost>0 与 pandas 分叉而漏检）
            sql_pct = sql_rows[sym][1]
            if pct == pct:
                assert sql_pct == pytest.approx(pct, abs=1e-9), sym
            else:
                assert sql_pct is None, sym

    def test_daily_return_matches_pct_change(self, warehouse):
        con, prices = warehouse["con"], warehouse["prices"]
        got = con.execute(
            "SELECT date, daily_return FROM marts.fact_prices WHERE symbol='AAA' ORDER BY date"
        ).df()
        expected = prices["AAA"]["close"].pct_change(fill_method=None).to_numpy()
        assert np.allclose(
            got["daily_return"].to_numpy()[1:], expected[1:], atol=1e-12
        )

    def test_backtest_drawdown_matches_pandas(self, warehouse):
        from quantai.analysis import drawdown_curve

        con, bt = warehouse["con"], warehouse["bt"]
        got = con.execute(
            "SELECT date, drawdown FROM marts.fact_backtest_equity WHERE run_id='bt1' ORDER BY date"
        ).df()
        expected = drawdown_curve(bt.equity).to_numpy()
        assert np.allclose(got["drawdown"].to_numpy(), expected, atol=1e-12)

    def test_short_position_negative_market_value(self, warehouse):
        con = warehouse["con"]
        mv = con.execute(
            "SELECT market_value FROM marts.fact_positions WHERE symbol='BBB'"
        ).fetchone()[0]
        assert mv < 0  # 空头市值为负（与 pandas 口径一致）

    def test_fact_news_sentiment_join_null_for_unscored(self, warehouse):
        con = warehouse["con"]
        rows = dict(
            con.execute("SELECT link, sentiment FROM marts.fact_news ORDER BY link").fetchall()
        )
        assert rows["https://t.test/n1"] == pytest.approx(0.7)  # 打过分的带分
        assert rows["https://t.test/n2"] is None  # 没打分 = NULL，不冒充 0

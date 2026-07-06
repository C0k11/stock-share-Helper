"""analyst.py 单测：brief 组装（离线，全注入）+ LLM 注入接缝 + 仓库缺失诚实降级。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantai.agents.analyst import build_brief, generate_commentary
from quantai.data.news import entries_to_items
from quantai.portfolio import Portfolio, PortfolioAnalyzer, Position


def _df(vals, start="2023-01-02"):
    close = pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)), dtype=float)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close})


@pytest.fixture
def snap():
    prices = {"AAA": _df(np.linspace(90, 100, 250))}
    bench = _df(np.linspace(100, 110, 250))
    pf = Portfolio(cash=500.0, positions=[Position(symbol="AAA", shares=10, cost_basis=90, open_date="2025-01-02")])
    return PortfolioAnalyzer(prices, bench).analyze(pf)


class TestBuildBrief:
    def test_contains_all_sections_and_real_numbers(self, snap):
        news = {"AAA": entries_to_items([{"title": "AAA wins deal", "link": "https://t/1"}], "AAA", "test")}
        watch = [{"symbol": "SPY", "last": 500.0, "day_pct": 0.01, "rsi": 55.0, "vol": 0.12}]
        brief = build_brief(snap, watch, news, warehouse_con=None, as_of="2026-07-06")
        assert "真实持仓快照" in brief and "AAA" in brief
        assert f"{snap.total_value:.2f}" in brief  # 引用真实数字
        assert "AAA wins deal" in brief
        assert "SPY" in brief and "55.0" in brief
        assert "仓库未初始化" in brief  # con=None 诚实降级

    def test_empty_watchlist_and_news_honest(self, snap):
        brief = build_brief(snap, [], {}, None)
        assert "无自选股数据" in brief
        assert "无新闻数据" in brief

    def test_warehouse_section_queries_marts(self, snap):
        import duckdb

        con = duckdb.connect(":memory:")
        con.execute("CREATE SCHEMA marts")
        con.execute(
            """CREATE TABLE marts.fact_prices AS
               SELECT * FROM (VALUES
                 ('AAA', DATE '2026-07-01', 100.0, CAST(NULL AS DOUBLE)),
                 ('AAA', DATE '2026-07-02', 110.0, -0.05)
               ) t(symbol, date, close, pct_from_52w_high)"""
        )
        con.execute(
            """CREATE TABLE marts.fact_backtest_results AS
               SELECT 'bt1' AS run_id, 0.7 AS sharpe, -0.21 AS max_drawdown"""
        )
        brief = build_brief(snap, [], {}, warehouse_con=con)
        con.close()
        assert "近5日涨跌榜" in brief and "AAA +10.00%" in brief
        assert "回测 bt1: Sharpe 0.7" in brief


class TestCommentary:
    def test_llm_injection_seam(self, snap):
        class FakeLLM:
            def generate(self, user, system=None):
                assert "真实持仓快照" in user  # LLM 真的看到了系统数据
                assert "只引用简报中出现的数字" in system  # 诚实约束在 system prompt
                return "【组合体检】ok"

        brief = build_brief(snap, [], {}, None)
        out = generate_commentary(brief, FakeLLM())
        assert out.startswith("【组合体检】")

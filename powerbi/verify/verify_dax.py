"""DAX vs pandas 对账：从同一批 CSV 算期望值，锚定 Power BI 语义模型的正确性。

套路：规格不变、引擎换掉、结果对账。
这里的"规格"是 tableau/DASHBOARD_SPEC.md 的表计算语义，三条硬约定：
1. 窗口 = 20 个**交易行**（该标的实际有 bar 的行），不是 20 个日历天；
2. WINDOW_STDEV 是样本标准差 → pandas ddof=1 / DAX STDEVX.S；
3. 窗口不足 20 行 → 空值（pandas rolling 默认 NaN / DAX 返回 BLANK）。

产出 powerbi/verify/expected_values.csv（check_id, description, expected）。
Power BI 侧的 QA 页把对应 DAX 度量与这些值并排；浮点相对误差 > 1e-6 即 DAX 有错，
禁止反向修改期望值迁就 DAX。

运行：venv311\\Scripts\\python.exe powerbi\\verify\\verify_dax.py
（数据量 1.6 万行级，pandas 向量化单进程毫秒级完成——这一步不是 CPU 瓶颈，
不需要多进程；诚实说明以免"吃满核心"变成表演。）
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

EXPORTS = Path(__file__).resolve().parents[2] / "tableau" / "exports"
OUT = Path(__file__).resolve().parent / "expected_values.csv"

TABLES = [
    "dim_date", "dim_symbol", "fact_prices", "fact_signals", "fact_news",
    "fact_event_odds", "fact_backtest_equity", "fact_backtest_results",
    "fact_positions", "fact_trades",
]


def main() -> int:
    checks: list[tuple[str, str, str]] = []

    def add(check_id: str, desc: str, value) -> None:
        if isinstance(value, float):
            checks.append((check_id, desc, f"{value:.10g}"))
        else:
            checks.append((check_id, desc, str(value)))

    dfs = {t: pd.read_csv(EXPORTS / f"{t}.csv", encoding="utf-8") for t in TABLES}

    # ---- 1. 每张表行数（dim_symbol 模型侧比 CSV 多 1 行：M 末尾 append 的 (untagged)）----
    for t in TABLES:
        if t == "dim_symbol":
            add(f"rowcount_{t}", f"{t}.csv rows + 1 M-appended (untagged)", len(dfs[t]) + 1)
        else:
            add(f"rowcount_{t}", f"{t}.csv row count", len(dfs[t]))

    # ---- 2. SPY 最后 5 个交易行的 Rolling Vol 20D (Ann) / MA20 / MA50 / MA200 ----
    px = dfs["fact_prices"].copy()
    px["date"] = pd.to_datetime(px["date"])
    spy = px[px["symbol"] == "SPY"].sort_values("date").reset_index(drop=True)
    spy["vol20"] = spy["daily_return"].rolling(20).std(ddof=1) * math.sqrt(252)
    for w in (20, 50, 200):
        spy[f"ma{w}"] = spy["close"].rolling(w).mean()
    tail = spy.tail(5)
    for _, r in tail.iterrows():
        d = r["date"].date()
        add(f"spy_vol20_{d}", f"SPY Rolling Vol 20D (Ann) @ {d}", float(r["vol20"]))
    for w in (20, 50, 200):
        for _, r in tail.iterrows():
            d = r["date"].date()
            add(f"spy_ma{w}_{d}", f"SPY MA{w} @ {d}", float(r[f"ma{w}"]))

    # ---- 3. 区间末归一化（起点=100，起点=各自序列首行） ----
    spy_indexed_end = spy["close"].iloc[-1] / spy["close"].iloc[0] * 100
    add("spy_indexed_end", "SPY Indexed 100 at last date", float(spy_indexed_end))
    eq = dfs["fact_backtest_equity"].copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date")
    add("equity_indexed_end", "Equity Indexed 100 at last date",
        float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] * 100))

    # ---- 4. 回测 7 指标（单行直读——验证的是 Power BI 格式化没算错） ----
    bt = dfs["fact_backtest_results"].iloc[0]
    for col in ("total_return", "cagr", "annual_volatility", "sharpe",
                "max_drawdown", "win_rate", "total_turnover"):
        add(f"bt_{col}", f"fact_backtest_results.{col}", float(bt[col]))

    # ---- 5. 最新持仓快照 4 数 ----
    pos = dfs["fact_positions"].copy()
    latest = pos[pos["as_of"] == pos["as_of"].max()]
    add("pos_latest_as_of", "latest positions as_of", pos["as_of"].max())
    add("pos_market_value", "latest market_value", float(latest["market_value"].sum()))
    add("pos_cost_value", "latest cost_value", float(latest["cost_value"].sum()))
    add("pos_unrealized_pnl", "latest unrealized_pnl", float(latest["unrealized_pnl"].sum()))
    add("pos_unrealized_pnl_pct", "latest pnl / |cost|",
        float(latest["unrealized_pnl"].sum() / abs(latest["cost_value"].sum())))

    # ---- 6. signal_strength 五桶 ----
    sig = dfs["fact_signals"]["signal_strength"].value_counts()
    for bucket in ("strong_short", "strong_long", "weak_short", "weak_long", "neutral"):
        add(f"sig_{bucket}", f"signal_strength == {bucket}", int(sig.get(bucket, 0)))

    # ---- 7. 新闻打标覆盖率 ----
    news = dfs["fact_news"]
    scored = int(news["sentiment"].notna().sum())
    add("news_total", "fact_news rows", len(news))
    add("news_scored", "fact_news rows with sentiment", scored)
    add("news_coverage", "scored / total", float(scored / len(news)))

    # ---- 8. 关系传播测点（走 dim→fact 关系过滤，DAX 侧禁止 REMOVEFILTERS 维度表）----
    # 判据：删掉任何一条日期/符号关系，这组必须 FAIL——它们验证模型接线，不是度量算术。
    px = dfs["fact_prices"]
    add("rel_prices_20260727", "fact_prices rows via dim_date=2026-07-27",
        int((pd.to_datetime(px["date"]) == "2026-07-27").sum()))
    sg2 = dfs["fact_signals"].copy()
    sg2["ym"] = pd.to_datetime(sg2["date"]).dt.strftime("%Y-%m")
    for sym in ("AAPL", "CRM"):
        sub = sg2[(sg2["symbol"] == sym) & (sg2["ym"] == "2024-11")]
        add(f"rel_composite_{sym}_202411",
            f"Composite Signal via dim_date.year_month=2024-11 + dim_symbol={sym}",
            float(sub["composite_signal"].mean()))
    add("rel_composite_DRAM_202411",
        "Composite Signal DRAM 2024-11 - no rows, must be BLANK", "BLANK")
    nw2 = dfs["fact_news"]
    for d in ("2026-07-24", "2026-07-25", "2026-07-26"):
        add(f"rel_news_{d}", f"fact_news rows via dim_date={d}",
            int((pd.to_datetime(nw2["date"]) == d).sum()))
    add("rel_odds_20260715", "fact_event_odds rows via dim_date=2026-07-15",
        int((pd.to_datetime(dfs["fact_event_odds"]["as_of"]) == "2026-07-15").sum()))
    add("rel_positions_20260727", "fact_positions rows via dim_date=2026-07-27",
        int((pd.to_datetime(dfs["fact_positions"]["as_of"]) == "2026-07-27").sum()))
    add("rel_equity_20260727", "fact_backtest_equity rows via dim_date=2026-07-27",
        int((pd.to_datetime(dfs["fact_backtest_equity"]["date"]) == "2026-07-27").sum()))
    add("rel_news_SPY", "fact_news rows via dim_symbol=SPY",
        int((nw2["symbol"] == "SPY").sum()))

    # ---- 8b. 空白成员防回归（fact_news.symbol 有 500 空值，M 层重映射为 (untagged)）----
    # 注意：普通 DISTINCTCOUNT(dim_symbol[symbol])=35 抓不到回归——RI 违规造出的虚拟空白行
    # 也被 DISTINCTCOUNT 计入，修复前(34实体+1虚拟空白)与修复后(35实体)都返回 35，永远绿。
    # 改用三点组合：NOBLANK 证明 (untagged) 成员存在；untagged 行数证明 500 条真实新闻挂上了；
    # canary 证明没有任何 fact_news 行落在 RI 虚拟空白成员上（任何新的未匹配值都会让它变红）。
    n_null_sym = int(nw2["symbol"].isna().sum())
    add("rel_news_untagged",
        "fact_news rows via dim_symbol=(untagged) - null symbols remapped in M",
        n_null_sym)
    add("dim_symbol_noblank",
        "DISTINCTCOUNTNOBLANK(dim_symbol.symbol) - 34 real symbols + (untagged)",
        int(dfs["dim_symbol"]["symbol"].nunique()) + 1)
    add("blank_member_canary",
        "fact_news rows on RI-violation blank member - must be BLANK", "BLANK")

    # ---- 9. 编码金丝雀：event_title 必须含弯引号（U+2019），乱码则 Power Query 读错 ----
    ev = dfs["fact_event_odds"]
    curly = ev["event_title"].astype(str).str.contains("’").sum()
    add("encoding_curly_quote_rows", "event_title rows containing U+2019", int(curly))

    out = pd.DataFrame(checks, columns=["check_id", "description", "expected"])
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"[verify] {len(out)} expected values -> {OUT}")
    print(out.to_string(index=False, max_colwidth=48))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

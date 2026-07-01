"""pandas ETL：把项目数据（价格 / 日历 / 持仓 / 成交 / 信号 / 回测）装载进 `raw` 层。

分工（ELT，非 ETL 的 T 在这里）：
- **这里只做 Extract + Load**：接收 DataFrame/领域对象 → 原样写 `raw.*`（附 `loaded_at`
  审计列），**不做业务变换**——清洗/改名/建模全部在 dbt SQL（staging → marts），
  变换逻辑因此可被 dbt test 断言、可被 SQL 阅读者审计。
- **幂等**：每个 loader 都是「先删本批次键，再插入」（delete-then-insert by batch key），
  重跑不产生重复行（测试断言）。
- **可测**：全部函数吃注入的连接（`connect(":memory:")`）+ 合成数据，离线单测。
"""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd
from duckdb import DuckDBPyConnection

from quantai.backtest.engine import BacktestResult
from quantai.data.calendar import TradingCalendar
from quantai.portfolio.loader import Portfolio

_AUDIT = "loaded_at TIMESTAMP DEFAULT current_timestamp"

#: 全部 raw 表 DDL（幂等）。集中登记，`init_raw_tables` 一次建齐——
#: 某类数据还没产生时（如还没跑过 live 就没有成交），dbt 仍能对空表建模，
#: 而不是 Catalog Error（实测踩过的坑：CLI 首跑无 trades 表 → dbt build 失败）。
_DDL: dict[str, str] = {
    "prices": f"""CREATE TABLE IF NOT EXISTS raw.prices (
        symbol VARCHAR NOT NULL,
        date DATE NOT NULL,
        open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT,
        {_AUDIT})""",
    "trading_days": f"""CREATE TABLE IF NOT EXISTS raw.trading_days (
        date DATE NOT NULL,
        is_early_close BOOLEAN NOT NULL,
        {_AUDIT})""",
    "positions": f"""CREATE TABLE IF NOT EXISTS raw.positions (
        as_of DATE NOT NULL,
        symbol VARCHAR NOT NULL,
        shares DOUBLE NOT NULL,
        cost_basis DOUBLE NOT NULL,
        open_date DATE NOT NULL,
        {_AUDIT})""",
    "portfolio_cash": f"""CREATE TABLE IF NOT EXISTS raw.portfolio_cash (
        as_of DATE NOT NULL,
        cash DOUBLE NOT NULL,
        {_AUDIT})""",
    "trades": f"""CREATE TABLE IF NOT EXISTS raw.trades (
        run_id VARCHAR NOT NULL,
        seq INTEGER NOT NULL,
        symbol VARCHAR NOT NULL,
        action VARCHAR NOT NULL,
        price DOUBLE,
        shares DOUBLE,
        trace_id VARCHAR,
        {_AUDIT})""",
    "signals": f"""CREATE TABLE IF NOT EXISTS raw.signals (
        symbol VARCHAR NOT NULL,
        date DATE NOT NULL,
        trend_signal INTEGER, momentum_signal INTEGER, ma_cross_signal INTEGER,
        breakout_signal INTEGER, composite_signal DOUBLE, signal_strength VARCHAR,
        {_AUDIT})""",
    "backtest_runs": f"""CREATE TABLE IF NOT EXISTS raw.backtest_runs (
        run_id VARCHAR NOT NULL,
        strategy VARCHAR, symbol VARCHAR, fill_timing VARCHAR,
        total_return DOUBLE, cagr DOUBLE, annual_volatility DOUBLE, sharpe DOUBLE,
        max_drawdown DOUBLE, win_rate DOUBLE, total_turnover DOUBLE,
        start_date VARCHAR, end_date VARCHAR, trading_days INTEGER,
        {_AUDIT})""",
    "backtest_equity": f"""CREATE TABLE IF NOT EXISTS raw.backtest_equity (
        run_id VARCHAR NOT NULL,
        date DATE NOT NULL,
        equity DOUBLE NOT NULL,
        daily_return DOUBLE,
        {_AUDIT})""",
}


def init_raw_tables(con: DuckDBPyConnection) -> None:
    """把全部 raw 表建齐（空表也建）——保证 dbt 全模型可编译，无论数据是否已产生。"""
    for ddl in _DDL.values():
        con.execute(ddl)


def _ensure(con: DuckDBPyConnection, ddl: str) -> None:
    con.execute(ddl)


# --------------------------------------------------------------------------- #
# prices
# --------------------------------------------------------------------------- #
def load_prices(con: DuckDBPyConnection, prices: Mapping[str, pd.DataFrame]) -> int:
    """{symbol: OHLCV df} → `raw.prices`。批次键 = symbol（重跑该 symbol 全量替换）。

    输入 df 需含 close（open/high/low/volume 可缺，落 NULL）；索引为日期。
    """
    _ensure(
        con,
        f"""CREATE TABLE IF NOT EXISTS raw.prices (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT,
            {_AUDIT}
        )""",
    )
    total = 0
    for symbol, df in prices.items():
        if df is None or len(df) == 0:
            continue
        frame = pd.DataFrame(index=df.index)
        for col in ("open", "high", "low", "close", "volume"):
            frame[col] = df[col] if col in df.columns else None
        frame = frame.reset_index(names="date")
        frame.insert(0, "symbol", str(symbol).upper())
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.date
        con.execute("DELETE FROM raw.prices WHERE symbol = ?", [str(symbol).upper()])
        con.register("_stg_prices", frame)
        con.execute(
            """INSERT INTO raw.prices (symbol, date, open, high, low, close, volume)
               SELECT symbol, date, open, high, low, close, volume FROM _stg_prices"""
        )
        con.unregister("_stg_prices")
        total += len(frame)
    return total


# --------------------------------------------------------------------------- #
# NYSE 交易日历（dim_date 的数据源）
# --------------------------------------------------------------------------- #
def load_trading_days(con: DuckDBPyConnection, start: str, end: str) -> int:
    """NYSE 交易日 → `raw.trading_days`（重跑全量替换）。

    dbt 的 `dim_date` 用它补齐「是否交易日/周几/年月季」等属性；日历真相来自
    `pandas_market_calendars`（含一次性休市/半日市），不手搓规则。
    """
    _ensure(
        con,
        f"""CREATE TABLE IF NOT EXISTS raw.trading_days (
            date DATE NOT NULL,
            is_early_close BOOLEAN NOT NULL,
            {_AUDIT}
        )""",
    )
    cal = TradingCalendar()
    days = cal.get_trading_days(start, end)
    early = set(cal.early_closes(start, end))
    frame = pd.DataFrame(
        {"date": list(days), "is_early_close": [d in early for d in days]}
    )
    con.execute("DELETE FROM raw.trading_days")
    con.register("_stg_days", frame)
    con.execute(
        "INSERT INTO raw.trading_days (date, is_early_close) SELECT date, is_early_close FROM _stg_days"
    )
    con.unregister("_stg_days")
    return len(frame)


# --------------------------------------------------------------------------- #
# positions（真实持仓快照）
# --------------------------------------------------------------------------- #
def load_positions(
    con: DuckDBPyConnection, portfolio: Portfolio, as_of: str
) -> int:
    """真实持仓 lots + 现金 → `raw.positions` / `raw.portfolio_cash`。

    批次键 = as_of（同一快照日重跑替换；不同日期追加 → 持仓历史随时间积累）。
    """
    _ensure(
        con,
        f"""CREATE TABLE IF NOT EXISTS raw.positions (
            as_of DATE NOT NULL,
            symbol VARCHAR NOT NULL,
            shares DOUBLE NOT NULL,
            cost_basis DOUBLE NOT NULL,
            open_date DATE NOT NULL,
            {_AUDIT}
        )""",
    )
    _ensure(
        con,
        f"""CREATE TABLE IF NOT EXISTS raw.portfolio_cash (
            as_of DATE NOT NULL,
            cash DOUBLE NOT NULL,
            {_AUDIT}
        )""",
    )
    con.execute("DELETE FROM raw.positions WHERE as_of = ?", [as_of])
    con.execute("DELETE FROM raw.portfolio_cash WHERE as_of = ?", [as_of])
    for p in portfolio.positions:
        con.execute(
            "INSERT INTO raw.positions (as_of, symbol, shares, cost_basis, open_date) VALUES (?,?,?,?,?)",
            [as_of, p.symbol, p.shares, p.cost_basis, str(p.open_date)],
        )
    con.execute(
        "INSERT INTO raw.portfolio_cash (as_of, cash) VALUES (?,?)", [as_of, portfolio.cash]
    )
    return len(portfolio.positions)


# --------------------------------------------------------------------------- #
# trades（live PaperBroker 成交流水）
# --------------------------------------------------------------------------- #
def load_trades(
    con: DuckDBPyConnection, trades: Iterable[Mapping], run_id: str
) -> int:
    """成交记录（`PaperBroker.orders` 的 dict 形状）→ `raw.trades`。批次键 = run_id。

    dict 键：ticker / action / price / shares / trace_id（缺失落 NULL）。
    """
    _ensure(
        con,
        f"""CREATE TABLE IF NOT EXISTS raw.trades (
            run_id VARCHAR NOT NULL,
            seq INTEGER NOT NULL,
            symbol VARCHAR NOT NULL,
            action VARCHAR NOT NULL,
            price DOUBLE,
            shares DOUBLE,
            trace_id VARCHAR,
            {_AUDIT}
        )""",
    )
    con.execute("DELETE FROM raw.trades WHERE run_id = ?", [run_id])
    n = 0
    for i, t in enumerate(trades):
        con.execute(
            "INSERT INTO raw.trades (run_id, seq, symbol, action, price, shares, trace_id) VALUES (?,?,?,?,?,?,?)",
            [
                run_id,
                i,
                str(t.get("ticker", "")).upper(),
                str(t.get("action", "")),
                t.get("price"),
                t.get("shares"),
                t.get("trace_id"),
            ],
        )
        n += 1
    return n


# --------------------------------------------------------------------------- #
# signals（SignalGenerator 输出）
# --------------------------------------------------------------------------- #
def load_signals(
    con: DuckDBPyConnection, symbol: str, signals: pd.DataFrame
) -> int:
    """`SignalGenerator.generate` 的输出 → `raw.signals`。批次键 = symbol。

    列：trend/momentum/ma_cross/breakout/composite_signal + signal_strength（枚举文本）。
    """
    _ensure(
        con,
        f"""CREATE TABLE IF NOT EXISTS raw.signals (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            trend_signal INTEGER, momentum_signal INTEGER, ma_cross_signal INTEGER,
            breakout_signal INTEGER, composite_signal DOUBLE, signal_strength VARCHAR,
            {_AUDIT}
        )""",
    )
    frame = signals.copy()
    frame["signal_strength"] = frame["signal_strength"].astype("string")
    frame = frame.reset_index(names="date")
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.date
    frame.insert(0, "symbol", str(symbol).upper())
    con.execute("DELETE FROM raw.signals WHERE symbol = ?", [str(symbol).upper()])
    con.register("_stg_signals", frame)
    con.execute(
        """INSERT INTO raw.signals
           (symbol, date, trend_signal, momentum_signal, ma_cross_signal,
            breakout_signal, composite_signal, signal_strength)
           SELECT symbol, date, trend_signal, momentum_signal, ma_cross_signal,
                  breakout_signal, composite_signal, signal_strength
           FROM _stg_signals"""
    )
    con.unregister("_stg_signals")
    return len(frame)


# --------------------------------------------------------------------------- #
# backtest（run_backtest 结果：指标行 + 净值曲线）
# --------------------------------------------------------------------------- #
def load_backtest(
    con: DuckDBPyConnection,
    run_id: str,
    result: BacktestResult,
    strategy: str = "",
    symbol: str = "",
) -> int:
    """`BacktestResult` → `raw.backtest_runs`（指标 1 行）+ `raw.backtest_equity`（曲线）。

    批次键 = run_id。metrics 展开为列（与 `PerformanceReport` 字段一一对应）。
    """
    _ensure(
        con,
        f"""CREATE TABLE IF NOT EXISTS raw.backtest_runs (
            run_id VARCHAR NOT NULL,
            strategy VARCHAR, symbol VARCHAR, fill_timing VARCHAR,
            total_return DOUBLE, cagr DOUBLE, annual_volatility DOUBLE, sharpe DOUBLE,
            max_drawdown DOUBLE, win_rate DOUBLE, total_turnover DOUBLE,
            start_date VARCHAR, end_date VARCHAR, trading_days INTEGER,
            {_AUDIT}
        )""",
    )
    _ensure(
        con,
        f"""CREATE TABLE IF NOT EXISTS raw.backtest_equity (
            run_id VARCHAR NOT NULL,
            date DATE NOT NULL,
            equity DOUBLE NOT NULL,
            daily_return DOUBLE,
            {_AUDIT}
        )""",
    )
    m = result.metrics
    con.execute("DELETE FROM raw.backtest_runs WHERE run_id = ?", [run_id])
    con.execute("DELETE FROM raw.backtest_equity WHERE run_id = ?", [run_id])
    con.execute(
        """INSERT INTO raw.backtest_runs
           (run_id, strategy, symbol, fill_timing, total_return, cagr, annual_volatility,
            sharpe, max_drawdown, win_rate, total_turnover, start_date, end_date, trading_days)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            run_id, strategy, str(symbol).upper(), result.fill_timing,
            m.total_return, m.cagr, m.annual_volatility, m.sharpe,
            m.max_drawdown, m.win_rate, result.total_turnover,
            m.start, m.end, m.trading_days,
        ],
    )
    eq = result.equity
    rets = result.returns.reindex(eq.index)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(eq.index).tz_localize(None).date,
            "equity": eq.to_numpy(dtype=float),
            "daily_return": rets.to_numpy(dtype=float),
        }
    )
    frame.insert(0, "run_id", run_id)
    con.register("_stg_bt", frame)
    con.execute(
        "INSERT INTO raw.backtest_equity (run_id, date, equity, daily_return) "
        "SELECT run_id, date, equity, daily_return FROM _stg_bt"
    )
    con.unregister("_stg_bt")
    return 1 + len(frame)

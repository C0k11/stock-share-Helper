"""quantai.warehouse — 本地数据仓库（DuckDB）装载层。

- `db.connect`：config 驱动的库连接 + 三层 schema（raw / staging / marts）。
- `etl`：pandas Extract-Load（原样落 raw + 审计列，幂等重跑）；
  变换（清洗/星型建模）在 dbt SQL：`warehouse/`（staging → marts）。

Tableau/BI 只查 `marts.*`（dim_symbol / dim_date + fact_*）。
"""

from quantai.warehouse.db import SCHEMAS, connect
from quantai.warehouse.etl import (
    load_backtest,
    load_news,
    load_news_scores,
    load_positions,
    load_prices,
    load_signals,
    load_trades,
    load_trading_days,
)

__all__ = [
    "connect",
    "SCHEMAS",
    "load_prices",
    "load_trading_days",
    "load_positions",
    "load_trades",
    "load_signals",
    "load_backtest",
    "load_news",
    "load_news_scores",
]

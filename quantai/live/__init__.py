"""quantai.live —— 实盘/模拟事件循环层（子系统 C 的运行时）。

把 `agents` 大脑接到一个事件驱动的运行时上：`DataFeed` 推行情 -> `TradingEngine` 分发 ->
策略决策 -> 券商撮合。**只管运行时编排**（线程、队列、数据源），决策逻辑在 `agents/`、
撮合成本在 `execution/`。

已迁：events / engine / data_feed。（broker / strategy_adapter / runner 后续增量。）
"""
from __future__ import annotations

from quantai.live.data_feed import (
    DataFeed,
    SimulatedDataFeed,
    YFinanceDataFeed,
    create_data_feed,
)
from quantai.live.engine import TradingEngine
from quantai.live.events import Event, EventType

__all__ = [
    "Event",
    "EventType",
    "TradingEngine",
    "DataFeed",
    "YFinanceDataFeed",
    "SimulatedDataFeed",
    "create_data_feed",
]

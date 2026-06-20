"""实盘事件类型 —— 忠实迁移自 `src/trading/event.py`。

事件驱动架构的最小词汇表：行情进来（MARKET_DATA）-> 策略产出意图（SIGNAL）->
下单（ORDER）-> 成交（FILL）；另有 LOG（Agent 思考/系统状态，给 UI/TTS 用）与 ERROR。
`priority` 给前端用：0 纯文本 / 1 气泡 / 2 播报。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(Enum):
    MARKET_DATA = "MARKET_DATA"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    ERROR = "ERROR"
    LOG = "LOG"  # Agent 思考过程 / 内部状态


@dataclass
class Event:
    type: EventType
    timestamp: datetime
    payload: Any
    priority: int = 0  # 0: 低(纯文本) / 1: 中(气泡) / 2: 高(播报)

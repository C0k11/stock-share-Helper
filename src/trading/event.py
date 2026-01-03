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
    LOG = "LOG"  # Agent 思考过程/内部状态


@dataclass
class Event:
    type: EventType
    timestamp: datetime
    payload: Any
    priority: int = 0  # 0: Low (Text only), 1: Medium (Bubble), 2: High (Speak)

"""交易规则：止损 / 止盈 / 调仓阈值 -> 动作与交易信号。

从旧 `src/strategy/rules.py` 迁移，逻辑不变，加类型标注。纯函数式（基于当前价/入场价/仓位），
无时间序列、无 lookahead。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Action(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    ADD = "add"
    REDUCE = "reduce"
    CLEAR = "clear"


@dataclass
class TradeSignal:
    symbol: str
    action: Action
    target_position: float
    current_position: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""


class TradingRules:
    """根据止损止盈与调仓阈值决定交易动作。"""

    def __init__(
        self,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = 0.25,
        rebalance_threshold: float = 0.05,
        max_single_position: float = 0.5,
    ) -> None:
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.rebalance_threshold = rebalance_threshold
        self.max_single_position = max_single_position

    def check_stop_loss(self, current_price: float, entry_price: float) -> bool:
        if entry_price <= 0:
            return False
        return (current_price - entry_price) / entry_price <= -self.stop_loss_pct

    def check_take_profit(self, current_price: float, entry_price: float) -> bool:
        if entry_price <= 0:
            return False
        return (current_price - entry_price) / entry_price >= self.take_profit_pct

    def should_rebalance(self, current_position: float, target_position: float) -> bool:
        return abs(current_position - target_position) >= self.rebalance_threshold

    def determine_action(
        self,
        current_position: float,
        target_position: float,
        current_price: float,
        entry_price: Optional[float] = None,
    ) -> Action:
        if entry_price and current_position > 0:
            if self.check_stop_loss(current_price, entry_price):
                return Action.CLEAR
            if self.check_take_profit(current_price, entry_price):
                return Action.REDUCE
        if not self.should_rebalance(current_position, target_position):
            return Action.HOLD
        if target_position > current_position:
            return Action.BUY if current_position == 0 else Action.ADD
        return Action.CLEAR if target_position == 0 else Action.REDUCE

    def generate_trade_signal(
        self,
        symbol: str,
        current_position: float,
        target_position: float,
        current_price: float,
        entry_price: Optional[float] = None,
        reason: str = "",
    ) -> TradeSignal:
        action = self.determine_action(
            current_position, target_position, current_price, entry_price
        )
        stop_loss = take_profit = None
        if action in (Action.BUY, Action.ADD):
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
        return TradeSignal(
            symbol=symbol,
            action=action,
            target_position=target_position,
            current_position=current_position,
            entry_price=current_price if action == Action.BUY else entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
        )

"""
交易规则模块 - 入场/离场条件、止损止盈
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class Action(Enum):
    """交易动作"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    ADD = "add"         # 加仓
    REDUCE = "reduce"   # 减仓
    CLEAR = "clear"     # 清仓


@dataclass
class TradeSignal:
    """交易信号"""
    symbol: str
    action: Action
    target_position: float
    current_position: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""


class TradingRules:
    """交易规则引擎"""
    
    def __init__(
        self,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = 0.25,
        rebalance_threshold: float = 0.05,
        max_single_position: float = 0.5
    ):
        """
        Args:
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
            rebalance_threshold: 调仓阈值
            max_single_position: 单标的最大仓位
        """
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.rebalance_threshold = rebalance_threshold
        self.max_single_position = max_single_position
    
    def check_stop_loss(
        self,
        current_price: float,
        entry_price: float
    ) -> bool:
        """检查是否触发止损"""
        if entry_price <= 0:
            return False
        return_pct = (current_price - entry_price) / entry_price
        return return_pct <= -self.stop_loss_pct
    
    def check_take_profit(
        self,
        current_price: float,
        entry_price: float
    ) -> bool:
        """检查是否触发止盈"""
        if entry_price <= 0:
            return False
        return_pct = (current_price - entry_price) / entry_price
        return return_pct >= self.take_profit_pct
    
    def should_rebalance(
        self,
        current_position: float,
        target_position: float
    ) -> bool:
        """检查是否需要调仓"""
        diff = abs(current_position - target_position)
        return diff >= self.rebalance_threshold
    
    def determine_action(
        self,
        current_position: float,
        target_position: float,
        current_price: float,
        entry_price: Optional[float] = None
    ) -> Action:
        """
        决定交易动作
        
        Args:
            current_position: 当前仓位
            target_position: 目标仓位
            current_price: 当前价格
            entry_price: 入场价格
        """
        # 1. 检查止损
        if entry_price and current_position > 0:
            if self.check_stop_loss(current_price, entry_price):
                return Action.CLEAR
        
        # 2. 检查止盈
        if entry_price and current_position > 0:
            if self.check_take_profit(current_price, entry_price):
                return Action.REDUCE
        
        # 3. 检查是否需要调仓
        if not self.should_rebalance(current_position, target_position):
            return Action.HOLD
        
        # 4. 决定方向
        if target_position > current_position:
            if current_position == 0:
                return Action.BUY
            else:
                return Action.ADD
        else:
            if target_position == 0:
                return Action.CLEAR
            else:
                return Action.REDUCE
    
    def generate_trade_signal(
        self,
        symbol: str,
        current_position: float,
        target_position: float,
        current_price: float,
        entry_price: Optional[float] = None,
        reason: str = ""
    ) -> TradeSignal:
        """生成交易信号"""
        action = self.determine_action(
            current_position, target_position, current_price, entry_price
        )
        
        # 计算止损止盈价位
        stop_loss = None
        take_profit = None
        
        if action in [Action.BUY, Action.ADD]:
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
            reason=reason
        )
    
    def format_recommendation(self, signal: TradeSignal) -> Dict:
        """格式化交易建议"""
        action_map = {
            Action.BUY: "买入建仓",
            Action.SELL: "卖出",
            Action.HOLD: "持有观望",
            Action.ADD: "加仓",
            Action.REDUCE: "减仓",
            Action.CLEAR: "清仓"
        }
        
        return {
            "symbol": signal.symbol,
            "action": action_map.get(signal.action, "观望"),
            "action_code": signal.action.value,
            "target_position": round(signal.target_position * 100, 1),
            "current_position": round(signal.current_position * 100, 1),
            "position_change": round((signal.target_position - signal.current_position) * 100, 1),
            "stop_loss": round(signal.stop_loss, 2) if signal.stop_loss else None,
            "take_profit": round(signal.take_profit, 2) if signal.take_profit else None,
            "reason": signal.reason
        }

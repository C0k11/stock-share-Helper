"""
回测引擎
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from loguru import logger

from .costs import CostModel
from .metrics import PerformanceMetrics


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 100000.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    rebalance_frequency: str = "weekly"  # daily, weekly, monthly
    rebalance_threshold: float = 0.05


@dataclass
class BacktestResult:
    """回测结果"""
    equity_curve: pd.Series = None
    positions_history: pd.DataFrame = None
    trades: List[Dict] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    config: BacktestConfig = None


class BacktestEngine:
    """回测引擎"""
    
    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        cost_model: Optional[CostModel] = None
    ):
        self.config = config or BacktestConfig()
        self.cost_model = cost_model or CostModel()
        self.metrics_calculator = PerformanceMetrics()
    
    def run(
        self,
        price_data: Dict[str, pd.DataFrame],
        signal_func: Callable[[Dict[str, pd.DataFrame], str], Dict[str, float]],
        symbols: Optional[List[str]] = None
    ) -> BacktestResult:
        """
        运行回测
        
        Args:
            price_data: 价格数据 {symbol: DataFrame}
            signal_func: 信号函数，输入(价格数据, 日期)，输出{symbol: target_weight}
            symbols: 标的列表
        
        Returns:
            BacktestResult
        """
        if symbols is None:
            symbols = list(price_data.keys())
        
        # 对齐日期
        dates = self._get_aligned_dates(price_data, symbols)
        
        if self.config.start_date:
            dates = [d for d in dates if str(d) >= self.config.start_date]
        if self.config.end_date:
            dates = [d for d in dates if str(d) <= self.config.end_date]
        
        # 初始化
        capital = self.config.initial_capital
        positions = {s: 0.0 for s in symbols}  # 持仓股数
        weights = {s: 0.0 for s in symbols}    # 持仓权重
        
        equity_history = []
        positions_history = []
        trades = []
        last_rebalance = None
        
        for i, date in enumerate(dates):
            # 获取当日价格
            prices = {}
            for s in symbols:
                df = price_data[s]
                if date in df.index:
                    prices[s] = df.loc[date, "close"]
            
            if not prices:
                continue
            
            # 计算当前组合价值
            portfolio_value = capital
            for s in symbols:
                if s in prices and positions[s] > 0:
                    portfolio_value += positions[s] * prices[s]
            
            # 检查是否需要再平衡
            should_rebalance = self._should_rebalance(date, last_rebalance)
            
            if should_rebalance:
                # 获取目标权重
                target_weights = signal_func(price_data, str(date))
                
                # 执行再平衡
                new_positions, trade_costs, trade_list = self._rebalance(
                    portfolio_value, positions, target_weights, prices
                )
                
                positions = new_positions
                weights = target_weights
                capital -= trade_costs
                trades.extend(trade_list)
                last_rebalance = date
            
            # 记录历史
            equity_history.append({
                "date": date,
                "portfolio_value": portfolio_value,
                "cash": capital
            })
            
            positions_history.append({
                "date": date,
                **{f"{s}_weight": weights.get(s, 0) for s in symbols}
            })
        
        # 构建结果
        equity_df = pd.DataFrame(equity_history).set_index("date")
        positions_df = pd.DataFrame(positions_history).set_index("date")
        
        result = BacktestResult(
            equity_curve=equity_df["portfolio_value"],
            positions_history=positions_df,
            trades=trades,
            config=self.config
        )
        
        # 计算绩效指标
        result.metrics = self.metrics_calculator.compute_all(result.equity_curve)
        
        return result
    
    def _get_aligned_dates(
        self,
        price_data: Dict[str, pd.DataFrame],
        symbols: List[str]
    ) -> List:
        """获取对齐的日期序列"""
        all_dates = set()
        for s in symbols:
            if s in price_data:
                all_dates.update(price_data[s].index.tolist())
        return sorted(all_dates)
    
    def _should_rebalance(self, current_date, last_rebalance) -> bool:
        """判断是否需要再平衡"""
        if last_rebalance is None:
            return True
        
        freq = self.config.rebalance_frequency
        
        if freq == "daily":
            return True
        elif freq == "weekly":
            # 每周一再平衡
            return current_date.weekday() == 0
        elif freq == "monthly":
            # 每月第一个交易日
            return current_date.day <= 7 and current_date.weekday() == 0
        
        return False
    
    def _rebalance(
        self,
        portfolio_value: float,
        current_positions: Dict[str, float],
        target_weights: Dict[str, float],
        prices: Dict[str, float]
    ) -> tuple:
        """
        执行再平衡
        
        Returns:
            (new_positions, total_costs, trade_list)
        """
        new_positions = {}
        total_costs = 0.0
        trade_list = []
        
        for symbol, target_weight in target_weights.items():
            if symbol not in prices:
                new_positions[symbol] = current_positions.get(symbol, 0)
                continue
            
            price = prices[symbol]
            target_value = portfolio_value * target_weight
            target_shares = target_value / price
            
            current_shares = current_positions.get(symbol, 0)
            shares_diff = target_shares - current_shares
            
            if abs(shares_diff) > 0.01:  # 忽略微小差异
                trade_value = abs(shares_diff * price)
                cost = self.cost_model.compute_cost(trade_value)
                total_costs += cost
                
                trade_list.append({
                    "symbol": symbol,
                    "shares": shares_diff,
                    "price": price,
                    "value": shares_diff * price,
                    "cost": cost,
                    "action": "buy" if shares_diff > 0 else "sell"
                })
            
            new_positions[symbol] = target_shares
        
        return new_positions, total_costs, trade_list


def run_simple_backtest(
    price_data: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    initial_capital: float = 100000
) -> BacktestResult:
    """简单回测（固定权重）的便捷函数"""
    engine = BacktestEngine(BacktestConfig(initial_capital=initial_capital))
    
    def fixed_weight_signal(data, date):
        return weights
    
    return engine.run(price_data, fixed_weight_signal)

"""
成本模型 - 佣金、滑点、税费
"""

from typing import Optional
from loguru import logger


class CostModel:
    """交易成本模型"""
    
    def __init__(
        self,
        commission_rate: float = 0.0005,
        min_commission: float = 1.0,
        slippage_bps: float = 5.0,
        stamp_duty: float = 0.0,  # 印花税（A股/港股）
        platform_fee: float = 0.0  # 平台费（港股）
    ):
        """
        Args:
            commission_rate: 佣金率
            min_commission: 最低佣金
            slippage_bps: 滑点（基点）
            stamp_duty: 印花税率
            platform_fee: 平台费
        """
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage_bps = slippage_bps
        self.stamp_duty = stamp_duty
        self.platform_fee = platform_fee
    
    def compute_cost(
        self,
        trade_value: float,
        is_sell: bool = False
    ) -> float:
        """
        计算交易成本
        
        Args:
            trade_value: 交易金额
            is_sell: 是否为卖出（影响印花税）
        
        Returns:
            总成本
        """
        # 佣金
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        # 滑点
        slippage = trade_value * (self.slippage_bps / 10000)
        
        # 印花税（通常卖出时收取）
        stamp = trade_value * self.stamp_duty if is_sell else 0
        
        # 平台费
        platform = self.platform_fee
        
        total = commission + slippage + stamp + platform
        
        return total
    
    def compute_round_trip_cost(self, trade_value: float) -> float:
        """计算往返交易成本（买入+卖出）"""
        buy_cost = self.compute_cost(trade_value, is_sell=False)
        sell_cost = self.compute_cost(trade_value, is_sell=True)
        return buy_cost + sell_cost
    
    @classmethod
    def us_stock(cls) -> "CostModel":
        """美股成本模型"""
        return cls(
            commission_rate=0.0005,
            min_commission=1.0,
            slippage_bps=5.0,
            stamp_duty=0.0,
            platform_fee=0.0
        )
    
    @classmethod
    def hk_stock(cls) -> "CostModel":
        """港股成本模型"""
        return cls(
            commission_rate=0.001,
            min_commission=50.0,  # HKD
            slippage_bps=10.0,
            stamp_duty=0.001,    # 0.1%
            platform_fee=15.0    # HKD
        )
    
    @classmethod
    def cn_stock(cls) -> "CostModel":
        """A股成本模型"""
        return cls(
            commission_rate=0.0003,
            min_commission=5.0,   # CNY
            slippage_bps=5.0,
            stamp_duty=0.001,    # 0.1%（卖出）
            platform_fee=0.0
        )
    
    def describe(self) -> str:
        """描述成本模型"""
        return (
            f"佣金率: {self.commission_rate:.2%}, "
            f"最低佣金: {self.min_commission}, "
            f"滑点: {self.slippage_bps}bps, "
            f"印花税: {self.stamp_duty:.2%}"
        )

"""交易成本模型（US-only）：佣金 + 滑点。

从旧 `src/backtest/costs.py` 的 `CostModel` 迁移并 US 化：
- 移除 `hk_stock()` / `cn_stock()` 工厂与 `stamp_duty` / `platform_fee`（HK/CN 专属；US 本就为 0）。
- 重命名为 `TransactionCosts`，与配置层的 `quantai.config.CostModel`（参数 schema）区分：
  schema 描述"成本参数"，本类是"成本计算器"。
"""

from __future__ import annotations


class TransactionCosts:
    """美股单边交易成本 = max(金额*佣金率, 最低佣金) + 金额*滑点。"""

    def __init__(
        self,
        commission_rate: float = 0.0005,
        min_commission: float = 1.0,
        slippage_bps: float = 5.0,
    ) -> None:
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage_bps = slippage_bps

    def cost(self, trade_value: float) -> float:
        """单边成交金额对应的总成本。"""
        commission = max(trade_value * self.commission_rate, self.min_commission)
        slippage = trade_value * (self.slippage_bps / 10000.0)
        return commission + slippage

    def round_trip(self, trade_value: float) -> float:
        """往返成本（买入 + 卖出）。"""
        return self.cost(trade_value) * 2

    @classmethod
    def from_config(cls, costs) -> "TransactionCosts":
        """从 `quantai.config` 的 backtest.costs 构造。"""
        return cls(
            commission_rate=costs.commission_rate,
            min_commission=costs.min_commission,
            slippage_bps=costs.slippage_bps,
        )

    def describe(self) -> str:
        return (
            f"佣金率: {self.commission_rate:.2%}, "
            f"最低佣金: {self.min_commission}, "
            f"滑点: {self.slippage_bps}bps"
        )

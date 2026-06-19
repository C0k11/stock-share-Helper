"""quantai.execution —— 执行层（成本 / 交易规则 / 执行状态机 / 撮合模拟）。

US-only：成本模型移除 HK/CN（印花税/平台费）。
撮合是否偷看未来取决于回测喂的是哪根 K 线（见 simulator 文档）。

用法：
    from quantai.execution import TransactionCosts, TradingRules, Action, TickerExecutionState, ExecutionSimulator
"""

from .costs import TransactionCosts
from .rules import Action, TradeSignal, TradingRules
from .simulator import ExecutionResult, ExecutionSimulator
from .state import TickerExecutionState

__all__ = [
    "TransactionCosts",
    "TradingRules",
    "Action",
    "TradeSignal",
    "TickerExecutionState",
    "ExecutionSimulator",
    "ExecutionResult",
]

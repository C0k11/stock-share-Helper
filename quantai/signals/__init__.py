"""quantai.signals —— 规则信号层（把特征/价格转成方向信号）。

无 lookahead：因果窗口；成交时点由 backtest 的 next_open 保证。

用法：
    from quantai.signals import SignalGenerator
"""

from .generator import SignalGenerator

__all__ = ["SignalGenerator"]

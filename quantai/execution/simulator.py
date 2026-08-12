"""成交撮合模拟器：给定某根 K 线的 OHLC，按撮合模式产出成交价/是否成交。

从旧 `src/execution/simulator.py` 迁移，加类型标注；并新增 "open" 模式（修复同根撮合 lookahead）。

撮合模式：
- "open"   ：按开盘价 +/-滑点成交（市价开盘单）。**默认**——纸面/实盘喂 t+1 日 K 线时即为 next_open，
             与回测 fill_timing="next_open" 同口径，不偷看决策当日收盘。
- "close"  ：按收盘价 +/-滑点成交（MOC）。若喂决策同根(t 日)的 close，即同根 lookahead，仅供复现旧版。
- "passive"：开盘价基础上挂被动限价，触及才成交，否则错过。
- "midpoint"：(high+low)/2 限价。

lookahead 关系：本类只是"给定一根 K 线如何成交"的纯模型，
**是否偷看未来取决于回测/实盘循环喂的是哪一根 K 线**。口径要求：用 t 日信息决策，
喂 **t+1 日** 的 OHLC 来撮合（next_open）。默认 "open" + 喂 t+1 K 线 = 在 open[t+1] 成交，即 next_open，
与回测一致；若用 "close" + 喂同根(t 日)，即旧版虚高的来源。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExecutionResult:
    filled: bool
    price: float
    note: str


class ExecutionSimulator:
    """单笔买/卖的撮合模型，并累计成交统计。"""

    def __init__(
        self,
        mode: str = "open",
        slippage_bps: float = 10.0,
        limit_threshold_bps: float = 50.0,
    ) -> None:
        self.mode = str(mode or "open").strip().lower()
        self.slippage = float(slippage_bps) / 10000.0
        self.limit_k = float(limit_threshold_bps) / 10000.0
        self.stats: dict[str, float] = {
            "total_orders": 0.0,
            "filled_orders": 0.0,
            "missed_orders": 0.0,
            "total_slippage_cost": 0.0,
        }

    def _bump(self, key: str, amount: float = 1.0) -> None:
        self.stats[key] = self.stats.get(key, 0.0) + amount

    def execute_buy(
        self,
        *,
        close: float,
        open: Optional[float] = None,
        high: Optional[float] = None,
        low: Optional[float] = None,
    ) -> ExecutionResult:
        self._bump("total_orders")
        c = float(close)

        if self.mode == "open":
            # next_open 对齐：按开盘价成交（喂 t+1 K 线时 = open[t+1]）；open 缺失才回退到 close。
            base = float(open) if open is not None else c
            final_price = base * (1.0 + self.slippage)
            self._bump("filled_orders")
            self._bump("total_slippage_cost", final_price - base)
            return ExecutionResult(True, final_price, "open_fill" if open is not None else "open_fill_fallback_close")

        if self.mode == "close":
            final_price = c * (1.0 + self.slippage)
            self._bump("filled_orders")
            self._bump("total_slippage_cost", final_price - c)
            return ExecutionResult(True, final_price, "moc_fill")

        if self.mode == "passive":
            if open is None or high is None or low is None:
                final_price = c * (1.0 + self.slippage)
                self._bump("filled_orders")
                self._bump("total_slippage_cost", final_price - c)
                return ExecutionResult(True, final_price, "moc_fill_fallback")
            limit_price = float(open) * (1.0 - self.limit_k)
            if float(low) <= limit_price:
                self._bump("filled_orders")
                return ExecutionResult(True, limit_price, "limit_fill")
            self._bump("missed_orders")
            return ExecutionResult(False, c, "missed_limit")

        if self.mode == "midpoint":
            if open is None or high is None or low is None:
                return ExecutionResult(True, c * (1.0 + self.slippage), "moc_fill_fallback")
            limit_price = (float(high) + float(low)) / 2.0
            if float(low) <= limit_price:
                return ExecutionResult(True, limit_price, "limit_fill")
            return ExecutionResult(False, c, "missed_limit")

        raise ValueError(f"Unknown execution mode: {self.mode}")

    def execute_sell(
        self,
        *,
        close: float,
        open: Optional[float] = None,
        high: Optional[float] = None,
        low: Optional[float] = None,
    ) -> ExecutionResult:
        self._bump("total_orders")
        c = float(close)

        if self.mode == "open":
            base = float(open) if open is not None else c
            final_price = base * (1.0 - self.slippage)
            self._bump("filled_orders")
            self._bump("total_slippage_cost", base - final_price)
            return ExecutionResult(True, final_price, "open_fill" if open is not None else "open_fill_fallback_close")

        if self.mode == "close":
            final_price = c * (1.0 - self.slippage)
            self._bump("filled_orders")
            self._bump("total_slippage_cost", c - final_price)
            return ExecutionResult(True, final_price, "moc_fill")

        if self.mode == "passive":
            if open is None or high is None or low is None:
                final_price = c * (1.0 - self.slippage)
                self._bump("filled_orders")
                self._bump("total_slippage_cost", c - final_price)
                return ExecutionResult(True, final_price, "moc_fill_fallback")
            limit_price = float(open) * (1.0 + self.limit_k)
            if float(high) >= limit_price:
                self._bump("filled_orders")
                return ExecutionResult(True, limit_price, "limit_fill")
            self._bump("missed_orders")
            return ExecutionResult(False, c, "missed_limit")

        if self.mode == "midpoint":
            if open is None or high is None or low is None:
                return ExecutionResult(True, c * (1.0 - self.slippage), "moc_fill_fallback")
            limit_price = (float(high) + float(low)) / 2.0
            if float(high) >= limit_price:
                return ExecutionResult(True, limit_price, "limit_fill")
            return ExecutionResult(False, c, "missed_limit")

        raise ValueError(f"Unknown execution mode: {self.mode}")

    @staticmethod
    def execution_cost_rate(*, side: str, close: float, exec_price: float) -> float:
        """成交价相对收盘价的滑点率（buy 为正向成本，sell 取反）。"""
        c = float(close)
        if abs(c) < 1e-12:
            return 0.0
        rel = (float(exec_price) / c) - 1.0
        s = str(side or "").strip().lower()
        if s == "buy":
            return float(rel)
        if s == "sell":
            return float(-rel)
        return 0.0

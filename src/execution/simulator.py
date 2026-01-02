from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ExecutionResult:
    filled: bool
    price: float
    note: str


class ExecutionSimulator:
    def __init__(self, mode: str = "close", slippage_bps: float = 10.0, limit_threshold_bps: float = 50.0):
        self.mode = str(mode or "close").strip().lower()
        self.slippage = float(slippage_bps) / 10000.0
        self.limit_k = float(limit_threshold_bps) / 10000.0

        self.stats: Dict[str, float] = {
            "total_orders": 0.0,
            "filled_orders": 0.0,
            "missed_orders": 0.0,
            "total_slippage_cost": 0.0,
        }

    def execute_buy(
        self,
        *,
        close: float,
        open: Optional[float] = None,
        high: Optional[float] = None,
        low: Optional[float] = None,
    ) -> ExecutionResult:
        self.stats["total_orders"] = float(self.stats.get("total_orders", 0.0)) + 1.0

        c = float(close)
        if self.mode == "close":
            base_price = c
            final_price = base_price * (1.0 + self.slippage)
            self.stats["filled_orders"] = float(self.stats.get("filled_orders", 0.0)) + 1.0
            self.stats["total_slippage_cost"] = float(self.stats.get("total_slippage_cost", 0.0)) + float(final_price - base_price)
            return ExecutionResult(filled=True, price=float(final_price), note="moc_fill")

        if self.mode == "passive":
            if open is None or high is None or low is None:
                base_price = c
                final_price = base_price * (1.0 + self.slippage)
                self.stats["filled_orders"] = float(self.stats.get("filled_orders", 0.0)) + 1.0
                self.stats["total_slippage_cost"] = float(self.stats.get("total_slippage_cost", 0.0)) + float(final_price - base_price)
                return ExecutionResult(filled=True, price=float(final_price), note="moc_fill_fallback")
            base_price = float(open)
            limit_price = base_price * (1.0 - float(self.limit_k))
            if float(low) <= float(limit_price):
                self.stats["filled_orders"] = float(self.stats.get("filled_orders", 0.0)) + 1.0
                return ExecutionResult(filled=True, price=float(limit_price), note="limit_fill")
            self.stats["missed_orders"] = float(self.stats.get("missed_orders", 0.0)) + 1.0
            return ExecutionResult(filled=False, price=c, note="missed_limit")

        if self.mode == "midpoint":
            if open is None or high is None or low is None:
                return ExecutionResult(filled=True, price=c * (1.0 + self.slippage), note="moc_fill_fallback")
            limit_price = (float(high) + float(low)) / 2.0
            if float(low) <= float(limit_price):
                return ExecutionResult(filled=True, price=float(limit_price), note="limit_fill")
            return ExecutionResult(filled=False, price=c, note="missed_limit")

        raise ValueError(f"Unknown execution mode: {self.mode}")

    def execute_sell(
        self,
        *,
        close: float,
        open: Optional[float] = None,
        high: Optional[float] = None,
        low: Optional[float] = None,
    ) -> ExecutionResult:
        self.stats["total_orders"] = float(self.stats.get("total_orders", 0.0)) + 1.0

        c = float(close)
        if self.mode == "close":
            base_price = c
            final_price = base_price * (1.0 - self.slippage)
            self.stats["filled_orders"] = float(self.stats.get("filled_orders", 0.0)) + 1.0
            self.stats["total_slippage_cost"] = float(self.stats.get("total_slippage_cost", 0.0)) + float(base_price - final_price)
            return ExecutionResult(filled=True, price=float(final_price), note="moc_fill")

        if self.mode == "passive":
            if open is None or high is None or low is None:
                base_price = c
                final_price = base_price * (1.0 - self.slippage)
                self.stats["filled_orders"] = float(self.stats.get("filled_orders", 0.0)) + 1.0
                self.stats["total_slippage_cost"] = float(self.stats.get("total_slippage_cost", 0.0)) + float(base_price - final_price)
                return ExecutionResult(filled=True, price=float(final_price), note="moc_fill_fallback")
            base_price = float(open)
            limit_price = base_price * (1.0 + float(self.limit_k))
            if float(high) >= float(limit_price):
                self.stats["filled_orders"] = float(self.stats.get("filled_orders", 0.0)) + 1.0
                return ExecutionResult(filled=True, price=float(limit_price), note="limit_fill")
            self.stats["missed_orders"] = float(self.stats.get("missed_orders", 0.0)) + 1.0
            return ExecutionResult(filled=False, price=c, note="missed_limit")

        if self.mode == "midpoint":
            if open is None or high is None or low is None:
                return ExecutionResult(filled=True, price=c * (1.0 - self.slippage), note="moc_fill_fallback")
            limit_price = (float(high) + float(low)) / 2.0
            if float(high) >= float(limit_price):
                return ExecutionResult(filled=True, price=float(limit_price), note="limit_fill")
            return ExecutionResult(filled=False, price=c, note="missed_limit")

        raise ValueError(f"Unknown execution mode: {self.mode}")

    @staticmethod
    def execution_cost_rate(*, side: str, close: float, exec_price: float) -> float:
        s = str(side or "").strip().lower()
        c = float(close)
        p = float(exec_price)
        if abs(c) < 1e-12:
            return 0.0
        rel = (p / c) - 1.0
        if s == "buy":
            return float(rel)
        if s == "sell":
            return float(-rel)
        return 0.0

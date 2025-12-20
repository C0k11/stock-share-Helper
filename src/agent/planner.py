from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PlannerDecision:
    strategy: str
    risk_budget: float
    inputs: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "risk_budget": float(self.risk_budget),
            "inputs": self.inputs,
        }


class Planner:
    def __init__(
        self,
        defensive_regimes: Optional[set[str]] = None,
        aggressive_regimes: Optional[set[str]] = None,
    ) -> None:
        self.defensive_regimes = defensive_regimes or {"risk_off"}
        self.aggressive_regimes = aggressive_regimes or {"risk_on"}

    def decide(self, *, market_regime: Optional[Dict[str, Any]] = None) -> PlannerDecision:
        regime = None
        score = None
        if isinstance(market_regime, dict):
            regime = str(market_regime.get("regime") or "").strip()
            try:
                score = float(market_regime.get("score"))
            except Exception:
                score = None

        inputs = {
            "market_regime": {
                "regime": regime,
                "score": score,
            }
        }

        if regime in self.defensive_regimes:
            return PlannerDecision(strategy="defensive", risk_budget=0.2, inputs=inputs)

        if regime in self.aggressive_regimes:
            return PlannerDecision(strategy="aggressive_long", risk_budget=1.0, inputs=inputs)

        return PlannerDecision(strategy="cash_preservation", risk_budget=0.4, inputs=inputs)

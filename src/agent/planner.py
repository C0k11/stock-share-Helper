from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


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
        *,
        policy: str = "rule",
        sft_model_path: str = "models/planner_sft_v1.pt",
    ) -> None:
        self.defensive_regimes = defensive_regimes or {"risk_off"}
        self.aggressive_regimes = aggressive_regimes or {"risk_on"}
        self.policy = str(policy or "rule").strip().lower()
        self.sft_model_path = str(sft_model_path or "").strip()
        self._sft: Optional[_PlannerSFTBundle] = None

        if self.policy == "sft" and self.sft_model_path:
            self._sft = _try_load_sft(Path(self.sft_model_path))

    def decide(
        self,
        *,
        market_regime: Optional[Dict[str, Any]] = None,
        features: Optional[Dict[str, float]] = None,
        date_str: str = "",
        nav_csv: str = "",
        signals_csv: str = "",
    ) -> PlannerDecision:
        regime = None
        score = None
        if isinstance(market_regime, dict):
            regime = str(market_regime.get("regime") or "").strip()
            try:
                score = float(market_regime.get("score"))
            except Exception:
                score = None

        prev = {}
        if date_str and signals_csv:
            prev = _load_prev_signals_features(Path(signals_csv), date_str)
        if (not prev) and date_str and nav_csv:
            prev = _load_prev_nav_features(Path(nav_csv), date_str)
        feats_in = dict(features or {})
        for k, v in prev.items():
            feats_in.setdefault(k, float(v))

        inputs = {
            "market_regime": {
                "regime": regime,
                "score": score,
            },
            "features": feats_in,
        }

        if self.policy == "sft" and self._sft is not None:
            strategy, probs = self._sft.predict_strategy(feats_in)
            return PlannerDecision(strategy=strategy, risk_budget=_risk_budget_for(strategy), inputs={**inputs, "probs": probs})

        if regime in self.defensive_regimes:
            return PlannerDecision(strategy="defensive", risk_budget=0.2, inputs=inputs)

        if regime in self.aggressive_regimes:
            return PlannerDecision(strategy="aggressive_long", risk_budget=1.0, inputs=inputs)

        return PlannerDecision(strategy="cash_preservation", risk_budget=0.4, inputs=inputs)


def _risk_budget_for(strategy: str) -> float:
    s = str(strategy or "").strip().lower()
    if s == "aggressive_long":
        return 1.0
    if s == "defensive":
        return 0.2
    return 0.4


def _load_prev_nav_features(nav_path: Path, date_str: str) -> Dict[str, float]:
    if (not date_str) or (not nav_path.exists()):
        return {}
    try:
        import csv

        rows: List[Dict[str, str]] = []
        with open(nav_path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                d = str(row.get("date") or "").strip()
                if not d:
                    continue
                rows.append({k: str(v) for k, v in row.items()})

        prev_row: Optional[Dict[str, str]] = None
        for row in rows:
            d = str(row.get("date") or "").strip()
            if d and d < str(date_str):
                if (prev_row is None) or (d > str(prev_row.get("date") or "")):
                    prev_row = row

        if prev_row is None:
            return {}

        def sf(x: Any) -> float:
            try:
                return float(x)
            except Exception:
                return 0.0

        return {
            "prev_gross_exposure": sf(prev_row.get("gross_exposure")),
            "prev_net_exposure": sf(prev_row.get("net_exposure")),
            "prev_abs_exposure_mean": sf(prev_row.get("abs_exposure_mean")),
            "prev_long_count": sf(prev_row.get("long_count")),
            "prev_short_count": sf(prev_row.get("short_count")),
        }
    except Exception:
        return {}


def _load_prev_signals_features(signals_path: Path, date_str: str) -> Dict[str, float]:
    if (not date_str) or (not signals_path.exists()):
        return {}
    try:
        import csv

        by_date: Dict[str, List[float]] = {}
        longs: Dict[str, int] = {}
        shorts: Dict[str, int] = {}

        with open(signals_path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                d = str(row.get("date") or "").strip()
                if not d:
                    continue
                tp_raw = row.get("target_position")
                try:
                    tp = float(tp_raw)
                except Exception:
                    tp = 0.0
                by_date.setdefault(d, []).append(tp)
                if tp > 0:
                    longs[d] = int(longs.get(d, 0)) + 1
                if tp < 0:
                    shorts[d] = int(shorts.get(d, 0)) + 1

        prev_d = ""
        for d in by_date.keys():
            if d < str(date_str) and d > prev_d:
                prev_d = d
        if not prev_d:
            return {}

        tps = by_date.get(prev_d, [])
        if not tps:
            return {}

        gross = float(sum(abs(x) for x in tps))
        net = float(sum(tps))
        abs_mean = float(sum(abs(x) for x in tps) / max(1, len(tps)))
        return {
            "prev_gross_exposure": gross,
            "prev_net_exposure": net,
            "prev_abs_exposure_mean": abs_mean,
            "prev_long_count": float(longs.get(prev_d, 0)),
            "prev_short_count": float(shorts.get(prev_d, 0)),
        }
    except Exception:
        return {}


class _LightPlanner(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, int(output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PlannerSFTBundle:
    def __init__(
        self,
        *,
        model: _LightPlanner,
        feature_names: List[str],
        mean: List[float],
        std: List[float],
        idx_to_label: Dict[int, str],
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.mean = list(mean)
        self.std = list(std)
        self.idx_to_label = dict(idx_to_label)

    def predict_strategy(self, feats: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        x_list: List[float] = []
        for i, name in enumerate(self.feature_names):
            v = float(feats.get(name, 0.0) or 0.0)
            mu = float(self.mean[i]) if i < len(self.mean) else 0.0
            sd = float(self.std[i]) if i < len(self.std) else 1.0
            if sd <= 1e-12:
                sd = 1.0
            x_list.append((v - mu) / sd)

        x = torch.tensor([x_list], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x)
            probs_t = torch.softmax(logits, dim=-1)[0]
            idx = int(torch.argmax(probs_t).item())
            probs = {self.idx_to_label.get(i, str(i)): float(probs_t[i].item()) for i in range(int(probs_t.shape[0]))}
        label = self.idx_to_label.get(idx, "cash_preservation")
        return str(label), probs


def _try_load_sft(path: Path) -> Optional[_PlannerSFTBundle]:
    if not path.exists():
        return None
    try:
        payload = torch.load(str(path), map_location="cpu")
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    feature_names = payload.get("feature_names")
    mean = payload.get("scaler_mean")
    std = payload.get("scaler_std")
    idx_to_label = payload.get("idx_to_label")
    state = payload.get("model_state")

    if not (isinstance(feature_names, list) and isinstance(mean, list) and isinstance(std, list) and isinstance(idx_to_label, dict) and isinstance(state, dict)):
        return None

    try:
        model = _LightPlanner(input_dim=len(feature_names), output_dim=len(idx_to_label), dropout=float(payload.get("dropout", 0.2)))
        model.load_state_dict(state)
        model.eval()
        idx_map = {int(k): str(v) for k, v in idx_to_label.items()}
        return _PlannerSFTBundle(model=model, feature_names=[str(x) for x in feature_names], mean=[float(x) for x in mean], std=[float(x) for x in std], idx_to_label=idx_map)
    except Exception:
        return None

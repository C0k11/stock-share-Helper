"""Planner —— 日级市场状态评估（rule 规则 / sft 轻量 MLP）。

迁移自 `src/agent/planner.py` + `strategy.py::_planner_assess` 的 regime 映射。

与旧版的差异：
- **torch 懒加载**：规则路径与 `assess_regime` 的规则分支完全不需要 torch，
  只有 SFT MLP 推理时才 `import torch`，因此本模块可在无 GPU/无 torch 环境导入与单测。
- **去掉 CSV 旁路**：旧 `decide()` 会从 nav.csv/signals.csv 读"昨日仓位"特征，那是
  walkforward 的 plumbing，应由调用方（backtest/evolution）预先算好放进 `features`，
  不属于 agent 大脑。其余行为等价。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from quantai.agents.base import Regime, flatten_features
from quantai.config.schema import PlannerConfig


# --------------------------------------------------------------------------- #
# 纯逻辑（无 torch）
# --------------------------------------------------------------------------- #
@dataclass
class PlannerDecision:
    strategy: str
    risk_budget: float
    inputs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "risk_budget": float(self.risk_budget),
            "inputs": self.inputs,
        }


def risk_budget_for(strategy: str) -> float:
    s = str(strategy or "").strip().lower()
    if s == "aggressive_long":
        return 1.0
    if s == "defensive":
        return 0.2
    return 0.4


def strategy_to_regime(strategy: str) -> str:
    """SFT 输出的 strategy -> orchestrator 用的 regime 字符串。"""
    s = str(strategy or "").strip().lower()
    if s == "aggressive_long":
        return Regime.AGGRESSIVE
    if s == "defensive":
        return Regime.DEFENSIVE
    return Regime.CASH_PRESERVATION


def assess_regime_rule(
    features: Dict[str, Any],
    *,
    cash_vol_ann_pct: float = 120.0,
    cash_ret_5d_pct: float = -10.0,
    defensive_vol_ann_pct: float = 80.0,
    defensive_ret_5d_pct: float = -5.0,
) -> str:
    """规则状态机（等价迁移自 `_planner_assess` 的规则分支）：
    高波动/大跌 -> cash_preservation；中等 -> defensive；否则 aggressive。纯函数。
    """
    tech = features.get("technical") if isinstance(features.get("technical"), dict) else {}
    vol = _as_float(tech.get("volatility_20d", 20.0), 20.0)
    ret_5d = _as_float(tech.get("return_5d", 0.0), 0.0)

    if vol > cash_vol_ann_pct or ret_5d < cash_ret_5d_pct:
        return Regime.CASH_PRESERVATION
    if vol > defensive_vol_ann_pct or ret_5d < defensive_ret_5d_pct:
        return Regime.DEFENSIVE
    return Regime.AGGRESSIVE


# --------------------------------------------------------------------------- #
# Planner（rule + SFT，torch 懒加载）
# --------------------------------------------------------------------------- #
class Planner:
    def __init__(
        self,
        *,
        policy: str = "rule",
        sft_model_path: str = "",
        cash_vol_ann_pct: float = 120.0,
        cash_ret_5d_pct: float = -10.0,
        defensive_vol_ann_pct: float = 80.0,
        defensive_ret_5d_pct: float = -5.0,
        defensive_regimes: Optional[set[str]] = None,
        aggressive_regimes: Optional[set[str]] = None,
    ) -> None:
        self.policy = str(policy or "rule").strip().lower()
        self.sft_model_path = str(sft_model_path or "").strip()
        self.cash_vol_ann_pct = float(cash_vol_ann_pct)
        self.cash_ret_5d_pct = float(cash_ret_5d_pct)
        self.defensive_vol_ann_pct = float(defensive_vol_ann_pct)
        self.defensive_ret_5d_pct = float(defensive_ret_5d_pct)
        self.defensive_regimes = defensive_regimes or {"risk_off"}
        self.aggressive_regimes = aggressive_regimes or {"risk_on"}

        self._sft: Optional["_PlannerSFTBundle"] = None
        self._sft_tried = False

    @classmethod
    def from_config(cls, cfg: PlannerConfig) -> "Planner":
        return cls(
            policy=cfg.policy,
            sft_model_path=cfg.sft_model_path,
            cash_vol_ann_pct=cfg.cash_vol_ann_pct,
            cash_ret_5d_pct=cfg.cash_ret_5d_pct,
            defensive_vol_ann_pct=cfg.defensive_vol_ann_pct,
            defensive_ret_5d_pct=cfg.defensive_ret_5d_pct,
        )

    # --- orchestrator 主接口 --- #
    def assess_regime(self, features: Dict[str, Any]) -> str:
        """features -> regime（aggressive/defensive/cash_preservation）。

        SFT 策略可用则走 MLP，否则回退规则；任何 SFT 异常都安全回退规则。
        """
        if self.policy == "sft":
            bundle = self._ensure_sft()
            if bundle is not None:
                try:
                    strategy, _ = bundle.predict_strategy(flatten_features(features))
                    return strategy_to_regime(strategy)
                except Exception:
                    pass
        return assess_regime_rule(
            features,
            cash_vol_ann_pct=self.cash_vol_ann_pct,
            cash_ret_5d_pct=self.cash_ret_5d_pct,
            defensive_vol_ann_pct=self.defensive_vol_ann_pct,
            defensive_ret_5d_pct=self.defensive_ret_5d_pct,
        )

    # --- 兼容旧 decide()（regime + features -> PlannerDecision） --- #
    def decide(
        self,
        *,
        market_regime: Optional[Dict[str, Any]] = None,
        features: Optional[Dict[str, float]] = None,
    ) -> PlannerDecision:
        regime = None
        score = None
        if isinstance(market_regime, dict):
            regime = str(market_regime.get("regime") or "").strip()
            try:
                score = float(market_regime.get("score"))
            except Exception:
                score = None

        feats_in = dict(features or {})
        reg_l = str(regime or "").strip().lower()
        feats_in.setdefault("market_regime_score", float(score) if score is not None else 0.0)
        feats_in.setdefault("market_regime_is_risk_off", 1.0 if reg_l == "risk_off" else 0.0)
        feats_in.setdefault("market_regime_is_risk_on", 1.0 if reg_l == "risk_on" else 0.0)

        inputs = {
            "market_regime": {"regime": regime, "score": score},
            "features": feats_in,
        }

        if self.policy == "sft":
            bundle = self._ensure_sft()
            if bundle is not None:
                strategy, probs = bundle.predict_strategy(feats_in)
                return PlannerDecision(
                    strategy=strategy,
                    risk_budget=risk_budget_for(strategy),
                    inputs={**inputs, "probs": probs},
                )

        if regime in self.defensive_regimes:
            return PlannerDecision(strategy="defensive", risk_budget=0.2, inputs=inputs)
        if regime in self.aggressive_regimes:
            return PlannerDecision(strategy="aggressive_long", risk_budget=1.0, inputs=inputs)
        return PlannerDecision(strategy="cash_preservation", risk_budget=0.4, inputs=inputs)

    def _ensure_sft(self) -> Optional["_PlannerSFTBundle"]:
        if self._sft_tried:
            return self._sft
        self._sft_tried = True
        if self.policy == "sft" and self.sft_model_path:
            self._sft = _try_load_sft(Path(self.sft_model_path))
        return self._sft


# --------------------------------------------------------------------------- #
# SFT MLP（torch 懒加载）
# --------------------------------------------------------------------------- #
def _build_planner_net(input_dim: int, output_dim: int = 3, dropout: float = 0.2):
    """构造 3 层 MLP（懒加载 torch）。"""
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(int(input_dim), 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(float(dropout)),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, int(output_dim)),
    )


class _PlannerSFTBundle:
    def __init__(
        self,
        *,
        model: Any,
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
        import torch

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
            probs = {
                self.idx_to_label.get(i, str(i)): float(probs_t[i].item())
                for i in range(int(probs_t.shape[0]))
            }
        label = self.idx_to_label.get(idx, "cash_preservation")
        return str(label), probs


def _try_load_sft(path: Path) -> Optional[_PlannerSFTBundle]:
    if not path.exists():
        return None
    try:
        import torch

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

    if not (
        isinstance(feature_names, list)
        and isinstance(mean, list)
        and isinstance(std, list)
        and isinstance(idx_to_label, dict)
        and isinstance(state, dict)
    ):
        return None

    try:
        model = _build_planner_net(
            input_dim=len(feature_names),
            output_dim=len(idx_to_label),
            dropout=float(payload.get("dropout", 0.2)),
        )
        model.load_state_dict(state)
        model.eval()
        idx_map = {int(k): str(v) for k, v in idx_to_label.items()}
        return _PlannerSFTBundle(
            model=model,
            feature_names=[str(x) for x in feature_names],
            mean=[float(x) for x in mean],
            std=[float(x) for x in std],
            idx_to_label=idx_map,
        )
    except Exception:
        return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

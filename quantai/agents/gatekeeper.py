"""Gatekeeper -- RL 门控 MLP（Q 值阈值过滤交易）。

迁移自 `src/agent/gatekeeper.py` + `strategy.py::_gatekeeper_approve`。

两处改动：
- **torch 懒加载**：`approve` 的兜底门是纯规则，只有加载/推理 RL MLP 时才 import torch。
- **删 random**：旧 `_gatekeeper_approve` 在无模型时用
  `return random.random() > 0.3`（70% 概率放行）——不可复现、是伪装的"门控"。
  新版无随机：`require_model=True` 时无模型即拒绝（架构建议，最诚实）；
  `require_model=False` 时用**确定性波动门**（vol <= vol_trigger 才放行）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from quantai.agents.base import flatten_features
from quantai.config.schema import GatekeeperConfig


@dataclass
class GateDecision:
    allow: bool
    q_allow: float
    threshold: float
    inputs: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow": bool(self.allow),
            "q_allow": float(self.q_allow),
            "threshold": float(self.threshold),
            "inputs": self.inputs,
        }


class Gatekeeper:
    def __init__(
        self,
        *,
        model_path: str = "",
        threshold: float = 0.0,
        require_model: bool = True,
        vol_trigger_ann_pct: float = 120.0,
    ) -> None:
        self.model_path = str(model_path or "").strip()
        self.threshold = float(threshold)
        self.require_model = bool(require_model)
        self.vol_trigger_ann_pct = float(vol_trigger_ann_pct)

        self._model: Any = None
        self._feature_names: List[str] = []
        self._mean: List[float] = []
        self._std: List[float] = []
        self._loaded = False

        if self.model_path:
            self._load(Path(self.model_path))

    @classmethod
    def from_config(cls, cfg: GatekeeperConfig) -> "Gatekeeper":
        return cls(
            model_path=cfg.model_path,
            threshold=cfg.threshold,
            require_model=cfg.require_model,
            vol_trigger_ann_pct=cfg.vol_trigger_ann_pct,
        )

    @property
    def model_loaded(self) -> bool:
        return self._model is not None and bool(self._feature_names)

    # --- orchestrator 主接口（修后：无随机） --- #
    def approve(self, features: Dict[str, Any]) -> bool:
        """features -> 是否放行交易。

        优先 RL Q 值；无模型时：require_model -> 拒绝；否则确定性波动门。
        """
        if self.model_loaded:
            try:
                d = self.decide(feats=flatten_features(features))
                return bool(d.allow)
            except Exception:
                # 模型存在但推理异常：保守拒绝（不放行未知风险）。
                return False

        if self.require_model:
            return False

        # 确定性兜底（取代旧 random）：波动过高则拒绝，否则放行。
        flat = flatten_features(features)
        vol = float(flat.get("volatility_ann_pct", 0.0) or 0.0)
        return vol <= self.vol_trigger_ann_pct

    # --- 兼容旧 decide/predict（RL MLP，torch 懒加载） --- #
    def predict(self, *, feats: Dict[str, float]) -> float:
        if not self.model_loaded:
            return 0.0
        import torch

        x_list = self._vectorize(feats)
        x = torch.tensor([x_list], dtype=torch.float32)
        with torch.no_grad():
            q = float(self._model(x)[0].item())
        return float(q)

    def decide(
        self, *, feats: Dict[str, float], threshold: Optional[float] = None
    ) -> GateDecision:
        thr = float(self.threshold if threshold is None else threshold)
        q = float(self.predict(feats=feats))
        allow = bool(q > thr)
        return GateDecision(allow=allow, q_allow=q, threshold=thr, inputs={"features": dict(feats)})

    # --- 内部：加载 RL MLP（懒 torch） --- #
    def _load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            import torch

            payload = torch.load(str(path), map_location="cpu")
        except Exception:
            return
        if not isinstance(payload, dict):
            return

        feat = payload.get("feature_names")
        mean = payload.get("scaler_mean")
        std = payload.get("scaler_std")
        state = payload.get("model_state")
        drop = payload.get("dropout", 0.2)

        if not (
            isinstance(feat, list)
            and isinstance(mean, list)
            and isinstance(std, list)
            and isinstance(state, dict)
        ):
            return

        self._feature_names = [str(x) for x in feat]
        self._mean = [float(x) for x in mean]
        self._std = [float(x) for x in std]

        try:
            m = _build_gate_net(input_dim=len(self._feature_names), dropout=float(drop))
            m.load_state_dict(state)
            m.eval()
        except Exception:
            self._feature_names = []
            return

        self._model = m
        self._loaded = True

    def _vectorize(self, feats: Dict[str, float]) -> List[float]:
        x: List[float] = []
        for i, name in enumerate(self._feature_names):
            v = float(feats.get(name, 0.0) or 0.0)
            mu = float(self._mean[i]) if i < len(self._mean) else 0.0
            sd = float(self._std[i]) if i < len(self._std) else 1.0
            if sd <= 1e-12:
                sd = 1.0
            x.append((v - mu) / sd)
        return x


def _build_gate_net(input_dim: int, dropout: float = 0.2):
    """构造门控 MLP（懒加载 torch）：输出单标量 Q 值。"""
    import torch

    class _Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(int(input_dim), 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x).squeeze(-1)

    return _Net()

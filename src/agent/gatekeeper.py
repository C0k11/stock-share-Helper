from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


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


class _LightGatekeeper(torch.nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class Gatekeeper:
    def __init__(self, *, model_path: str, threshold: float = 0.0) -> None:
        self.model_path = str(model_path or "").strip()
        self.threshold = float(threshold)
        self._bundle: Optional[Dict[str, Any]] = None
        self._model: Optional[_LightGatekeeper] = None
        self._feature_names: List[str] = []
        self._mean: List[float] = []
        self._std: List[float] = []

        if self.model_path:
            self._load(Path(self.model_path))

    def _load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
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

        if not (isinstance(feat, list) and isinstance(mean, list) and isinstance(std, list) and isinstance(state, dict)):
            return

        self._feature_names = [str(x) for x in feat]
        self._mean = [float(x) for x in mean]
        self._std = [float(x) for x in std]

        m = _LightGatekeeper(input_dim=len(self._feature_names), dropout=float(drop))
        try:
            m.load_state_dict(state)
            m.eval()
        except Exception:
            return

        self._bundle = payload
        self._model = m

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

    def predict(self, *, feats: Dict[str, float]) -> float:
        if self._model is None or not self._feature_names:
            return 0.0
        x_list = self._vectorize(feats)
        x = torch.tensor([x_list], dtype=torch.float32)
        with torch.no_grad():
            q = float(self._model(x)[0].item())
        return float(q)

    def decide(self, *, feats: Dict[str, float], threshold: Optional[float] = None) -> GateDecision:
        thr = float(self.threshold if threshold is None else threshold)
        q = float(self.predict(feats=feats))
        allow = bool(q > thr)
        return GateDecision(allow=allow, q_allow=q, threshold=thr, inputs={"features": dict(feats)})

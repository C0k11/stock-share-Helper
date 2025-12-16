import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class MarketRAG:
    def __init__(self, data_dir: str = "data/daily"):
        self.data_dir = Path(data_dir)

        self.dimension = 3
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

        self._build_index()

    def _iter_feature_files(self) -> List[Path]:
        if not self.data_dir.exists():
            return []
        return sorted(list(self.data_dir.rglob("etf_features_*.json")))

    def _parse_payload_items(self, payload: Any) -> List[Tuple[str, Dict[str, Any]]]:
        out: List[Tuple[str, Dict[str, Any]]] = []

        def push_item(symbol: Any, feats: Any) -> None:
            if not symbol:
                return
            if not isinstance(feats, dict):
                return
            out.append((str(symbol).strip(), feats))

        if isinstance(payload, dict):
            items = payload.get("items")
            if isinstance(items, list):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    symbol = it.get("symbol") or it.get("ticker")
                    feats = it.get("features") if isinstance(it.get("features"), dict) else it
                    push_item(symbol, feats)
                return out

            for _k, v in payload.items():
                if isinstance(v, list):
                    for it in v:
                        if not isinstance(it, dict):
                            continue
                        symbol = it.get("symbol") or it.get("ticker")
                        feats = it.get("features") if isinstance(it.get("features"), dict) else it
                        push_item(symbol, feats)
                elif isinstance(v, dict):
                    for sym, feats in v.items():
                        push_item(sym, feats)
            return out

        if isinstance(payload, list):
            for it in payload:
                if not isinstance(it, dict):
                    continue
                symbol = it.get("symbol") or it.get("ticker")
                feats = it.get("features") if isinstance(it.get("features"), dict) else it
                push_item(symbol, feats)

        return out

    def _feats_to_vec(self, feats: Dict[str, Any]) -> np.ndarray:
        def _to_float(x: Any) -> float:
            try:
                return float(x)
            except Exception:
                return 0.0

        def _get(d: Any, *path: str) -> Any:
            cur = d
            for k in path:
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(k)
            return cur

        def _to_pct(v: Any) -> float:
            x = _to_float(v)
            if abs(x) <= 2.5:
                return x * 100.0
            return x

        change_5d = feats.get("change_5d_pct")
        if change_5d is None:
            change_5d = feats.get("return_5d")
        if change_5d is None:
            change_5d = _get(feats, "technical", "return_5d")
        change_5d_pct = _to_pct(change_5d)

        vol_ann = feats.get("volatility_ann_pct")
        if vol_ann is None:
            vol_ann = feats.get("volatility_20d")
        if vol_ann is None:
            vol_ann = _get(feats, "technical", "volatility_20d")
        vol_ann_pct = _to_pct(vol_ann)

        dd20 = feats.get("drawdown_20d_pct")
        if dd20 is None:
            dd20 = feats.get("max_drawdown_20d")
        if dd20 is None:
            dd20 = _get(feats, "technical", "max_drawdown_20d")
        if dd20 is None:
            dd20 = feats.get("drawdown")
        if dd20 is None:
            dd20 = _get(feats, "technical", "drawdown")
        dd20_pct = _to_pct(dd20)

        return np.array([change_5d_pct, vol_ann_pct, dd20_pct], dtype=np.float32)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            return x
        return (x - self._mean) / self._std

    def _build_index(self) -> None:
        files = self._iter_feature_files()
        if not files:
            return

        vectors: List[np.ndarray] = []
        meta: List[Dict[str, Any]] = []

        for fp in files:
            date = fp.stem.split("_")[-1]
            try:
                payload = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue

            for symbol, feats in self._parse_payload_items(payload):
                vec = self._feats_to_vec(feats)
                vectors.append(vec)
                meta.append(
                    {
                        "date": date,
                        "ticker": symbol,
                        "features": {
                            "change_5d_pct": float(vec[0]),
                            "volatility_ann_pct": float(vec[1]),
                            "drawdown_20d_pct": float(vec[2]),
                        },
                        "summary": f"chg5d={vec[0]:.2f}%, vol={vec[1]:.1f}%, dd20d={vec[2]:.1f}%",
                    }
                )

        if not vectors:
            return

        data_np = np.stack(vectors, axis=0).astype(np.float32)

        mean = data_np.mean(axis=0)
        std = data_np.std(axis=0)
        std = np.where(std <= 1e-6, 1.0, std)
        self._mean = mean.astype(np.float32)
        self._std = std.astype(np.float32)

        data_np = self._normalize(data_np)

        try:
            import faiss  # type: ignore
        except Exception as e:
            raise ImportError(
                "FAISS is required for MarketRAG. Install with: pip install faiss-cpu"
            ) from e

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(data_np)
        self.metadata = meta

    def retrieve(
        self,
        current_features: Dict[str, Any],
        k: int = 3,
        ticker: Optional[str] = None,
        exclude_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.index is None or len(self.metadata) == 0:
            return []

        q = self._feats_to_vec(current_features)
        qn = self._normalize(q.reshape(1, -1).astype(np.float32))

        k = int(max(1, k))
        probe = int(min(len(self.metadata), max(k * 12, 50)))
        distances, indices = self.index.search(qn, probe)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0].tolist(), indices[0].tolist()):
            if idx == -1:
                continue
            if idx >= len(self.metadata):
                continue

            m = dict(self.metadata[idx])
            if ticker is not None and str(m.get("ticker")) != str(ticker):
                continue
            if exclude_date is not None and str(m.get("date")) == str(exclude_date):
                continue

            m["distance"] = float(dist)
            results.append(m)
            if len(results) >= k:
                break

        return results

"""QuantAI API 客户端 —— UI 与 HTTP 之间的薄封装（可单测）。

UI（streamlit / 桌面）不直接拼 HTTP，统一走 `QuantAIClient`。它只调 `quantai.api` 暴露的
**真实** endpoint。可注入 `transport`（如 `httpx.ASGITransport(app=create_app())`）让客户端在
进程内直连 app，单测无需起服务器。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


class QuantAIClient:
    """对 quantai.api 的薄客户端。"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        timeout: float = 10.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    # --- 低层 --- #
    def _get(self, path: str, params: Optional[dict] = None) -> Dict[str, Any]:
        r = self._client.get(path, params=params)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: Optional[dict] = None) -> Dict[str, Any]:
        r = self._client.post(path, json=payload or {})
        r.raise_for_status()
        return r.json()

    # --- health --- #
    def health(self) -> Dict[str, Any]:
        return self._get("/api/v1/health")

    # --- live --- #
    def start_live(
        self,
        tickers: List[str],
        *,
        source: str = "simulated",
        cash: float = 100000.0,
        interval_sec: float = 1.0,
        allow_short: bool = False,
        seed: Optional[int] = None,
        collect: bool = False,
    ) -> Dict[str, Any]:
        return self._post(
            "/api/v1/live/start",
            {
                "tickers": tickers,
                "source": source,
                "cash": cash,
                "interval_sec": interval_sec,
                "allow_short": allow_short,
                "seed": seed,
                "collect": collect,
            },
        )

    def stop_live(self) -> Dict[str, Any]:
        return self._post("/api/v1/live/stop")

    def live_status(self) -> Dict[str, Any]:
        return self._get("/api/v1/live/status")

    def live_trades(self, limit: int = 200) -> Dict[str, Any]:
        return self._get("/api/v1/live/trades", params={"limit": limit})

    # --- feedback --- #
    def submit_feedback(self, ref_id: str, score: int, comment: str = "") -> Dict[str, Any]:
        return self._post("/api/v1/feedback", {"ref_id": ref_id, "score": score, "comment": comment})

    # --- evolution --- #
    def build_dataset(self) -> Dict[str, Any]:
        return self._post("/api/v1/evolution/build-dataset")

    def active_adapters(self) -> Dict[str, Any]:
        return self._get("/api/v1/evolution/active")

    # --- chat --- #
    def chat(self, message: str, context: Optional[dict] = None) -> Dict[str, Any]:
        return self._post("/api/v1/chat", {"message": message, "context": context or {}})

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "QuantAIClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

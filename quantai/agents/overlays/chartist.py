"""Chartist overlay —— VLM 看 K 线给方向分（+1 支持 / -1 反对 / 0 中性）。

迁移自 `strategy.py::_chartist_overlay`。拆成两块：
- **纯打分**（`score_from_vlm`）：把 VLM 输出 {signal, confidence} + 提案方向 映射成 -1/0/1，
  含置信度阈值过滤。等价迁移自旧码 3061-3077 行，可完整单测。
- **编排接缝**（`assess`）：真正跑 VLM（渲染 K 线图 + Qwen2.5-VL 推理）通过注入的
  `analyzer` 完成（鸭子类型：`.analyze(ticker, asof=...) -> {"signal","confidence",...} | None`）。
  默认未注入 / 未启用 -> 返回 0（中性，不影响决策）。

注：具体的 Qwen2.5-VL 后端（mplfinance 渲染 + VLM 生成）见 `vlm_chartist.QwenVLChartist`
（重依赖懒加载、GPU 可选），经 `analyzer` 注入接入本 overlay。
"""
from __future__ import annotations

from typing import Any, Optional

from quantai.agents.base import Action
from quantai.config.schema import ChartistConfig


class ChartistOverlay:
    def __init__(
        self, *, enabled: bool = False, confidence_threshold: float = 0.7
    ) -> None:
        self.enabled = bool(enabled)
        self.confidence_threshold = float(confidence_threshold)

    @classmethod
    def from_config(cls, cfg: ChartistConfig) -> "ChartistOverlay":
        return cls(enabled=cfg.enabled, confidence_threshold=cfg.confidence_threshold)

    def score_from_vlm(
        self, signal: str, confidence: float, proposed_action: str
    ) -> int:
        """VLM 信号 + 提案方向 -> 方向分（纯逻辑）。

        置信度不过阈值 -> 0；BULLISH 与 BUY 同向得 +1、与 SELL 反向得 -1（BEARISH 对称）。
        """
        try:
            conf = float(confidence)
        except Exception:
            conf = 0.0
        if conf <= self.confidence_threshold:
            return 0

        sig = str(signal or "").strip().upper()
        act = Action.normalize(proposed_action)

        if sig == "BULLISH":
            if act == Action.BUY:
                return 1
            if act == Action.SELL:
                return -1
            return 0
        if sig == "BEARISH":
            if act == Action.SELL:
                return 1
            if act == Action.BUY:
                return -1
            return 0
        return 0

    def assess(
        self,
        ticker: str,
        proposed_action: str,
        *,
        analyzer: Optional[Any] = None,
        asof: str = "",
    ) -> int:
        """跑 VLM（经注入的 analyzer）并打分；未启用/无 analyzer/异常 -> 0。"""
        if not self.enabled or analyzer is None:
            return 0
        try:
            out = analyzer.analyze(ticker, asof=asof)
        except Exception:
            return 0
        if not isinstance(out, dict):
            return 0
        return self.score_from_vlm(
            out.get("signal", ""), out.get("confidence", 0.0), proposed_action
        )

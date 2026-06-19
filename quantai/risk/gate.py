"""风险闸门：对"提议的动作/仓位"做最终裁决（回撤强平 / 恐慌事件 / 仓位上限 / 高波动降仓）。

从旧 `src/risk/gate.py` 迁移，逻辑不变，加类型标注。纯函数式裁决：输入当前特征/新闻信号
/提议，输出 (最终动作, 最终仓位, 决策轨迹)。无时间序列、无 lookahead。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

#: 触发强制减仓的"恐慌"事件类型
PANIC_EVENTS = ("regulation_crackdown", "war_breakout", "exchange_investigation")


class RiskGate:
    """在策略提议之上叠加硬性风控约束，返回裁决后的动作与仓位。"""

    def __init__(
        self,
        max_drawdown_limit_pct: float = -8.0,
        vol_reduce_trigger_ann_pct: float = 30.0,
        max_pos_limit: float = 0.5,
    ) -> None:
        self.MAX_DRAWDOWN_LIMIT = float(max_drawdown_limit_pct)
        self.MAX_POS_LIMIT = float(max_pos_limit)
        self.VOL_REDUCE_TRIGGER = float(vol_reduce_trigger_ann_pct)

    def adjudicate(
        self,
        features: Any,
        news_signals: Optional[list],
        proposed_action: Any,
        proposed_pos: Any,
    ) -> tuple[str, float, list[str]]:
        final_action = str(proposed_action or "HOLD").upper()
        try:
            final_pos = float(proposed_pos)
        except (TypeError, ValueError):
            final_pos = 0.0
        final_pos = float(np.clip(final_pos, 0.0, 1.0))
        trace: list[str] = []

        dd = features.get("drawdown_20d_pct", 0) if isinstance(features, dict) else 0
        try:
            dd = float(dd)
        except (TypeError, ValueError):
            dd = 0.0

        # 1. 回撤击穿 -> 强制清仓
        if dd < self.MAX_DRAWDOWN_LIMIT:
            trace.append(f"[RISK] Drawdown {dd}% hits limit {self.MAX_DRAWDOWN_LIMIT}%. FORCE CLEAR.")
            return "CLEAR", 0.0, trace

        # 2. 恐慌事件 -> 强制减仓
        for sig in news_signals or []:
            if not isinstance(sig, dict):
                continue
            if sig.get("event_type") in PANIC_EVENTS:
                try:
                    impact = float(sig.get("impact_equity", 0))
                except (TypeError, ValueError):
                    impact = 0.0
                if impact < 0:
                    trace.append(f"[RISK] Critical Event: {sig.get('event_type')}. FORCE REDUCE.")
                    return "REDUCE", float(min(final_pos, 0.1)), trace

        # 3. 仓位上限
        if final_pos > self.MAX_POS_LIMIT:
            trace.append(f"[RISK] Cap position {final_pos} -> {self.MAX_POS_LIMIT}.")
            final_pos = self.MAX_POS_LIMIT

        # 4. 高波动降仓
        vol = features.get("volatility_ann_pct", 0) if isinstance(features, dict) else 0
        try:
            vol = float(vol)
        except (TypeError, ValueError):
            vol = 0.0
        if vol > self.VOL_REDUCE_TRIGGER:
            scale = max(0.5, 1.0 - ((vol - self.VOL_REDUCE_TRIGGER) / 100.0))
            new_pos = round(final_pos * scale, 2)
            if new_pos < final_pos:
                trace.append(f"[RISK] High Vol {vol}%. Scaled pos {final_pos} -> {new_pos}.")
                final_pos = new_pos

        if final_action == "CLEAR":
            final_pos = 0.0
        if not trace:
            trace.append("[RISK] Proposal Approved.")

        final_pos = float(np.clip(final_pos, 0.0, self.MAX_POS_LIMIT))
        return final_action, final_pos, trace

import numpy as np


class RiskGate:
    def __init__(
        self,
        max_drawdown_limit_pct: float = -8.0,
        vol_reduce_trigger_ann_pct: float = 30.0,
        max_pos_limit: float = 0.5,
    ):
        self.MAX_DRAWDOWN_LIMIT = float(max_drawdown_limit_pct)
        self.MAX_POS_LIMIT = float(max_pos_limit)
        self.VOL_REDUCE_TRIGGER = float(vol_reduce_trigger_ann_pct)

        self.PANIC_EVENTS = [
            "regulation_crackdown",
            "war_breakout",
            "exchange_investigation",
        ]

    def adjudicate(self, features, news_signals, proposed_action, proposed_pos):
        final_action = str(proposed_action or "HOLD").upper()
        try:
            final_pos = float(proposed_pos)
        except Exception:
            final_pos = 0.0

        final_pos = float(np.clip(final_pos, 0.0, 1.0))
        trace = []

        dd = features.get("drawdown_20d_pct", 0) if isinstance(features, dict) else 0
        try:
            dd = float(dd)
        except Exception:
            dd = 0.0

        if dd < float(self.MAX_DRAWDOWN_LIMIT):
            trace.append(
                f"[RISK] Drawdown {dd}% hits limit {self.MAX_DRAWDOWN_LIMIT}%. FORCE CLEAR."
            )
            return "CLEAR", 0.0, trace

        if news_signals:
            for sig in news_signals:
                if not isinstance(sig, dict):
                    continue
                if sig.get("event_type") in self.PANIC_EVENTS:
                    try:
                        impact = float(sig.get("impact_equity", 0))
                    except Exception:
                        impact = 0.0
                    if impact < 0:
                        trace.append(
                            f"[RISK] Critical Event: {sig.get('event_type')}. FORCE REDUCE."
                        )
                        final_action = "REDUCE"
                        final_pos = min(final_pos, 0.1)
                        return final_action, float(final_pos), trace

        if final_pos > float(self.MAX_POS_LIMIT):
            trace.append(f"[RISK] Cap position {final_pos} -> {self.MAX_POS_LIMIT}.")
            final_pos = float(self.MAX_POS_LIMIT)

        vol = features.get("volatility_ann_pct", 0) if isinstance(features, dict) else 0
        try:
            vol = float(vol)
        except Exception:
            vol = 0.0

        if vol > float(self.VOL_REDUCE_TRIGGER):
            scale = 1.0 - ((vol - float(self.VOL_REDUCE_TRIGGER)) / 100.0)
            scale = max(0.5, float(scale))
            new_pos = round(float(final_pos) * scale, 2)
            if new_pos < final_pos:
                trace.append(f"[RISK] High Vol {vol}%. Scaled pos {final_pos} -> {new_pos}.")
                final_pos = new_pos

        if final_action == "CLEAR":
            final_pos = 0.0

        if not trace:
            trace.append("[RISK] Proposal Approved.")

        final_pos = float(np.clip(final_pos, 0.0, float(self.MAX_POS_LIMIT)))
        return final_action, final_pos, trace

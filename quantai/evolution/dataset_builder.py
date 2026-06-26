"""偏好数据集构建 —— 迁自 `src/rl/online_learning.py::PreferenceLogger`。

数据飞轮的"建数据集"端：决策先 `log_decision`（出结果前），平仓后 `log_outcome` 落成
completed_decisions.jsonl；`generate_preference_pairs` 把**相似上下文、不同盈亏**的决策配成
DPO 的 (chosen / rejected) 偏好对，供 `evolution/trainer.py` 走离线 DPO 训练。

诚实命名：旧名 `PreferenceLogger` 改 `PreferenceBuilder`（它不止记录，核心是"建偏好对数据集"）。
目录 config 驱动（`EvolutionConfig.preferences_dir`），替换硬编码 `data/dpo_preferences`。
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from quantai.config.schema import EvolutionConfig


class PreferenceBuilder:
    """决策→结果→DPO 偏好对（chosen/rejected）。"""

    def __init__(
        self,
        save_dir: str = "data/evolution/preferences",
        *,
        min_pnl_diff: float = 100.0,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.min_pnl_diff = float(min_pnl_diff)
        self._lock = threading.Lock()
        self.pending_decisions: Dict[str, Dict] = {}  # trade_id -> 决策信息

    @classmethod
    def from_config(cls, cfg: EvolutionConfig) -> "PreferenceBuilder":
        return cls(save_dir=cfg.preferences_dir, min_pnl_diff=cfg.min_pnl_diff)

    @property
    def completed_file(self) -> Path:
        return self.save_dir / "completed_decisions.jsonl"

    def log_decision(
        self,
        trade_id: str,
        context: Dict[str, Any],
        decision: str,
        reasoning: str,
        expert: str,
    ) -> None:
        """出结果前记录一次决策（挂起，等 `log_outcome`）。"""
        with self._lock:
            self.pending_decisions[str(trade_id)] = {
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "decision": decision,
                "reasoning": reasoning,
                "expert": expert,
            }

    def log_outcome(self, trade_id: str, pnl: float, hold_bars: int, exit_reason: str) -> None:
        """平仓后补上结果，落成 completed_decisions.jsonl。"""
        with self._lock:
            info = self.pending_decisions.pop(str(trade_id), None)
            if info is None:
                return
            info["outcome"] = {
                "pnl": float(pnl),
                "hold_bars": int(hold_bars),
                "exit_reason": str(exit_reason),
                "is_profitable": float(pnl) > 0,
            }
            with self.completed_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(info, ensure_ascii=False) + "\n")

    def _load_completed(self) -> List[Dict[str, Any]]:
        if not self.completed_file.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in self.completed_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    def generate_preference_pairs(
        self, min_pnl_diff: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """把相似上下文（同 ticker + 同小时）里盈亏差≥阈值的最好/最差决策配成偏好对。"""
        threshold = self.min_pnl_diff if min_pnl_diff is None else float(min_pnl_diff)
        decisions = self._load_completed()

        groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for d in decisions:
            ticker = (d.get("context") or {}).get("ticker", "UNKNOWN")
            hour = str(d.get("timestamp", ""))[:13]  # YYYY-MM-DDTHH
            groups[(ticker, hour)].append(d)

        pairs: List[Dict[str, Any]] = []
        for group in groups.values():
            if len(group) < 2:
                continue
            ordered = sorted(group, key=lambda x: (x.get("outcome") or {}).get("pnl", 0.0))
            worst, best = ordered[0], ordered[-1]
            worst_pnl = (worst.get("outcome") or {}).get("pnl", 0.0)
            best_pnl = (best.get("outcome") or {}).get("pnl", 0.0)
            if best_pnl - worst_pnl >= threshold:
                pairs.append(
                    {
                        "chosen": {
                            "decision": best.get("decision"),
                            "reasoning": best.get("reasoning"),
                            "pnl": best_pnl,
                        },
                        "rejected": {
                            "decision": worst.get("decision"),
                            "reasoning": worst.get("reasoning"),
                            "pnl": worst_pnl,
                        },
                        "context": best.get("context"),
                    }
                )
        return pairs

    def export_pairs(self, out_path: Optional[str] = None, min_pnl_diff: Optional[float] = None) -> Path:
        """把生成的偏好对落成 JSONL，返回路径（给 trainer 喂 DPO）。"""
        pairs = self.generate_preference_pairs(min_pnl_diff=min_pnl_diff)
        target = Path(out_path) if out_path else (self.save_dir / "preference_pairs.jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        return target

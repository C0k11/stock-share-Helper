"""经验采集 + 奖励整形 —— 迁自 `src/rl/online_learning.py` 的采集部分。

**诚实定位**（B-1）：本文件**只做采集与奖励计算**，不含任何梯度更新。`ExperienceBuffer`
把 (state, action, reward, next_state, metadata) 落 JSONL；`RewardShaper` 从盈亏/回撤/类
Sharpe/动作质量算 shaped reward。真训练在 `evolution/trainer.py` 走**离线** DPO/SFT。

与旧版差异：目录 config 驱动（`EvolutionConfig.experiences_dir`），替换硬编码 `data/rl_experiences`。
"""
from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


class ExperienceBuffer:
    """存放交易经验、落盘 JSONL、启动时回载。**只采集，不训练。**"""

    def __init__(
        self,
        max_size: int = 10000,
        save_dir: str = "data/evolution/experiences",
        file_name: str = "experiences.jsonl",
    ) -> None:
        self.buffer: Deque[dict] = deque(maxlen=max_size)
        self.save_dir = Path(save_dir)
        self.file_name = str(file_name or "experiences.jsonl")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._load_existing()

    def _load_existing(self) -> None:
        exp_file = self.save_dir / self.file_name
        if not exp_file.exists():
            return
        try:
            text = exp_file.read_text(encoding="utf-8")
        except Exception:
            return
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                self.buffer.append(obj)

    def add(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        exp = {
            "timestamp": datetime.now().isoformat(),
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "metadata": metadata or {},
        }
        with self._lock:
            self.buffer.append(exp)
            exp_file = self.save_dir / self.file_name
            try:
                with exp_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(exp, ensure_ascii=False) + "\n")
            except Exception:
                pass

    def sample(self, batch_size: int = 32, *, rng: Optional[np.random.Generator] = None) -> List[Dict[str, Any]]:
        """随机采一批（训练用）。可注入 `rng` 让采样可复现。"""
        with self._lock:
            n = len(self.buffer)
            if n <= batch_size:
                return list(self.buffer)
            gen = rng if rng is not None else np.random.default_rng()
            idx = gen.choice(n, batch_size, replace=False)
            return [self.buffer[int(i)] for i in idx]

    def __len__(self) -> int:
        return len(self.buffer)


class RewardShaper:
    """从交易结果整形奖励：PnL（即时）+ 回撤惩罚 + 类 Sharpe + 动作质量奖励。纯逻辑。"""

    def __init__(
        self,
        pnl_scale: float = 0.001,
        sharpe_weight: float = 0.3,
        drawdown_penalty: float = -2.0,
        max_drawdown_trigger: float = -0.05,
    ) -> None:
        self.pnl_scale = pnl_scale
        self.sharpe_weight = sharpe_weight
        self.drawdown_penalty = drawdown_penalty
        self.max_drawdown_trigger = max_drawdown_trigger
        self.pnl_history: List[float] = []

    def compute_reward(
        self,
        realized_pnl: float,
        unrealized_pnl: float,
        drawdown_pct: float,
        position_held_bars: int = 1,
        action_taken: str = "HOLD",
    ) -> Tuple[float, Dict[str, float]]:
        components: Dict[str, float] = {}

        # 1. PnL 奖励（缩放）
        components["pnl"] = realized_pnl * self.pnl_scale

        # 2. 回撤惩罚（超过触发阈值才罚）
        dd_reward = 0.0
        if drawdown_pct < self.max_drawdown_trigger and self.max_drawdown_trigger != 0:
            dd_reward = self.drawdown_penalty * (abs(drawdown_pct) / abs(self.max_drawdown_trigger))
        components["drawdown"] = dd_reward

        # 3. 类 Sharpe（近 20 笔均值/标准差）
        self.pnl_history.append(realized_pnl)
        if len(self.pnl_history) > 20:
            recent = self.pnl_history[-20:]
            std = float(np.std(recent)) + 1e-8
            components["sharpe"] = (float(np.mean(recent)) / std) * self.sharpe_weight
        else:
            components["sharpe"] = 0.0

        # 4. 动作质量奖励
        if action_taken in ("BUY", "SELL") and realized_pnl > 0:
            components["action_quality"] = 0.1
        elif action_taken == "HOLD" and abs(unrealized_pnl) < 100:
            components["action_quality"] = 0.02
        else:
            components["action_quality"] = 0.0

        return float(sum(components.values())), components

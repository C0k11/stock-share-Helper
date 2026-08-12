"""经验采集编排器 —— 迁自 `src/rl/online_learning.py::OnlineLearningManager`。

**诚实命名（B-1）**：旧名 `OnlineLearningManager` 暗示"在线学习/在线更新"，但实际**从未接线
在线实时梯度**——闭环只做采集。故重命名 `ExperienceCollector`，并把"在线梯度"做成**显式
未实现占位** `online_gradient_step()`（调用即 `NotImplementedError`，指向离线 DPO），而不是
旧版那个静默打印的 `_trigger_update` TODO。真训练走 `evolution/trainer.py` 的离线 DPO/SFT。

职责：on_trade_complete / on_step / on_joint_step 采集经验 + 算奖励 + 记指标 + 喂偏好构建器。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from quantai.config.schema import EvolutionConfig
from quantai.evolution.dataset_builder import PreferenceBuilder
from quantai.evolution.experience import ExperienceBuffer, RewardShaper


class ExperienceCollector:
    """采集交易/步级经验，算 shaped reward，落盘并喂偏好构建器。**不做在线梯度。**"""

    def __init__(
        self,
        *,
        experiences_dir: str = "data/evolution/experiences",
        max_experiences: int = 100_000,
        update_interval_trades: int = 50,
        min_experiences: int = 100,
        reward_shaper: Optional[RewardShaper] = None,
        preferences: Optional[PreferenceBuilder] = None,
        enabled: bool = True,
    ) -> None:
        self.experience_buffer = ExperienceBuffer(
            max_size=max_experiences, save_dir=experiences_dir, file_name="experiences.jsonl"
        )
        self.step_buffer = ExperienceBuffer(
            max_size=max_experiences, save_dir=experiences_dir, file_name="step_experiences.jsonl"
        )
        self.joint_buffer = ExperienceBuffer(
            max_size=max_experiences, save_dir=experiences_dir, file_name="joint_step_experiences.jsonl"
        )
        self.reward_shaper = reward_shaper or RewardShaper()
        self.preferences = preferences
        self.enabled = bool(enabled)

        self.update_interval = int(update_interval_trades)
        self.min_experiences = int(min_experiences)
        self.trade_count = 0
        self.last_update_trade = 0
        self._metrics: Dict[str, List[float]] = {"rewards": [], "pnl": [], "win_rate": []}

        # B-1：在线实时梯度从未接线。恒 False，仅作诚实标注。
        self.online_gradient_enabled = False

    @classmethod
    def from_config(
        cls, cfg: EvolutionConfig, *, preferences: Optional[PreferenceBuilder] = None, enabled: bool = True
    ) -> "ExperienceCollector":
        shaper = RewardShaper(
            pnl_scale=cfg.reward_pnl_scale,
            sharpe_weight=cfg.reward_sharpe_weight,
            drawdown_penalty=cfg.reward_drawdown_penalty,
            max_drawdown_trigger=cfg.reward_max_drawdown_trigger,
        )
        return cls(
            experiences_dir=cfg.experiences_dir,
            max_experiences=cfg.max_experiences,
            update_interval_trades=cfg.update_interval_trades,
            min_experiences=cfg.min_experiences,
            reward_shaper=shaper,
            preferences=preferences,
            enabled=enabled,
        )

    # ----------------------------------------------------------------- #
    # 采集
    # ----------------------------------------------------------------- #
    def on_trade_complete(
        self,
        *,
        trade_id: str,
        state: Dict[str, Any],
        action: str,
        pnl: float,
        drawdown_pct: float,
        hold_bars: int,
        exit_reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        self.trade_count += 1
        reward, components = self.reward_shaper.compute_reward(
            realized_pnl=pnl,
            unrealized_pnl=0.0,
            drawdown_pct=drawdown_pct,
            position_held_bars=hold_bars,
            action_taken=action,
        )
        md: Dict[str, Any] = {
            "trade_id": trade_id,
            "pnl": pnl,
            "hold_bars": hold_bars,
            "exit_reason": exit_reason,
            "reward_components": components,
        }
        if isinstance(metadata, dict):
            md.update(metadata)
        self.experience_buffer.add(state=state, action=action, reward=reward, metadata=md)

        if self.preferences is not None:
            self.preferences.log_outcome(
                trade_id=trade_id, pnl=pnl, hold_bars=hold_bars, exit_reason=exit_reason
            )

        self._metrics["rewards"].append(reward)
        self._metrics["pnl"].append(float(pnl))
        self._metrics["win_rate"].append(1.0 if pnl > 0 else 0.0)

    def on_step(
        self,
        *,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        self.step_buffer.add(
            state=state, action=str(action or "HOLD"), reward=float(reward), next_state=next_state, metadata=metadata
        )

    def on_joint_step(
        self,
        *,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        self.joint_buffer.add(
            state=state, action=str(action or "HOLD"), reward=float(reward), next_state=next_state, metadata=metadata
        )

    # ----------------------------------------------------------------- #
    # 离线数据准备（真训练在 trainer.py）
    # ----------------------------------------------------------------- #
    def should_prepare_update(self) -> bool:
        """是否攒够经验 + 距上次够久，可以准备一批离线训练数据。"""
        if len(self.experience_buffer) < self.min_experiences:
            return False
        return (self.trade_count - self.last_update_trade) >= self.update_interval

    def prepare_offline_batch(
        self, batch_size: int = 64, *, rng: Optional[np.random.Generator] = None
    ) -> Dict[str, Any]:
        """采一批经验 + 生成偏好对（供离线 DPO）。**不做任何梯度更新。**"""
        batch = self.experience_buffer.sample(batch_size, rng=rng)
        pairs = self.preferences.generate_preference_pairs() if self.preferences is not None else []
        self.last_update_trade = self.trade_count
        return {"batch": batch, "pairs": pairs}

    def online_gradient_step(self, *args: Any, **kwargs: Any) -> None:
        """在线实时梯度更新——**未实现**（B-1 诚实占位）。

        旧 `online_learning.py::_trigger_update` 留了 `# TODO: gradient updates` 却静默打印
        假装在更新。这里改成**调用即报错**，杜绝"半成品冒充功能"。真训练请走离线：
        `evolution.trainer.EvolutionTrainer` 或 `scripts/train_dpo.py`。
        """
        raise NotImplementedError(
            "Online realtime gradient update is NOT implemented (honest placeholder, B-1). "
            "Use offline DPO: evolution.trainer.EvolutionTrainer / scripts/train_dpo.py."
        )

    def get_metrics(self, window: int = 100) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for key, values in self._metrics.items():
            recent = values[-window:] if len(values) > window else values
            if recent:
                result[f"mean_{key}"] = float(np.mean(recent))
                result[f"std_{key}"] = float(np.std(recent))
        result["total_trades"] = float(self.trade_count)
        result["total_experiences"] = float(len(self.experience_buffer))
        return result

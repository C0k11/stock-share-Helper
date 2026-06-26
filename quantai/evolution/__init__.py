"""quantai.evolution —— 数据飞轮（子系统 D 的"真"部分：采集 → 建数据集 → 离线训练 → 热切换）。

**诚实定位**：本层只做 ① 采集（决策轨迹 + 经验 + 盈亏回填）② 建 DPO 偏好对
③ 编排**离线** DPO 训练 + 写 active adapter 指针。**没有在线实时梯度**——旧 `OnlineLearningManager`
的在线更新从未接线，这里重命名 `ExperienceCollector` 并把在线梯度做成显式 `NotImplementedError` 占位。

四件：
    recorder        -> EvolutionRecorder：决策轨迹 / 盈亏 / 反馈落 JSONL（PaperBroker 注入它回填 PnL）
    experience      -> ExperienceBuffer（只采集）+ RewardShaper（纯奖励整形）
    dataset_builder -> PreferenceBuilder：决策→结果→DPO (chosen/rejected) 偏好对
    collector       -> ExperienceCollector：采集编排（诚实命名）；online_gradient_step 是未实现占位
    trainer         -> EvolutionTrainer：偏好对 → 离线 DPO → active adapter 指针（重活懒导入）
"""
from __future__ import annotations

from quantai.evolution.collector import ExperienceCollector
from quantai.evolution.dataset_builder import PreferenceBuilder
from quantai.evolution.experience import ExperienceBuffer, RewardShaper
from quantai.evolution.recorder import EvolutionRecorder
from quantai.evolution.trainer import EvolutionTrainer

__all__ = [
    "EvolutionRecorder",
    "ExperienceBuffer",
    "RewardShaper",
    "PreferenceBuilder",
    "ExperienceCollector",
    "EvolutionTrainer",
]

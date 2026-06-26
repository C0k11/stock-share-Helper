"""决策轨迹记录器 —— 迁自 `src/learning/recorder.py`。

数据飞轮的"采集"端之一：把每次 Agent 决策（`record`）和事后实现盈亏（`log_outcome`）/
人工反馈（`log_feedback`）按天落成 JSONL。`live` 的 `PaperBroker` 通过注入 `recorder` 调
`log_outcome(ref_id, outcome, comment)` 把成交盈亏回填到对应决策的 `ref_id`。

与旧版差异：
- **删除 `update_reward()` 空转方法**（C-3：方法体只有 `return`，是伪装成功能的 no-op）。
- 目录 **config 驱动**（`EvolutionConfig.trajectories_dir`），替换硬编码 `data/evolution/trajectories`。
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from quantai.config.schema import EvolutionConfig


@dataclass
class Trajectory:
    id: str
    timestamp: str
    agent_id: str
    context: str
    action: str
    outcome: Optional[float] = None
    feedback: str = ""
    type: str = "trajectory"


@dataclass
class FeedbackRecord:
    ref_id: str
    timestamp: str
    score: int
    comment: str = ""
    type: str = "feedback"


@dataclass
class OutcomeRecord:
    ref_id: str
    timestamp: str
    outcome: float
    comment: str = ""
    type: str = "outcome"


class EvolutionRecorder:
    """按天把决策轨迹 / 盈亏 / 反馈落成 JSONL（线程安全）。"""

    def __init__(self, trajectories_dir: str = "data/evolution/trajectories") -> None:
        self.trajectory_dir = Path(trajectories_dir)
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: EvolutionConfig) -> "EvolutionRecorder":
        return cls(trajectories_dir=cfg.trajectories_dir)

    def _current_file(self) -> Path:
        return self.trajectory_dir / f"{datetime.now().strftime('%Y%m%d')}.jsonl"

    def _write(self, data: dict) -> None:
        fp = self._current_file()
        fp.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(data, ensure_ascii=False)
        with self._lock:
            with fp.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def record(
        self,
        *,
        agent_id: str,
        context: str,
        action: str,
        outcome: Optional[float] = None,
        feedback: str = "",
    ) -> str:
        """记录一次决策（出结果前），返回 `ref_id` 供事后 `log_outcome` 回填。"""
        record_id = uuid.uuid4().hex
        rec = Trajectory(
            id=record_id,
            timestamp=datetime.now().isoformat(),
            agent_id=str(agent_id),
            context=str(context),
            action=str(action),
            outcome=outcome,
            feedback=str(feedback or ""),
        )
        self._write(asdict(rec))
        return record_id

    def log_feedback(self, *, ref_id: str, score: int, comment: str = "") -> None:
        self._write(
            asdict(
                FeedbackRecord(
                    ref_id=str(ref_id),
                    timestamp=datetime.now().isoformat(),
                    score=int(score),
                    comment=str(comment or ""),
                )
            )
        )

    def log_outcome(self, *, ref_id: str, outcome: float, comment: str = "") -> None:
        """事后把实现盈亏回填到决策 `ref_id`（PaperBroker 平仓时调）。"""
        self._write(
            asdict(
                OutcomeRecord(
                    ref_id=str(ref_id),
                    timestamp=datetime.now().isoformat(),
                    outcome=float(outcome),
                    comment=str(comment or ""),
                )
            )
        )

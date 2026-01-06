import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


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
    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        self.trajectory_dir = repo_root / "data" / "evolution" / "trajectories"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _current_file(self) -> Path:
        day = datetime.now().strftime("%Y%m%d")
        return self.trajectory_dir / f"{day}.jsonl"

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
        fb = FeedbackRecord(
            ref_id=str(ref_id),
            timestamp=datetime.now().isoformat(),
            score=int(score),
            comment=str(comment or ""),
        )
        self._write(asdict(fb))

    def log_outcome(self, *, ref_id: str, outcome: float, comment: str = "") -> None:
        oc = OutcomeRecord(
            ref_id=str(ref_id),
            timestamp=datetime.now().isoformat(),
            outcome=float(outcome),
            comment=str(comment or ""),
        )
        self._write(asdict(oc))

    def update_reward(self, *, original_action: str, reward_score: float) -> None:
        _ = original_action
        _ = reward_score
        return


recorder = EvolutionRecorder()

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Trajectory:
    timestamp: str
    agent_id: str
    context: str
    action: str
    outcome: Optional[float] = None
    feedback: str = ""


class EvolutionRecorder:
    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        self.trajectory_dir = repo_root / "data" / "evolution" / "trajectories"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _current_file(self) -> Path:
        day = datetime.now().strftime("%Y%m%d")
        return self.trajectory_dir / f"{day}.jsonl"

    def record(
        self,
        *,
        agent_id: str,
        context: str,
        action: str,
        outcome: Optional[float] = None,
        feedback: str = "",
    ) -> None:
        rec = Trajectory(
            timestamp=datetime.now().isoformat(),
            agent_id=str(agent_id),
            context=str(context),
            action=str(action),
            outcome=outcome,
            feedback=str(feedback or ""),
        )
        fp = self._current_file()
        fp.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(rec), ensure_ascii=False)
        with self._lock:
            with fp.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def update_reward(self, *, original_action: str, reward_score: float) -> None:
        _ = original_action
        _ = reward_score
        return


recorder = EvolutionRecorder()

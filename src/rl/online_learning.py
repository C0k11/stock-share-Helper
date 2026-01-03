"""
Online Reinforcement Learning Framework for Live Trading
Enables the multi-agent system to learn from real-time trading outcomes

Architecture:
1. Experience Buffer: Stores (state, action, reward, next_state) tuples from live trading
2. Reward Shaper: Calculates rewards from P&L, Sharpe, drawdown
3. Policy Updater: Periodic gradient updates to Gatekeeper Q-network
4. Preference Logger: Logs chosen/rejected pairs for DPO alignment
"""

import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ExperienceBuffer:
    """Stores trading experiences for online learning"""
    
    def __init__(self, max_size: int = 10000, save_dir: Optional[str] = None):
        self.buffer: deque = deque(maxlen=max_size)
        self.save_dir = Path(save_dir) if save_dir else Path(__file__).parent.parent.parent / "data" / "rl_experiences"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._load_existing()
    
    def _load_existing(self) -> None:
        """Load existing experiences from disk"""
        exp_file = self.save_dir / "experiences.jsonl"
        if exp_file.exists():
            try:
                with open(exp_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self.buffer.append(json.loads(line))
            except Exception:
                pass
    
    def add(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new experience to the buffer"""
        exp = {
            "timestamp": datetime.now().isoformat(),
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "metadata": metadata or {}
        }
        
        with self._lock:
            self.buffer.append(exp)
            
            # Append to file
            exp_file = self.save_dir / "experiences.jsonl"
            with open(exp_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(exp, ensure_ascii=False) + "\n")
    
    def sample(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Sample a batch of experiences for training"""
        with self._lock:
            if len(self.buffer) < batch_size:
                return list(self.buffer)
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)


class RewardShaper:
    """
    Shapes rewards from trading outcomes
    
    Reward components:
    - Realized P&L (immediate)
    - Risk-adjusted return (Sharpe-like)
    - Drawdown penalty
    - Position sizing quality
    """
    
    def __init__(
        self,
        pnl_scale: float = 0.001,
        sharpe_weight: float = 0.3,
        drawdown_penalty: float = -2.0,
        max_drawdown_trigger: float = -0.05
    ):
        self.pnl_scale = pnl_scale
        self.sharpe_weight = sharpe_weight
        self.drawdown_penalty = drawdown_penalty
        self.max_drawdown_trigger = max_drawdown_trigger
        
        # Running statistics for normalization
        self.pnl_history: List[float] = []
        self.returns_history: List[float] = []
    
    def compute_reward(
        self,
        realized_pnl: float,
        unrealized_pnl: float,
        drawdown_pct: float,
        position_held_bars: int = 1,
        action_taken: str = "HOLD"
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute shaped reward for a trading step
        
        Returns:
            reward: Total shaped reward
            components: Breakdown of reward components
        """
        components = {}
        
        # 1. P&L reward (scaled)
        pnl_reward = realized_pnl * self.pnl_scale
        components["pnl"] = pnl_reward
        
        # 2. Drawdown penalty
        dd_reward = 0.0
        if drawdown_pct < self.max_drawdown_trigger:
            dd_reward = self.drawdown_penalty * (abs(drawdown_pct) / abs(self.max_drawdown_trigger))
        components["drawdown"] = dd_reward
        
        # 3. Risk-adjusted component (simplified Sharpe)
        self.pnl_history.append(realized_pnl)
        if len(self.pnl_history) > 20:
            recent_pnls = self.pnl_history[-20:]
            mean_pnl = np.mean(recent_pnls)
            std_pnl = np.std(recent_pnls) + 1e-8
            sharpe_reward = (mean_pnl / std_pnl) * self.sharpe_weight
            components["sharpe"] = sharpe_reward
        else:
            components["sharpe"] = 0.0
        
        # 4. Action quality bonus
        if action_taken in ("BUY", "SELL") and realized_pnl > 0:
            components["action_quality"] = 0.1
        elif action_taken == "HOLD" and abs(unrealized_pnl) < 100:
            components["action_quality"] = 0.02
        else:
            components["action_quality"] = 0.0
        
        # Total reward
        reward = sum(components.values())
        return reward, components


class PreferenceLogger:
    """
    Logs preference pairs for DPO training
    
    Each preference pair contains:
    - chosen: The action/response that led to positive outcome
    - rejected: The action/response that led to negative outcome
    - context: The market context at decision time
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else Path(__file__).parent.parent.parent / "data" / "dpo_preferences"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.pending_decisions: Dict[str, Dict] = {}  # trade_id -> decision info
    
    def log_decision(
        self,
        trade_id: str,
        context: Dict[str, Any],
        decision: str,
        reasoning: str,
        expert: str
    ) -> None:
        """Log a decision before we know the outcome"""
        with self._lock:
            self.pending_decisions[trade_id] = {
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "decision": decision,
                "reasoning": reasoning,
                "expert": expert
            }
    
    def log_outcome(
        self,
        trade_id: str,
        pnl: float,
        hold_bars: int,
        exit_reason: str
    ) -> None:
        """Log outcome and potentially create preference pair"""
        with self._lock:
            if trade_id not in self.pending_decisions:
                return
            
            decision_info = self.pending_decisions.pop(trade_id)
            decision_info["outcome"] = {
                "pnl": pnl,
                "hold_bars": hold_bars,
                "exit_reason": exit_reason,
                "is_profitable": pnl > 0
            }
            
            # Save completed decision
            outcomes_file = self.save_dir / "completed_decisions.jsonl"
            with open(outcomes_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(decision_info, ensure_ascii=False) + "\n")
    
    def generate_preference_pairs(self, min_pnl_diff: float = 100.0) -> List[Dict[str, Any]]:
        """
        Generate preference pairs from completed decisions
        
        Pairs decisions with similar context but different outcomes
        """
        outcomes_file = self.save_dir / "completed_decisions.jsonl"
        if not outcomes_file.exists():
            return []
        
        decisions = []
        with open(outcomes_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    decisions.append(json.loads(line))
        
        pairs = []
        # Group by similar context (simplified: same ticker within same hour)
        from collections import defaultdict
        groups = defaultdict(list)
        
        for d in decisions:
            ticker = d.get("context", {}).get("ticker", "UNKNOWN")
            hour = d.get("timestamp", "")[:13]  # YYYY-MM-DDTHH
            groups[(ticker, hour)].append(d)
        
        for key, group in groups.items():
            if len(group) < 2:
                continue
            
            # Find best and worst in each group
            sorted_group = sorted(group, key=lambda x: x.get("outcome", {}).get("pnl", 0))
            worst = sorted_group[0]
            best = sorted_group[-1]
            
            worst_pnl = worst.get("outcome", {}).get("pnl", 0)
            best_pnl = best.get("outcome", {}).get("pnl", 0)
            
            if best_pnl - worst_pnl >= min_pnl_diff:
                pairs.append({
                    "chosen": {
                        "decision": best.get("decision"),
                        "reasoning": best.get("reasoning"),
                        "pnl": best_pnl
                    },
                    "rejected": {
                        "decision": worst.get("decision"),
                        "reasoning": worst.get("reasoning"),
                        "pnl": worst_pnl
                    },
                    "context": best.get("context")
                })
        
        return pairs


class OnlineLearningManager:
    """
    Orchestrates online learning across all components
    
    Responsibilities:
    - Coordinate experience collection
    - Trigger periodic model updates
    - Log metrics for monitoring
    """
    
    def __init__(
        self,
        update_interval_trades: int = 100,
        min_experiences: int = 500,
        enable_updates: bool = False  # Set True when ready for live updates
    ):
        self.experience_buffer = ExperienceBuffer()
        self.reward_shaper = RewardShaper()
        self.preference_logger = PreferenceLogger()
        
        self.update_interval = update_interval_trades
        self.min_experiences = min_experiences
        self.enable_updates = enable_updates
        
        self.trade_count = 0
        self.last_update_trade = 0
        
        self._metrics: Dict[str, List[float]] = {
            "rewards": [],
            "pnl": [],
            "win_rate": []
        }
    
    def on_trade_complete(
        self,
        trade_id: str,
        state: Dict[str, Any],
        action: str,
        pnl: float,
        drawdown_pct: float,
        hold_bars: int,
        exit_reason: str
    ) -> None:
        """Called when a trade is closed"""
        self.trade_count += 1
        
        # Compute reward
        reward, components = self.reward_shaper.compute_reward(
            realized_pnl=pnl,
            unrealized_pnl=0,
            drawdown_pct=drawdown_pct,
            position_held_bars=hold_bars,
            action_taken=action
        )
        
        # Store experience
        self.experience_buffer.add(
            state=state,
            action=action,
            reward=reward,
            metadata={
                "trade_id": trade_id,
                "pnl": pnl,
                "hold_bars": hold_bars,
                "exit_reason": exit_reason,
                "reward_components": components
            }
        )
        
        # Log outcome for DPO
        self.preference_logger.log_outcome(
            trade_id=trade_id,
            pnl=pnl,
            hold_bars=hold_bars,
            exit_reason=exit_reason
        )
        
        # Update metrics
        self._metrics["rewards"].append(reward)
        self._metrics["pnl"].append(pnl)
        self._metrics["win_rate"].append(1.0 if pnl > 0 else 0.0)
        
        # Check if we should trigger an update
        if self.enable_updates and self._should_update():
            self._trigger_update()
    
    def _should_update(self) -> bool:
        """Check if we should trigger a model update"""
        if len(self.experience_buffer) < self.min_experiences:
            return False
        if self.trade_count - self.last_update_trade < self.update_interval:
            return False
        return True
    
    def _trigger_update(self) -> None:
        """Trigger a model update (placeholder for actual implementation)"""
        self.last_update_trade = self.trade_count
        
        # Sample experiences for training
        batch = self.experience_buffer.sample(batch_size=64)
        
        # Generate preference pairs
        pairs = self.preference_logger.generate_preference_pairs()
        
        # Log update trigger
        print(f"[Online RL] Update triggered at trade {self.trade_count}")
        print(f"[Online RL] Batch size: {len(batch)}, Preference pairs: {len(pairs)}")
        
        # TODO: Implement actual gradient updates
        # This would involve:
        # 1. Update Gatekeeper Q-network using TD learning
        # 2. Fine-tune experts using DPO on preference pairs
        # 3. Update Planner based on regime outcomes
    
    def get_metrics(self, window: int = 100) -> Dict[str, float]:
        """Get recent performance metrics"""
        result = {}
        for key, values in self._metrics.items():
            recent = values[-window:] if len(values) > window else values
            if recent:
                result[f"mean_{key}"] = float(np.mean(recent))
                result[f"std_{key}"] = float(np.std(recent))
        result["total_trades"] = self.trade_count
        result["total_experiences"] = len(self.experience_buffer)
        return result


# Global instance
_online_learning_manager: Optional[OnlineLearningManager] = None


def get_online_learning_manager() -> OnlineLearningManager:
    """Get the global online learning manager"""
    global _online_learning_manager
    if _online_learning_manager is None:
        _online_learning_manager = OnlineLearningManager()
    return _online_learning_manager

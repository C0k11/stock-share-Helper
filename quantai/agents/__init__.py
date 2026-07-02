"""quantai.agents —— 多 Agent 大脑（拆自 150KB 的 src/trading/strategy.py）。

编排顺序（orchestrator）：
    Planner（市场状态） -> Gatekeeper（RL 门控） -> HeuristicRouter（选专家）
    -> Expert（scalper/analyst/news 出方向） -> Chartist（VLM 看图加减分）
    -> MacroGovernor（宏观闸） -> System2Debate（Critic+Judge 复核） -> FinalDecision

设计原则（与 llm/ 一致）：
- 纯决策逻辑与重依赖（torch / LLM 推理 / VLM）解耦：重活通过参数注入，
  所有 agent 的核心逻辑都能用假对象单测，CI 无需 GPU。
- 「诚实命名」：路由是规则不是学习门控 -> `HeuristicRouter`；宏观闸由真实
  VIX/10Y 收益率信号确定性给闸（旧版 random 已删）-> 见 `MacroGovernor`。

子模块按需导入（planner/gatekeeper 的 torch 为懒加载），`import quantai.agents`
不会拉起 torch。
"""
from __future__ import annotations

from quantai.agents.base import (
    Account,
    Action,
    AgentContext,
    ExpertDecision,
    FinalDecision,
    Position,
    Regime,
    flatten_features,
)
from quantai.agents.debate import System2Debate
from quantai.agents.experts import (
    AnalystExpert,
    LLMExpert,
    NewsExpert,
    ScalperExpert,
    make_expert,
)
from quantai.agents.gatekeeper import GateDecision, Gatekeeper
from quantai.agents.orchestrator import AgentOrchestrator
from quantai.agents.overlays import ChartistOverlay, MacroGovernor
from quantai.agents.planner import (
    Planner,
    PlannerDecision,
    assess_regime_rule,
    risk_budget_for,
    strategy_to_regime,
)
from quantai.agents.router import HeuristicRouter

__all__ = [
    # base
    "Account",
    "Action",
    "AgentContext",
    "ExpertDecision",
    "FinalDecision",
    "Position",
    "Regime",
    "flatten_features",
    # router
    "HeuristicRouter",
    # planner
    "Planner",
    "PlannerDecision",
    "assess_regime_rule",
    "risk_budget_for",
    "strategy_to_regime",
    # gatekeeper
    "Gatekeeper",
    "GateDecision",
    # experts
    "LLMExpert",
    "ScalperExpert",
    "AnalystExpert",
    "NewsExpert",
    "make_expert",
    # overlays
    "ChartistOverlay",
    "MacroGovernor",
    # debate
    "System2Debate",
    # orchestrator
    "AgentOrchestrator",
]

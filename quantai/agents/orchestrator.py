"""AgentOrchestrator —— 组合所有 agent 的"大脑"（取代旧 `MultiAgentStrategy.on_bar` 的决策段）。

这是子系统 B（历史推理回测）与 C（实盘）**共用**的同一个大脑：给定一个 `AgentContext`
（特征 + 持仓 + 账户）和可选的注入 `llm`，按固定管线产出 `FinalDecision`。

管线（忠实迁移自 on_bar 的 1-5 步）：
    1. Planner.assess_regime：cash_preservation -> 直接不交易；
    2. Gatekeeper.approve：拒绝 -> 不交易；
    3. HeuristicRouter.route：选专家（或 all_agents_mode 委员会）；
    4. Expert.decide：出 BUY/SELL/HOLD + 理由；
    5. Chartist + MacroGovernor + System2Debate：复核 / 可能否决或改判。

与旧版的边界划分：**只做决策**。下单撮合、账户更新、新闻抓取、推理锁/线程、性能
backlog 降级都属于 `live/`，不在这里（保持大脑纯净、可被回测与实盘复用）。
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from quantai.agents.base import Action, AgentContext, FinalDecision, Regime
from quantai.agents.debate import System2Debate
from quantai.agents.experts import LLMExpert, make_expert
from quantai.agents.gatekeeper import Gatekeeper
from quantai.agents.overlays import ChartistOverlay, MacroGovernor
from quantai.agents.planner import Planner
from quantai.agents.router import HeuristicRouter
from quantai.config.schema import AppConfig

ExpertFactory = Callable[..., LLMExpert]


class AgentOrchestrator:
    def __init__(
        self,
        *,
        planner: Planner,
        gatekeeper: Gatekeeper,
        router: HeuristicRouter,
        system2: System2Debate,
        chartist: ChartistOverlay,
        macro: MacroGovernor,
        all_agents_mode: bool = False,
        committee_policy: str = "conservative",
        expert_factory: ExpertFactory = make_expert,
        chartist_analyzer: Optional[Any] = None,
    ) -> None:
        self.planner = planner
        self.gatekeeper = gatekeeper
        self.router = router
        self.system2 = system2
        self.chartist = chartist
        self.macro = macro
        self.all_agents_mode = bool(all_agents_mode)
        self.committee_policy = str(committee_policy or "conservative").strip().lower()
        self.expert_factory = expert_factory
        self.chartist_analyzer = chartist_analyzer

    @classmethod
    def from_config(
        cls,
        cfg: AppConfig,
        *,
        news_adapter_available: bool = False,
        chartist_analyzer: Optional[Any] = None,
    ) -> "AgentOrchestrator":
        a = cfg.agents
        return cls(
            planner=Planner.from_config(a.planner),
            gatekeeper=Gatekeeper.from_config(a.gatekeeper),
            router=HeuristicRouter.from_config(
                a.router, news_adapter_available=news_adapter_available
            ),
            system2=System2Debate.from_config(a.system2),
            chartist=ChartistOverlay.from_config(a.chartist),
            macro=MacroGovernor.from_config(a.macro_enabled),
            all_agents_mode=a.all_agents_mode,
            chartist_analyzer=chartist_analyzer,
        )

    def decide(self, ctx: AgentContext, *, llm: Optional[Any] = None) -> FinalDecision:
        trace: dict[str, Any] = {}

        # 1. Planner --------------------------------------------------------- #
        regime = self.planner.assess_regime(ctx.features)
        trace["regime"] = regime
        if regime == Regime.CASH_PRESERVATION:
            return FinalDecision(
                action=Action.HOLD,
                approved=False,
                reason="planner_cash_preservation",
                regime=regime,
                trace=trace,
            )

        # 2. Gatekeeper ------------------------------------------------------ #
        if not self.gatekeeper.approve(ctx.features):
            return FinalDecision(
                action=Action.HOLD,
                approved=False,
                reason="gatekeeper_rejected",
                regime=regime,
                trace=trace,
            )

        # 3. Router ---------------------------------------------------------- #
        expert_name, router_meta = self.router.route(ctx.features)
        trace["router"] = router_meta

        # 4. Expert(s) ------------------------------------------------------- #
        if self.all_agents_mode:
            action, analysis, expert_name = self._committee(ctx, llm)
        else:
            dec = self.expert_factory(expert_name).decide(ctx, llm=llm)
            action, analysis = dec.decision, dec.analysis
        trace["expert"] = expert_name
        trace["expert_action"] = action

        # 5. Overlays + System2 --------------------------------------------- #
        chart_score = self.chartist.assess(
            ctx.ticker, action, analyzer=self.chartist_analyzer, asof=ctx.asof
        )
        macro_gear, macro_label = self.macro.assess()
        trace["chart_score"] = chart_score
        trace["macro_label"] = macro_label

        approved, final_action, reason = self.system2.run(
            ctx,
            proposed_action=action,
            proposed_analysis=analysis,
            proposed_expert=expert_name,
            chart_score=chart_score,
            macro_gear=macro_gear,
            macro_label=macro_label,
            llm=llm,
        )

        final = final_action if approved else Action.HOLD
        is_trade = approved and final in {Action.BUY, Action.SELL}
        return FinalDecision(
            action=final,
            approved=is_trade,
            reason=reason,
            expert=expert_name,
            regime=regime,
            chart_score=int(chart_score),
            macro_label=macro_label,
            trace=trace,
        )

    def _committee(
        self, ctx: AgentContext, llm: Optional[Any]
    ) -> Tuple[str, str, str]:
        """all_agents_mode：同时跑 scalper + analyst，按 committee_policy 合议。

        忠实迁移自 on_bar 969-998：conservative=两者一致才动作；aggressive=任一动作即采纳。
        """
        dec_s = self.expert_factory("scalper").decide(ctx, llm=llm)
        dec_a = self.expert_factory("analyst").decide(ctx, llm=llm)
        act_s = Action.normalize(dec_s.decision)
        act_a = Action.normalize(dec_a.decision)

        if self.committee_policy == "aggressive":
            if act_s in {Action.BUY, Action.SELL}:
                return act_s, dec_s.analysis, "committee"
            if act_a in {Action.BUY, Action.SELL}:
                return act_a, dec_a.analysis, "committee"
            return Action.HOLD, "committee: no_action", "committee"

        if act_s == act_a and act_s in {Action.BUY, Action.SELL}:
            return act_s, f"committee agree: {dec_s.analysis}", "committee"
        return Action.HOLD, "committee: disagree_or_hold", "committee"

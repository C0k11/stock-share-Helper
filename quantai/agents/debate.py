"""System2 辩论 —— Critic + Judge 二次复核（迁移自 `strategy.py::_system2_debate`）。

System1（专家）给出快决策后，对高风险方向（默认 BUY/SELL）触发一轮"慢思考"：
- **Critic**：站在反方挑刺，输出 {accept, suggested_decision, pro, con, reasons}；
- **Judge**：综合提案 + critic，给最终 {final_decision, rationale}。

设计：
- LLM 推理通过注入的 `llm`（鸭子类型 `.chat(messages, adapter=...)`）完成，prompt 构建、
  JSON 解析、裁决聚合都是纯逻辑，可用假 llm 单测。
- 解析用 `quantai.llm.json_utils.repair_and_parse_json`（鲁棒救 JSON）。
- `lenient`（忠实迁移）：解析失败/模型忙时不硬性 HOLD，而是放行原提案。
- `buy_only`（忠实迁移）：仅对 BUY/SELL 触发辩论，HOLD 跳过（省推理）。
"""
from __future__ import annotations

import json
from typing import Any, Optional, Tuple

from quantai.agents.base import Action, AgentContext
from quantai.config.schema import System2Config
from quantai.llm.json_utils import repair_and_parse_json
from quantai.llm.prompts import build_messages

_CRITIC_SYS = (
    "You are a strict trading decision critic.\n"
    "Rules: return STRICT JSON only; no markdown; no extra text; no newlines.\n"
    "You MUST respect position semantics: if current position is SHORT (shares<0), "
    "reducing risk/exposure means BUY (cover), not SELL. If LONG (shares>0), reducing "
    "exposure means SELL.\n"
    "Keep 'pro' and 'con' concise (<= 30 words each).\n"
    'Response Format (STRICT JSON ONLY): {"accept": true|false, "suggested_decision": '
    '"BUY"|"SELL"|"HOLD"|"CLEAR", "pro": "...", "con": "...", "reasons": [..3 strings..]}'
)

_JUDGE_SYS = (
    "You are a strict trading decision judge.\n"
    "Rules: return STRICT JSON only; no markdown; no extra text; no newlines.\n"
    "You MUST respect position semantics: if current position is SHORT (shares<0), BUY "
    "reduces short exposure and SELL increases short exposure. If LONG (shares>0), SELL "
    "reduces exposure and BUY increases exposure.\n"
    "Keep 'rationale' concise (<= 60 words).\n"
    'Response Format (STRICT JSON ONLY): {"final_decision": "BUY"|"SELL"|"HOLD"|"CLEAR", '
    '"rationale": "..."}'
)


class System2Debate:
    def __init__(
        self,
        *,
        enabled: bool = True,
        buy_only: bool = True,
        lenient: bool = False,
        adapter: str = "system2",
    ) -> None:
        self.enabled = bool(enabled)
        self.buy_only = bool(buy_only)
        self.lenient = bool(lenient)
        self.adapter = str(adapter)

    @classmethod
    def from_config(cls, cfg: System2Config, *, adapter: str = "system2") -> "System2Debate":
        return cls(
            enabled=cfg.enabled, buy_only=cfg.buy_only, lenient=cfg.lenient, adapter=adapter
        )

    # --- 触发判定（纯逻辑） --- #
    def should_run(self, proposed_action: str) -> bool:
        if not self.enabled:
            return False
        if self.buy_only and Action.normalize(proposed_action) not in {Action.BUY, Action.SELL}:
            return False
        return True

    # --- prompt 构建（纯逻辑） --- #
    def build_critic_messages(
        self,
        ctx: AgentContext,
        *,
        proposed_expert: str,
        proposed_action: str,
        proposed_analysis: str,
        chart_score: int = 0,
        macro_gear: float = 0.0,
        macro_label: str = "",
    ) -> list[dict[str, str]]:
        action_up = Action.normalize(proposed_action)
        tech = ctx.technical
        sig = ctx.signal
        pos = ctx.position
        lines = [
            f"Ticker: {str(ctx.ticker).upper()}",
            f"Date: {ctx.asof}",
            f"Close: {tech.get('close', '')}",
            f"Price vs MA20: {tech.get('price_vs_ma20', '')}",
            f"Return 5d: {tech.get('return_5d', '')}",
            f"Volatility 20d: {tech.get('volatility_20d', '')}",
            f"Volume ratio: {tech.get('vol_ratio', '')}",
            f"Composite signal: {sig.get('composite', '')}",
            f"Chartist score: {int(chart_score)}",
            f"Macro regime: {str(macro_label)} (gear={macro_gear})",
            f"Current position shares: {pos.shares:g} ({pos.side})",
            "Action semantics: BUY increases long / reduces short; SELL reduces long / "
            "increases short; HOLD no trade; CLEAR means close position to FLAT.",
            "",
            f"Proposed by expert: {str(proposed_expert or '').strip() or 'unknown'}",
            f"Proposed decision: {action_up}",
            f"Proposed analysis: {str(proposed_analysis or '').strip()}",
        ]
        return build_messages("\n".join(lines), _CRITIC_SYS)

    def build_judge_messages(
        self,
        ctx: AgentContext,
        *,
        proposed_action: str,
        proposed_analysis: str,
        critic_json: dict,
    ) -> list[dict[str, str]]:
        action_up = Action.normalize(proposed_action)
        pos = ctx.position
        user = "\n".join(
            [
                f"Ticker: {str(ctx.ticker).upper()}",
                f"Current position shares: {pos.shares:g} ({pos.side})",
                "Proposal JSON: "
                + json.dumps(
                    {"decision": action_up, "analysis": str(proposed_analysis or "").strip()},
                    ensure_ascii=False,
                ),
                "Critic JSON: " + json.dumps(critic_json, ensure_ascii=False),
            ]
        )
        return build_messages(user, _JUDGE_SYS)

    # --- 裁决聚合（纯逻辑） --- #
    def aggregate(
        self, final_decision: str, proposed_action: str, rationale: str = ""
    ) -> Tuple[bool, str, str]:
        final_dec = str(final_decision or "").strip().upper()
        action_up = Action.normalize(proposed_action)
        if final_dec in {"CLEAR", "HOLD"}:
            return False, Action.HOLD, rationale or "system2_hold"
        if final_dec in {"BUY", "SELL"}:
            return True, final_dec, rationale
        return True, action_up, rationale

    def _on_parse_fail(self, stage: str, proposed_action: str) -> Tuple[bool, str, str]:
        reason = f"{stage}_parse_failed"
        if self.lenient:
            return True, Action.normalize(proposed_action), reason
        return False, Action.HOLD, reason

    # --- 主接口（注入 llm） --- #
    def run(
        self,
        ctx: AgentContext,
        *,
        proposed_action: str,
        proposed_analysis: str = "",
        proposed_expert: str = "",
        chart_score: int = 0,
        macro_gear: float = 0.0,
        macro_label: str = "",
        llm: Optional[Any] = None,
    ) -> Tuple[bool, str, str]:
        """跑 Critic -> Judge，返回 (approved, action, reason)。

        - 不触发（disabled / buy_only 且非 BUY/SELL）：放行原提案。
        - 无 llm：放行原提案（不让"没模型"变成全盘 HOLD）。
        - 解析失败：lenient 放行，否则 HOLD。
        """
        action_up = Action.normalize(proposed_action)
        if not self.should_run(action_up):
            return True, action_up, "system2_skipped"
        if llm is None:
            return True, action_up, "system2_no_llm"

        critic_msgs = self.build_critic_messages(
            ctx,
            proposed_expert=proposed_expert,
            proposed_action=action_up,
            proposed_analysis=proposed_analysis,
            chart_score=chart_score,
            macro_gear=macro_gear,
            macro_label=macro_label,
        )
        critic_json = repair_and_parse_json(_one_line(self._infer(llm, critic_msgs)))
        if not isinstance(critic_json, dict):
            return self._on_parse_fail("critic", action_up)

        judge_msgs = self.build_judge_messages(
            ctx,
            proposed_action=action_up,
            proposed_analysis=proposed_analysis,
            critic_json=critic_json,
        )
        judge_json = repair_and_parse_json(_one_line(self._infer(llm, judge_msgs)))
        if not isinstance(judge_json, dict):
            return self._on_parse_fail("judge", action_up)

        final_dec = str(judge_json.get("final_decision") or "").strip().upper()
        rationale = str(judge_json.get("rationale") or "").strip()
        return self.aggregate(final_dec, action_up, rationale)

    def _infer(self, llm: Any, messages: list[dict[str, str]]) -> str:
        try:
            return str(llm.chat(messages, adapter=self.adapter) or "")
        except Exception:
            return ""


def _one_line(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ").strip()

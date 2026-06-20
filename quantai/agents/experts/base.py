"""专家基类 `LLMExpert` —— scalper/analyst/news 的共同骨架。

迁移自 `strategy.py::_model_infer`（prompt 构建 + 调模型 + 解析）与 `_heuristic_infer`
（无模型时的动量兜底）。

两处与旧版的差异：
- **重活注入**：真正的 LLM 推理通过注入的 `llm`（鸭子类型，需有 `.chat(messages, adapter=...)`）
  完成；不注入则走确定性启发式。于是专家逻辑（prompt 拼装、解析、回退）可用假 llm 单测。
- **删 random**：旧 `_heuristic_infer` 有 `if random.random() < 0.3: 随机改判` 的
  "demo 随机"，会让回测/实盘不可复现。新版启发式是**纯确定性动量规则**。
"""
from __future__ import annotations

from typing import Any, Optional

from quantai.agents.base import Action, AgentContext, ExpertDecision
from quantai.llm.prompts import build_messages, parse_decision

#: 所有专家共用的输出约束（忠实迁移自 _model_infer 的 system prompt）。
DEFAULT_SYSTEM_PROMPT = (
    'Return ONLY a single-line JSON object: '
    '{"decision": "BUY|SELL|HOLD", "analysis": "brief reason"}. '
    "No markdown, no extra text before/after JSON. "
    "Decision must respect allow_short and current position context."
)


class LLMExpert:
    """单个 MoE 专家：拼 prompt -> 调注入的 LLM(指定 adapter) -> 解析；无模型则启发式。"""

    name: str = "expert"
    default_adapter: str = "scalper"

    def __init__(
        self, *, adapter: Optional[str] = None, system_prompt: Optional[str] = None
    ) -> None:
        self.adapter = str(adapter or self.default_adapter)
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT

    # --- prompt 构建（纯逻辑） --- #
    def build_user_prompt(self, ctx: AgentContext) -> str:
        tech = ctx.technical
        acc = ctx.account
        pos = ctx.position
        t = ctx.ticker
        return (
            f"Ticker: {t}\n"
            f"Close: {_f(tech.get('close')):.2f}\n"
            f"Return 5d: {_f(tech.get('return_5d')):.2f}%\n"
            f"Volatility: {_f(tech.get('volatility_20d')):.1f}%\n\n"
            f"Account:\n"
            f"- cash: {acc.cash:.2f}\n"
            f"- equity: {acc.equity:.2f}\n"
            f"- gross_exposure: {acc.gross_exposure:.2f}\n"
            f"- leverage: {acc.leverage:.2f}\n\n"
            f"Position ({t}):\n"
            f"- shares: {pos.shares:.4f} (positive=long, negative=short)\n"
            f"- avg_price: {pos.avg_price:.2f}\n\n"
            f"Trading rules:\n"
            f"- Output decision BUY|SELL|HOLD.\n"
            f"- BUY means buy more if long/flat OR cover (reduce) if currently short.\n"
            f"- SELL means sell/reduce if long OR open/increase short if flat/short.\n"
            f"- allow_short: {str(ctx.allow_short).lower()}\n\n"
            f"Decide BUY/SELL/HOLD for next 5 days."
        )

    def build_messages(self, ctx: AgentContext) -> list[dict[str, str]]:
        return build_messages(self.build_user_prompt(ctx), self.system_prompt)

    # --- 确定性启发式兜底（无 random） --- #
    def heuristic_decide(self, ctx: AgentContext) -> ExpertDecision:
        tech = ctx.technical
        ret_5d = _f(tech.get("return_5d"))
        price_vs_ma = _f(tech.get("price_vs_ma20"))

        if ret_5d > 3 and price_vs_ma > 2:
            decision = Action.BUY
            analysis = f"Momentum: +{ret_5d:.1f}% 5d, above MA20"
        elif ret_5d < -3 and price_vs_ma < -2:
            decision = Action.SELL
            analysis = f"Weakness: {ret_5d:.1f}% 5d, below MA20"
        else:
            decision = Action.HOLD
            analysis = "No clear signal"

        return ExpertDecision(
            decision=decision,
            analysis=analysis,
            expert=self.name,
            meta={"source": "heuristic", "adapter": self.adapter},
        )

    # --- 主接口 --- #
    def decide(self, ctx: AgentContext, llm: Optional[Any] = None) -> ExpertDecision:
        """有可用 LLM 则模型推理，否则确定性启发式；任何推理异常都安全回退启发式。"""
        if llm is None or not getattr(llm, "is_loaded", True):
            return self.heuristic_decide(ctx)
        try:
            raw = llm.chat(self.build_messages(ctx), adapter=self.adapter)
        except Exception:
            return self.heuristic_decide(ctx)
        if not str(raw or "").strip():
            return self.heuristic_decide(ctx)
        parsed = parse_decision(raw)
        return ExpertDecision(
            decision=parsed["decision"],
            analysis=parsed["analysis"],
            expert=self.name,
            meta={"source": "model", "adapter": self.adapter},
        )


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

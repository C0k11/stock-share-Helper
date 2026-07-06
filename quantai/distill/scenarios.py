"""蒸馏场景构造：真实市场数据 + analysis/ 指标 → 教师模型的分析任务 prompt。

纯逻辑（不碰网络）：输入价格 DataFrame，输出 `Scenario`（消息列表 + 元数据）。
指标全部复用 `quantai.analysis`（与仪表盘/仓库同一套数字，教师看到的就是系统算的）。

场景 = 一个 symbol 在某个"截止日"的分析任务。同一份数据可出多类任务（趋势判断 /
风险评估 / 回踩买点 / 波动环境），task 模板见 `_TASKS`。scenario_id 确定性
（symbol+date+task），重跑可去重。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

import pandas as pd

from quantai.analysis import (
    atr,
    bollinger,
    macd,
    moving_average_system,
    pullback,
    realized_volatility,
    roc,
    rsi,
)

_SYSTEM_PROMPT = (
    "你是一名专业的美股量化分析师。基于给定的行情与技术指标数据做分析，要求：\n"
    "1) 结论明确（看多/看空/中性 + 置信度），2) 理由必须引用给定的具体数值，"
    "3) 明确列出主要风险与失效条件，4) 不编造数据中不存在的信息，不确定就说不确定。\n"
    "输出结构：【结论】【依据】【风险】三段。"
)

#: 任务模板：key -> 用户侧任务指令
_TASKS: dict[str, str] = {
    "trend": "请判断该股当前的趋势状态（多头/空头/震荡），并给出你对未来 1-4 周方向的看法。",
    "risk": "请评估当前持有该股的风险暴露（波动环境、回撤风险、需要警惕的技术位）。",
    "pullback": "该股当前是否处于上升趋势中的回踩买点？请结合均线与回撤幅度论证。",
    "volatility": "请分析该股当前的波动率环境（挤压还是扩张），及其对仓位管理的含义。",
}


@dataclass
class Scenario:
    """一条蒸馏任务：喂给教师模型的完整对话上下文。"""

    scenario_id: str
    symbol: str
    as_of: str
    task: str
    messages: list[dict] = field(default_factory=list)  # [{"role","content"}...]
    meta: dict = field(default_factory=dict)


def _fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}" if x == x else "n/a"


def build_indicator_brief(df: pd.DataFrame, symbol: str) -> tuple[str, dict]:
    """价格 df → 指标简报文本 +（数值 dict 供 meta/审计）。

    df 需含 close（high/low 缺失用 close 兜底，仅影响 ATR/pullback 口径）。
    指标不足热身时诚实显示 n/a（教师被要求不编造）。
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close

    mas = moving_average_system(close, windows=(20, 50, 200)).iloc[-1]
    pb = pullback(close, high).iloc[-1]
    bb = bollinger(close).iloc[-1]
    vals = {
        "last_close": float(close.iloc[-1]),
        "ret_5d_pct": float(roc(close, 5).iloc[-1]),
        "ret_20d_pct": float(roc(close, 20).iloc[-1]),
        "rsi_14": float(rsi(close, 14).iloc[-1]),
        "macd_hist": float(macd(close)["macd_histogram"].iloc[-1]),
        "sma_20": float(mas["sma_20"]),
        "sma_50": float(mas["sma_50"]),
        "sma_200": float(mas["sma_200"]),
        "ma_alignment": float(mas["ma_alignment"]),
        "bb_percent_b": float(bb["bb_percent_b"]),
        "bb_bandwidth": float(bb["bb_bandwidth"]),
        "atr_14": float(atr(high, low, close, 14).iloc[-1]),
        "realized_vol_20_ann": float(realized_volatility(close, 20).iloc[-1]),
        "retrace_from_20d_high": float(pb["retrace_from_high"]),
        "in_uptrend": bool(pb["in_uptrend"]),
        "is_pullback": bool(pb["pullback"]),
    }
    brief = (
        f"标的：{symbol}（截至 {df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]}）\n"
        f"最新收盘：{_fmt(vals['last_close'])}\n"
        f"近5日/20日涨跌幅：{_fmt(vals['ret_5d_pct'])}% / {_fmt(vals['ret_20d_pct'])}%\n"
        f"RSI(14)：{_fmt(vals['rsi_14'], 1)}　MACD 柱：{_fmt(vals['macd_hist'], 4)}\n"
        f"均线：MA20={_fmt(vals['sma_20'])} MA50={_fmt(vals['sma_50'])} MA200={_fmt(vals['sma_200'])}"
        f"　多头排列度：{_fmt(vals['ma_alignment'])}\n"
        f"布林 %B：{_fmt(vals['bb_percent_b'])}　带宽：{_fmt(vals['bb_bandwidth'], 4)}\n"
        f"ATR(14)：{_fmt(vals['atr_14'])}　年化波动率(20D)：{_fmt(vals['realized_vol_20_ann'] * 100, 1)}%\n"
        f"距 20 日高点回落：{_fmt(vals['retrace_from_20d_high'] * 100, 1)}%"
        f"　趋势向上：{'是' if vals['in_uptrend'] else '否'}"
        f"　回踩形态：{'是' if vals['is_pullback'] else '否'}"
    )
    return brief, vals


class ScenarioBuilder:
    """{symbol: 价格 df} → 蒸馏场景流。

    Args:
        min_bars: 低于该根数的标的直接跳过（指标基本全 n/a，教师没料可分析）。
        tasks: 要生成的任务子集（默认全部 `_TASKS`）。
    """

    def __init__(self, min_bars: int = 60, tasks: Optional[list[str]] = None) -> None:
        self.min_bars = min_bars
        unknown = set(tasks or []) - set(_TASKS)
        if unknown:
            raise ValueError(f"未知任务模板 {sorted(unknown)}，可选 {sorted(_TASKS)}")
        self.tasks = list(tasks) if tasks else list(_TASKS)

    def build(self, prices: dict[str, pd.DataFrame]) -> Iterator[Scenario]:
        for symbol, df in prices.items():
            if df is None or len(df) < self.min_bars or "close" not in df.columns:
                continue
            brief, vals = build_indicator_brief(df, symbol)
            as_of = str(df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1])
            for task in self.tasks:
                yield Scenario(
                    scenario_id=f"{symbol}_{as_of}_{task}",
                    symbol=symbol,
                    as_of=as_of,
                    task=task,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": f"{brief}\n\n任务：{_TASKS[task]}"},
                    ],
                    meta={"indicators": vals},
                )


def weak_baseline_answer(scenario: Scenario) -> str:
    """确定性弱基线回答（DPO 的 rejected 侧，按任务类型分支）。

    诚实说明：这不是真实学生模型的输出，而是刻意模板化的坏例——
    指标/报告任务给"不引用数值的空话"，打分任务给"破坏 JSON 约定的散文"，
    引导学生学会引用数据、遵守输出格式。后续可替换为学生模型采样。
    """
    if scenario.task == "news_scoring":
        return "整体来看这些新闻有好有坏，市场情绪比较复杂，建议投资者自行判断。"
    if scenario.task == "market_report":
        return (
            "【组合体检】市场近期波动。\n【个股要点】各股表现不一。\n"
            "【风险提示】股市有风险。\n【今日关注】持续关注市场动态。"
        )
    return (
        f"【结论】{scenario.symbol} 后市可能上涨也可能下跌，建议观望。\n"
        "【依据】市场有不确定性，技术指标仅供参考。\n"
        "【风险】股市有风险，投资需谨慎。"
    )


# --------------------------------------------------------------------------- #
# 生产同款任务场景（与线上 prompt 常量同源，学生练的就是要上岗的活）
# --------------------------------------------------------------------------- #
def build_news_scoring_scenarios(
    news_items: list, as_of: str, batch_size: int = 8
) -> Iterator[Scenario]:
    """真实头条 → 新闻打分任务场景（system/user 与 `news_scorer` 生产路径同源）。

    每 batch_size 条一个场景（与生产的"一次调用打一批"一致）。
    """
    from quantai.agents.news_scorer import SCORING_SYSTEM_PROMPT, build_scoring_prompt

    items = [it for it in news_items if getattr(it, "title", "")]
    for b, i in enumerate(range(0, len(items), batch_size)):
        batch = items[i : i + batch_size]
        yield Scenario(
            scenario_id=f"news_{as_of}_{b}",
            symbol="_NEWS_",
            as_of=as_of,
            task="news_scoring",
            messages=[
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": build_scoring_prompt(batch)},
            ],
            meta={"n_items": len(batch), "links": [it.link for it in batch]},
        )


def build_market_report_scenario(
    prices: dict, as_of: str, min_bars: int = 60, max_symbols: int = 8
) -> Optional[Scenario]:
    """多标的指标简报 → 四段式综合报告任务（system 与 `analyst` 生产报告同源）。

    诚实边界：这是生产日报简报的**近似**（多标的指标节）——真实持仓/现金等隐私
    数据**不进训练集**；教师学的是"给定一篮子指标出四段分析"的通用能力。
    """
    from quantai.agents.analyst import REPORT_SYSTEM_PROMPT

    briefs = []
    for sym, df in list(prices.items())[:max_symbols]:
        if df is None or len(df) < min_bars or "close" not in df.columns:
            continue
        briefs.append(build_indicator_brief(df, sym)[0])
    if not briefs:
        return None
    user = (
        "# 市场数据简报（多标的技术指标）\n\n" + "\n\n".join(briefs)
        + "\n\n基于以上数据出具分析报告。"
    )
    return Scenario(
        scenario_id=f"report_{as_of}",
        symbol="_MARKET_",
        as_of=as_of,
        task="market_report",
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        meta={"n_symbols": len(briefs)},
    )

"""v3 期权教学场景：真实链面 + BS 引擎产物 → 教师决策论证任务。

覆盖用户点名的主题：
- ``premium_selling``（玩权利金）：卖 covered call / cash-secured put 的时机、
  行权价选择、被行权风险；
- ``options_timing``（玩期权的时间）：IV 环境决定买权还是卖权、周权还是月权；
- ``zero_dte``（末日期权）：0-7 天到期的 theta/gamma 双刃、纪律与限损；
- ``hedge_review``（对冲评审）：保护性 put / 备兑 call 方案值不值、怎么改。

铁律与 v2 一脉相承：**期权数学全部由 BS 引擎算好写进 prompt**，教师产出的是
"引用数字的决策论证"语料——学生将来学的是引用与判断，绝不是心算 Black-Scholes。
期权链没有历史存档，场景恒为"当下快照"（与新闻打分任务同性质）；每天跑一次
就随时间自然积累多市况链面。
"""

from __future__ import annotations

from typing import Iterator, Optional

from quantai.analysis.options import (
    bs_greeks,
    chain_stats,
    covered_call_plan,
    protective_put_plan,
)
from quantai.distill.scenarios import Scenario

OPTIONS_SYSTEM_PROMPT = (
    "你是一名专业的美股期权策略分析师。基于给定的标的行情、期权链面与 "
    "Black-Scholes 引擎计算结果做分析，要求：\n"
    "1) 结论明确，理由必须引用给定的具体数值（IV、权利金、Greeks、保费占比等）；\n"
    "2) 定价与希腊字母一律以给定引擎数值为准，不要自行推算；\n"
    "3) 明确列出风险与失效条件（被行权、时间损耗、流动性、IV 变化）；\n"
    "4) 不编造数据中不存在的信息，数据不足处明说。输出【结论】【依据】【风险】三段。"
)

OPT_TASKS: dict[str, str] = {
    "premium_selling": (
        "当前链面适不适合卖权利金（covered call / cash-secured put）？"
        "结合 IV 水平、给定的备兑方案与行权价距离，论证卖方时机与被行权风险。"
    ),
    "options_timing": (
        "若要用期权表达对该标的的方向观点，现在是买期权还是卖期权的时机？"
        "论证 IV 环境（贵还是便宜）与到期期限选择（周权 vs 月权）的取舍。"
    ),
    "hedge_review": (
        "评审给定的保护性 put 与备兑 call 方案：保费/收益是否划算？"
        "行权价与期限是否合理？给出改进方向并说明代价。"
    ),
    "zero_dte": (
        "评估用给定的近端到期（末日/周内）期权在该标的上做短线的风险收益比："
        "theta 与 gamma 的双刃、时间价值蒸发速度、若参与应有的仓位纪律与限损。"
    ),
}


def _fmt_rows(rows: list[dict], spot: float, n: int = 5) -> str:
    """ATM 附近 ±n 档报价表（有报价的行）。"""
    priced = [r for r in rows if (r.get("strike") or 0) > 0
              and ((r.get("bid") or 0) > 0 or (r.get("lastPrice") or 0) > 0)]
    priced.sort(key=lambda r: abs(r["strike"] - spot))
    lines = []
    for r in sorted(priced[:n * 2], key=lambda r: r["strike"]):
        iv = r.get("impliedVolatility")
        lines.append(
            f"  行权 {r['strike']:.0f}: bid {r.get('bid') or 0:.2f} / ask {r.get('ask') or 0:.2f}"
            f"，IV {iv:.0%}" if iv else
            f"  行权 {r['strike']:.0f}: bid {r.get('bid') or 0:.2f} / ask {r.get('ask') or 0:.2f}，IV n/a"
        )
    return "\n".join(lines) if lines else "  （无有效报价）"


def build_option_brief(
    symbol: str, spot: float, chain: dict, daily_vals: dict,
    nearest_chain: Optional[dict] = None,
) -> str:
    """标的行情摘要 + 链面 + BS 引擎产物 → 期权简报（纯组装可测）。"""
    stats = chain_stats(chain["calls"], chain["puts"], spot)
    days = chain["days_to_expiry"]
    T = max(days, 1) / 365.0
    iv = stats.get("atm_iv")

    lines = [
        f"标的：{symbol}　现价：{spot:.2f}",
        f"日线摘要：RSI(14) {daily_vals.get('rsi_14', float('nan')):.1f}"
        f"　20日涨跌 {daily_vals.get('ret_20d_pct', float('nan')):+.1f}%"
        f"　年化波动率(20D) {daily_vals.get('realized_vol_20_ann', float('nan')) * 100:.0f}%"
        f"　趋势向上：{'是' if daily_vals.get('in_uptrend') else '否'}",
        "",
        f"## 期权链（到期 {chain['expiry']}，{days} 天）",
        f"ATM 隐含波动率：{iv:.0%}" if iv is not None else "ATM 隐含波动率：n/a",
        f"Put/Call 成交比：{stats['pc_volume_ratio']:.2f}"
        if stats.get("pc_volume_ratio") is not None else "Put/Call 成交比：无量",
        "Calls（ATM 附近）：", _fmt_rows(chain["calls"], spot),
        "Puts（ATM 附近）：", _fmt_rows(chain["puts"], spot),
    ]
    if iv:
        g = bs_greeks(spot, spot, T, iv, "call")
        lines += [
            "",
            f"## BS 引擎（ATM，IV {iv:.0%}，{days} 天）",
            f"ATM call delta {g['delta']:.2f}，gamma {g['gamma']:.4f}，"
            f"theta {g['theta_per_day']:.3f}/日，vega {g['vega_per_pct']:.3f}/IV点",
        ]
    pp = protective_put_plan(100, spot, chain["puts"], days)
    cc = covered_call_plan(100, spot, chain["calls"], days)
    if pp or cc:
        lines += ["", "## 对冲台（100 股口径，引擎计算）"]
        if pp:
            lines.append(
                f"保护性 put：行权 {pp['strike']:.0f}，保费 {pp['cost_pct']:.1%}，"
                f"覆盖部分最大回撤锁定 {pp['max_loss_pct_covered']:.1%}"
            )
        if cc:
            lines.append(
                f"备兑 call：行权 {cc['strike']:.0f}，收租 {cc['income_pct']:.1%}"
                f"（年化 {cc['annualized_pct']:.0%}），上行封顶 +{cc['upside_capped_pct']:.1%}"
            )
    if nearest_chain and nearest_chain["expiry"] != chain["expiry"]:
        nstats = chain_stats(nearest_chain["calls"], nearest_chain["puts"], spot)
        niv = nstats.get("atm_iv")
        lines += [
            "",
            f"## 近端到期链（{nearest_chain['expiry']}，{nearest_chain['days_to_expiry']} 天——末日/周内口径）",
            f"ATM 隐含波动率：{niv:.0%}" if niv is not None else "ATM 隐含波动率：n/a",
            "Calls：", _fmt_rows(nearest_chain["calls"], spot, n=3),
            "Puts：", _fmt_rows(nearest_chain["puts"], spot, n=3),
        ]
        if niv:
            nT = max(nearest_chain["days_to_expiry"], 1) / 365.0
            ng = bs_greeks(spot, spot, nT, niv, "call")
            lines.append(
                f"近端 ATM call theta {ng['theta_per_day']:.3f}/日，gamma {ng['gamma']:.4f}"
                f"（对比远端 theta 放大倍数自明）"
            )
    return "\n".join(lines)


def build_option_scenarios(
    symbol: str, spot: float, chain: dict, daily_vals: dict,
    nearest_chain: Optional[dict] = None, as_of: str = "",
) -> Iterator[Scenario]:
    """一个标的的链面 → 四类期权任务场景（zero_dte 仅在有近端链时产出）。"""
    brief = build_option_brief(symbol, spot, chain, daily_vals, nearest_chain)
    for task, ask in OPT_TASKS.items():
        if task == "zero_dte" and (
            nearest_chain is None or nearest_chain["days_to_expiry"] > 7
        ):
            continue  # 没有真实近端链就不出末日题（不拿月权冒充 0DTE）
        yield Scenario(
            scenario_id=f"{symbol}_{as_of}_opt_{task}",
            symbol=symbol,
            as_of=as_of,
            task=f"opt_{task}",
            messages=[
                {"role": "system", "content": OPTIONS_SYSTEM_PROMPT},
                {"role": "user", "content": f"{brief}\n\n任务：{ask}"},
            ],
            meta={
                "kind": "options",
                "expiry": chain["expiry"],
                "days_to_expiry": chain["days_to_expiry"],
                "spot": spot,
            },
        )

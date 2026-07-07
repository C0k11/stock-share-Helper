"""盘中战术台：确定性操作建议引擎（规则层，秒级刷新；LLM 观点走驻留模型另出）。

工作台"实时作战台"的大脑：日线五因子计分（与 distill.journal 的 rule_v1 同门）
叠加**盘中修正**（卖压/贴地/放量急跌），输出每标的一张操作卡——
动作（加仓/持有/减持/清仓/关注买点/观望）+ 止损参考位 + 引用真实数值的理由。

诚实边界：
- 这是**分析参考不是投资建议**（渲染层必须带此标注）；
- 全部结论由给定数据确定性推导，可复现、可单测；
- 止损参考位是技术位（MA20 / 日内低点），不是任何形式的收益承诺。
"""

from __future__ import annotations

from typing import Optional

#: 动作字典（渲染层按此上色：红=减仓向，绿=加仓向，灰=中性）
ACTION_ADD = "加仓"
ACTION_HOLD = "持有"
ACTION_TRIM = "减持"
ACTION_EXIT = "清仓"
ACTION_ENTRY = "关注买点"
ACTION_WATCH = "观望"

#: 驻留 LLM 战术观点的 system prompt（工作台实时综述用；与日报/快评 prompt 区分——
#: 这里要求落到"标的+动作+参考位"的可执行颗粒度）
TACTICAL_SYSTEM_PROMPT = (
    "你是盘中交易台的战术分析师。基于给定的实时作战台数据（每标的的操作卡、"
    "盘中卖压指标与新闻情绪）做综述：\n"
    "1) 只引用给定数据中的数字与事实，不编造；\n"
    "2) 输出【盘面判断】【操作建议】【风险】三段；操作建议必须落到具体标的：\n"
    "   动作（加仓/持有/减持/清仓/观望）+ 参考价位（如止损位），与操作卡冲突时说明理由；\n"
    "3) 持仓标的优先讨论；4) 短句克制，这是分析参考不是投资建议。"
)

_TACTICAL_SYSTEM_PROMPT_EN = (
    "You are the tactical analyst on an intraday trading desk. Summarize the live "
    "tactics-board data (per-symbol action cards, intraday sell-pressure metrics, "
    "news sentiment). Rules: 1) cite only numbers and facts present in the given "
    "data — never invent; 2) answer in English with three sections "
    "[Market Read] [Actions] [Risks]; actions must name specific symbols with an "
    "action (add/hold/trim/exit/watch) and a reference level such as a stop, and "
    "explain any disagreement with the rule cards; 3) discuss held positions first; "
    "4) short, restrained sentences — analysis reference, not investment advice. "
    "The data brief may be written in Chinese; still answer in English."
)


def tactical_system_prompt(lang: str = "zh") -> str:
    return _TACTICAL_SYSTEM_PROMPT_EN if lang == "en" else TACTICAL_SYSTEM_PROMPT


#: 动作词渲染映射（内部 canonical 值保持中文，UI 层按语言翻）
ACTION_LABELS_EN = {
    ACTION_ADD: "Add", ACTION_HOLD: "Hold", ACTION_TRIM: "Trim",
    ACTION_EXIT: "Exit", ACTION_ENTRY: "Entry watch", ACTION_WATCH: "Watch",
}


def action_label(action: str, lang: str = "zh") -> str:
    return ACTION_LABELS_EN.get(action, action) if lang == "en" else action


#: 操作卡理由/风险模板（advise(lang=...) 选用；en 缺条目回退 zh）
_TXT = {
    "zh": {
        "ma_bull": "均线多头排列 {v:.2f}", "ma_bear": "均线空头排列 {v:.2f}",
        "uptrend": "20 日趋势向上", "macd": "MACD 柱 {v:+.3f}", "r20": "20 日 {v:+.1f}%",
        "rsi_hot": "RSI {v:.0f} 超买", "retrace": "距 20 日高点 -{v:.0f}%",
        "vol_hot": "年化波动率 {v:.0f}% 偏高，仓位应打折",
        "sell_pressure": "盘中抛压主导（跌量 {d:.0f}%、VWAP 下方 {p:.1f}%）",
        "at_low": "贴日内低点", "vol_crash": "放量急跌（{x:.1f}× 均量、{c:+.1f}%）",
        "take_profit": "可考虑部分止盈锁定利润",
    },
    "en": {
        "ma_bull": "MAs in bullish stack {v:.2f}", "ma_bear": "MAs in bearish stack {v:.2f}",
        "uptrend": "20d trend up", "macd": "MACD hist {v:+.3f}", "r20": "20d {v:+.1f}%",
        "rsi_hot": "RSI {v:.0f} overbought", "retrace": "-{v:.0f}% off 20d high",
        "vol_hot": "annualized vol {v:.0f}% high — size down",
        "sell_pressure": "intraday sell pressure ({d:.0f}% down-volume, {p:.1f}% below VWAP)",
        "at_low": "at day low", "vol_crash": "high-volume drop ({x:.1f}× avg vol, {c:+.1f}%)",
        "take_profit": "consider partial profit taking",
    },
}


def _txt(lang: str, key: str, **kw) -> str:
    s = _TXT.get(lang, {}).get(key) or _TXT["zh"][key]
    return s.format(**kw)


def _f(vals: dict, key: str) -> Optional[float]:
    v = vals.get(key)
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def advise(
    symbol: str,
    daily_vals: dict,
    intraday: Optional[dict] = None,
    held: bool = False,
    avg_cost: Optional[float] = None,
    lang: str = "zh",
) -> dict:
    """日线因子 + 盘中修正 → 一张操作卡。

    Args:
        daily_vals: `build_indicator_brief` 的数值 dict（**收盘口径**——调用方负责
            剔除当日进行中的半根日线，盘中信息由 `intraday` 入口进来，不混口径）。
        intraday: `intraday_stats` 输出（收盘后/无当日数据传 None）。
        held / avg_cost: 是否实际持仓及成本（持仓才给止损位与减清仓动作）。

    Returns:
        {symbol, action, score, last, day_pct, stop, reasons, risks, held}
    """
    reasons: list[str] = []
    risks: list[str] = []
    score = 0

    ma = _f(daily_vals, "ma_alignment")
    if ma is not None:
        if ma >= 0.67:
            score += 1
            reasons.append(_txt(lang, "ma_bull", v=ma))
        elif ma <= 0.33:
            score -= 1
            reasons.append(_txt(lang, "ma_bear", v=ma))
    if daily_vals.get("in_uptrend"):
        score += 1
        reasons.append(_txt(lang, "uptrend"))
    macd_h = _f(daily_vals, "macd_hist")
    if macd_h is not None:
        score += 1 if macd_h > 0 else -1
        reasons.append(_txt(lang, "macd", v=macd_h))
    r20 = _f(daily_vals, "ret_20d_pct")
    if r20 is not None:
        score += 1 if r20 > 0 else -1
        reasons.append(_txt(lang, "r20", v=r20))
    rsi = _f(daily_vals, "rsi_14")
    if rsi is not None and rsi >= 70:
        # 阈值与 rule_v1 对齐（70）：同族引擎对同一数据必须给同一分
        score -= 1
        risks.append(_txt(lang, "rsi_hot", v=rsi))
    # retrace_from_20d_high 生产口径恒为正（1 - close/rolling_high ∈ [0,1)，
    # 越大回撤越深）——旧判 `<= -0.15` 是永假死代码（审查实锤，rule_v1 同病同修）
    retrace = _f(daily_vals, "retrace_from_20d_high")
    if retrace is not None and retrace >= 0.15:
        score -= 1
        risks.append(_txt(lang, "retrace", v=retrace * 100))
    vol = _f(daily_vals, "realized_vol_20_ann")
    if vol is not None and vol >= 0.6:
        # 第五因子（波动）：漏掉它会让作战台在高波动票上比 rule_v1 激进整整一档
        score -= 1
        risks.append(_txt(lang, "vol_hot", v=vol * 100))

    # ---- 盘中修正（只降不升：盘中噪音不该独立制造买入信号） ----
    last = _f(daily_vals, "last_close")
    day_pct = None
    if intraday:
        last = _f(intraday, "last") or last
        day_pct = _f(intraday, "chg_pct")
        dvs = _f(intraday, "down_volume_share")
        vsv = _f(intraday, "vs_vwap_pct")
        if dvs is not None and vsv is not None and dvs > 0.65 and vsv < -0.01:
            score -= 1
            risks.append(_txt(lang, "sell_pressure", d=dvs * 100, p=abs(vsv) * 100))
        pos = _f(intraday, "close_position")
        if pos is not None and pos < 0.12:
            risks.append(_txt(lang, "at_low"))
        vva = _f(intraday, "vol_vs_avg")
        if vva is not None and day_pct is not None and vva > 1.5 and day_pct < -0.03:
            score -= 1
            risks.append(_txt(lang, "vol_crash", x=vva, c=day_pct * 100))

    # ---- 动作与参考位 ----
    sma20 = _f(daily_vals, "sma_20")
    day_low = _f(intraday or {}, "day_low")
    stop: Optional[float] = None
    if held:
        if score >= 3:
            action = ACTION_ADD
        elif score >= 0:
            action = ACTION_HOLD
        elif score >= -2:
            action = ACTION_TRIM
        else:
            action = ACTION_EXIT
        # 止损参考：价在 MA20 上方给 MA20；已破 MA20 给日内低点（跌破=趋势恶化确认）
        if last is not None and sma20 is not None:
            stop = sma20 if last >= sma20 else (day_low if day_low is not None else None)
        elif day_low is not None:
            stop = day_low
        if rsi is not None and rsi >= 75 and score >= 1:
            risks.append(_txt(lang, "take_profit"))
    else:
        if score >= 3 or (score >= 2 and daily_vals.get("is_pullback")):
            action = ACTION_ENTRY
        else:
            action = ACTION_WATCH

    return {
        "symbol": symbol,
        "action": action,
        "score": score,
        "last": last,
        "day_pct": day_pct,
        "stop": stop,
        "reasons": reasons,
        "risks": risks,
        "held": held,
        "avg_cost": avg_cost,
    }


def advice_row(adv: dict, lang: str = "zh") -> dict:
    """操作卡 → 表格行（渲染层 dataframe 直接吃；表头/动作词按语言）。"""
    from quantai.ui.i18n import tr

    stop = adv.get("stop")
    held_tag = (" [持仓]" if lang == "zh" else " [HELD]") if adv.get("held") else ""
    return {
        tr(lang, "card.symbol"): adv["symbol"] + held_tag,
        tr(lang, "card.last"): f"{adv['last']:.2f}" if adv.get("last") is not None else "—",
        tr(lang, "card.day"): f"{adv['day_pct'] * 100:+.2f}%" if adv.get("day_pct") is not None else "—",
        tr(lang, "card.action"): action_label(adv["action"], lang),
        tr(lang, "card.stop"): f"{stop:.2f}" if stop is not None else "—",
        tr(lang, "card.reasons"): "；".join(adv.get("reasons", [])[:3]) or "—",
        tr(lang, "card.risks"): "；".join(adv.get("risks", [])[:2]) or "—",
    }


#: 警报前缀（UI 按 startswith 决定 warning/info 样式，两语都要认）
ALERT_MARKS = ("[警报]", "[ALERT]")


def alerts(advices: list[dict], lang: str = "zh") -> list[str]:
    """从一批操作卡里挑需要立刻看见的警报（持仓的减/清仓与风险项优先；无状态，
    只看本轮——跨轮变化检测未实现，别当成"止损位变动提醒"用）。"""
    zh = lang != "en"
    out = []
    for a in advices:
        if a.get("held") and a["action"] in (ACTION_TRIM, ACTION_EXIT):
            act = action_label(a["action"], lang)
            if zh:
                stop = f"，止损参考 {a['stop']:.2f}" if a.get("stop") is not None else ""
                out.append(f"[警报] {a['symbol']} 建议{act}{stop}（计分 {a['score']:+d}）")
            else:
                stop = f", stop ref {a['stop']:.2f}" if a.get("stop") is not None else ""
                out.append(f"[ALERT] {a['symbol']} suggests {act}{stop} (score {a['score']:+d})")
        elif a.get("held") and a.get("risks"):
            out.append(f"{'[关注]' if zh else '[WATCH]'} {a['symbol']}：{a['risks'][0]}")
        elif a["action"] == ACTION_ENTRY:
            if zh:
                out.append(f"[机会] {a['symbol']} 出现关注买点（计分 {a['score']:+d}）")
            else:
                out.append(f"[SETUP] {a['symbol']} entry setup (score {a['score']:+d})")
    return out


def build_tactical_brief(
    advices: list[dict],
    scored_news: Optional[list[dict]] = None,
    as_of: str = "",
) -> str:
    """操作卡 + 新闻情绪 → 驻留 LLM 的战术简报（纯组装可测）。"""
    lines = [f"# 实时作战台数据{f'（{as_of}）' if as_of else ''}", "", "## 操作卡（规则引擎）"]
    for a in advices:
        head = f"- {a['symbol']}{'（持仓）' if a.get('held') else ''}: "
        seg = head + (f"现价 {a['last']:.2f}" if a.get("last") is not None else "现价 n/a")
        if a.get("day_pct") is not None:
            seg += f"（当日 {a['day_pct'] * 100:+.2f}%）"
        seg += f"，动作={a['action']}（计分 {a['score']:+d}）"
        if a.get("stop") is not None:
            seg += f"，止损参考 {a['stop']:.2f}"
        if a.get("reasons"):
            seg += f"；依据：{'；'.join(a['reasons'][:3])}"
        if a.get("risks"):
            seg += f"；风险：{'；'.join(a['risks'][:2])}"
        lines.append(seg)
    lines.append("")
    lines.append("## 新闻情绪（标题级 [-1,1]）")
    if scored_news:
        for r in scored_news[:12]:
            it = r["item"]
            s = r.get("sentiment")
            tag = f"{s:+.2f}" if s is not None else "未打分"
            sym = getattr(it, "symbol", None) or "-"
            lines.append(f"- [{sym}] ({tag}) {getattr(it, 'title', '')}")
    else:
        lines.append("（本轮无新闻数据）")
    return "\n".join(lines)

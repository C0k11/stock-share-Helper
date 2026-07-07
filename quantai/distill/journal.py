"""每日决策日志飞轮：真实行情 → 规则学生判断 → 教师独立分析 + 打分 → 训练资产。

每个交易日积累一批带**三重标注**的样本（"有点 RL 意思"的离线数据采集闭环）：

1. **学生判断**：当前是透明规则引擎 ``rule_v1``（引用真实指标数值，确定性可复现）；
   学生模型通过盲评上岗后可替换为模型采样。
2. **教师独立答案**：与生产 prompt 同源（``scenarios._SYSTEM_PROMPT`` + decision 任务），
   干净的 SFT 资料——与既有蒸馏批完全同契约。
3. **教师评分**：0-10 + 点评 = 偏好信号；学生低分样本进 DPO 的 rejected 侧
   （教师答案为 chosen）。
4. **实现结局**：fwd 5/20 日收益由 :func:`backfill_outcomes` 在行情走出来之后回填，
   只进 ``meta.outcome``（结局排序 DPO 的地面真值），**绝不进 prompt**。

与既有蒸馏管线的关系：v1 失败的直接教训是"同日快照零市况多样性"；journal 是
**增量日采**——每天一个快照，随时间自然覆盖多市况，且每条都带学生对照与结局。

成本闸：真实教师调用只走 ``scripts/journal.py --run --confirm-spend``（定时任务的
命令行里显式写这两个开关 = 用户的常设授权）；**每交易日一个文件天然幂等**——
同一 as_of 重复运行直接跳过（周末/节假日 as_of 不前进，自动零花费）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from quantai.distill.scenarios import Scenario, ScenarioBuilder

STUDENT_KIND = "rule_v1"

REVIEW_SYSTEM_PROMPT = (
    "你是一名资深美股量化投资总监，负责评审系统给出的每日操作判断。"
    "先基于给定的行情与指标数据独立形成你自己的看法，再对【系统判断】打分。"
    "评分标准（各占约四分之一）：方向与数据的一致性、是否引用具体数值、"
    "风险与失效条件的完整性、操作建议的可执行性。不编造数据中不存在的信息。\n"
    "输出格式（严格遵守）：第一行只写 `评分: N`（N 为 0-10 的整数），"
    "随后【点评】段指出系统判断的优缺点，最后【正确做法】段给出你自己的判断与操作。"
)

_ACTIONS = ("加仓", "持有", "观望", "减仓")


def _fmt(x, nd: int = 2) -> str:
    try:
        return f"{float(x):.{nd}f}" if float(x) == float(x) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def rule_student_answer(symbol: str, vals: dict) -> str:
    """确定性规则学生：指标 dict → 与教师同格式的操作判断（引用真实数值）。

    诚实边界：这是"系统当前判断"的透明基线（趋势/动量/超买/回撤/波动五因子计分），
    不是模型输出；它的职责是给教师一个**可打分的对照**，教师低分时它就是
    DPO 的 rejected 侧。NaN 因子跳过计分并如实说明。
    """
    score = 0
    reasons: list[str] = []
    risks: list[str] = []

    ma = vals.get("ma_alignment")
    if ma is not None and ma == ma:
        if ma >= 0.67:
            score += 1
            reasons.append(f"均线多头排列度 {_fmt(ma)}（偏多）")
        elif ma <= 0.33:
            score -= 1
            reasons.append(f"均线排列度 {_fmt(ma)} 偏空头")
        else:
            reasons.append(f"均线排列度 {_fmt(ma)} 中性")
    else:
        reasons.append("均线排列度数据不足（n/a），该因子不计分")

    if vals.get("in_uptrend"):
        score += 1
        reasons.append("20 日趋势向上")
    else:
        reasons.append("20 日趋势未向上")

    macd_h = vals.get("macd_hist")
    if macd_h is not None and macd_h == macd_h:
        score += 1 if macd_h > 0 else -1
        reasons.append(f"MACD 柱 {_fmt(macd_h, 4)}（{'正' if macd_h > 0 else '负'}动量）")

    r20 = vals.get("ret_20d_pct")
    if r20 is not None and r20 == r20:
        score += 1 if r20 > 0 else -1
        reasons.append(f"近 20 日涨跌 {_fmt(r20)}%")

    rsi = vals.get("rsi_14")
    if rsi is not None and rsi == rsi:
        reasons.append(f"RSI(14)={_fmt(rsi, 1)}")
        if rsi >= 70:
            score -= 1
            risks.append(f"RSI {_fmt(rsi, 1)} 超买，追高风险")
        elif rsi <= 30:
            risks.append(f"RSI {_fmt(rsi, 1)} 超卖，可能仍在下跌途中（不抄底赌反转）")

    # retrace_from_20d_high 恒为正（1 - close/rolling_high，越大回撤越深）——
    # 旧判 `<= -0.15` 永假（审查实锤：该因子上线以来从未触发过）
    retrace = vals.get("retrace_from_20d_high")
    if retrace is not None and retrace == retrace and retrace >= 0.15:
        score -= 1
        risks.append(f"距 20 日高点已回落 {_fmt(retrace * 100, 1)}%，深回撤中")

    vol = vals.get("realized_vol_20_ann")
    if vol is not None and vol == vol and vol >= 0.6:
        score -= 1
        risks.append(f"年化波动率 {_fmt(vol * 100, 1)}% 偏高，仓位应打折")

    if score >= 3:
        action = "加仓"
    elif score >= 1:
        action = "持有"
    elif score >= -1:
        action = "观望"
    else:
        action = "减仓"

    # 失效条件按现价与 MA20 的相对位置给（价已在 MA20 下方时"跌破失效"是废话——
    # 教师评审实锤点名过，改为反向站稳条件）
    sma20 = vals.get("sma_20")
    last = vals.get("last_close")
    if all(x is not None and x == x for x in (sma20, last)):
        if last >= sma20:
            risks.append(f"若收盘跌破 MA20（{_fmt(sma20)}）则本判断失效，应重估")
        else:
            risks.append(f"现价已在 MA20（{_fmt(sma20)}）下方，若放量站稳 MA20 则本判断失效，应重估")
    else:
        risks.append("MA20 数据不足，无法给出均线失效位")

    return (
        f"【结论】{symbol} 下一交易日操作建议：{action}（规则计分 {score:+d}）。\n"
        f"【依据】{'；'.join(reasons)}。\n"
        f"【风险】{'；'.join(risks)}。"
    )


def build_review_messages(scenario: Scenario, student_answer: str) -> list[dict]:
    """场景 + 学生判断 → 教师评审对话（独立于教师自己作答的那次调用）。"""
    return [
        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{scenario.messages[-1]['content']}\n\n"
                f"【系统判断】\n{student_answer}\n\n请按格式评审。"
            ),
        },
    ]


def parse_review_score(text: str) -> Optional[int]:
    """从评审文本提取 0-10 整数分。格式漂移时退回 `N/10` 写法；都没有 -> None（诚实缺失）。"""
    for pat in (r"评分\s*[:：]\s*(\d{1,2})", r"\b(\d{1,2})\s*/\s*10"):
        m = re.search(pat, text or "")
        if m:
            n = int(m.group(1))
            if 0 <= n <= 10:
                return n
    return None


# --------------------------------------------------------------------------- #
# 日采主流程
# --------------------------------------------------------------------------- #
def run_daily_journal(
    prices: dict,
    client,
    out_dir: str | Path,
    cap: int = 16,
    min_bars: int = 60,
    workers: int = 1,
    force: bool = False,
) -> dict:
    """当日快照 → decision 场景 × (学生 + 教师作答 + 教师评审) → journal JSONL。

    幂等按 **scenario_id 跨全部历史文件去重**（不是文件级）：每个标的以自己的
    最新 K 线日成 id，已采过的直接过滤——周末/节假日股票 as_of 不前进自动零花费；
    7×24 标的（如 BTC-USD）天天有新 as_of 不会把整批股票拖成重复采集，也不会
    因"当日文件已存在"误跳过收盘后的正式快照（文件按最大 as_of 命名、**追加**写）。
    `force=True` 跳过去重并覆盖当日文件（会重复花钱，仅手动排查用）。
    单场景失败不炸整批（进 failures 如实汇报）。
    """
    builder = ScenarioBuilder(min_bars=min_bars, tasks=["decision"])
    candidates = list(builder.build(prices))
    if not candidates:
        return {"written": 0, "skipped": False, "failures": [], "reason": "no scenarios（数据不足？）"}

    if force:
        scenarios = candidates[: max(0, cap)]
    else:
        seen: set[str] = set()
        for _, recs in load_journal(out_dir):
            seen.update(r.get("scenario_id", "") for r in recs)
        scenarios = [sc for sc in candidates if sc.scenario_id not in seen][: max(0, cap)]
        if not scenarios:
            return {
                "written": 0, "skipped": True, "failures": [],
                "as_of": max(sc.as_of for sc in candidates),
                "reason": "全部场景已采集过（幂等零花费）",
            }

    as_of = max(sc.as_of for sc in scenarios)
    path = Path(out_dir) / f"journal_{as_of}.jsonl"

    records: list[Optional[dict]] = [None] * len(scenarios)
    failures: list[dict] = []

    def _one(idx: int, sc: Scenario) -> None:
        try:
            student = rule_student_answer(sc.symbol, sc.meta.get("indicators", {}))
            teacher = client.chat(sc.messages)
            if not (teacher or "").strip():
                raise RuntimeError("empty teacher answer")
            review_text = client.chat(build_review_messages(sc, student))
            records[idx] = {
                "scenario_id": sc.scenario_id,
                "symbol": sc.symbol,
                "as_of": sc.as_of,
                "task": sc.task,
                "messages": sc.messages,
                "student_answer": student,
                "student_kind": STUDENT_KIND,
                "teacher_answer": teacher,
                "review": {"score": parse_review_score(review_text), "text": review_text},
                "meta": {
                    "indicators": sc.meta.get("indicators", {}),
                    "outcome": None,  # backfill_outcomes 在行情走出来后回填
                    "teacher_model": getattr(client, "model", "unknown"),
                },
            }
        except Exception as exc:  # noqa: BLE001 - 单场景失败不中断整批
            failures.append({"scenario_id": sc.scenario_id, "error": str(exc)})

    if workers <= 1:
        for i, sc in enumerate(scenarios):
            _one(i, sc)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda t: _one(*t), enumerate(scenarios)))

    kept = [r for r in records if r is not None]
    n = 0
    if kept:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if force else "a"  # 正常路径追加（同名文件可能已有别的 as_of 批次）
        with path.open(mode, encoding="utf-8") as f:
            for rec in kept:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return {
        "written": n,
        "skipped": False,
        "failures": failures,
        "as_of": as_of,
        "path": str(path),
        "usage": dict(getattr(client, "usage_totals", {})),
        "scored": sum(1 for r in kept if r["review"]["score"] is not None),
    }


# --------------------------------------------------------------------------- #
# 结局回填 + 训练格式导出
# --------------------------------------------------------------------------- #
def load_journal(journal_dir: str | Path) -> list[tuple[Path, list[dict]]]:
    """读全部 journal 文件（按文件名升序）。"""
    out = []
    for p in sorted(Path(journal_dir).glob("journal_*.jsonl")):
        recs = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        out.append((p, recs))
    return out


def backfill_outcomes(journal_dir: str | Path, prices: dict) -> dict:
    """给已成熟的 journal 记录回填 fwd 5/20 日实现收益（只写 meta.outcome）。

    部分成熟部分回填（够 5 根填 fwd_5d，fwd_20d 留 None 等下次）；已填齐的跳过。
    零教师调用、零花费；幂等（重跑无新数据时 updated=0）。
    """
    files = updated = scanned = 0
    for path, records in load_journal(journal_dir):
        changed = False
        for r in records:
            scanned += 1
            meta = r.setdefault("meta", {})
            oc = meta.get("outcome") or {}
            if oc.get("fwd_5d") is not None and oc.get("fwd_20d") is not None:
                continue
            df = prices.get(r.get("symbol"))
            if df is None or getattr(df, "empty", True) or "close" not in df.columns:
                continue
            future = df.loc[r["as_of"]:]["close"].dropna()
            if future.empty:
                continue
            # 窗口校验：pandas 标签切片会把越界起点静默钳到窗口开头——若抓来的
            # 价格窗口不够老（没覆盖 as_of），iloc[0] 就不是 as_of 当天的收盘，
            # 算出的"前瞻收益"整条是错的还写进训练标签（审查实锤）。基准日必须
            # 恰好等于 as_of，否则跳过等更长窗口。
            first = future.index[0]
            first_date = str(first.date() if hasattr(first, "date") else first)
            if first_date != str(r["as_of"]):
                continue
            base = float(future.iloc[0])
            if base <= 0:
                continue
            new = {
                "fwd_5d": oc.get("fwd_5d") if oc.get("fwd_5d") is not None
                else (float(future.iloc[5] / base - 1) if len(future) > 5 else None),
                "fwd_20d": oc.get("fwd_20d") if oc.get("fwd_20d") is not None
                else (float(future.iloc[20] / base - 1) if len(future) > 20 else None),
            }
            if new != oc:
                meta["outcome"] = new
                changed = True
                updated += 1
        if changed:
            # 原子重写：journal 是花了真钱的唯一副本，mode='w' 直写在写一半崩溃
            # /断电时会截断毁档——先写临时文件再 os.replace（同盘原子替换）。
            import os

            tmp = path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(tmp, path)
            files += 1
    return {"files_rewritten": files, "records_scanned": scanned, "records_updated": updated}


def journal_to_sft_records(records: list[dict]) -> list[dict]:
    """journal 记录 → SFT 行（conversations 契约，同 `scenario_to_sft_record`）。"""
    out = []
    for r in records:
        if not (r.get("teacher_answer") or "").strip():
            continue
        out.append(
            {
                "conversations": r["messages"]
                + [{"role": "assistant", "content": r["teacher_answer"]}],
                "meta": {
                    **(r.get("meta") or {}),
                    "scenario_id": r["scenario_id"],
                    "symbol": r["symbol"],
                    "task": r["task"],
                    "review_score": (r.get("review") or {}).get("score"),
                },
            }
        )
    return out


def journal_to_dpo_records(records: list[dict], max_student_score: int = 6) -> list[dict]:
    """journal 记录 → DPO 偏好对（教师=chosen，**低分**学生=rejected）。

    只有教师明确给出低分（score <= max_student_score）的样本才成对——
    学生答得好（高分）或没打出分（None）的不硬造偏好，宁缺毋滥。
    """
    out = []
    for r in records:
        score = (r.get("review") or {}).get("score")
        if score is None or score > max_student_score:
            continue
        if not (r.get("teacher_answer") or "").strip():
            continue
        out.append(
            {
                "prompt": r["messages"],
                "chosen": r["teacher_answer"],
                "rejected": r["student_answer"],
                "meta": {
                    **(r.get("meta") or {}),
                    "scenario_id": r["scenario_id"],
                    "symbol": r["symbol"],
                    "task": r["task"],
                    "review_score": score,
                    "rejected_kind": r.get("student_kind", STUDENT_KIND),
                },
            }
        )
    return out

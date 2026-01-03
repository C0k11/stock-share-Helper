from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class NarrationResult:
    run_id: str
    system: str
    date: str
    ticker: str
    narrative: str
    source_file: str


def _repo_root() -> Path:
    # src/analysis/narrator.py -> <repo>/src/analysis/narrator.py
    return Path(__file__).resolve().parents[2]


def _read_json(fp: Path) -> Any:
    return json.loads(fp.read_text(encoding="utf-8"))


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _pick_systems(run_dir: Path) -> List[str]:
    if not run_dir.exists():
        return []
    out: List[str] = []
    for p in run_dir.iterdir():
        if p.is_dir() and (p / "metrics.json").exists():
            out.append(p.name)
    out.sort()
    return out


def _iter_decision_files(system_dir: Path) -> Iterable[Path]:
    return sorted(system_dir.glob("decisions_*.json"), key=lambda p: p.name)


def _get_item_for(*, payload: Dict[str, Any], date: str, ticker: str) -> Optional[Dict[str, Any]]:
    days = payload.get("days") if isinstance(payload, dict) else None
    if not isinstance(days, dict):
        return None

    day = days.get(str(date))
    if not isinstance(day, dict):
        return None

    items = day.get("items") if isinstance(day.get("items"), dict) else None
    if not isinstance(items, dict):
        return None

    t = str(ticker).upper().strip()
    v = items.get(t)
    if isinstance(v, dict):
        return v

    # best-effort fallback
    for k, vv in items.items():
        if str(k).upper().strip() == t and isinstance(vv, dict):
            return vv

    return None


def _truncate(s: str, n: int) -> str:
    s = _safe_str(s)
    if len(s) <= n:
        return s
    return s[: max(0, n - 3)] + "..."


def _extract_final_action(item: Dict[str, Any]) -> Tuple[str, Optional[float], List[str]]:
    final = item.get("final") if isinstance(item.get("final"), dict) else {}
    action = _safe_str((final or {}).get("action") or "")
    tp = (final or {}).get("target_position")
    target_position: Optional[float] = None
    try:
        if tp is not None:
            target_position = float(tp)
    except Exception:
        target_position = None

    trace = (final or {}).get("trace")
    trace_list: List[str] = []
    if isinstance(trace, list):
        trace_list = [str(x) for x in trace if str(x).strip()]
    return action or "UNKNOWN", target_position, trace_list


def _extract_router(item: Dict[str, Any]) -> Dict[str, Any]:
    router = item.get("router") if isinstance(item.get("router"), dict) else {}
    out: Dict[str, Any] = {}
    if isinstance(router, dict):
        for k in ["expert", "news_count", "news_score", "volatility_ann_pct", "planner_strategy", "planner_gate", "expert_before_planner_gate"]:
            if k in router:
                out[k] = router.get(k)
    return out


def _extract_chartist(item: Dict[str, Any]) -> Dict[str, Any]:
    chartist = item.get("chartist") if isinstance(item.get("chartist"), dict) else {}
    out: Dict[str, Any] = {}
    if isinstance(chartist, dict):
        for k in ["signal", "confidence", "score", "reasoning"]:
            if k in chartist:
                out[k] = chartist.get(k)
    return out


def _extract_system2(item: Dict[str, Any]) -> Dict[str, Any]:
    system2 = item.get("system2") if isinstance(item.get("system2"), dict) else {}
    if not isinstance(system2, dict) or not system2:
        return {}

    out: Dict[str, Any] = {}
    for role in ["proposal", "critic", "judge"]:
        v = system2.get(role)
        if isinstance(v, dict):
            # keep small + stable keys; also keep raw if it's short
            d: Dict[str, Any] = {}
            for k in ["verdict", "rationale", "decision", "analysis", "action"]:
                if k in v:
                    d[k] = v.get(k)
            raw = v.get("raw")
            if isinstance(raw, str) and raw.strip():
                d["raw"] = _truncate(raw.strip(), 600)
            if d:
                out[role] = d
        elif isinstance(v, str) and v.strip():
            out[role] = _truncate(v.strip(), 600)

    return out


def narrate_trade_context(
    *,
    run_id: str,
    date: str,
    ticker: str,
    prefer_systems: Optional[List[str]] = None,
) -> NarrationResult:
    """
    Build a factual narrative for a given (run_id, date, ticker) by reading decisions_*.json.

    Notes:
    - decisions schema: payload["days"][date]["items"][TICKER] contains per-ticker record
    - We prefer golden_strict by default, but can accept any available system.
    """

    run_id_s = str(run_id).strip()
    date_s = str(date).strip()
    ticker_s = str(ticker).upper().strip()

    root = _repo_root()
    run_dir = root / "results" / run_id_s
    if not run_dir.exists():
        return NarrationResult(
            run_id=run_id_s,
            system="",
            date=date_s,
            ticker=ticker_s,
            narrative=f"老师，我找不到 Run ID 为 {run_id_s} 的记录呢……（路径不存在：{run_dir}）",
            source_file="",
        )

    systems = _pick_systems(run_dir)
    if not systems:
        return NarrationResult(
            run_id=run_id_s,
            system="",
            date=date_s,
            ticker=ticker_s,
            narrative=f"老师，我在 {run_id_s} 里没有发现任何系统输出呢……（缺少 metrics.json 子目录）",
            source_file="",
        )

    prefs = prefer_systems if isinstance(prefer_systems, list) and prefer_systems else ["golden_strict", "baseline_fast"]
    order: List[str] = []
    for p in prefs:
        if p in systems and p not in order:
            order.append(p)
    for s in systems:
        if s not in order:
            order.append(s)

    chosen_item: Optional[Dict[str, Any]] = None
    chosen_file: Optional[Path] = None
    chosen_system: Optional[str] = None

    for sys_name in order:
        sys_dir = run_dir / sys_name
        if not sys_dir.exists():
            continue
        for fp in _iter_decision_files(sys_dir):
            try:
                payload = _read_json(fp)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            item = _get_item_for(payload=payload, date=date_s, ticker=ticker_s)
            if isinstance(item, dict):
                chosen_item = item
                chosen_file = fp
                chosen_system = sys_name
                break
        if chosen_item is not None:
            break

    if chosen_item is None or chosen_file is None or chosen_system is None:
        return NarrationResult(
            run_id=run_id_s,
            system=order[0] if order else systems[0],
            date=date_s,
            ticker=ticker_s,
            narrative=f"老师，我在 {date_s} 的档案里没有找到关于 {ticker_s} 的决策记录呢……（可能休市、或该标的未在当日 universe 内）",
            source_file="",
        )

    action, target_pos, trace = _extract_final_action(chosen_item)
    parsed = chosen_item.get("parsed") if isinstance(chosen_item.get("parsed"), dict) else {}
    parsed_decision = _safe_str((parsed or {}).get("decision") or "")
    parsed_analysis = _safe_str((parsed or {}).get("analysis") or "")

    chartist = _extract_chartist(chosen_item)
    router = _extract_router(chosen_item)
    system2 = _extract_system2(chosen_item)

    lines: List[str] = []
    lines.append(f"【交易档案】Run: {run_id_s} | 系统: {chosen_system} | 日期: {date_s} | 标的: {ticker_s}")
    lines.append(f"来源文件: {chosen_file.as_posix()}")
    lines.append("")

    lines.append(f"最终决策(final.action): {action}" + (f" (target_position={target_pos})" if target_pos is not None else ""))
    if trace:
        lines.append("最终 trace:")
        for x in trace[:8]:
            lines.append(f"- {x}")

    if parsed_decision or parsed_analysis:
        lines.append("")
        if parsed_decision:
            lines.append(f"模型提案(parsed.decision): {parsed_decision}")
        if parsed_analysis:
            lines.append(f"模型简述(parsed.analysis): {_truncate(parsed_analysis, 220)}")

    if router:
        lines.append("")
        lines.append("Router 元信息:")
        for k, v in router.items():
            lines.append(f"- {k}: {v}")

    if chartist:
        lines.append("")
        lines.append("Chartist(视觉) 信号:")
        lines.append(f"- signal: {chartist.get('signal')}")
        lines.append(f"- confidence: {chartist.get('confidence')}")
        if "score" in chartist:
            lines.append(f"- score: {chartist.get('score')}")
        if chartist.get("reasoning"):
            lines.append(f"- reasoning: {_truncate(_safe_str(chartist.get('reasoning')), 320)}")

    if system2:
        lines.append("")
        lines.append("System-2 Debate 记录:")
        for role, payload in system2.items():
            lines.append(f"- {role}: {json.dumps(payload, ensure_ascii=False)}")

    narrative = "\n".join(lines).strip() + "\n"
    return NarrationResult(
        run_id=run_id_s,
        system=str(chosen_system),
        date=date_s,
        ticker=ticker_s,
        narrative=narrative,
        source_file=str(chosen_file),
    )

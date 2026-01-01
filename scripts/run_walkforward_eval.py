#!/usr/bin/env python

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _parse_ymd(s: str) -> Tuple[int, int, int]:
    a = str(s).strip().split("-")
    if len(a) != 3:
        raise ValueError(f"invalid date: {s}")
    return int(a[0]), int(a[1]), int(a[2])


def _iter_dates_inclusive(start_ymd: str, end_ymd: str) -> List[str]:
    import datetime as _dt

    y, m, d = _parse_ymd(start_ymd)
    a = _dt.date(y, m, d)
    y, m, d = _parse_ymd(end_ymd)
    b = _dt.date(y, m, d)
    if b < a:
        a, b = b, a
    out: List[str] = []
    cur = a
    while cur <= b:
        out.append(cur.isoformat())
        cur = cur + _dt.timedelta(days=1)
    return out


def _iter_decision_records(payload: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    if not isinstance(payload, dict):
        return

    if isinstance(payload.get("days"), dict):
        days = payload.get("days")
        if isinstance(days, dict):
            for date_str, day in days.items():
                if not isinstance(day, dict):
                    continue
                items = day.get("items") if isinstance(day.get("items"), dict) else {}
                for sym, rec in items.items():
                    if not isinstance(rec, dict):
                        continue
                    yield str(date_str), str(sym).upper().strip(), rec
        return

    date_str = str(payload.get("date") or "")
    items = payload.get("items") if isinstance(payload.get("items"), dict) else {}
    for sym, rec in items.items():
        if not isinstance(rec, dict):
            continue
        yield date_str, str(sym).upper().strip(), rec


def _get_expert(rec: Dict[str, Any], default_expert: str) -> str:
    e = rec.get("expert")
    if isinstance(e, str) and e.strip():
        return e.strip()
    return str(default_expert).strip() or "unknown"


def _get_final(rec: Dict[str, Any]) -> Dict[str, Any]:
    final = rec.get("final") if isinstance(rec.get("final"), dict) else {}
    return final


def _get_position(rec: Dict[str, Any]) -> float:
    return _to_float(_get_final(rec).get("target_position"))


def _get_router(rec: Dict[str, Any]) -> Dict[str, Any]:
    r = rec.get("router")
    return r if isinstance(r, dict) else {}


def _load_stock_close_map(daily_dir: Path, dates: List[str]) -> Tuple[List[str], Dict[Tuple[str, str], float]]:
    close_map: Dict[Tuple[str, str], float] = {}
    available_dates: List[str] = []

    for d in sorted(set([str(x) for x in dates if str(x).strip()])):
        fp = daily_dir / f"stock_features_{d}.json"
        if not fp.exists():
            continue
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue

        items = payload.get("items") if isinstance(payload.get("items"), list) else []
        any_row = False
        for it in items:
            if not isinstance(it, dict):
                continue
            sym = str(it.get("symbol") or "").upper().strip()
            if not sym:
                continue
            tech = it.get("technical") if isinstance(it.get("technical"), dict) else {}
            close = tech.get("close")
            if close is None:
                continue
            close_map[(str(d), sym)] = _to_float(close)
            any_row = True
        if any_row:
            available_dates.append(str(d))

    available_dates = sorted(set(available_dates))
    return available_dates, close_map


def _forward_return_from_closes(
    date_str: str,
    sym: str,
    trading_dates: List[str],
    close_map: Dict[Tuple[str, str], float],
    horizon: int,
) -> Optional[float]:
    if horizon <= 0:
        return 0.0
    try:
        idx = trading_dates.index(str(date_str))
    except ValueError:
        return None
    j = idx + int(horizon)
    if j >= len(trading_dates):
        return None
    d0 = str(date_str)
    d1 = str(trading_dates[j])
    sym_u = str(sym).upper().strip()
    c0 = close_map.get((d0, sym_u))
    c1 = close_map.get((d1, sym_u))
    if c0 is None or c1 is None:
        return None
    if abs(float(c0)) < 1e-12:
        return None
    return float(c1) / float(c0) - 1.0


def _detect_strong_news_days(signals_dir: Path, dates: List[str], min_abs_impact: float) -> set:
    strong: set = set()
    for d in sorted(set([str(x) for x in dates if str(x).strip()])):
        fp = signals_dir / f"signals_{d}.json"
        if not fp.exists():
            continue
        try:
            arr = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(arr, list):
            continue
        ok = False
        for it in arr:
            if not isinstance(it, dict):
                continue
            sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
            ie = sig.get("impact_equity")
            if ie is None:
                continue
            if abs(_to_float(ie)) >= float(min_abs_impact):
                ok = True
                break
        if ok:
            strong.add(str(d))
    return strong


def _planner_allow_dates_from_moe(moe_path: Path) -> set:
    try:
        payload = json.loads(moe_path.read_text(encoding="utf-8"))
    except Exception:
        return set()

    days = payload.get("days") if isinstance(payload, dict) else None
    if not isinstance(days, dict):
        return set()

    out: set = set()
    for d, day in days.items():
        if not isinstance(day, dict):
            continue
        strat = ""
        planner = day.get("planner") if isinstance(day.get("planner"), dict) else {}
        s = planner.get("strategy")
        if isinstance(s, str) and s.strip():
            strat = s.strip()
        if not strat:
            items = day.get("items") if isinstance(day.get("items"), dict) else {}
            for _sym, rec in items.items():
                if isinstance(rec, dict):
                    router = rec.get("router") if isinstance(rec.get("router"), dict) else {}
                    rs = router.get("planner_strategy")
                    if isinstance(rs, str) and rs.strip():
                        strat = rs.strip()
                        break
        if strat == "aggressive_long":
            out.add(str(d))
    return out


def _gatekeeper_stats_from_decisions(*, payload: Dict[str, Any], dates: List[str]) -> Dict[str, float]:
    days = payload.get("days") if isinstance(payload, dict) else None
    if not isinstance(days, dict) or not dates:
        return {"gate_present_rate": 0.0, "gate_allow_rate": 0.0}

    present_days = 0
    allow_days = 0
    for d in dates:
        day = days.get(str(d))
        if not isinstance(day, dict):
            continue
        gk = day.get("gatekeeper") if isinstance(day.get("gatekeeper"), dict) else None
        if not isinstance(gk, dict):
            continue
        present_days += 1
        if bool(gk.get("allow")):
            allow_days += 1

    total_days = len(set([str(d) for d in dates]))
    present_rate = float(present_days) / float(total_days) if total_days else 0.0
    allow_rate = float(allow_days) / float(present_days) if present_days else 1.0
    return {"gate_present_rate": float(present_rate), "gate_allow_rate": float(allow_rate)}


def _run_inference(
    *,
    cfg: Dict[str, Any],
    out_path: Path,
    start: str,
    end: str,
    universe: str,
    daily_dir: Path,
    base_model: str,
    load_4bit: bool,
    macro_file: str = "",
    chart_signals_file: str = "",
    chart_confidence: Optional[float] = None,
    chart_mode: str = "",
) -> None:
    args: List[str] = [
        sys.executable,
        "scripts/run_trading_inference.py",
        "--date-range",
        str(start),
        str(end),
        "--progress",
        "--daily-dir",
        str(daily_dir),
        "--universe",
        str(universe),
        "--model",
        str(base_model),
        "--out",
        str(out_path),
    ]

    if load_4bit:
        args.append("--load-in-4bit")
    else:
        args.append("--no-load-4bit")

    if bool(cfg.get("moe_mode")):
        args.append("--moe-mode")
        if str(cfg.get("moe_scalper") or "").strip():
            args.extend(["--moe-scalper", str(cfg.get("moe_scalper"))])
        if str(cfg.get("moe_analyst") or "").strip():
            args.extend(["--moe-analyst", str(cfg.get("moe_analyst"))])
        if str(cfg.get("planner_mode") or "").strip():
            args.extend(["--planner-mode", str(cfg.get("planner_mode"))])
        if str(cfg.get("planner_sft_model") or "").strip():
            args.extend(["--planner-sft-model", str(cfg.get("planner_sft_model"))])
        if str(cfg.get("planner_rl_model") or "").strip():
            args.extend(["--planner-rl-model", str(cfg.get("planner_rl_model"))])
        if cfg.get("planner_rl_threshold") is not None:
            args.extend(["--planner-rl-threshold", str(cfg.get("planner_rl_threshold"))])
        if cfg.get("moe_any_news") is False:
            args.append("--no-moe-any-news")
        if cfg.get("moe_any_news") is True:
            args.append("--moe-any-news")
        if cfg.get("moe_news_threshold") is not None:
            args.extend(["--moe-news-threshold", str(cfg.get("moe_news_threshold"))])
        if cfg.get("moe_vol_threshold") is not None:
            args.extend(["--moe-vol-threshold", str(cfg.get("moe_vol_threshold"))])
    else:
        if bool(cfg.get("use_lora")):
            args.append("--use-lora")
        if str(cfg.get("adapter") or "").strip():
            args.extend(["--adapter", str(cfg.get("adapter"))])
        if str(cfg.get("planner_mode") or "").strip():
            args.extend(["--planner-mode", str(cfg.get("planner_mode"))])
        if str(cfg.get("planner_sft_model") or "").strip():
            args.extend(["--planner-sft-model", str(cfg.get("planner_sft_model"))])
        if str(cfg.get("planner_rl_model") or "").strip():
            args.extend(["--planner-rl-model", str(cfg.get("planner_rl_model"))])
        if cfg.get("planner_rl_threshold") is not None:
            args.extend(["--planner-rl-threshold", str(cfg.get("planner_rl_threshold"))])

    if bool(cfg.get("allow_clear")):
        args.append("--allow-clear")

    if cfg.get("risk_max_drawdown") is not None:
        args.extend(["--risk-max-drawdown", str(cfg.get("risk_max_drawdown"))])
    if cfg.get("risk_vol_limit") is not None:
        args.extend(["--risk-vol-limit", str(cfg.get("risk_vol_limit"))])
    if str(cfg.get("risk_watch_market") or "").strip():
        args.extend(["--risk-watch-market", str(cfg.get("risk_watch_market"))])

    if cfg.get("max_new_tokens") is not None:
        args.extend(["--max-new-tokens", str(cfg.get("max_new_tokens"))])
    if cfg.get("temperature") is not None:
        args.extend(["--temperature", str(cfg.get("temperature"))])

    if str(macro_file or "").strip():
        args.extend(["--macro-file", str(macro_file).strip()])

    if str(chart_signals_file or "").strip():
        args.extend(["--chart-signals-file", str(chart_signals_file).strip()])
        if chart_confidence is not None:
            args.extend(["--chart-confidence", str(float(chart_confidence))])
        if str(chart_mode or "").strip():
            args.extend(["--chart-mode", str(chart_mode).strip()])

    subprocess.run(args, check=True)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_system_metrics(
    *,
    system_key: str,
    decision_path: Path,
    default_expert: str,
    daily_dir: Path,
    start: str,
    end: str,
    horizons: List[int],
    trade_cost_bps: float,
    min_news_abs_impact: float,
    high_vol_ann_pct_threshold: float,
    planner_allow_dates: Optional[set],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    payload = json.loads(decision_path.read_text(encoding="utf-8"))

    dates = _iter_dates_inclusive(start, end)
    trading_dates, close_map = _load_stock_close_map(daily_dir, dates)

    gk_stats = _gatekeeper_stats_from_decisions(payload=payload, dates=dates)

    strong_news_days = _detect_strong_news_days(daily_dir, dates, float(min_news_abs_impact))

    cost_rate = float(trade_cost_bps) / 10000.0 if float(trade_cost_bps) > 0 else 0.0

    # Build per-day, per-ticker records first, then compute turnover in chronological order.
    per_day_rows: Dict[str, List[Dict[str, Any]]] = {d: [] for d in dates}
    pos_map: Dict[Tuple[str, str], float] = {}
    fee_map: Dict[Tuple[str, str], float] = {}

    total_items = 0
    analyst_items = 0

    for d, sym, rec in _iter_decision_records(payload):
        if d not in per_day_rows:
            continue
        total_items += 1
        expert = _get_expert(rec, default_expert)
        if expert == "analyst":
            analyst_items += 1

        router = _get_router(rec)
        news_count = router.get("news_count")
        news_count_i = int(news_count) if isinstance(news_count, int) else int(_to_float(news_count))
        news_score = _to_float(router.get("news_score"))

        vol_ann = router.get("volatility_ann_pct")
        vol_ann_f = _to_float(vol_ann)

        planner_strategy = router.get("planner_strategy")
        planner_strategy_s = str(planner_strategy).strip() if isinstance(planner_strategy, str) else ""

        pos = _get_position(rec)
        sym_u = str(sym).upper().strip()
        pos_map[(str(d), sym_u)] = float(pos)

        planner_allow = False
        if planner_allow_dates is not None:
            planner_allow = str(d) in set(planner_allow_dates)

        per_day_rows[str(d)].append(
            {
                "date": str(d),
                "ticker": sym_u,
                "expert": str(expert),
                "target_position": float(pos),
                "turnover": 0.0,
                "fee": 0.0,
                "news_count": int(news_count_i),
                "news_score": float(news_score),
                "volatility_ann_pct": float(vol_ann_f),
                "has_strong_news_day": bool(str(d) in strong_news_days),
                "planner_allow": bool(planner_allow),
                "planner_strategy": str(planner_strategy_s),
            }
        )

    # Compute turnover + fee in deterministic date order (independent from cost)
    if dates:
        tickers: List[str] = sorted(set([sym for (_d, sym) in pos_map.keys()]))
        last_pos: Dict[str, float] = {t: 0.0 for t in tickers}
        turnover_map: Dict[Tuple[str, str], float] = {}
        for d in dates:
            for t in tickers:
                cur = float(pos_map.get((str(d), str(t)), 0.0))
                prev = float(last_pos.get(str(t), 0.0))
                turnover = abs(cur - prev)
                turnover_map[(str(d), str(t))] = float(turnover)
                fee_map[(str(d), str(t))] = float(turnover) * float(cost_rate)
                last_pos[str(t)] = float(cur)

        for d in dates:
            for r in per_day_rows.get(str(d), []):
                sym_u = str(r.get("ticker") or "").upper().strip()
                cur = float(pos_map.get((str(d), sym_u), float(r.get("target_position") or 0.0)))
                turnover = float(turnover_map.get((str(d), sym_u), 0.0))
                fee = float(fee_map.get((str(d), sym_u), 0.0))
                r["target_position"] = float(cur)
                r["turnover"] = float(turnover)
                r["fee"] = float(fee)

    # Flatten decision-level rows and compute forward returns / pnl
    decision_rows: List[Dict[str, Any]] = []
    for d in dates:
        for r in per_day_rows.get(str(d), []):
            sym = str(r.get("ticker") or "")
            pos = float(r.get("target_position") or 0.0)
            fee = float(r.get("fee") or 0.0)

            fr_by_h: Dict[str, Optional[float]] = {}
            pnl_by_h: Dict[str, float] = {}
            for h in horizons:
                fr = _forward_return_from_closes(str(d), sym, trading_dates, close_map, int(h))
                fr_by_h[str(h)] = fr
                if fr is None:
                    pnl_by_h[str(h)] = 0.0
                else:
                    pnl_by_h[str(h)] = float(pos) * float(fr)

            # Fees are charged on turnover; apply to h=1 net only.
            pnl_h1_net = float(pnl_by_h.get("1", 0.0)) - float(fee)

            out = dict(r)
            out.update(
                {
                    "system": system_key,
                    "fr_h1": fr_by_h.get("1"),
                    "fr_h5": fr_by_h.get("5"),
                    "fr_h10": fr_by_h.get("10"),
                    "pnl_h1": float(pnl_by_h.get("1", 0.0)),
                    "pnl_h5": float(pnl_by_h.get("5", 0.0)),
                    "pnl_h10": float(pnl_by_h.get("10", 0.0)),
                    "pnl_h1_net": float(pnl_h1_net),
                }
            )
            decision_rows.append(out)

    df_decisions = pd.DataFrame(decision_rows)

    # Aggregate metrics from decision-level rows
    pnl_sum_by_h: Dict[str, float] = {str(h): 0.0 for h in horizons}
    pnl_sum_net_by_h: Dict[str, float] = {str(h): 0.0 for h in horizons}
    fees_total: float = 0.0
    trade_count: int = 0

    nav: float = 1.0
    peak: float = 1.0
    max_drawdown: float = 0.0

    by_date: Dict[str, List[Dict[str, Any]]] = {d: [] for d in dates}
    for r in decision_rows:
        by_date.setdefault(str(r.get("date") or ""), []).append(r)

    for d in dates:
        rows = by_date.get(str(d), [])
        day_fee = sum(float(r.get("fee") or 0.0) for r in rows)
        fees_total += float(day_fee)

        # trade_count counts turnover events (rebalance actions)
        day_trades = sum(1 for r in rows if float(r.get("turnover") or 0.0) > 1e-12)
        trade_count += int(day_trades)

        day_ret_h1 = 0.0
        for h in horizons:
            key = f"pnl_h{int(h)}"
            day_pnl = sum(float(r.get(key) or 0.0) for r in rows)
            pnl_sum_by_h[str(h)] += float(day_pnl)
            pnl_sum_net_by_h[str(h)] += float(day_pnl)
            if int(h) == 1:
                day_ret_h1 = float(day_pnl)

        # Fees applied to h=1 net only
        pnl_sum_net_by_h["1"] = float(pnl_sum_net_by_h.get("1", 0.0)) - float(day_fee)

        nav = nav * (1.0 + (float(day_ret_h1) - float(day_fee)))
        peak = max(peak, nav)
        dd = (nav / peak) - 1.0
        max_drawdown = min(max_drawdown, dd)

    analyst_coverage = float(analyst_items) / float(total_items) if total_items else 0.0

    allow_rate = 0.0
    if planner_allow_dates is not None:
        allow_rate = float(len(set(planner_allow_dates).intersection(set(dates)))) / float(len(set(dates))) if dates else 0.0

    metrics: Dict[str, Any] = {
        "system": system_key,
        "range": {"start": start, "end": end},
        "horizons": horizons,
        "pnl_sum": {str(k): float(v) for k, v in pnl_sum_by_h.items()},
        "pnl_sum_net": {str(k): float(v) for k, v in pnl_sum_net_by_h.items()},
        "trade_count": int(trade_count),
        "fees_total": float(fees_total),
        "nav": {"start": 1.0, "end": float(nav)},
        "max_drawdown": float(max_drawdown),
        "analyst_coverage": float(analyst_coverage),
        "planner_allow_rate": allow_rate,
        "gate_present_rate": float(gk_stats.get("gate_present_rate", 0.0)),
        "gate_allow_rate": float(gk_stats.get("gate_allow_rate", 0.0)),
        "inputs": {"decision_json": str(decision_path), "sha256": _hash_file(decision_path)},
    }

    return metrics, df_decisions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-config", default="configs/baseline_fast_v1.yaml")
    p.add_argument("--golden-config", default="configs/golden_strict_v1.yaml")
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--chart-signals-file",
        default="",
        help="Optional Chartist signals jsonl injected into GOLDEN run only (baseline remains control)",
    )
    p.add_argument(
        "--chart-confidence",
        type=float,
        default=0.7,
        help="Chartist confidence threshold for overlay (default: 0.7)",
    )
    p.add_argument(
        "--chart-mode",
        default="standard",
        choices=["standard", "conservative"],
        help="Chartist overlay mode: standard=upgrade+block, conservative=block only",
    )
    p.add_argument(
        "--macro-file",
        default="",
        help="Optional macro features CSV (Date,Global_Risk_Score). Passed through to baseline and golden inference.",
    )
    p.add_argument("--windows", nargs="*", default=[])
    p.add_argument("--override-moe-analyst", default="", help="Override golden inference.moe_analyst adapter path")
    p.add_argument("--override-moe-scalper", default="", help="Override golden inference.moe_scalper adapter path")
    p.add_argument(
        "--override-moe-any-news",
        default="",
        choices=["", "true", "false"],
        help="Override golden inference.moe_any_news (true/false)",
    )
    p.add_argument(
        "--override-moe-news-threshold",
        type=float,
        default=None,
        help="Override golden inference.moe_news_threshold",
    )
    p.add_argument(
        "--override-moe-vol-threshold",
        type=float,
        default=None,
        help="Override golden inference.moe_vol_threshold",
    )
    p.add_argument(
        "--override-planner-mode",
        "--planner-mode",
        dest="override_planner_mode",
        default="",
        choices=["", "off", "rule", "sft"],
        help="Override golden inference.planner_mode",
    )
    p.add_argument("--override-planner-sft-model", default="", help="Override golden inference.planner_sft_model")
    p.add_argument("--override-planner-rl-model", default="", help="Override golden inference.planner_rl_model")
    p.add_argument(
        "--override-planner-rl-threshold",
        type=float,
        default=None,
        help="Override golden inference.planner_rl_threshold",
    )
    p.add_argument(
        "--override-risk-max-drawdown",
        type=float,
        default=None,
        help="Override golden inference.risk_max_drawdown",
    )
    p.add_argument("--force-rerun", action="store_true", default=False)
    args = p.parse_args()

    base_cfg = yaml.safe_load(Path(str(args.baseline_config)).read_text(encoding="utf-8"))
    gold_cfg = yaml.safe_load(Path(str(args.golden_config)).read_text(encoding="utf-8"))

    gold_infer = gold_cfg.get("inference") if isinstance(gold_cfg, dict) else None
    if not isinstance(gold_infer, dict):
        gold_infer = {}
        if isinstance(gold_cfg, dict):
            gold_cfg["inference"] = gold_infer

    if str(args.override_moe_analyst or "").strip():
        print(f"Overriding golden inference.moe_analyst: {str(args.override_moe_analyst).strip()}")
        gold_infer["moe_analyst"] = str(args.override_moe_analyst).strip()

    if str(args.override_moe_scalper or "").strip():
        print(f"Overriding golden inference.moe_scalper: {str(args.override_moe_scalper).strip()}")
        gold_infer["moe_scalper"] = str(args.override_moe_scalper).strip()

    if str(args.override_planner_mode or "").strip():
        print(f"Overriding golden inference.planner_mode: {str(args.override_planner_mode).strip()}")
        gold_infer["planner_mode"] = str(args.override_planner_mode).strip()

    if str(args.override_planner_sft_model or "").strip():
        print(f"Overriding golden inference.planner_sft_model: {str(args.override_planner_sft_model).strip()}")
        gold_infer["planner_sft_model"] = str(args.override_planner_sft_model).strip()

    if str(args.override_planner_rl_model or "").strip():
        print(f"Overriding golden inference.planner_rl_model: {str(args.override_planner_rl_model).strip()}")
        gold_infer["planner_rl_model"] = str(args.override_planner_rl_model).strip()

    if args.override_planner_rl_threshold is not None:
        print(f"Overriding golden inference.planner_rl_threshold: {float(args.override_planner_rl_threshold)}")
        gold_infer["planner_rl_threshold"] = float(args.override_planner_rl_threshold)

    if args.override_risk_max_drawdown is not None:
        print(f"Overriding golden inference.risk_max_drawdown: {float(args.override_risk_max_drawdown)}")
        gold_infer["risk_max_drawdown"] = float(args.override_risk_max_drawdown)

    if str(args.override_moe_any_news or "").strip():
        val = str(args.override_moe_any_news).strip().lower()
        b = True if val == "true" else False
        print(f"Overriding golden inference.moe_any_news: {b}")
        gold_infer["moe_any_news"] = bool(b)
        gold_infer["moe_mode"] = True

    if args.override_moe_news_threshold is not None:
        print(f"Overriding golden inference.moe_news_threshold: {float(args.override_moe_news_threshold)}")
        gold_infer["moe_news_threshold"] = float(args.override_moe_news_threshold)
        gold_infer["moe_mode"] = True

    if args.override_moe_vol_threshold is not None:
        print(f"Overriding golden inference.moe_vol_threshold: {float(args.override_moe_vol_threshold)}")
        gold_infer["moe_vol_threshold"] = float(args.override_moe_vol_threshold)
        gold_infer["moe_mode"] = True

    start = str((gold_cfg.get("date_range") or {}).get("start") or "").strip()
    end = str((gold_cfg.get("date_range") or {}).get("end") or "").strip()
    if not start or not end:
        raise SystemExit("missing date_range in golden config")

    universe = str(gold_cfg.get("universe") or "stock")
    daily_dir = Path(str(gold_cfg.get("daily_dir") or "data/daily"))
    base_model = str(gold_cfg.get("base_model") or "Qwen/Qwen2.5-7B-Instruct")
    load_4bit = bool(gold_cfg.get("load_4bit") is True)

    protocol = gold_cfg.get("protocol") if isinstance(gold_cfg.get("protocol"), dict) else {}
    horizons = [int(x) for x in (protocol.get("horizons") or [1, 5, 10])]
    trade_cost_bps = float(protocol.get("trade_cost_bps") or 0.0)
    min_news_abs_impact = float(protocol.get("min_news_abs_impact") or 0.5)
    high_news_score_threshold = float(protocol.get("high_news_score_threshold") or 0.8)
    high_news_count_threshold = int(protocol.get("high_news_count_threshold") or 1)
    high_vol_thr = float(protocol.get("high_vol_ann_pct_threshold") or 25.0)

    windows: List[Tuple[str, str]] = []
    if args.windows:
        if len(args.windows) % 2 != 0:
            raise SystemExit("--windows expects pairs: START END [START END ...]")
        for i in range(0, len(args.windows), 2):
            windows.append((str(args.windows[i]), str(args.windows[i + 1])))
    else:
        windows.append((start, end))

    run_dir = Path("results") / str(args.run_id)
    (run_dir / "baseline_fast").mkdir(parents=True, exist_ok=True)
    (run_dir / "golden_strict").mkdir(parents=True, exist_ok=True)

    Path(run_dir / "baseline_fast" / "config.yaml").write_text(yaml.safe_dump(base_cfg, sort_keys=True), encoding="utf-8")
    Path(run_dir / "golden_strict" / "config.yaml").write_text(yaml.safe_dump(gold_cfg, sort_keys=True), encoding="utf-8")

    combined: Dict[str, Any] = {
        "run_id": str(args.run_id),
        "protocol": {
            "horizons": horizons,
            "trade_cost_bps": trade_cost_bps,
            "min_news_abs_impact": min_news_abs_impact,
            "high_news_score_threshold": high_news_score_threshold,
            "high_news_count_threshold": high_news_count_threshold,
            "high_vol_ann_pct_threshold": high_vol_thr,
        },
        "outputs": {
            "baseline_fast": {
                "daily_csv": str((run_dir / "baseline_fast" / "daily.csv").as_posix()),
                "metrics_json": str((run_dir / "baseline_fast" / "metrics.json").as_posix()),
            },
            "golden_strict": {
                "daily_csv": str((run_dir / "golden_strict" / "daily.csv").as_posix()),
                "metrics_json": str((run_dir / "golden_strict" / "metrics.json").as_posix()),
            },
        },
        "windows": [],
    }

    baseline_daily_all: List[pd.DataFrame] = []
    golden_daily_all: List[pd.DataFrame] = []

    for w_start, w_end in windows:
        base_out = run_dir / "baseline_fast" / f"decisions_{w_start}_{w_end}.json".replace(":", "")
        gold_out = run_dir / "golden_strict" / f"decisions_{w_start}_{w_end}.json".replace(":", "")

        if bool(args.force_rerun):
            if base_out.exists():
                base_out.unlink()
            if gold_out.exists():
                gold_out.unlink()

        if not base_out.exists():
            _run_inference(
                cfg=(base_cfg.get("inference") or {}),
                out_path=base_out,
                start=str(w_start),
                end=str(w_end),
                universe=str(universe),
                daily_dir=daily_dir,
                base_model=str(base_model),
                load_4bit=bool(load_4bit),
                macro_file=str(getattr(args, "macro_file", "") or "").strip(),
            )

        if not gold_out.exists():
            _run_inference(
                cfg=(gold_cfg.get("inference") or {}),
                out_path=gold_out,
                start=str(w_start),
                end=str(w_end),
                universe=str(universe),
                daily_dir=daily_dir,
                base_model=str(base_model),
                load_4bit=bool(load_4bit),
                macro_file=str(getattr(args, "macro_file", "") or "").strip(),
                chart_signals_file=str(getattr(args, "chart_signals_file", "") or "").strip(),
                chart_confidence=float(getattr(args, "chart_confidence", 0.7) or 0.7),
                chart_mode=str(getattr(args, "chart_mode", "standard") or "standard").strip(),
            )

        planner_allow_dates = _planner_allow_dates_from_moe(gold_out)

        base_metrics, base_daily = _compute_system_metrics(
            system_key="baseline_fast",
            decision_path=base_out,
            default_expert="scalper",
            daily_dir=daily_dir,
            start=str(w_start),
            end=str(w_end),
            horizons=horizons,
            trade_cost_bps=trade_cost_bps,
            min_news_abs_impact=min_news_abs_impact,
            high_vol_ann_pct_threshold=high_vol_thr,
            planner_allow_dates=None,
        )

        gold_metrics, gold_daily = _compute_system_metrics(
            system_key="golden_strict",
            decision_path=gold_out,
            default_expert="unknown",
            daily_dir=daily_dir,
            start=str(w_start),
            end=str(w_end),
            horizons=horizons,
            trade_cost_bps=trade_cost_bps,
            min_news_abs_impact=min_news_abs_impact,
            high_vol_ann_pct_threshold=high_vol_thr,
            planner_allow_dates=planner_allow_dates,
        )

        base_daily_path = run_dir / "baseline_fast" / f"daily_{w_start}_{w_end}.csv"
        gold_daily_path = run_dir / "golden_strict" / f"daily_{w_start}_{w_end}.csv"
        base_daily_path.write_text(base_daily.to_csv(index=False), encoding="utf-8")
        gold_daily_path.write_text(gold_daily.to_csv(index=False), encoding="utf-8")

        baseline_daily_all.append(base_daily)
        golden_daily_all.append(gold_daily)

        (run_dir / "baseline_fast" / f"metrics_{w_start}_{w_end}.json").write_text(
            json.dumps(base_metrics, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (run_dir / "golden_strict" / f"metrics_{w_start}_{w_end}.json").write_text(
            json.dumps(gold_metrics, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        combined["windows"].append(
            {
                "start": str(w_start),
                "end": str(w_end),
                "baseline_fast": base_metrics,
                "golden_strict": gold_metrics,
            }
        )

    # Write combined daily.csv for each system
    if baseline_daily_all:
        df_base = pd.concat(baseline_daily_all, ignore_index=True)
        (run_dir / "baseline_fast" / "daily.csv").write_text(df_base.to_csv(index=False), encoding="utf-8")
    if golden_daily_all:
        df_gold = pd.concat(golden_daily_all, ignore_index=True)
        (run_dir / "golden_strict" / "daily.csv").write_text(df_gold.to_csv(index=False), encoding="utf-8")

    # Write combined per-system metrics.json (aggregate across windows)
    # Since daily.csv is combined, recompute summary from daily.csv to avoid window weighting issues.
    def _aggregate_metrics_from_daily(system_key: str, daily_csv: Path, system_dir: Path) -> Dict[str, Any]:
        if not daily_csv.exists():
            return {}
        df = pd.read_csv(daily_csv)
        range_start = str(start)
        range_end = str(end)
        if "date" in df.columns and len(df):
            try:
                dt = pd.to_datetime(df["date"], errors="coerce")
                if dt.notna().any():
                    range_start = str(dt.min().date())
                    range_end = str(dt.max().date())
            except Exception:
                pass
        for c in ["turnover", "fee", "pnl_h1", "pnl_h5", "pnl_h10"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        if "expert" in df.columns:
            df["expert"] = df["expert"].astype(str)
        if "planner_allow" in df.columns:
            df["planner_allow"] = df["planner_allow"].astype(str).str.lower().isin(["true", "1", "yes"])

        horizons_local = horizons
        pnl_sum = {str(h): float(df.get(f"pnl_h{int(h)}", 0.0).sum()) for h in horizons_local}
        fees_total_local = float(df.get("fee", 0.0).sum())
        pnl_sum_net = dict(pnl_sum)
        if "1" in pnl_sum_net:
            pnl_sum_net["1"] = float(pnl_sum_net["1"]) - float(fees_total_local)

        trade_count_local = int((df.get("turnover", 0.0) > 1e-12).sum())
        analyst_cov_local = float((df.get("expert", "").astype(str) == "analyst").sum()) / float(len(df)) if len(df) else 0.0
        planner_allow_rate_local = float(df.get("planner_allow", False).astype(bool).groupby(df.get("date")).max().mean()) if ("planner_allow" in df.columns and "date" in df.columns and len(df)) else 0.0

        gate_present_days = 0
        gate_allow_days = 0
        total_days = 0
        if "date" in df.columns and len(df):
            uniq_dates = sorted(set([str(x) for x in df["date"].astype(str).tolist() if str(x).strip()]))
            total_days = len(uniq_dates)
            day_status: Dict[str, Dict[str, bool]] = {d: {"present": False, "allow": False} for d in uniq_dates}
            for fp in sorted(system_dir.glob("decisions_*.json")):
                try:
                    payload = json.loads(fp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                days = payload.get("days") if isinstance(payload, dict) else None
                if not isinstance(days, dict):
                    continue
                for d in uniq_dates:
                    if d not in days:
                        continue
                    day = days.get(d)
                    if not isinstance(day, dict):
                        continue
                    gk = day.get("gatekeeper") if isinstance(day.get("gatekeeper"), dict) else None
                    if not isinstance(gk, dict):
                        continue
                    day_status[d]["present"] = True
                    if bool(gk.get("allow")):
                        day_status[d]["allow"] = True

            gate_present_days = sum(1 for d in uniq_dates if bool(day_status.get(d, {}).get("present")))
            gate_allow_days = sum(1 for d in uniq_dates if bool(day_status.get(d, {}).get("present")) and bool(day_status.get(d, {}).get("allow")))

        gate_present_rate_local = float(gate_present_days) / float(total_days) if total_days else 0.0
        gate_allow_rate_local = float(gate_allow_days) / float(gate_present_days) if gate_present_days else 1.0

        return {
            "system": system_key,
            "range": {"start": str(range_start), "end": str(range_end)},
            "horizons": horizons_local,
            "pnl_sum": pnl_sum,
            "pnl_sum_net": pnl_sum_net,
            "trade_count": int(trade_count_local),
            "fees_total": float(fees_total_local),
            "analyst_coverage": float(analyst_cov_local),
            "planner_allow_rate": float(planner_allow_rate_local),
            "gate_present_rate": float(gate_present_rate_local),
            "gate_allow_rate": float(gate_allow_rate_local),
        }

    base_combined_metrics = _aggregate_metrics_from_daily(
        "baseline_fast", run_dir / "baseline_fast" / "daily.csv", run_dir / "baseline_fast"
    )
    gold_combined_metrics = _aggregate_metrics_from_daily(
        "golden_strict", run_dir / "golden_strict" / "daily.csv", run_dir / "golden_strict"
    )

    (run_dir / "baseline_fast" / "metrics.json").write_text(
        json.dumps(base_combined_metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "golden_strict" / "metrics.json").write_text(
        json.dumps(gold_combined_metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    (run_dir / "metrics.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "run_id.txt").write_text(str(args.run_id) + "\n", encoding="utf-8")

    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()

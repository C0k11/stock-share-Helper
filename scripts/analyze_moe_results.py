#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


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
                    yield str(date_str), str(sym), rec
        return

    date_str = str(payload.get("date") or "")
    items = payload.get("items") if isinstance(payload.get("items"), dict) else {}
    for sym, rec in items.items():
        if not isinstance(rec, dict):
            continue
        yield date_str, str(sym), rec


def _load_decision_map(payload: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for d, sym, rec in _iter_decision_records(payload):
        if not d:
            continue
        out[(str(d), str(sym).upper().strip())] = rec
    return out


def _load_forward_return_map(report_path: str, strategy: str) -> Dict[Tuple[str, str], float]:
    fp = Path(report_path)
    data = json.loads(fp.read_text(encoding="utf-8"))

    strategies = data.get("strategies") if isinstance(data.get("strategies"), dict) else {}
    strat = strategies.get(strategy) if isinstance(strategies.get(strategy), dict) else None
    if strat is None:
        raise SystemExit(f"strategy not found in report: {strategy}")

    trades = strat.get("trades") if isinstance(strat.get("trades"), list) else []
    out: Dict[Tuple[str, str], float] = {}
    for t in trades:
        if not isinstance(t, dict):
            continue
        date = str(t.get("date") or "")
        sym = str(t.get("ticker") or t.get("symbol") or "").upper().strip()
        if not date or not sym:
            continue
        fr = t.get("forward_return")
        if fr is None:
            fr = t.get("realized_return")
        out[(date, sym)] = _to_float(fr)
    return out


def _load_close_map(daily_dir: str, dates: List[str]) -> Tuple[List[str], Dict[Tuple[str, str], float]]:
    root = Path(daily_dir)
    close_map: Dict[Tuple[str, str], float] = {}
    available_dates: List[str] = []

    for d in sorted(set([str(x) for x in dates if str(x).strip()])):
        fp = root / f"stock_features_{d}.json"
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
            technical = it.get("technical") if isinstance(it.get("technical"), dict) else {}
            close = technical.get("close")
            if close is None:
                continue
            close_map[(str(d), sym)] = _to_float(close)
            any_row = True
        if any_row:
            available_dates.append(str(d))

    available_dates = sorted(set(available_dates))
    return available_dates, close_map


def _compute_forward_return_from_closes(
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


def _get_expert(rec: Dict[str, Any]) -> str:
    e = rec.get("expert")
    if isinstance(e, str) and e.strip():
        return e.strip()
    return "unknown"


def _get_position(rec: Dict[str, Any]) -> float:
    final = rec.get("final") if isinstance(rec.get("final"), dict) else {}
    return _to_float(final.get("target_position"))


def _get_action(rec: Dict[str, Any]) -> str:
    final = rec.get("final") if isinstance(rec.get("final"), dict) else {}
    a = final.get("action")
    if isinstance(a, str) and a.strip():
        return a.strip().upper()
    return ""


def _bucket_action(action: str) -> str:
    a = str(action or "").upper().strip()
    if not a:
        return "OTHER"
    if a in {"BUY", "LONG"}:
        return "BUY"
    if a in {"SELL", "SHORT", "REDUCE"}:
        return "SELL"
    if a in {"CLEAR", "EXIT"}:
        return "CLEAR"
    if a in {"HOLD", "KEEP"}:
        return "HOLD"
    return a


def _get_parsed_decision(rec: Dict[str, Any]) -> str:
    parsed = rec.get("parsed") if isinstance(rec.get("parsed"), dict) else {}
    d = parsed.get("decision")
    if isinstance(d, str) and d.strip():
        return d.strip().upper()

    label = parsed.get("label") if isinstance(parsed.get("label"), dict) else {}
    a = label.get("action")
    if isinstance(a, str) and a.strip():
        return a.strip().upper()
    return ""


def _get_risk_reason(rec: Dict[str, Any]) -> str:
    final = rec.get("final") if isinstance(rec.get("final"), dict) else {}
    trace = final.get("trace") if isinstance(final.get("trace"), list) else []
    parts: List[str] = []
    for t in trace:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s:
            continue
        if "Proposal Approved" in s:
            continue
        parts.append(s)
    return " | ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--moe", required=True, help="Path to moe_race_*.json")
    p.add_argument("--baseline", default="", help="Optional baseline decision json for A/B comparison")
    p.add_argument("--daily-dir", default="data/daily")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--report", default="", help="Optional backtest report JSON path with forward_return")
    p.add_argument("--strategy", default="v1_1_news")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--diagnose-analyst", action="store_true", default=False)
    p.add_argument("--diagnose-out", default="", help="Optional path to write analyst diagnostics JSON")
    p.add_argument("--diagnose-limit", type=int, default=200)
    args = p.parse_args()

    moe_path = Path(args.moe)
    payload = json.loads(moe_path.read_text(encoding="utf-8"))

    baseline_payload: Optional[Dict[str, Any]] = None
    baseline_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if str(args.baseline).strip():
        base_path = Path(str(args.baseline))
        baseline_payload = json.loads(base_path.read_text(encoding="utf-8"))
        baseline_map = _load_decision_map(baseline_payload)

    fr_map: Dict[Tuple[str, str], float] = {}
    if str(args.report).strip():
        fr_map = _load_forward_return_map(str(args.report), str(args.strategy))

    all_dates: List[str] = []
    for d, _, _ in _iter_decision_records(payload):
        if d:
            all_dates.append(str(d))
    if baseline_payload is not None:
        for d, _, _ in _iter_decision_records(baseline_payload):
            if d:
                all_dates.append(str(d))
    trading_dates, close_map = _load_close_map(str(args.daily_dir), all_dates)

    moe_map = _load_decision_map(payload)

    experts = ["analyst", "scalper", "unknown"]
    stats: Dict[str, Dict[str, float]] = {
        e: {
            "items": 0.0,
            "items_scored": 0.0,
            "trades": 0.0,
            "trades_scored": 0.0,
            "wins": 0.0,
            "pnl_sum": 0.0,
            "pnl_abs_sum": 0.0,
            "missing_forward": 0.0,
        }
        for e in experts
    }

    total_items = 0
    total_analyst = 0

    action_stats: Dict[str, Dict[str, float]] = {}
    action_stats_by_expert: Dict[str, Dict[str, Dict[str, float]]] = {}

    for (date_str, sym_u), rec in moe_map.items():
        e = _get_expert(rec)
        if e not in stats:
            stats[e] = {
                "items": 0.0,
                "items_scored": 0.0,
                "trades": 0.0,
                "trades_scored": 0.0,
                "wins": 0.0,
                "pnl_sum": 0.0,
                "pnl_abs_sum": 0.0,
                "missing_forward": 0.0,
            }

        total_items += 1
        if e == "analyst":
            total_analyst += 1

        st = stats[e]
        st["items"] += 1.0

        pos = _get_position(rec)
        if abs(pos) > 1e-9:
            st["trades"] += 1.0

        fr = fr_map.get((str(date_str), sym_u))
        if fr is None:
            fr = _compute_forward_return_from_closes(
                date_str=str(date_str),
                sym=sym_u,
                trading_dates=trading_dates,
                close_map=close_map,
                horizon=int(args.horizon),
            )
        if fr is None:
            st["missing_forward"] += 1.0
            continue

        st["items_scored"] += 1.0
        if abs(pos) > 1e-9:
            st["trades_scored"] += 1.0

        pnl = float(pos) * float(fr)
        st["pnl_sum"] += pnl
        st["pnl_abs_sum"] += abs(pnl)
        if abs(pos) > 1e-9 and pnl > 0:
            st["wins"] += 1.0

        if abs(pos) > 1e-9:
            a_bucket = _bucket_action(_get_action(rec))

            if a_bucket not in action_stats:
                action_stats[a_bucket] = {
                    "trades_scored": 0.0,
                    "wins": 0.0,
                    "pnl_sum": 0.0,
                }
            action_stats[a_bucket]["trades_scored"] += 1.0
            action_stats[a_bucket]["pnl_sum"] += pnl
            if pnl > 0:
                action_stats[a_bucket]["wins"] += 1.0

            if e not in action_stats_by_expert:
                action_stats_by_expert[e] = {}
            if a_bucket not in action_stats_by_expert[e]:
                action_stats_by_expert[e][a_bucket] = {
                    "trades_scored": 0.0,
                    "wins": 0.0,
                    "pnl_sum": 0.0,
                }
            action_stats_by_expert[e][a_bucket]["trades_scored"] += 1.0
            action_stats_by_expert[e][a_bucket]["pnl_sum"] += pnl
            if pnl > 0:
                action_stats_by_expert[e][a_bucket]["wins"] += 1.0

    def safe_div(a: float, b: float) -> float:
        return a / b if b else 0.0

    print("=== MoE Scorecard ===")
    print(f"items_total={total_items}")
    print(f"analyst_coverage={safe_div(float(total_analyst), float(total_items)):.4f}")
    print(f"trading_days={len(trading_dates)}")
    print(f"horizon_days={int(args.horizon)}")

    rows: List[Tuple[str, Dict[str, float]]] = []
    for k, v in stats.items():
        if v.get("items", 0.0) <= 0:
            continue
        rows.append((k, v))
    rows.sort(key=lambda x: x[0])

    for e, st in rows:
        items = st["items"]
        items_scored = st["items_scored"]
        trades = st["trades"]
        trades_scored = st["trades_scored"]
        wins = st["wins"]
        pnl_sum = st["pnl_sum"]
        missing = st["missing_forward"]
        winrate = safe_div(wins, trades_scored)
        avg_pnl_per_trade = safe_div(pnl_sum, trades_scored)
        avg_abs_pnl_per_trade = safe_div(st["pnl_abs_sum"], trades_scored)

        print("---")
        print(f"expert={e}")
        print(f"items={int(items)}")
        print(f"items_scored={int(items_scored)}")
        print(f"trades={int(trades)}")
        print(f"trades_scored={int(trades_scored)}")
        print(f"winrate={winrate:.4f}")
        print(f"pnl_sum={pnl_sum:.6f}")
        print(f"avg_pnl_per_trade={avg_pnl_per_trade:.6f}")
        print(f"avg_abs_pnl_per_trade={avg_abs_pnl_per_trade:.6f}")
        print(f"missing_forward={int(missing)}")

    if action_stats:
        print("=== Action Breakdown (MoE, scored trades) ===")
        for a in sorted(action_stats.keys()):
            st = action_stats[a]
            ts = float(st.get("trades_scored") or 0.0)
            wins = float(st.get("wins") or 0.0)
            pnl_sum = float(st.get("pnl_sum") or 0.0)
            print(
                f"action={a} trades_scored={int(ts)} winrate={safe_div(wins, ts):.4f} "
                f"pnl_sum={pnl_sum:.6f} avg_pnl={safe_div(pnl_sum, ts):.6f}"
            )

        print("=== Action Breakdown by Expert (MoE, scored trades) ===")
        for e in sorted(action_stats_by_expert.keys()):
            buckets = action_stats_by_expert[e]
            for a in sorted(buckets.keys()):
                st = buckets[a]
                ts = float(st.get("trades_scored") or 0.0)
                wins = float(st.get("wins") or 0.0)
                pnl_sum = float(st.get("pnl_sum") or 0.0)
                print(
                    f"expert={e} action={a} trades_scored={int(ts)} winrate={safe_div(wins, ts):.4f} "
                    f"pnl_sum={pnl_sum:.6f} avg_pnl={safe_div(pnl_sum, ts):.6f}"
                )

    if baseline_payload is None:
        return

    base_stats = {
        "items": 0.0,
        "items_scored": 0.0,
        "trades": 0.0,
        "trades_scored": 0.0,
        "wins": 0.0,
        "pnl_sum": 0.0,
        "pnl_abs_sum": 0.0,
        "missing_forward": 0.0,
        "missing_pair": 0.0,
    }

    base_action_stats: Dict[str, Dict[str, float]] = {}

    for (date_str, sym_u), rec in baseline_map.items():
        base_stats["items"] += 1.0
        pos = _get_position(rec)
        if abs(pos) > 1e-9:
            base_stats["trades"] += 1.0
        fr = fr_map.get((str(date_str), sym_u))
        if fr is None:
            fr = _compute_forward_return_from_closes(
                date_str=str(date_str),
                sym=sym_u,
                trading_dates=trading_dates,
                close_map=close_map,
                horizon=int(args.horizon),
            )
        if fr is None:
            base_stats["missing_forward"] += 1.0
            continue

        base_stats["items_scored"] += 1.0
        if abs(pos) > 1e-9:
            base_stats["trades_scored"] += 1.0

        pnl = float(pos) * float(fr)
        base_stats["pnl_sum"] += pnl
        base_stats["pnl_abs_sum"] += abs(pnl)
        if abs(pos) > 1e-9 and pnl > 0:
            base_stats["wins"] += 1.0

        if abs(pos) > 1e-9:
            a_bucket = _bucket_action(_get_action(rec))
            if a_bucket not in base_action_stats:
                base_action_stats[a_bucket] = {
                    "trades_scored": 0.0,
                    "wins": 0.0,
                    "pnl_sum": 0.0,
                }
            base_action_stats[a_bucket]["trades_scored"] += 1.0
            base_action_stats[a_bucket]["pnl_sum"] += pnl
            if pnl > 0:
                base_action_stats[a_bucket]["wins"] += 1.0

    moe_total = {"pnl_sum": 0.0, "wins": 0.0, "trades_scored": 0.0, "items_scored": 0.0}
    for st in stats.values():
        moe_total["pnl_sum"] += float(st.get("pnl_sum") or 0.0)
        moe_total["wins"] += float(st.get("wins") or 0.0)
        moe_total["trades_scored"] += float(st.get("trades_scored") or 0.0)
        moe_total["items_scored"] += float(st.get("items_scored") or 0.0)

    print("=== A/B Global ===")
    print(f"moe_items_scored={int(moe_total['items_scored'])}")
    print(f"moe_trades_scored={int(moe_total['trades_scored'])}")
    print(f"moe_winrate={safe_div(moe_total['wins'], moe_total['trades_scored']):.4f}")
    print(f"moe_pnl_sum={moe_total['pnl_sum']:.6f}")
    print(f"baseline_items_scored={int(base_stats['items_scored'])}")
    print(f"baseline_trades_scored={int(base_stats['trades_scored'])}")
    print(f"baseline_winrate={safe_div(base_stats['wins'], base_stats['trades_scored']):.4f}")
    print(f"baseline_pnl_sum={base_stats['pnl_sum']:.6f}")
    print(f"delta_pnl_sum={(moe_total['pnl_sum'] - base_stats['pnl_sum']):.6f}")

    if base_action_stats:
        print("=== Action Breakdown (Baseline, scored trades) ===")
        for a in sorted(base_action_stats.keys()):
            st = base_action_stats[a]
            ts = float(st.get("trades_scored") or 0.0)
            wins = float(st.get("wins") or 0.0)
            pnl_sum = float(st.get("pnl_sum") or 0.0)
            print(
                f"action={a} trades_scored={int(ts)} winrate={safe_div(wins, ts):.4f} "
                f"pnl_sum={pnl_sum:.6f} avg_pnl={safe_div(pnl_sum, ts):.6f}"
            )

    analyst_deltas: List[Dict[str, Any]] = []
    delta_pos = 0
    delta_neg = 0
    delta_zero = 0
    delta_missing_pair = 0
    delta_sum = 0.0

    for (date_str, sym_u), rec in moe_map.items():
        if _get_expert(rec) != "analyst":
            continue
        base_rec = baseline_map.get((str(date_str), sym_u))
        if base_rec is None:
            delta_missing_pair += 1
            continue

        fr = fr_map.get((str(date_str), sym_u))
        if fr is None:
            fr = _compute_forward_return_from_closes(
                date_str=str(date_str),
                sym=sym_u,
                trading_dates=trading_dates,
                close_map=close_map,
                horizon=int(args.horizon),
            )
        if fr is None:
            delta_missing_pair += 1
            continue

        moe_pos = _get_position(rec)
        base_pos = _get_position(base_rec)
        moe_pnl = float(moe_pos) * float(fr)
        base_pnl = float(base_pos) * float(fr)
        delta = float(moe_pnl) - float(base_pnl)
        delta_sum += delta
        if delta > 1e-12:
            delta_pos += 1
        elif delta < -1e-12:
            delta_neg += 1
        else:
            delta_zero += 1

        analyst_deltas.append(
            {
                "date": str(date_str),
                "ticker": str(sym_u),
                "fr": float(fr),
                "moe_expert": "analyst",
                "moe_action": _get_action(rec),
                "moe_pos": float(moe_pos),
                "base_action": _get_action(base_rec),
                "base_pos": float(base_pos),
                "moe_pnl": float(moe_pnl),
                "base_pnl": float(base_pnl),
                "delta": float(delta),
            }
        )

    print("=== Attribution (Analyst vs Baseline) ===")
    print(f"analyst_items={int(stats.get('analyst', {}).get('items', 0.0))}")
    print(f"paired_scored={len(analyst_deltas)}")
    print(f"delta_pos={delta_pos}")
    print(f"delta_neg={delta_neg}")
    print(f"delta_zero={delta_zero}")
    print(f"delta_missing_pair={delta_missing_pair}")
    print(f"delta_sum={delta_sum:.6f}")
    print(f"delta_avg={safe_div(delta_sum, float(len(analyst_deltas))):.6f}")

    top_n = int(args.top_n)
    if top_n > 0 and analyst_deltas:
        analyst_deltas_sorted = sorted(analyst_deltas, key=lambda x: float(x.get('delta') or 0.0))
        worst = analyst_deltas_sorted[:top_n]
        best = list(reversed(analyst_deltas_sorted[-top_n:]))

        print("=== Attribution Top Wins ===")
        for r in best:
            print(
                f"{r['date']} {r['ticker']} delta={r['delta']:.6f} fr={r['fr']:.4f} "
                f"moe({r['moe_action']} {r['moe_pos']:.2f}) base({r['base_action']} {r['base_pos']:.2f})"
            )

        print("=== Attribution Top Losses ===")
        for r in worst:
            print(
                f"{r['date']} {r['ticker']} delta={r['delta']:.6f} fr={r['fr']:.4f} "
                f"moe({r['moe_action']} {r['moe_pos']:.2f}) base({r['base_action']} {r['base_pos']:.2f})"
            )

    if bool(args.diagnose_analyst):
        rows: List[Dict[str, Any]] = []
        for r in analyst_deltas:
            key = (str(r.get("date") or ""), str(r.get("ticker") or "").upper().strip())
            moe_rec = moe_map.get(key)
            base_rec = baseline_map.get(key)
            if moe_rec is None or base_rec is None:
                continue

            rows.append(
                {
                    "date": key[0],
                    "ticker": key[1],
                    "fr": r.get("fr"),
                    "analyst_parsed_decision": _get_parsed_decision(moe_rec),
                    "baseline_parsed_decision": _get_parsed_decision(base_rec),
                    "analyst_final_action": _get_action(moe_rec),
                    "baseline_final_action": _get_action(base_rec),
                    "analyst_final_target": _get_position(moe_rec),
                    "baseline_final_target": _get_position(base_rec),
                    "analyst_risk_reason": _get_risk_reason(moe_rec),
                    "baseline_risk_reason": _get_risk_reason(base_rec),
                    "analyst_trace": (moe_rec.get("final") or {}).get("trace") if isinstance(moe_rec.get("final"), dict) else [],
                    "baseline_trace": (base_rec.get("final") or {}).get("trace") if isinstance(base_rec.get("final"), dict) else [],
                    "delta": r.get("delta"),
                }
            )

        rows.sort(key=lambda x: (str(x.get("date") or ""), str(x.get("ticker") or "")))
        print("=== Analyst Sample Diagnostics ===")
        if str(args.diagnose_out).strip():
            out_path = Path(str(args.diagnose_out))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved: {out_path}")
        else:
            lim = int(args.diagnose_limit)
            for row in rows[: max(lim, 0)]:
                print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()

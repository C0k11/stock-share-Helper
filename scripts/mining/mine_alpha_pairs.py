import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


ALPHA_DAY_TYPES_DEFAULT = ["ALPHA_DAY", "DEFENSIVE_ALPHA", "DEFENSIVE_ALPHA_GOLD"]


@dataclass
class DecisionItem:
    action: str
    target_position: float
    expert: str
    planner_strategy: str
    raw_decision: str


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_str(x: Any) -> str:
    return str(x or "").strip()


def _find_single_decisions_file(system_dir: Path) -> Optional[Path]:
    cands = sorted(system_dir.glob("decisions_*.json"))
    if not cands:
        return None
    return cands[-1]


def _load_decisions_map(decisions_path: Path) -> Dict[Tuple[str, str], DecisionItem]:
    obj = json.loads(decisions_path.read_text(encoding="utf-8"))
    days = obj.get("days") if isinstance(obj, dict) else None
    if not isinstance(days, dict):
        return {}

    out: Dict[Tuple[str, str], DecisionItem] = {}
    for date, day_obj in days.items():
        if not isinstance(day_obj, dict):
            continue
        items = day_obj.get("items")
        if not isinstance(items, dict):
            continue
        for ticker, it in items.items():
            if not isinstance(it, dict):
                continue
            final = it.get("final") if isinstance(it.get("final"), dict) else {}
            parsed = it.get("parsed") if isinstance(it.get("parsed"), dict) else {}
            router = it.get("router") if isinstance(it.get("router"), dict) else {}

            action = _safe_str(final.get("action"))
            target_position = _safe_float(final.get("target_position"), 0.0)
            expert = _safe_str(it.get("expert") or router.get("expert"))
            planner_strategy = _safe_str(router.get("planner_strategy"))
            raw_decision = _safe_str(parsed.get("decision"))

            # Fallbacks
            if not action:
                action = raw_decision
            if not action:
                action = "UNKNOWN"

            out[(str(date), str(ticker))] = DecisionItem(
                action=action,
                target_position=target_position,
                expert=expert,
                planner_strategy=planner_strategy,
                raw_decision=raw_decision,
            )

    return out


def _load_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError(f"daily.csv missing date/ticker columns: {path}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["ticker"] = df["ticker"].astype(str)

    for c in ["pnl_h1_net", "news_score", "news_count", "volatility_ann_pct", "turnover", "fee", "target_position"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    if "planner_strategy" not in df.columns:
        df["planner_strategy"] = ""
    if "expert" not in df.columns:
        df["expert"] = ""

    return df


def _load_alpha_days(path: Path, allowed_day_types: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"alpha_days.csv missing date column: {path}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "day_type" not in df.columns:
        raise ValueError(f"alpha_days.csv missing day_type column: {path}")
    df["day_type"] = df["day_type"].astype(str)

    return df[df["day_type"].isin(set(allowed_day_types))].copy()


def _make_side(system: str, daily_row: Dict[str, Any], decision: Optional[DecisionItem]) -> Dict[str, Any]:
    action = _safe_str(daily_row.get("action"))
    target_position = _safe_float(daily_row.get("target_position"), 0.0)
    expert = _safe_str(daily_row.get("expert"))
    planner_strategy = _safe_str(daily_row.get("planner_strategy"))

    if decision is not None:
        action = decision.action or action
        target_position = decision.target_position if decision is not None else target_position
        expert = decision.expert or expert
        planner_strategy = decision.planner_strategy or planner_strategy

    return {
        "system": system,
        "action": action,
        "target_position": float(target_position),
        "quantity": float(target_position),
        "expert": expert,
        "planner_strategy": planner_strategy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine action-pair alpha samples from alpha_days + baseline/golden runs")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing baseline_fast/ and golden_strict/ subdirs",
    )
    parser.add_argument(
        "--alpha-days",
        default=None,
        help="Path to alpha_days.csv (default: <run-dir>/golden_strict/alpha_days.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output alpha_pairs.json path (default: <run-dir>/alpha_pairs.json)",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=0.002,
        help="Significant pnl diff threshold for POSITIVE/NEGATIVE samples",
    )
    parser.add_argument(
        "--behavior-eps",
        type=float,
        default=1e-9,
        help="Treat |diff| <= behavior_eps as ~0 for BEHAVIOR_DIFF",
    )
    parser.add_argument(
        "--day-types",
        default=",".join(ALPHA_DAY_TYPES_DEFAULT),
        help="Comma-separated day_type filter (default: ALPHA_DAY,DEFENSIVE_ALPHA,DEFENSIVE_ALPHA_GOLD)",
    )
    parser.add_argument(
        "--max-per-day",
        type=int,
        default=0,
        help="If >0, keep only top-N tickers per day by |diff_pnl| (after filtering)",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    base_dir = run_dir / "baseline_fast"
    gold_dir = run_dir / "golden_strict"

    alpha_days_path = Path(args.alpha_days) if args.alpha_days else (gold_dir / "alpha_days.csv")
    output_path = Path(args.output) if args.output else (run_dir / "alpha_pairs.json")

    base_daily_path = base_dir / "daily.csv"
    gold_daily_path = gold_dir / "daily.csv"

    if not alpha_days_path.exists():
        raise SystemExit(f"Missing alpha_days.csv: {alpha_days_path}")
    if not base_daily_path.exists():
        raise SystemExit(f"Missing baseline daily.csv: {base_daily_path}")
    if not gold_daily_path.exists():
        raise SystemExit(f"Missing golden daily.csv: {gold_daily_path}")

    allowed_day_types = [s.strip() for s in str(args.day_types).split(",") if s.strip()]

    alpha_days = _load_alpha_days(alpha_days_path, allowed_day_types)
    base_daily = _load_daily(base_daily_path)
    gold_daily = _load_daily(gold_daily_path)

    base_decisions_path = _find_single_decisions_file(base_dir)
    gold_decisions_path = _find_single_decisions_file(gold_dir)

    base_decisions = _load_decisions_map(base_decisions_path) if base_decisions_path else {}
    gold_decisions = _load_decisions_map(gold_decisions_path) if gold_decisions_path else {}

    # Join baseline vs golden on (date,ticker)
    base_daily = base_daily.rename(columns={c: f"base_{c}" for c in base_daily.columns if c not in ["date", "ticker"]})
    gold_daily = gold_daily.rename(columns={c: f"gold_{c}" for c in gold_daily.columns if c not in ["date", "ticker"]})

    merged = base_daily.merge(gold_daily, on=["date", "ticker"], how="inner")
    merged = merged[merged["date"].isin(set(alpha_days["date"].tolist()))].copy()

    if merged.empty:
        raise SystemExit("No overlapping rows between baseline/golden daily.csv for selected alpha days")

    merged["diff_pnl"] = merged["gold_pnl_h1_net"] - merged["base_pnl_h1_net"]

    # Attach per-day context
    alpha_days_small = alpha_days.set_index("date").to_dict(orient="index")

    records: List[Dict[str, Any]] = []

    # Iterate per day for optional max-per-day filtering
    for date, df_day in merged.groupby("date"):
        day_ctx = alpha_days_small.get(str(date), {})

        day_rows = df_day.copy()

        # Extract actions from decisions files
        def _action_for(system: str, ticker: str) -> DecisionItem:
            if system == "baseline_fast":
                return base_decisions.get((str(date), str(ticker)))
            return gold_decisions.get((str(date), str(ticker)))

        rows_out: List[Dict[str, Any]] = []
        for _, r in day_rows.iterrows():
            ticker = str(r["ticker"])
            diff = float(r["diff_pnl"])

            base_dec = _action_for("baseline_fast", ticker)
            gold_dec = _action_for("golden_strict", ticker)

            base_action = base_dec.action if base_dec else ""
            gold_action = gold_dec.action if gold_dec else ""

            sample_type = ""
            dpo_candidate = False

            if diff > float(args.diff_threshold):
                sample_type = "POSITIVE_SAMPLE"
                dpo_candidate = True
                winner = _make_side(
                    "golden_strict",
                    {
                        "action": gold_action,
                        "target_position": r.get("gold_target_position"),
                        "expert": r.get("gold_expert"),
                        "planner_strategy": r.get("gold_planner_strategy"),
                    },
                    gold_dec,
                )
                loser = _make_side(
                    "baseline_fast",
                    {
                        "action": base_action,
                        "target_position": r.get("base_target_position"),
                        "expert": r.get("base_expert"),
                        "planner_strategy": r.get("base_planner_strategy"),
                    },
                    base_dec,
                )
                pair = {"winner": winner, "loser": loser}

            elif diff < -float(args.diff_threshold):
                sample_type = "NEGATIVE_SAMPLE"
                dpo_candidate = True
                winner = _make_side(
                    "baseline_fast",
                    {
                        "action": base_action,
                        "target_position": r.get("base_target_position"),
                        "expert": r.get("base_expert"),
                        "planner_strategy": r.get("base_planner_strategy"),
                    },
                    base_dec,
                )
                loser = _make_side(
                    "golden_strict",
                    {
                        "action": gold_action,
                        "target_position": r.get("gold_target_position"),
                        "expert": r.get("gold_expert"),
                        "planner_strategy": r.get("gold_planner_strategy"),
                    },
                    gold_dec,
                )
                pair = {"winner": winner, "loser": loser}

            else:
                if abs(diff) <= float(args.behavior_eps) and base_action and gold_action and (base_action != gold_action):
                    sample_type = "BEHAVIOR_DIFF"
                    dpo_candidate = False
                    pair = {
                        "winner": _make_side(
                            "golden_strict",
                            {
                                "action": gold_action,
                                "target_position": r.get("gold_target_position"),
                                "expert": r.get("gold_expert"),
                                "planner_strategy": r.get("gold_planner_strategy"),
                            },
                            gold_dec,
                        ),
                        "loser": _make_side(
                            "baseline_fast",
                            {
                                "action": base_action,
                                "target_position": r.get("base_target_position"),
                                "expert": r.get("base_expert"),
                                "planner_strategy": r.get("base_planner_strategy"),
                            },
                            base_dec,
                        ),
                    }
                else:
                    continue

            ctx = {
                "day_type": _safe_str(day_ctx.get("day_type")),
                "excess_return": _safe_float(day_ctx.get("excess_return"), 0.0),
                "strategy_return_h1": _safe_float(day_ctx.get("strategy_return_h1"), 0.0),
                "market_return_h1": _safe_float(day_ctx.get("market_return_h1"), 0.0),
                "avg_vol": _safe_float(day_ctx.get("avg_vol"), 0.0),
                "total_news_vol": _safe_float(day_ctx.get("total_news_vol"), 0.0),
                "max_news_impact": _safe_float(day_ctx.get("max_news_impact"), 0.0),
                "ticker_news_score": _safe_float(r.get("gold_news_score"), 0.0),
                "ticker_news_count": _safe_float(r.get("gold_news_count"), 0.0),
                "ticker_volatility_ann_pct": _safe_float(r.get("gold_volatility_ann_pct"), 0.0),
            }

            rows_out.append(
                {
                    "date": str(date),
                    "ticker": ticker,
                    "diff_pnl": float(diff),
                    "type": sample_type,
                    "dpo_candidate": bool(dpo_candidate),
                    "pair": pair,
                    "context": ctx,
                }
            )

        # Optional top-N per day
        if int(args.max_per_day) > 0 and rows_out:
            rows_out = sorted(rows_out, key=lambda x: abs(float(x.get("diff_pnl", 0.0))), reverse=True)[: int(args.max_per_day)]

        records.extend(rows_out)

    # Global sort: (date asc, |diff| desc)
    records = sorted(records, key=lambda x: (str(x.get("date")), -abs(float(x.get("diff_pnl", 0.0)))))

    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # Minimal console summary
    pos = sum(1 for r in records if r.get("type") == "POSITIVE_SAMPLE")
    neg = sum(1 for r in records if r.get("type") == "NEGATIVE_SAMPLE")
    beh = sum(1 for r in records if r.get("type") == "BEHAVIOR_DIFF")
    print(f"Saved: {output_path}")
    print(f"records={len(records)} positive={pos} negative={neg} behavior_diff={beh}")


if __name__ == "__main__":
    main()

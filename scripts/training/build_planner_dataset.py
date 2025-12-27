import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _load_latest_daily_csv(system_dir: Path) -> Optional[Path]:
    p = system_dir / "daily.csv"
    if p.exists():
        return p
    cands = sorted(system_dir.glob("daily_*.csv"))
    if not cands:
        return None
    return cands[-1]


def _load_latest_decisions_json(system_dir: Path) -> Optional[Path]:
    cands = sorted(system_dir.glob("decisions_*.json"))
    if not cands:
        return None
    return cands[-1]


def _load_decisions_by_date(path: Path) -> Dict[str, Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid decisions json: {path}")

    if isinstance(obj.get("days"), dict):
        out: Dict[str, Dict[str, Any]] = {}
        for d, day in obj["days"].items():
            if isinstance(day, dict):
                out[str(d)] = day
        return out

    if isinstance(obj.get("items"), dict):
        d = str(obj.get("date") or "")
        if not d:
            raise ValueError(f"Single-day decisions missing date: {path}")
        return {d: obj}

    raise ValueError(f"Unrecognized decisions json format: {path}")


def _mode_str(values: List[str]) -> str:
    vals = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    if not vals:
        return ""
    c = Counter(vals)
    return c.most_common(1)[0][0]


def _extract_day_metrics(day_obj: Dict[str, Any]) -> Dict[str, Any]:
    items = day_obj.get("items") if isinstance(day_obj.get("items"), dict) else {}

    planner_obj = day_obj.get("planner") if isinstance(day_obj.get("planner"), dict) else {}
    planner_inputs = planner_obj.get("inputs") if isinstance(planner_obj.get("inputs"), dict) else {}
    mr = planner_inputs.get("market_regime") if isinstance(planner_inputs.get("market_regime"), dict) else {}
    mr_regime = str(mr.get("regime") or "").strip().lower()
    mr_score = _safe_float(mr.get("score"), default=0.0)

    planner_strats: List[str] = []
    actions: List[str] = []
    force_clear = 0

    for _, it in items.items():
        if not isinstance(it, dict):
            continue
        router = it.get("router") if isinstance(it.get("router"), dict) else {}
        planner_strats.append(str(router.get("planner_strategy") or ""))

        final = it.get("final") if isinstance(it.get("final"), dict) else {}
        act = str(final.get("action") or "").strip().upper()
        if act:
            actions.append(act)

        trace = final.get("trace") if isinstance(final.get("trace"), list) else []
        if any("FORCE CLEAR" in str(x) for x in trace):
            force_clear += 1

    action_counts = Counter(actions)
    return {
        "market_regime_score": float(mr_score),
        "market_regime_is_risk_off": 1 if mr_regime == "risk_off" else 0,
        "market_regime_is_risk_on": 1 if mr_regime == "risk_on" else 0,
        "decisions_count": len(items),
        "dec_planner_strategy": _mode_str(planner_strats),
        "dec_force_clear_count": int(force_clear),
        "dec_action_buy": int(action_counts.get("BUY", 0) + action_counts.get("LONG", 0)),
        "dec_action_sell": int(action_counts.get("SELL", 0) + action_counts.get("SHORT", 0)),
        "dec_action_clear": int(action_counts.get("CLEAR", 0) + action_counts.get("FLAT", 0)),
        "dec_action_hold": int(action_counts.get("HOLD", 0)),
    }


def build_dataset(*, run_dir: Path, system: str) -> pd.DataFrame:
    system_dir = run_dir / system
    daily_path = _load_latest_daily_csv(system_dir)
    if daily_path is None:
        raise FileNotFoundError(f"No daily csv found under: {system_dir}")

    df = pd.read_csv(daily_path)
    if "date" not in df.columns:
        raise ValueError(f"daily.csv missing date column: {daily_path}")

    decisions_path = _load_latest_decisions_json(system_dir)
    decisions_by_date: Dict[str, Dict[str, Any]] = {}
    if decisions_path is not None:
        decisions_by_date = _load_decisions_by_date(decisions_path)

    rows: List[Dict[str, Any]] = []
    for date_str, g in df.groupby("date"):
        g2 = g.copy()

        planner_strategy = ""
        if "planner_strategy" in g2.columns:
            planner_strategy = _mode_str([str(x) for x in g2["planner_strategy"].tolist()])

        target_pos = g2["target_position"] if "target_position" in g2.columns else pd.Series([0.0] * len(g2))
        target_pos = pd.to_numeric(target_pos, errors="coerce").fillna(0.0)

        news_count = g2["news_count"] if "news_count" in g2.columns else pd.Series([0.0] * len(g2))
        news_score = g2["news_score"] if "news_score" in g2.columns else pd.Series([0.0] * len(g2))
        vol = g2["volatility_ann_pct"] if "volatility_ann_pct" in g2.columns else pd.Series([0.0] * len(g2))

        news_count = pd.to_numeric(news_count, errors="coerce").fillna(0.0)
        news_score = pd.to_numeric(news_score, errors="coerce").fillna(0.0)
        vol = pd.to_numeric(vol, errors="coerce").fillna(0.0)

        has_strong = 0
        if "has_strong_news_day" in g2.columns:
            try:
                has_strong = int(bool(g2["has_strong_news_day"].astype(bool).any()))
            except Exception:
                has_strong = 0

        day_metrics = {
            "date": str(date_str),
            "system": system,
            "n_tickers": int(len(g2)),
            "vol_mean": float(vol.mean()),
            "vol_max": float(vol.max()),
            "news_count_sum": float(news_count.sum()),
            "news_count_mean": float(news_count.mean()),
            "news_score_mean": float(news_score.mean()),
            "news_score_max": float(news_score.max()),
            "has_strong_news_day": int(has_strong),
            "gross_exposure": float(target_pos.abs().sum()),
            "net_exposure": float(target_pos.sum()),
            "abs_exposure_mean": float(target_pos.abs().mean()),
            "long_count": int((target_pos > 0).sum()),
            "short_count": int((target_pos < 0).sum()),
        }

        dec_obj = decisions_by_date.get(str(date_str))
        if isinstance(dec_obj, dict):
            day_metrics.update(_extract_day_metrics(dec_obj))
        else:
            day_metrics.update(
                {
                    "decisions_count": 0,
                    "dec_planner_strategy": "",
                    "dec_force_clear_count": 0,
                    "dec_action_buy": 0,
                    "dec_action_sell": 0,
                    "dec_action_clear": 0,
                    "dec_action_hold": 0,
                }
            )

        y = planner_strategy or str(day_metrics.get("dec_planner_strategy") or "")
        day_metrics["y_planner_strategy"] = str(y)

        rows.append(day_metrics)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["date"] = out["date"].astype(str)
    out = out.sort_values(["date", "system"])

    out["prev_gross_exposure"] = out.groupby("system")["gross_exposure"].shift(1)
    out["prev_net_exposure"] = out.groupby("system")["net_exposure"].shift(1)
    out["prev_abs_exposure_mean"] = out.groupby("system")["abs_exposure_mean"].shift(1)
    out["prev_long_count"] = out.groupby("system")["long_count"].shift(1)
    out["prev_short_count"] = out.groupby("system")["short_count"].shift(1)
    out[
        [
            "prev_gross_exposure",
            "prev_net_exposure",
            "prev_abs_exposure_mean",
            "prev_long_count",
            "prev_short_count",
        ]
    ] = out[
        [
            "prev_gross_exposure",
            "prev_net_exposure",
            "prev_abs_exposure_mean",
            "prev_long_count",
            "prev_short_count",
        ]
    ].fillna(0.0)

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-run-dir", action="append", required=False)
    p.add_argument(
        "--run-dir",
        dest="results_run_dir",
        action="append",
        required=False,
        help="Alias of --results-run-dir (can be specified multiple times)",
    )
    p.add_argument("--system", default="golden_strict")
    p.add_argument("--out", default="data/training/planner_dataset_v1.csv")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--include-leaky-features", action="store_true", default=False)

    args = p.parse_args()

    if not getattr(args, "results_run_dir", None):
        raise SystemExit("Missing required argument: --run-dir/--results-run-dir")

    dfs: List[pd.DataFrame] = []
    for rd in args.results_run_dir:
        run_dir = Path(str(rd))
        dfs.append(build_dataset(run_dir=run_dir, system=str(args.system)))
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if not df.empty:
        df = df.drop_duplicates(subset=["date", "system"], keep="last").sort_values(["date", "system"])

    if str(args.start).strip():
        df = df[df["date"] >= str(args.start).strip()]
    if str(args.end).strip():
        df = df[df["date"] <= str(args.end).strip()]

    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not bool(args.include_leaky_features) and (not df.empty):
        leaky_cols = [
            c
            for c in df.columns
            if c.startswith("dec_") or ("turnover" in c) or ("fee" in c) or c.startswith("fr_") or c.startswith("pnl_")
        ]
        leaky_cols.extend(["gross_exposure", "net_exposure", "abs_exposure_mean", "long_count", "short_count"])
        df = df.drop(columns=leaky_cols, errors="ignore")

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} rows={len(df)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _load_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "ticker" not in df.columns:
        raise SystemExit(f"daily.csv missing date/ticker: {path}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["ticker"] = df["ticker"].astype(str)

    for c in ["target_position", "pnl_h1_net", "fee", "turnover"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    return df


def _load_alpha_days(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise SystemExit(f"alpha_days.csv missing date: {path}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def _quantiles(series: pd.Series) -> Dict[str, float]:
    if series is None or len(series) == 0:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    qs = series.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    return {
        "p50": _safe_float(qs.get(0.5), 0.0),
        "p90": _safe_float(qs.get(0.9), 0.0),
        "p95": _safe_float(qs.get(0.95), 0.0),
        "p99": _safe_float(qs.get(0.99), 0.0),
        "max": _safe_float(series.max(), 0.0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect baseline vs golden diffs from a run_dir")
    p.add_argument("--run-dir", required=True, help="Directory containing baseline_fast/ and golden_strict/")
    p.add_argument("--alpha-days", default="", help="Optional alpha_days.csv (default: <run-dir>/golden_strict/alpha_days.csv)")
    p.add_argument("--only-alpha-days", action="store_true", default=False)
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--top-n", type=int, default=25)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    base_path = run_dir / "baseline_fast" / "daily.csv"
    gold_path = run_dir / "golden_strict" / "daily.csv"

    if not base_path.exists():
        raise SystemExit(f"Missing: {base_path}")
    if not gold_path.exists():
        raise SystemExit(f"Missing: {gold_path}")

    base = _load_daily(base_path)
    gold = _load_daily(gold_path)

    m = base.merge(gold, on=["date", "ticker"], how="inner", suffixes=("_base", "_gold"))

    alpha_days_path = Path(args.alpha_days) if str(args.alpha_days).strip() else (run_dir / "golden_strict" / "alpha_days.csv")
    alpha_days: Optional[pd.DataFrame] = None
    if alpha_days_path.exists():
        alpha_days = _load_alpha_days(alpha_days_path)

    if bool(args.only_alpha_days):
        if alpha_days is None:
            raise SystemExit(f"--only-alpha-days requested but alpha_days.csv not found: {alpha_days_path}")
        m = m[m["date"].isin(set(alpha_days["date"].tolist()))].copy()

    if m.empty:
        raise SystemExit("No overlapping rows between baseline and golden daily.csv")

    m["tp_diff_abs"] = (m["target_position_gold"] - m["target_position_base"]).abs()
    m["pnl_diff"] = m["pnl_h1_net_gold"] - m["pnl_h1_net_base"]
    m["pnl_diff_abs"] = m["pnl_diff"].abs()

    eps = float(args.eps)

    tp_diff_rows = int((m["tp_diff_abs"] > eps).sum())
    pnl_diff_rows = int((m["pnl_diff_abs"] > eps).sum())

    tp_diff_days = int(m.loc[m["tp_diff_abs"] > eps, "date"].nunique())
    pnl_diff_days = int(m.loc[m["pnl_diff_abs"] > eps, "date"].nunique())

    print("# Run Diff Summary")
    print(f"run_dir: {run_dir.as_posix()}")
    print(f"rows: {len(m)}")
    print("")

    print("## Target Position Diff")
    print(f"tp_diff_rows (>eps): {tp_diff_rows}")
    print(f"tp_diff_days (>eps): {tp_diff_days}")
    print(f"tp_diff_abs_quantiles: {_quantiles(m['tp_diff_abs'])}")
    print("")

    print("## PnL Diff")
    print(f"pnl_diff_rows (>eps): {pnl_diff_rows}")
    print(f"pnl_diff_days (>eps): {pnl_diff_days}")
    print(f"pnl_diff_abs_quantiles: {_quantiles(m['pnl_diff_abs'])}")
    print("")

    print("## Top Rows by |PnL Diff|")
    top = m.sort_values("pnl_diff_abs", ascending=False).head(int(args.top_n)).copy()
    cols = [
        "date",
        "ticker",
        "pnl_h1_net_base",
        "pnl_h1_net_gold",
        "pnl_diff",
        "target_position_base",
        "target_position_gold",
        "tp_diff_abs",
    ]
    cols = [c for c in cols if c in top.columns]
    print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()

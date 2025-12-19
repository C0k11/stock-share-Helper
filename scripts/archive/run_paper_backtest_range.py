#!/usr/bin/env python

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


_DATE_RE = re.compile(r"etf_features_(\d{4}-\d{2}-\d{2})\.json$")


@dataclass
class RunCfg:
    daily_dir: Path
    outdir: Path
    base_model: str
    adapter: str
    max_new_tokens: int
    temperature: float
    load_4bit: bool
    initial_cash: float
    rebalance_threshold: float
    slippage_bps: float
    fee_bps: float
    mtm_next_day: int
    data_dir: Path


def _parse_date_from_features_path(p: Path) -> Optional[str]:
    m = _DATE_RE.search(p.name)
    if not m:
        return None
    return m.group(1)


def _collect_dates(daily_dir: Path, start_date: str, end_date: str) -> List[str]:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    dates: List[str] = []
    for fp in daily_dir.glob("etf_features_*.json"):
        ds = _parse_date_from_features_path(fp)
        if not ds:
            continue
        try:
            d = pd.to_datetime(ds)
        except Exception:
            continue
        if start <= d <= end:
            dates.append(ds)

    dates = sorted(set(dates))
    return dates


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def _infer_for_date(cfg: RunCfg, date_str: str, decision_path: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/run_trading_inference.py",
        "--date",
        date_str,
        "--daily-dir",
        str(cfg.daily_dir),
        "--base",
        cfg.base_model,
        "--adapter",
        cfg.adapter,
        "--max-new-tokens",
        str(cfg.max_new_tokens),
        "--temperature",
        str(cfg.temperature),
        "--out",
        str(decision_path),
    ]
    if cfg.load_4bit:
        cmd.append("--load-4bit")
    _run(cmd)


def _paper_for_date(cfg: RunCfg, date_str: str, decision_path: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/paper_trade_sim.py",
        "--date",
        date_str,
        "--daily-dir",
        str(cfg.daily_dir),
        "--decision",
        str(decision_path),
        "--outdir",
        str(cfg.outdir),
        "--state",
        "account_state.json",
        "--initial-cash",
        str(cfg.initial_cash),
        "--rebalance-threshold",
        str(cfg.rebalance_threshold),
        "--slippage-bps",
        str(cfg.slippage_bps),
        "--fee-bps",
        str(cfg.fee_bps),
        "--mtm-next-day",
        str(cfg.mtm_next_day),
        "--data-dir",
        str(cfg.data_dir),
    ]
    _run(cmd)


def _load_nav(nav_path: Path) -> pd.DataFrame:
    df = pd.read_csv(nav_path)
    if "date" not in df.columns or "value" not in df.columns:
        raise SystemExit(f"Invalid NAV csv schema: {nav_path}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    return df


def _max_drawdown(values: pd.Series) -> float:
    peak = values.cummax()
    dd = (values / peak) - 1.0
    return float(dd.min())


def _summarize(outdir: Path) -> Tuple[float, float, float, float, int, Path]:
    nav_path = outdir / "paper_nav.csv"
    if not nav_path.exists():
        raise SystemExit(f"Missing nav: {nav_path}")

    df = _load_nav(nav_path)
    if df.empty:
        raise SystemExit(f"Empty nav: {nav_path}")

    start_v = float(df.iloc[0]["value"])
    end_v = float(df.iloc[-1]["value"])
    total_ret = (end_v / start_v) - 1.0 if start_v > 0 else 0.0
    mdd = _max_drawdown(df["value"])
    n_days = int(df.shape[0])
    return start_v, end_v, total_ret, mdd, n_days, nav_path


def main() -> None:
    p = argparse.ArgumentParser(description="Rolling paper backtest: run trading inference + paper sim across a date range")
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end-date", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--daily-dir", default="data/daily")

    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--outdir", required=True)

    p.add_argument("--max-new-tokens", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--no-load-4bit", dest="load_4bit", action="store_false")
    p.set_defaults(load_4bit=True)

    p.add_argument("--initial-cash", type=float, default=100000.0)
    p.add_argument("--rebalance-threshold", type=float, default=0.02)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--fee-bps", type=float, default=1.0)
    p.add_argument("--mtm-next-day", type=int, default=1)
    p.add_argument("--data-dir", default="data")

    p.add_argument("--limit-days", type=int, default=0, help="Optional cap on number of days processed (0=all)")
    args = p.parse_args()

    cfg = RunCfg(
        daily_dir=Path(args.daily_dir),
        outdir=Path(args.outdir),
        base_model=str(args.base_model),
        adapter=str(args.adapter),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        load_4bit=bool(args.load_4bit),
        initial_cash=float(args.initial_cash),
        rebalance_threshold=float(args.rebalance_threshold),
        slippage_bps=float(args.slippage_bps),
        fee_bps=float(args.fee_bps),
        mtm_next_day=int(args.mtm_next_day),
        data_dir=Path(args.data_dir),
    )

    cfg.outdir.mkdir(parents=True, exist_ok=True)

    dates = _collect_dates(cfg.daily_dir, str(args.start_date), str(args.end_date))
    if not dates:
        raise SystemExit(
            f"No etf_features_YYYY-MM-DD.json found in range {args.start_date}..{args.end_date} under {cfg.daily_dir}"
        )

    if int(args.limit_days) > 0:
        dates = dates[: int(args.limit_days)]

    decisions_dir = cfg.outdir / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dates: {len(dates)} ({dates[0]}..{dates[-1]})")
    print(f"Outdir: {cfg.outdir}")
    print(f"Base: {cfg.base_model}")
    print(f"Adapter: {cfg.adapter}")
    print(f"max_new_tokens={cfg.max_new_tokens} temperature={cfg.temperature} load_4bit={cfg.load_4bit}")

    for i, d in enumerate(dates, start=1):
        decision_path = decisions_dir / f"trading_decision_{d}.json"
        print(f"\n[{i}/{len(dates)}] {d}")
        _infer_for_date(cfg, d, decision_path)
        _paper_for_date(cfg, d, decision_path)

    start_v, end_v, total_ret, mdd, n_days, nav_path = _summarize(cfg.outdir)
    print("\n=== Final Report ===")
    print(f"Days: {n_days}")
    print(f"Start value: {start_v:.2f}")
    print(f"End value:   {end_v:.2f}")
    print(f"Total return: {total_ret * 100.0:.2f}%")
    print(f"Max drawdown: {mdd * 100.0:.2f}%")
    print(f"NAV csv: {nav_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


def load_tickers(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Ticker file not found: {path}")
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            s = s.lstrip("\ufeff")
            if not s or s.startswith("#"):
                continue
            out.append(s.upper())
    return list(dict.fromkeys(out))


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).strip() for c in df.columns]

    df.columns = [str(c).strip().lower() for c in df.columns]

    if "date" in df.columns:
        df = df.set_index("date")

    if df.index.name is None:
        df.index.name = "date"

    df = df.reset_index()

    # Normalize date column name (yfinance uses 'Date' with capital D)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    keep = [c for c in ["date", "open", "high", "low", "close", "adj close", "volume"] if c in df.columns]
    df = df[keep]

    if "adj close" in df.columns and "adj_close" not in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})

    return df


def _try_read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    return df


def _max_date(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty or "date" not in df.columns:
        return None
    try:
        d = df["date"].max()
        if pd.isna(d):
            return None
        return str(pd.to_datetime(d).date())
    except Exception:
        return None


def _parse_date(s: str) -> datetime:
    try:
        return datetime.fromisoformat(str(s).strip()[:10])
    except Exception:
        raise SystemExit(f"Invalid date: {s} (expected YYYY-MM-DD)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical stock OHLCV via yfinance and save to data/raw/*.parquet")
    parser.add_argument("--tickers-file", default="data/tickers/phase6_expansion_large.txt")
    parser.add_argument(
        "--tickers",
        default="",
        help="Optional comma-separated tickers override (e.g. AAPL,NVDA,TSLA). If provided, tickers-file is ignored.",
    )
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--mode", choices=["append", "overwrite"], default="append")
    parser.add_argument(
        "--end-inclusive",
        action="store_true",
        default=True,
        help="Treat --end as inclusive by adding +1 day for yfinance (end is exclusive). Default True.",
    )
    parser.add_argument("--no-end-inclusive", dest="end_inclusive", action="store_false")
    parser.add_argument("--skip-existing", action="store_true", help="Alias for --mode append (kept for compatibility)")

    args = parser.parse_args()

    tickers_file = Path(args.tickers_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers_arg = str(args.tickers or "").strip()
    if tickers_arg:
        tickers = [t.strip().upper() for t in tickers_arg.split(",") if t.strip()]
    else:
        tickers = load_tickers(tickers_file)
    if not tickers:
        raise SystemExit("No tickers loaded.")

    start = str(args.start).strip()
    end_str = str(args.end).strip() or ""

    start_dt = _parse_date(start)
    end_dt = _parse_date(end_str) if end_str else datetime.now()
    if bool(args.end_inclusive):
        end_dt = end_dt + timedelta(days=1)

    if end_dt <= start_dt:
        raise SystemExit(f"Invalid range: start={start_dt.date()} end={end_dt.date()}")

    ok = 0
    failed = 0
    skipped = 0
    updated: Dict[str, str] = {}
    existing_max: Dict[str, Optional[str]] = {}

    for i, t in enumerate(tickers, start=1):
        out_path = out_dir / f"{t}.parquet"

        mode = str(args.mode)
        if bool(args.skip_existing):
            mode = "append"

        existing = _try_read_existing(out_path) if mode == "append" else pd.DataFrame()
        existing_max_date = _max_date(existing)
        existing_max[t] = existing_max_date

        fetch_start_dt = start_dt
        if mode == "append" and existing_max_date:
            try:
                fetch_start_dt = datetime.fromisoformat(existing_max_date) + timedelta(days=1)
            except Exception:
                fetch_start_dt = start_dt

        if mode == "append" and existing_max_date and fetch_start_dt >= end_dt:
            skipped += 1
            continue

        try:
            df = yf.download(
                t,
                start=str(fetch_start_dt.date()),
                end=str(end_dt.date()),
                interval=str(args.interval),
                progress=False,
                auto_adjust=False,
                actions=False,
            )

            df2 = normalize_df(df)
            if df2.empty:
                failed += 1
                continue

            if mode == "append" and not existing.empty:
                merged = pd.concat([existing, df2], ignore_index=True)
                if "date" in merged.columns:
                    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
                    merged = merged.dropna(subset=["date"])
                    merged = merged.sort_values("date")
                    merged = merged.drop_duplicates(subset=["date"], keep="last")
                df2 = merged

            df2.to_parquet(out_path, index=False)
            ok += 1
            updated[t] = _max_date(df2) or ""

        except Exception:
            failed += 1

        if float(args.sleep) > 0:
            time.sleep(float(args.sleep))

    print(
        json.dumps(
            {
                "tickers": len(tickers),
                "ok": ok,
                "failed": failed,
                "skipped": skipped,
                "out_dir": str(out_dir),
                "range": {
                    "start": str(start_dt.date()),
                    "end_exclusive": str(end_dt.date()),
                    "end_inclusive": bool(args.end_inclusive),
                },
                "existing_max_date": existing_max,
                "updated_max_date": updated,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

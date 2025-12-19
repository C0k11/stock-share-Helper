#!/usr/bin/env python

import argparse
import json
import collections
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_EXCLUDE = {"SPY", "QQQ", "TLT", "GLD", "IWM", "DIA", "VIX", "BTC-USD"}


def load_tickers_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"tickers-file not found: {path}")
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip().lstrip("\ufeff")
            if not s or s.startswith("#"):
                continue
            out.append(s.upper())
    return list(dict.fromkeys(out))


def normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Prefer explicit date-like columns
    date_col = None
    for c in ("date", "datetime", "timestamp", "time"):
        if c in df.columns:
            date_col = c
            break

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
    else:
        # If parquet has no date column, only accept already-datetime index.
        # Do NOT coerce RangeIndex/int index to datetime (would create 1970-01-01 fake dates).
        if isinstance(df.index, pd.DatetimeIndex):
            pass
        elif isinstance(df.index, pd.RangeIndex):
            return pd.DataFrame()
        else:
            # Try parse string-like index to datetime; reject if mostly invalid.
            idx = pd.to_datetime(df.index, errors="coerce")
            if idx.isna().mean() > 0.1:
                return pd.DataFrame()
            df.index = idx

    df = df.sort_index()

    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    return df


def get_label(ret_lookahead: float, buy_th: float, sell_th: float) -> str:
    if ret_lookahead >= buy_th:
        return "BUY"
    if ret_lookahead <= sell_th:
        return "SELL"
    return "HOLD"


def build_samples_for_symbol(
    *,
    symbol: str,
    df: pd.DataFrame,
    look_ahead_days: int,
    buy_th: float,
    sell_th: float,
    stride: int,
    max_lookback_rows: int,
) -> List[Dict[str, Any]]:
    df = normalize_price_df(df)
    if df is None or df.empty:
        return []

    if "close" not in df.columns:
        return []

    df = df.dropna(subset=["close"])
    if len(df) < max(30, look_ahead_days + 2):
        return []

    # focus on latest N rows to reduce sample explosion
    start_idx = max(0, len(df) - int(max_lookback_rows)) if int(max_lookback_rows) > 0 else 0

    out: List[Dict[str, Any]] = []

    closes = df["close"].astype(float)
    vols = df["volume"].astype(float) if "volume" in df.columns else None

    last_i = len(df) - look_ahead_days - 1
    for i in range(start_idx, last_i + 1, max(1, int(stride))):
        c0 = float(closes.iloc[i])
        c1 = float(closes.iloc[i + look_ahead_days])
        if c0 <= 0:
            continue
        ret = (c1 - c0) / c0
        label = get_label(ret, float(buy_th), float(sell_th))

        dt0 = df.index[i]
        vol0 = float(vols.iloc[i]) if vols is not None and not pd.isna(vols.iloc[i]) else None

        out.append(
            {
                "ticker": symbol,
                "date": str(dt0.date()),
                "label": label,
                "look_ahead_days": int(look_ahead_days),
                "ret_lookahead": round(float(ret), 6),
                "threshold_buy": float(buy_th),
                "threshold_sell": float(sell_th),
                "price_context": {
                    "close": round(c0, 4),
                    "volume": None if vol0 is None else int(vol0),
                },
            }
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rule-based stock training labels (BUY/SELL/HOLD) from OHLCV")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--tickers-file", default="data/tickers/phase6_expansion_large.txt")
    parser.add_argument("--out", default="data/finetune/trader_stock_raw_v1.json")

    parser.add_argument("--look-ahead-days", type=int, default=5)
    parser.add_argument("--buy-th", type=float, default=0.04)
    parser.add_argument("--sell-th", type=float, default=-0.03)

    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-lookback-rows", type=int, default=500)

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise SystemExit(f"raw-dir not found: {raw_dir}")

    tickers = load_tickers_file(Path(args.tickers_file))
    tickers = [t for t in tickers if t not in DEFAULT_EXCLUDE]

    samples: List[Dict[str, Any]] = []
    missing: List[str] = []

    for t in tickers:
        p = raw_dir / f"{t}.parquet"
        if not p.exists():
            missing.append(t)
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue

        ss = build_samples_for_symbol(
            symbol=t,
            df=df,
            look_ahead_days=int(args.look_ahead_days),
            buy_th=float(args.buy_th),
            sell_th=float(args.sell_th),
            stride=int(args.stride),
            max_lookback_rows=int(args.max_lookback_rows),
        )
        samples.extend(ss)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "raw_dir": str(raw_dir),
                    "tickers_file": str(args.tickers_file),
                    "tickers": len(tickers),
                    "missing_parquet": missing,
                    "look_ahead_days": int(args.look_ahead_days),
                    "buy_th": float(args.buy_th),
                    "sell_th": float(args.sell_th),
                    "stride": int(args.stride),
                    "max_lookback_rows": int(args.max_lookback_rows),
                },
                "samples": samples,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"saved_samples={len(samples)} out={out_path}")
    if samples:
        c = collections.Counter(s.get("label") for s in samples)
        n = len(samples)
        for k in ["BUY", "SELL", "HOLD"]:
            v = int(c.get(k, 0))
            pct = (v * 100.0 / n) if n else 0.0
            print(f"label_{k}={v} ({pct:.2f}%)")
    if missing:
        print(f"missing_parquet={len(missing)} example={missing[:10]}")


if __name__ == "__main__":
    main()

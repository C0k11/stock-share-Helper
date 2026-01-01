#!/usr/bin/env python

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _to_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    idx = idx.normalize()
    df = df.copy()
    df.index = idx
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df


def _load_symbol(storage, symbol: str, category: str) -> Optional[pd.DataFrame]:
    df = storage.load_price_data(symbol, category=category)
    if df is None or df.empty:
        return None
    df = _to_dt_index(df)
    if "close" not in df.columns:
        raise SystemExit(f"Missing 'close' column for {symbol} under data/{category}")
    return df


def _zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - mu) / sd.replace(0.0, np.nan)


def _clip01(s: pd.Series) -> pd.Series:
    return s.clip(lower=0.0, upper=1.0)


def _available_weighted_avg(parts: Dict[str, Tuple[pd.Series, float]]) -> pd.Series:
    total_w = 0.0
    acc = None
    for _, (s, w) in parts.items():
        if s is None:
            continue
        total_w += float(w)
        acc = (s * float(w)) if acc is None else (acc + s * float(w))
    if acc is None or total_w <= 0:
        raise SystemExit("No macro components available to build Global_Risk_Score")
    return acc / total_w


def build_macro_features(
    *,
    base_path: str,
    category: str,
    include_dxy: bool,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from src.data.storage import DataStorage

    storage = DataStorage(base_path=base_path)

    spy = _load_symbol(storage, "SPY", category)
    vix = _load_symbol(storage, "VIX", category)
    tnx = _load_symbol(storage, "TNX", category)

    if spy is None:
        raise SystemExit(f"Missing SPY data under data/{category}. Run scripts/download_data.py first.")

    if vix is None:
        raise SystemExit(f"Missing VIX data under data/{category}. Expect symbol saved as VIX (from ^VIX).")

    if tnx is None:
        raise SystemExit(f"Missing TNX data under data/{category}. Expect symbol saved as TNX (from ^TNX).")

    dxy = None
    if include_dxy:
        dxy = _load_symbol(storage, "DXY", category)

    idx = spy.index
    idx = idx.union(vix.index).union(tnx.index)
    if dxy is not None:
        idx = idx.union(dxy.index)
    idx = idx.sort_values()

    if start_date:
        idx = idx[idx >= pd.to_datetime(start_date)]
    if end_date:
        idx = idx[idx <= pd.to_datetime(end_date)]

    spy_close = spy["close"].reindex(idx).ffill()
    vix_close = vix["close"].reindex(idx).ffill()
    tnx_close = tnx["close"].reindex(idx).ffill()

    ma200 = spy_close.rolling(200).mean()
    bull_regime = (spy_close > ma200).astype(float)
    regime_risk = 1.0 - bull_regime

    vix_level_score = _clip01((vix_close - 15.0) / 15.0)
    vix_z20 = _zscore(vix_close, 20)
    vix_z_score = _clip01((vix_z20.fillna(0.0) - 1.0) / 3.0)
    stress_score = pd.concat([vix_level_score, vix_z_score], axis=1).max(axis=1)

    tnx_diff = tnx_close.diff(1)
    tnx_z20 = _zscore(tnx_diff, 20)
    tnx_pos_z = tnx_z20.where(tnx_z20 > 0.0, 0.0).fillna(0.0)
    rate_shock_score = _clip01(tnx_pos_z / 3.0)

    parts: Dict[str, Tuple[pd.Series, float]] = {
        "regime": (regime_risk, 0.5),
        "stress": (stress_score, 0.3),
        "rate_shock": (rate_shock_score, 0.2),
    }

    if dxy is not None and not dxy.empty:
        dxy_close = dxy["close"].reindex(idx).ffill()
        dxy_ret_20 = dxy_close.pct_change(20)
        dxy_z20 = _zscore(dxy_ret_20, 60)
        dxy_pos_z = dxy_z20.where(dxy_z20 > 0.0, 0.0).fillna(0.0)
        dxy_score = _clip01(dxy_pos_z / 3.0)
        parts["dxy"] = (dxy_score, 0.1)

    global_risk = _available_weighted_avg(parts).ffill().fillna(0.0)
    global_risk = _clip01(global_risk)

    out = pd.DataFrame(
        {
            "Date": idx.date,
            "Global_Risk_Score": global_risk.values,
        }
    )

    out = out.drop_duplicates(subset=["Date"], keep="last")

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default="data")
    parser.add_argument("--category", default="raw", choices=["raw", "processed", "cache", "features"])
    parser.add_argument("--include-dxy", action="store_true")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--out", default="data/macro_features.csv")
    args = parser.parse_args()

    df = build_macro_features(
        base_path=str(args.base_path),
        category=str(args.category),
        include_dxy=bool(args.include_dxy),
        start_date=args.start_date,
        end_date=args.end_date,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

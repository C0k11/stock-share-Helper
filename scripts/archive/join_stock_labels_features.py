#!/usr/bin/env python

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
    df = df.sort_index()
    return df


def compute_features_for_date(df: pd.DataFrame, target_date: pd.Timestamp) -> Optional[Dict[str, Any]]:
    """Compute technical features using data up to target_date (inclusive)."""
    df_slice = df.loc[:target_date]
    if df_slice.empty or len(df_slice) < 50:
        return None

    if "close" not in df_slice.columns:
        return None

    price = df_slice["close"].astype(float)
    curr = float(price.iloc[-1])
    if curr <= 0:
        return None

    # Moving averages
    ma20 = price.rolling(20).mean().iloc[-1]
    ma60 = price.rolling(60).mean().iloc[-1] if len(price) >= 60 else ma20

    # RSI (14-day)
    delta = price.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    # Volatility (20-day annualized)
    log_ret = np.log(price / price.shift(1))
    vol = float(log_ret.rolling(20).std().iloc[-1] * np.sqrt(252)) if len(log_ret) >= 20 else 0.0

    # Distance from MA20
    dist_ma20 = (curr - ma20) / ma20 if ma20 > 0 else 0.0

    # Volume trend
    vol_ratio = 1.0
    if "volume" in df_slice.columns:
        vol_col = df_slice["volume"].astype(float)
        vol_ma = vol_col.rolling(20).mean().iloc[-1]
        if vol_ma > 0:
            vol_ratio = float(vol_col.iloc[-1] / vol_ma)

    return {
        "price": round(curr, 2),
        "ma20": round(float(ma20), 2),
        "ma60": round(float(ma60), 2),
        "dist_ma20_pct": round(float(dist_ma20 * 100), 2),
        "rsi": round(rsi_val, 1),
        "volatility": round(vol, 3),
        "volume_ratio": round(vol_ratio, 2),
    }


def build_sft_conversation(ticker: str, date_str: str, feats: Dict[str, Any], label: str) -> Dict[str, Any]:
    """Build a conversation in Qwen Instruct SFT format."""
    prompt_text = f"""Analyze the following stock data for {ticker} on {date_str}:
Price: {feats['price']}
MA20: {feats['ma20']} (Dist: {feats['dist_ma20_pct']}%)
MA60: {feats['ma60']}
RSI: {feats['rsi']}
Volatility: {feats['volatility']}
Volume Ratio: {feats['volume_ratio']}

Based on the technical structure, what is the trading decision (BUY/SELL/HOLD) for the next 5 days?"""

    response = json.dumps({
        "decision": label,
        "ticker": ticker,
        "analysis": "Technical setup suggests current trend continuation."
    })

    return {
        "conversations": [
            {"from": "system", "value": "You are a professional quantitative trader. Analyze the market data and provide a trading signal."},
            {"from": "user", "value": prompt_text},
            {"from": "assistant", "value": response},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Join labels with dynamically computed features to produce Trader SFT dataset")
    parser.add_argument("--labels", default="data/finetune/trader_stock_raw_v1.json")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/finetune/trader_stock_sft_v1.json")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0=no limit)")

    args = parser.parse_args()

    labels_path = Path(args.labels)
    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)

    print(f"Loading labels from {labels_path}...")
    with open(labels_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = raw_data.get("samples", [])
    if int(args.limit) > 0:
        samples = samples[: int(args.limit)]

    # Group by ticker to reduce parquet IO
    by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        t = s.get("ticker")
        if t not in by_ticker:
            by_ticker[t] = []
        by_ticker[t].append(s)

    print(f"Processing {len(samples)} samples across {len(by_ticker)} tickers...")

    sft_dataset: List[Dict[str, Any]] = []
    skipped = 0

    for ticker, items in by_ticker.items():
        parquet_path = raw_dir / f"{ticker}.parquet"
        df = load_parquet(parquet_path)
        if df.empty:
            skipped += len(items)
            continue

        for item in items:
            date_str = item.get("date")
            label = item.get("label")
            if not date_str or not label:
                skipped += 1
                continue

            target_date = pd.to_datetime(date_str, errors="coerce")
            if pd.isna(target_date):
                skipped += 1
                continue

            feats = compute_features_for_date(df, target_date)
            if feats is None:
                skipped += 1
                continue

            conv = build_sft_conversation(ticker, date_str, feats, label)
            sft_dataset.append(conv)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sft_dataset, f, ensure_ascii=False, indent=2)

    print(f"saved_samples={len(sft_dataset)} skipped={skipped} out={out_path}")


if __name__ == "__main__":
    main()

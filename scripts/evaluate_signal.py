#!/usr/bin/env python

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.storage import DataStorage
from src.data.fetcher import DataFetcher


def _today_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")


def _parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("signals file must be a JSON list")
    return data


def _build_daily_returns(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    if "close" not in df.columns:
        raise ValueError("price dataframe must contain 'close' column")

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    s = df["close"].astype(float).sort_index()
    ret = s.pct_change() * 100.0
    out: Dict[str, float] = {}
    for ts, v in ret.dropna().items():
        out[ts.strftime("%Y-%m-%d")] = float(v)
    return out


def _find_tplus1_return(pub_date: str, daily_returns: Dict[str, float], *, max_lag_days: int) -> Tuple[Optional[str], Optional[float]]:
    d0 = _parse_date(pub_date)
    for i in range(1, max(1, int(max_lag_days)) + 1):
        d = (d0 + dt.timedelta(days=i)).strftime("%Y-%m-%d")
        if d in daily_returns:
            return d, daily_returns[d]
    return None, None


def _last_available_date(daily_returns: Dict[str, float]) -> Optional[str]:
    if not daily_returns:
        return None
    return max(daily_returns.keys())


def _max_pub_date(preds: List[Dict[str, Any]]) -> Optional[str]:
    dates = sorted({p["pub_date"] for p in preds if p.get("pub_date")})
    return dates[-1] if dates else None


def _ensure_price_coverage(
    *,
    storage: DataStorage,
    fetcher: DataFetcher,
    symbol: str,
    category: str,
    required_end_date: str,
) -> Optional[pd.DataFrame]:
    existing = storage.load_price_data(symbol, category=category)
    if existing is None or existing.empty:
        start = (dt.datetime.strptime(required_end_date, "%Y-%m-%d") - dt.timedelta(days=30)).strftime("%Y-%m-%d")
        logger.info(f"Auto-fetch {symbol}: no existing data; fetching from {start} to {required_end_date}")
        fetched = fetcher.fetch_price([symbol], start_date=start, end_date=required_end_date).get(symbol)
        if fetched is None or fetched.empty:
            return existing
        storage.save_price_data(symbol, fetched, category=category)
        return fetched

    if not isinstance(existing.index, pd.DatetimeIndex):
        existing = existing.copy()
        existing.index = pd.to_datetime(existing.index)

    last_dt = existing.index.max()
    last_str = last_dt.strftime("%Y-%m-%d")
    if last_str >= required_end_date:
        return existing

    # fetch a small overlap window to compute pct_change on the first new day
    start = (last_dt - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    logger.info(f"Auto-fetch {symbol}: extending from {start} to {required_end_date} (last={last_str})")
    fetched = fetcher.fetch_price([symbol], start_date=start, end_date=required_end_date).get(symbol)
    if fetched is None or fetched.empty:
        return existing

    if not isinstance(fetched.index, pd.DatetimeIndex):
        fetched = fetched.copy()
        fetched.index = pd.to_datetime(fetched.index)

    combined = existing.combine_first(fetched).sort_index()
    storage.save_price_data(symbol, combined, category=category)
    return combined


def _iter_predictions(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in signals:
        market = str(it.get("market") or "US").strip().upper()
        published_at = str(it.get("published_at") or "")
        pub_date = published_at[:10] if len(published_at) >= 10 else ""
        sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
        impact = sig.get("impact_equity")
        try:
            impact_i = int(impact)
        except Exception:
            impact_i = 0

        if impact_i == 0:
            continue
        if not pub_date:
            continue

        out.append(
            {
                "id": it.get("id"),
                "market": market,
                "pub_date": pub_date,
                "impact_equity": impact_i,
                "event_type": str(sig.get("event_type") or ""),
                "title": str(it.get("title") or ""),
            }
        )
    return out


def evaluate(
    signals_path: Path,
    *,
    us_symbol: str,
    cn_symbol: str,
    category: str,
    sample_n: int,
    max_lag_days: int,
    auto_fetch: bool,
    fetch_source: str,
) -> int:
    if not signals_path.exists():
        logger.error(f"Signals file not found: {signals_path}")
        return 2

    signals_raw = _load_json_list(signals_path)
    preds = _iter_predictions(signals_raw)

    storage = DataStorage(base_path="data")

    us_df = storage.load_price_data(us_symbol, category=category)
    cn_df = storage.load_price_data(cn_symbol, category=category)

    # Optional: auto-fetch price data up to needed evaluation horizon.
    max_pub = _max_pub_date(preds)
    if auto_fetch and max_pub:
        required_end = (_parse_date(max_pub) + dt.timedelta(days=max(1, int(max_lag_days)))).strftime("%Y-%m-%d")
        fetcher = DataFetcher(source=str(fetch_source))
        us_df = _ensure_price_coverage(storage=storage, fetcher=fetcher, symbol=us_symbol, category=category, required_end_date=required_end)
        cn_df = _ensure_price_coverage(storage=storage, fetcher=fetcher, symbol=cn_symbol, category=category, required_end_date=required_end)

    if us_df is None:
        logger.error(f"Missing US price data for {us_symbol} in data/{category}")
        return 3
    if cn_df is None:
        logger.error(f"Missing CN price data for {cn_symbol} in data/{category}")
        return 3

    us_ret = _build_daily_returns(us_df)
    cn_ret = _build_daily_returns(cn_df)

    logger.info(f"US symbol={us_symbol} last_return_date={_last_available_date(us_ret)}")
    logger.info(f"CN symbol={cn_symbol} last_return_date={_last_available_date(cn_ret)}")

    if preds:
        pub_dates = sorted({p["pub_date"] for p in preds if p.get("pub_date")})
        if pub_dates:
            logger.info(f"Signal pub_date range: {pub_dates[0]} .. {pub_dates[-1]}")

    total = 0
    wins = 0
    skipped = 0

    by_market: Dict[str, Dict[str, int]] = {"US": {"total": 0, "wins": 0, "skipped": 0}, "CN": {"total": 0, "wins": 0, "skipped": 0}}

    rows: List[Dict[str, Any]] = []

    for p in preds:
        market = p["market"]
        pub_date = p["pub_date"]
        impact = int(p["impact_equity"])

        if market == "CN":
            match_date, real = _find_tplus1_return(pub_date, cn_ret, max_lag_days=max_lag_days)
        else:
            match_date, real = _find_tplus1_return(pub_date, us_ret, max_lag_days=max_lag_days)

        if real is None or match_date is None:
            skipped += 1
            by_market["CN" if market == "CN" else "US"]["skipped"] += 1
            continue

        is_win = (impact > 0 and real > 0) or (impact < 0 and real < 0)
        total += 1
        wins += 1 if is_win else 0

        mkey = "CN" if market == "CN" else "US"
        by_market[mkey]["total"] += 1
        by_market[mkey]["wins"] += 1 if is_win else 0

        rows.append(
            {
                "match_date": match_date,
                "market": mkey,
                "ai": impact,
                "real_pct": real,
                "win": is_win,
                "event_type": p.get("event_type"),
                "title": p.get("title"),
            }
        )

    df_out = pd.DataFrame(rows)

    if sample_n > 0 and not df_out.empty:
        show = df_out.head(int(sample_n)).copy()
        print(f"{'MatchDate':<12} | {'Mkt':<2} | {'AI':>3} | {'Real%':>8} | {'Win':<4} | {'event_type':<22} | title")
        print("-" * 120)
        for _, r in show.iterrows():
            w = "WIN" if bool(r["win"]) else "LOSS"
            print(
                f"{str(r['match_date']):<12} | {str(r['market']):<2} | {int(r['ai']):>3} | {float(r['real_pct']):>7.2f}% | {w:<4} | {str(r['event_type']):<22} | {str(r['title'])[:80]}"
            )

    if total == 0:
        logger.warning("No evaluatable signals (impact_equity!=0) matched to market data")
        logger.info(f"Skipped (no market data match): {skipped}")
        return 0

    acc = wins / total * 100.0
    logger.info(f"Total evaluated: {total}")
    logger.info(f"Wins: {wins}")
    logger.info(f"Accuracy: {acc:.2f}%")
    logger.info(f"Skipped (no market data match): {skipped}")

    for m in ("US", "CN"):
        mt = by_market[m]["total"]
        mw = by_market[m]["wins"]
        ms = by_market[m]["skipped"]
        if mt > 0:
            logger.info(f"{m} accuracy: {mw}/{mt} = {mw/mt*100.0:.2f}% (skipped={ms})")
        else:
            logger.info(f"{m} accuracy: n/a (skipped={ms})")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate AI signals vs market ground truth (T+1 alignment)")
    parser.add_argument("--signals", default=None, help="Signals JSON path (default: data/daily/signals_YYYY-MM-DD.json)")
    parser.add_argument("--date", default=None, help="Date for default signals path (YYYY-MM-DD)")
    parser.add_argument("--us-symbol", default="SPY")
    parser.add_argument("--cn-symbol", default="510300")
    parser.add_argument("--category", default="raw", choices=["raw", "processed", "cache", "features"])
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--max-lag-days", type=int, default=7)
    parser.add_argument("--auto-fetch", action="store_true", help="Fetch/extend market price data automatically (network)")
    parser.add_argument("--fetch-source", default="yfinance", choices=["yfinance", "akshare"], help="DataFetcher source used by --auto-fetch")
    args = parser.parse_args()

    date_str = args.date or _today_str()
    signals_path = Path(args.signals) if args.signals else Path("data/daily") / f"signals_{date_str}.json"

    code = evaluate(
        signals_path,
        us_symbol=str(args.us_symbol),
        cn_symbol=str(args.cn_symbol),
        category=str(args.category),
        sample_n=int(args.sample),
        max_lag_days=int(args.max_lag_days),
        auto_fetch=bool(args.auto_fetch),
        fetch_source=str(args.fetch_source),
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()

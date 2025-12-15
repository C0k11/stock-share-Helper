#!/usr/bin/env python

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.storage import DataStorage
from src.features.technical import TechnicalFeatures
from src.features.regime import RegimeDetector
from src.strategy.signals import SignalGenerator
from src.strategy.position import PositionSizer


def _today_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")


def _parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def _as_date_str(x: Any) -> str:
    try:
        if isinstance(x, (dt.date, dt.datetime)):
            return x.strftime("%Y-%m-%d")
        return str(x)[:10]
    except Exception:
        return ""


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        v = int(x)
        return v
    except Exception:
        return None


def _roundf(x: Optional[float], nd: int = 4) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return None


def _compute_one(
    *,
    symbol: str,
    df: pd.DataFrame,
    asof_date: dt.date,
    regime_detector: RegimeDetector,
    spy_df: Optional[pd.DataFrame],
    vix_df: Optional[pd.DataFrame],
    risk_profile: str,
) -> Dict[str, Any]:
    if df is None or df.empty:
        raise ValueError(f"empty price data for {symbol}")

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    df = df[df.index.date <= asof_date]
    if df.empty:
        raise ValueError(f"no data on or before {asof_date} for {symbol}")

    last_dt = df.index[-1]
    last_date = _as_date_str(last_dt)

    close = _safe_float(df["close"].iloc[-1])

    tech = TechnicalFeatures(df).add_all().get_latest()

    # Normalize key indicators
    features = {
        "date": last_date,
        "close": _roundf(close, 4),
        "price_vs_ma20": _roundf(_safe_float(tech.get("price_vs_ma20")), 4),
        "price_vs_ma200": _roundf(_safe_float(tech.get("price_vs_ma200")), 4),
        "trend_alignment": _roundf(_safe_float(tech.get("trend_alignment")), 4),
        "breakout_20d_high": _safe_int(tech.get("breakout_20d_high")),
        "breakdown_20d_low": _safe_int(tech.get("breakdown_20d_low")),
        "return_5d": _roundf(_safe_float(tech.get("return_5d")), 6),
        "return_21d": _roundf(_safe_float(tech.get("return_21d")), 6),
        "return_63d": _roundf(_safe_float(tech.get("return_63d")), 6),
        "volatility_20d": _roundf(_safe_float(tech.get("volatility_20d")), 6),
        "vol_ratio": _roundf(_safe_float(tech.get("vol_ratio")), 4),
        "drawdown": _roundf(_safe_float(tech.get("drawdown")), 6),
        "max_drawdown_20d": _roundf(_safe_float(tech.get("max_drawdown_20d")), 6),
        "max_drawdown_60d": _roundf(_safe_float(tech.get("max_drawdown_60d")), 6),
    }

    # Regime (global, based on SPY/VIX when available)
    regime = None
    regime_score = None
    if spy_df is not None and not spy_df.empty:
        try:
            spy_asof = spy_df
            vix_asof = vix_df
            if isinstance(spy_asof.index, pd.DatetimeIndex):
                spy_asof = spy_asof[spy_asof.index.date <= asof_date]
            if vix_asof is not None and isinstance(vix_asof.index, pd.DatetimeIndex):
                vix_asof = vix_asof[vix_asof.index.date <= asof_date]
            rr = regime_detector.get_current_regime(spy_asof, vix_asof)
            regime = rr.get("regime")
            regime_score = rr.get("score")
        except Exception:
            regime = None
            regime_score = None

    # Signal strength (per-symbol)
    sg = SignalGenerator()
    sig = sg.get_current_signal(df)
    strength = sig.get("strength")

    # Target position (teacher-style)
    ps = PositionSizer()
    pos_df = ps.compute_final_position(
        df,
        signal_strength=pd.Series([strength], index=[df.index[-1]]),
        regime=pd.Series([regime or "transition"], index=[df.index[-1]]),
    )
    target_position = float(pos_df["target_position"].iloc[-1])
    target_position_profiled = float(ps.apply_risk_profile(target_position, profile=risk_profile))

    return {
        "symbol": symbol,
        "date": last_date,
        "market_regime": {
            "regime": regime,
            "score": _safe_float(regime_score),
        },
        "technical": features,
        "signal": {
            "strength": strength,
            "trend": sig.get("trend"),
            "momentum": sig.get("momentum"),
            "ma_cross": sig.get("ma_cross"),
            "breakout": sig.get("breakout"),
            "composite": _safe_float(sig.get("composite")),
        },
        "teacher": {
            "target_position": _roundf(target_position, 4),
            "target_position_profiled": _roundf(target_position_profiled, 4),
            "risk_profile": risk_profile,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build daily ETF/index feature snapshot (technical + regime + teacher target)")
    parser.add_argument("--date", default=None, help="Date label for output filename YYYY-MM-DD (default: today)")
    parser.add_argument("--start-date", default=None, help="Batch mode: start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end-date", default=None, help="Batch mode: end date YYYY-MM-DD (inclusive)")
    parser.add_argument("--outdir", default="data/daily")
    parser.add_argument("--category", default="raw", choices=["raw", "processed", "cache", "features"])
    parser.add_argument(
        "--symbols",
        default="SPY,QQQ,TLT,GLD,510300,510500,510050",
        help="Comma-separated ETF/index symbols (must exist in data/<category>/*.parquet)",
    )
    parser.add_argument("--risk-profile", default="balanced", choices=["conservative", "balanced", "aggressive"])
    args = parser.parse_args()

    if (args.start_date or args.end_date) and args.date:
        raise SystemExit("Use either --date or --start-date/--end-date")

    date_str = args.date or _today_str()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    storage = DataStorage(base_path="data")

    # Load SPY/VIX for regime
    spy_df = storage.load_price_data("SPY", category=args.category)
    vix_df = storage.load_price_data("VIX", category=args.category)
    regime_detector = RegimeDetector()

    price_map: Dict[str, pd.DataFrame] = {}
    all_dates: Set[dt.date] = set()
    for sym in symbols:
        df = storage.load_price_data(sym, category=args.category)
        if df is None or df.empty:
            logger.warning(f"Missing parquet for {sym} under data/{args.category}")
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        price_map[sym] = df
        all_dates.update(set(df.index.date))

    base_calendar_dates: Set[dt.date] = set()
    spy_base = price_map.get("SPY")
    if spy_base is not None and not spy_base.empty and isinstance(spy_base.index, pd.DatetimeIndex):
        base_calendar_dates = set(spy_base.index.date)
    else:
        base_calendar_dates = set(all_dates)

    if args.start_date or args.end_date:
        if not args.start_date or not args.end_date:
            raise SystemExit("Batch mode requires both --start-date and --end-date")
        start_d = _parse_date(str(args.start_date))
        end_d = _parse_date(str(args.end_date))
        if start_d > end_d:
            raise SystemExit("start-date must be <= end-date")
        target_dates = sorted([d for d in base_calendar_dates if start_d <= d <= end_d])
        if not target_dates:
            logger.warning(f"No trading dates found in range {start_d}..{end_d}")
            return
    else:
        target_dates = [_parse_date(date_str)]

    for td in target_dates:
        out_path = outdir / f"etf_features_{td.strftime('%Y-%m-%d')}.json"
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for sym in symbols:
            try:
                df = price_map.get(sym)
                if df is None or df.empty:
                    raise ValueError(f"missing parquet for {sym} under data/{args.category}")
                rec = _compute_one(
                    symbol=sym,
                    df=df,
                    asof_date=td,
                    regime_detector=regime_detector,
                    spy_df=spy_df,
                    vix_df=vix_df,
                    risk_profile=str(args.risk_profile),
                )
                results.append(rec)
            except Exception as e:
                logger.warning(f"Failed to build features for {sym} at {td}: {e}")
                errors.append({"symbol": sym, "error": str(e)})

        payload = {
            "date": td.strftime("%Y-%m-%d"),
            "generated_at": dt.datetime.now().astimezone().isoformat(),
            "category": str(args.category),
            "symbols": symbols,
            "items": results,
            "errors": errors,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved ETF features: {out_path} items={len(results)} errors={len(errors)}")


if __name__ == "__main__":
    main()

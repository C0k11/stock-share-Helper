#!/usr/bin/env python

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
            rr = regime_detector.get_current_regime(spy_df, vix_df)
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
    parser.add_argument("--outdir", default="data/daily")
    parser.add_argument("--category", default="raw", choices=["raw", "processed", "cache", "features"])
    parser.add_argument(
        "--symbols",
        default="SPY,QQQ,TLT,GLD,510300,510500,510050",
        help="Comma-separated ETF/index symbols (must exist in data/<category>/*.parquet)",
    )
    parser.add_argument("--risk-profile", default="balanced", choices=["conservative", "balanced", "aggressive"])
    args = parser.parse_args()

    date_str = args.date or _today_str()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"etf_features_{date_str}.json"

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    storage = DataStorage(base_path="data")

    # Load SPY/VIX for regime
    spy_df = storage.load_price_data("SPY", category=args.category)
    vix_df = storage.load_price_data("VIX", category=args.category)
    regime_detector = RegimeDetector()

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for sym in symbols:
        try:
            df = storage.load_price_data(sym, category=args.category)
            if df is None or df.empty:
                raise ValueError(f"missing parquet for {sym} under data/{args.category}")
            rec = _compute_one(
                symbol=sym,
                df=df,
                regime_detector=regime_detector,
                spy_df=spy_df,
                vix_df=vix_df,
                risk_profile=str(args.risk_profile),
            )
            results.append(rec)
        except Exception as e:
            logger.warning(f"Failed to build features for {sym}: {e}")
            errors.append({"symbol": sym, "error": str(e)})

    payload = {
        "date": date_str,
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

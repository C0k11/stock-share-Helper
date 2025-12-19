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
from src.features.regime import RegimeDetector
from src.features.technical import TechnicalFeatures
from src.strategy.position import PositionSizer
from src.strategy.signals import SignalGenerator


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
        return int(x)
    except Exception:
        return None


def _roundf(x: Optional[float], nd: int = 4) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return None


def _normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna(axis=0, subset=["close"]) if "close" in df.columns else df

    df = df.sort_index()
    return df


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

    df = _normalize_price_df(df)
    df = df[df.index.date <= asof_date]
    if df.empty:
        raise ValueError(f"no data on or before {asof_date} for {symbol}")

    required = {"close", "high", "low", "volume"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns for {symbol}: {missing}")

    last_dt = df.index[-1]
    last_date = _as_date_str(last_dt)

    close = _safe_float(df["close"].iloc[-1])

    tech = TechnicalFeatures(df).add_all().get_latest()

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

    regime = None
    regime_score = None
    if spy_df is not None and not spy_df.empty:
        try:
            spy_asof = _normalize_price_df(spy_df)
            vix_asof = _normalize_price_df(vix_df) if vix_df is not None else None
            spy_asof = spy_asof[spy_asof.index.date <= asof_date]
            if vix_asof is not None:
                vix_asof = vix_asof[vix_asof.index.date <= asof_date]
            rr = regime_detector.get_current_regime(spy_asof, vix_asof)
            regime = rr.get("regime")
            regime_score = rr.get("score")
        except Exception:
            regime = None
            regime_score = None

    sg = SignalGenerator()
    sig = sg.get_current_signal(df)
    strength = sig.get("strength")

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


def _load_tickers(path: Path) -> List[str]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily asset feature snapshot (technical + regime + teacher target)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--category", default="raw", choices=["raw", "processed", "cache", "features"])
    parser.add_argument("--tickers-file", required=True)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", required=True)
    parser.add_argument("--risk-profile", default="balanced", choices=["conservative", "balanced", "aggressive"])
    args = parser.parse_args()

    td = _parse_date(str(args.date))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    storage = DataStorage(base_path=str(args.data_dir))

    tickers = _load_tickers(Path(args.tickers_file))
    if not tickers:
        raise SystemExit("No tickers loaded.")

    spy_df = storage.load_price_data("SPY", category=str(args.category))
    vix_df = storage.load_price_data("VIX", category=str(args.category))
    regime_detector = RegimeDetector()

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for sym in tickers:
        try:
            df = storage.load_price_data(sym, category=str(args.category))
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
        "tickers_file": str(args.tickers_file),
        "items": results,
        "errors": errors,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved features: {out_path} items={len(results)} errors={len(errors)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import sys
import argparse
import hashlib
import json
import random
import pandas as pd
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def conversation_key(item: dict) -> str:
    meta = item.get("meta") or {}
    basis = "|".join(
        [
            str(meta.get("case_type", "")),
            str(meta.get("market", "")),
            str(meta.get("symbol", "")),
            str(meta.get("bar_interval", "")),
            str(meta.get("t0", "")),
        ]
    )
    if basis.strip("|"):
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()

    convs = item.get("conversations") or []
    user_text = ""
    for msg in convs:
        if msg.get("role") == "user":
            user_text = (msg.get("content") or "").strip()
            break
    norm = " ".join(user_text.split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def load_existing(out_path: Path):
    if not out_path.exists():
        return []
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"Failed to load existing dataset for append: {e}")
        return []


def parse_published_at(item: dict) -> datetime:
    meta = item.get("meta") or {}
    s = (meta.get("published_at") or "").strip()
    if not s:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        dt = parsedate_to_datetime(s)
    except Exception:
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def classify_risk_level(regime: str, features_latest: dict) -> str:
    vol20 = features_latest.get("volatility_20d")
    dd = features_latest.get("drawdown")

    if regime == "risk_off":
        return "high"

    if isinstance(dd, (int, float)) and dd <= -0.15:
        return "high"

    if isinstance(vol20, (int, float)) and vol20 >= 0.25:
        return "high"

    if isinstance(vol20, (int, float)) and vol20 >= 0.18:
        return "medium"

    return "low"


def main():
    parser = argparse.ArgumentParser(description="Build trading case finetune dataset (strict JSON)")
    parser.add_argument("--out", default="data/finetune/cases.json", help="输出json路径")
    parser.add_argument("--append", action="store_true", help="如果输出文件已存在，则追加新样本而不是覆盖")
    parser.add_argument("--dedup", action="store_true", help="合并时按case key去重")
    parser.add_argument("--split-val", action="store_true", help="同时输出验证集 val.json（从合并后的样本中切分）")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n", type=int, default=200, help="要生成的case数量")
    parser.add_argument("--us-ratio", type=float, default=0.60, help="US market ratio")

    parser.add_argument("--source", default="yfinance", choices=["yfinance", "akshare"], help="行情数据源")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--window", type=int, default=60, help="输入窗口bars数")

    parser.add_argument(
        "--us-symbols",
        nargs="+",
        default=["SPY", "QQQ", "IWM", "TLT", "GLD"],
        help="US symbols",
    )
    parser.add_argument(
        "--cn-symbols",
        nargs="+",
        default=["510300", "510500", "510050", "159915"],
        help="CN symbols (6-digit)",
    )

    args = parser.parse_args()

    random.seed(int(args.seed))

    if args.interval != "1d":
        raise SystemExit("This builder currently supports --interval 1d only")

    from src.data.fetcher import DataFetcher
    from src.data.storage import DataStorage
    from src.features.technical import TechnicalFeatures
    from src.features.regime import RegimeDetector
    from src.strategy.signals import SignalGenerator
    from src.strategy.position import PositionSizer
    from src.llm.finetune.dataset import FineTuneDataset

    fetcher = DataFetcher(source=args.source)
    storage = DataStorage(base_path="data")

    def load_or_fetch(symbol: str):
        df = storage.load_price_data(symbol)
        if df is not None:
            return df
        data = fetcher.fetch_price([symbol], start_date=args.start, end_date=args.end, interval=args.interval)
        df2 = data.get(str(symbol).strip())
        if df2 is None or df2.empty:
            return None
        storage.save_price_data(symbol, df2)
        return df2

    regime_detector = RegimeDetector()
    signal_gen = SignalGenerator()
    position_sizer = PositionSizer(target_volatility=0.10)

    spy_df = load_or_fetch("SPY")
    cn_regime_df = load_or_fetch("510300")

    ds = FineTuneDataset(data_path=str(Path(args.out).parent))

    min_required = max(int(args.window) + 5, 260)

    generated = 0
    attempts = 0
    max_attempts = int(args.n) * 20

    while generated < int(args.n) and attempts < max_attempts:
        attempts += 1

        use_us = random.random() < float(args.us_ratio)
        market = "US" if use_us else "CN"
        symbol = random.choice(args.us_symbols if use_us else args.cn_symbols)

        df = load_or_fetch(symbol)
        if df is None or df.empty or len(df) < min_required:
            continue

        idx_max = len(df) - 2
        if idx_max <= min_required:
            continue

        i = random.randint(min_required, idx_max)
        t0 = df.index[i]

        df_slice = df.iloc[: i + 1].copy()
        if len(df_slice) < min_required:
            continue

        try:
            sig = signal_gen.get_current_signal(df_slice)
            strength = sig.get("strength", "neutral")
        except Exception:
            strength = "neutral"
            sig = {}

        try:
            feats = TechnicalFeatures(df_slice).add_all().get_latest()
        except Exception:
            feats = {}

        if market == "US" and spy_df is not None and len(spy_df) > 60:
            r_slice = spy_df.loc[:t0]
        elif market == "CN" and cn_regime_df is not None and len(cn_regime_df) > 60:
            r_slice = cn_regime_df.loc[:t0]
        else:
            r_slice = None

        if r_slice is not None and len(r_slice) > 60:
            try:
                regime = regime_detector.get_current_regime(r_slice).get("regime", "transition")
            except Exception:
                regime = "transition"
        else:
            regime = "transition"

        if r_slice is not None and len(r_slice) >= len(df_slice):
            try:
                regime_hist = regime_detector.get_regime_history(r_slice).reindex(df_slice.index, method="ffill")
                regime_series = regime_hist["regime"]
            except Exception:
                regime_series = None
        else:
            regime_series = None

        try:
            signals_df = signal_gen.generate(df_slice)
            signal_strength_series = signals_df["signal_strength"]
        except Exception:
            signal_strength_series = None

        if regime_series is None:
            regime_series = df_slice["close"].copy()
            regime_series[:] = "transition"

        if signal_strength_series is None:
            signal_strength_series = df_slice["close"].copy()
            signal_strength_series[:] = "neutral"

        try:
            pos_df = position_sizer.compute_final_position(df_slice, signal_strength_series, regime_series)
            position_target = float(pos_df["target_position"].iloc[-1])
        except Exception:
            position_target = 0.3

        risk_level = classify_risk_level(regime, feats)

        actions = []
        if risk_level == "high" or regime == "risk_off":
            actions.append("reduce")
        if strength in ("strong_long", "weak_long") and risk_level != "high":
            actions.append("buy")
        elif strength in ("strong_short", "weak_short"):
            actions.append("sell")
        if not actions:
            actions.append("hold")

        window_df = df_slice.tail(int(args.window))[["open", "high", "low", "close", "volume"]].copy()
        window_csv = window_df.to_csv(index=True).strip()

        feat_keep = {k: feats.get(k) for k in ["price_vs_ma20", "price_vs_ma200", "volatility_20d", "drawdown", "trend_alignment"] if k in feats}

        t0_iso = pd.to_datetime(t0).to_pydatetime().replace(tzinfo=timezone.utc).isoformat()

        user = (
            f"Market: {market}\n"
            f"Symbol: {symbol}\n"
            f"Interval: {args.interval}\n"
            f"t0: {t0_iso}\n\n"
            f"Recent OHLCV (last {int(args.window)} bars):\n{window_csv}\n\n"
            f"Technical (latest): {json.dumps(feat_keep, ensure_ascii=False)}\n"
            f"Signal (latest): {json.dumps({'strength': strength, 'composite': sig.get('composite')}, ensure_ascii=False)}\n\n"
            "Output exactly ONE JSON object with these fields: "
            "risk_level (low/medium/high), regime (risk_on/transition/risk_off), "
            "signal_strength (strong_long/weak_long/neutral/weak_short/strong_short), "
            "position_target (0~1), actions (array), invalidation (string), summary (string). "
            "Do not include any additional keys."
        )

        invalidation = "If trend reverses (price falls below key moving averages) or volatility spikes, reduce exposure and reassess."
        summary = f"{market} {symbol}: regime={regime}, signal={strength}, risk={risk_level}."

        output_obj = {
            "risk_level": risk_level,
            "regime": regime,
            "signal_strength": strength,
            "position_target": round(max(0.0, min(1.0, position_target)), 3),
            "actions": actions,
            "invalidation": invalidation,
            "summary": summary,
        }

        meta = {
            "case_type": "trading_case",
            "market": market,
            "symbol": str(symbol),
            "bar_interval": str(args.interval),
            "t0": t0_iso,
            "published_at": t0_iso,
        }

        ds.add_trading_case_sample(context=user, output_json=output_obj, meta=meta)
        generated += 1

    logger.info(f"Generated trading cases: {generated} (attempts={attempts})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_data = ds.to_conversation_format()

    if args.append and out_path.exists():
        existing_train = load_existing(out_path)
        existing_val = []
        if args.split_val:
            val_path = out_path.parent / "cases_val.json"
            existing_val = load_existing(val_path)

        merged = existing_train + existing_val + new_data
        if args.dedup:
            seen = set()
            deduped = []
            for item in merged:
                key = conversation_key(item)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
            merged = deduped
            logger.info(f"Dedup applied: total={len(merged)}")

        final_data = merged
    else:
        final_data = new_data

    if args.split_val:
        data = list(final_data)
        data.sort(key=lambda x: (parse_published_at(x), conversation_key(x)))
        val_n = int(len(data) * float(args.val_ratio))
        val_n = max(1, val_n) if len(data) >= 2 else 0
        val_data = data[-val_n:] if val_n > 0 else []
        train_data = data[:-val_n] if val_n > 0 else data

        val_path = out_path.parent / "cases_val.json"
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved train={len(train_data)} to {out_path}")
        logger.info(f"Saved val={len(val_data)} to {val_path}")
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(final_data)} to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


def _load_tickers(*, tickers_file: str, tickers_csv: str) -> List[str]:
    tickers: List[str] = []

    if str(tickers_csv or "").strip():
        for t in str(tickers_csv).split(","):
            s = str(t or "").strip().upper()
            if s:
                tickers.append(s)

    if not tickers:
        p = Path(str(tickers_file))
        if not p.exists():
            raise SystemExit(f"Ticker file not found: {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            s = (line or "").strip()
            s = s.lstrip("\ufeff")
            if (not s) or s.startswith("#"):
                continue
            tickers.append(s.upper())

    return list(dict.fromkeys([t for t in tickers if t]))


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date")
        out = out.set_index("date")
    else:
        try:
            out.index = pd.to_datetime(out.index, errors="coerce")
            out = out.dropna(axis=0, how="any")
            out = out.sort_index()
        except Exception:
            return pd.DataFrame()

    rename_map: Dict[str, str] = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj_close": "Adj Close",
        "adjclose": "Adj Close",
        "volume": "Volume",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    for req in ["Open", "High", "Low", "Close"]:
        if req not in out.columns:
            return pd.DataFrame()

    if "Volume" not in out.columns:
        out["Volume"] = 0.0

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    out = out[keep]

    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"]).copy()

    return out


def _slice_lookback(df: pd.DataFrame, *, asof: str, lookback: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    asof_dt = pd.to_datetime(str(asof), errors="coerce")
    if pd.isna(asof_dt):
        return pd.DataFrame()

    df2 = df[df.index <= asof_dt].copy()
    if df2.empty:
        return pd.DataFrame()

    df2 = df2.tail(int(lookback)).copy()
    return df2


def _slice_history_for_indicators(df: pd.DataFrame, *, asof: str, bars: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    asof_dt = pd.to_datetime(str(asof), errors="coerce")
    if pd.isna(asof_dt):
        return pd.DataFrame()

    df2 = df[df.index <= asof_dt].copy()
    if df2.empty:
        return pd.DataFrame()

    return df2.tail(int(bars)).copy()


def _load_ohlcv_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"OHLCV csv not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise SystemExit(f"Failed to read OHLCV csv: {path}") from e
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _infer_ticker_col(df: pd.DataFrame) -> str:
    for c in ["ticker", "symbol"]:
        if c in df.columns:
            return c
    return ""


def _filter_csv_for_ticker(df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    tcol = _infer_ticker_col(df)
    if not tcol:
        return pd.DataFrame()
    sub = df[df[tcol].astype(str).str.upper().str.strip() == str(ticker).upper().strip()].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.drop(columns=[tcol], errors="ignore")
    return sub


def _compute_overlays(df: pd.DataFrame) -> Dict[str, pd.Series]:
    close = df["Close"].astype(float)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    mid = sma20
    sd = close.rolling(20).std()
    upper = mid + 2.0 * sd
    lower = mid - 2.0 * sd

    return {
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "bb_mid": mid,
        "bb_upper": upper,
        "bb_lower": lower,
    }


def _has_any_value(s: Optional[pd.Series]) -> bool:
    if s is None:
        return False
    try:
        return int(s.dropna().shape[0]) > 0
    except Exception:
        return False


def _ensure_mpl_backend() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
    except Exception:
        return


def _plot_one(
    *,
    df: pd.DataFrame,
    overlays: Dict[str, pd.Series],
    ticker: str,
    out_png: Path,
    title: str,
    include_bbands: bool,
    include_sma: bool,
    verbose: bool,
) -> bool:
    if df is None or df.empty:
        return False

    try:
        import mplfinance as mpf
    except Exception as e:
        raise SystemExit(
            "mplfinance is required. Install dependencies (e.g. pip install -r requirements.txt)."
        ) from e

    addplots = []
    if include_sma:
        sma20 = overlays.get("sma20")
        sma50 = overlays.get("sma50")
        sma200 = overlays.get("sma200")
        if _has_any_value(sma20):
            addplots.append(mpf.make_addplot(sma20, color="#4C78A8", width=1))
        if _has_any_value(sma50):
            addplots.append(mpf.make_addplot(sma50, color="#F58518", width=1))
        if _has_any_value(sma200):
            addplots.append(mpf.make_addplot(sma200, color="#54A24B", width=1))

    if include_bbands:
        bb_upper = overlays.get("bb_upper")
        bb_mid = overlays.get("bb_mid")
        bb_lower = overlays.get("bb_lower")
        if _has_any_value(bb_upper):
            addplots.append(mpf.make_addplot(bb_upper, color="#B279A2", width=1))
        if _has_any_value(bb_mid):
            addplots.append(mpf.make_addplot(bb_mid, color="#9D755D", width=1))
        if _has_any_value(bb_lower):
            addplots.append(mpf.make_addplot(bb_lower, color="#B279A2", width=1))

    out_png.parent.mkdir(parents=True, exist_ok=True)

    try:
        mpf.plot(
            df,
            type="candle",
            style="yahoo",
            volume=True,
            addplot=addplots if addplots else None,
            title=title,
            figsize=(10, 6),
            tight_layout=True,
            savefig=dict(fname=str(out_png), dpi=160, pad_inches=0.15),
        )
        return True
    except Exception as e:
        if bool(verbose):
            print(f"[plot_failed] ticker={ticker} err={repr(e)}")
        return False


def _png_to_b64(path: Path) -> str:
    b = path.read_bytes()
    return base64.b64encode(b).decode("ascii")


def _iter_dates(df: pd.DataFrame) -> Tuple[str, str]:
    if df is None or df.empty:
        return "", ""
    try:
        a = str(pd.to_datetime(df.index.min()).date())
        b = str(pd.to_datetime(df.index.max()).date())
        return a, b
    except Exception:
        return "", ""


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--tickers-file", default="data/tickers/phase6_expansion_plus_default.txt")
    ap.add_argument("--tickers", default="", help="Comma-separated tickers override")

    ap.add_argument("--raw-dir", default="data/raw", help="Folder containing TICKER.parquet")
    ap.add_argument("--ohlcv-csv", default="", help="Optional OHLCV csv (must include ticker/symbol column)")
    ap.add_argument("--out-dir", default="data/charts", help="Root output dir")

    ap.add_argument("--asof", required=True, help="As-of date YYYY-MM-DD")
    ap.add_argument("--lookback", type=int, default=60, help="Lookback bars (trading days)")

    ap.add_argument("--include-sma", action="store_true", default=True)
    ap.add_argument("--no-include-sma", dest="include_sma", action="store_false")

    ap.add_argument("--include-bbands", action="store_true", default=True)
    ap.add_argument("--no-include-bbands", dest="include_bbands", action="store_false")

    ap.add_argument("--out-jsonl", default="", help="Optional jsonl with base64 images")
    ap.add_argument("--verbose", action="store_true", default=False)

    args = ap.parse_args()

    _ensure_mpl_backend()

    tickers = _load_tickers(tickers_file=str(args.tickers_file), tickers_csv=str(args.tickers))

    raw_dir = Path(str(args.raw_dir))
    out_root = Path(str(args.out_dir)) / str(args.asof)

    df_csv: pd.DataFrame = pd.DataFrame()
    if str(args.ohlcv_csv or "").strip():
        df_csv = _load_ohlcv_csv(Path(str(args.ohlcv_csv)))

    rows_jsonl: List[Dict[str, Any]] = []

    ok = 0
    failed = 0
    skipped = 0

    for t in tickers:
        df_raw: Optional[pd.DataFrame] = None
        if not df_csv.empty:
            df_raw = _filter_csv_for_ticker(df_csv, ticker=t)
        else:
            fp = raw_dir / f"{t}.parquet"
            if not fp.exists():
                skipped += 1
                continue
            try:
                df_raw = pd.read_parquet(fp)
            except Exception:
                failed += 1
                continue

        if df_raw is None or df_raw.empty:
            skipped += 1
            continue

        df = _normalize_ohlcv(df_raw)
        lookback = int(args.lookback)
        hist_bars = max(lookback, 220)
        df_hist = _slice_history_for_indicators(df, asof=str(args.asof), bars=hist_bars)
        df60 = df_hist.tail(lookback).copy() if (df_hist is not None and not df_hist.empty) else pd.DataFrame()

        if df60.empty or len(df60) < max(10, lookback // 3):
            skipped += 1
            continue

        overlays_full = _compute_overlays(df_hist)
        overlays_aligned = {k: v.reindex(df60.index) for k, v in overlays_full.items()}

        start_d, end_d = _iter_dates(df60)
        title = f"{t} {start_d}..{end_d}"

        out_png = out_root / f"{t}.png"
        if _plot_one(
            df=df60,
            overlays=overlays_aligned,
            ticker=t,
            out_png=out_png,
            title=title,
            include_bbands=bool(args.include_bbands),
            include_sma=bool(args.include_sma),
            verbose=bool(args.verbose),
        ):
            ok += 1
            if str(args.out_jsonl or "").strip():
                rows_jsonl.append(
                    {
                        "ticker": t,
                        "asof": str(args.asof),
                        "lookback": int(args.lookback),
                        "start": start_d,
                        "end": end_d,
                        "png_path": str(out_png.as_posix()),
                        "image_base64": _png_to_b64(out_png),
                    }
                )
        else:
            failed += 1

    if str(args.out_jsonl or "").strip():
        out_jsonl = Path(str(args.out_jsonl))
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as f:
            for r in rows_jsonl:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "asof": str(args.asof),
                "lookback": int(args.lookback),
                "tickers": len(tickers),
                "ok": int(ok),
                "failed": int(failed),
                "skipped": int(skipped),
                "out_dir": str(out_root.as_posix()),
                "out_jsonl": str(args.out_jsonl or ""),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.strategy.execution import TickerExecutionState


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _fmt_date(x: Any) -> str:
    return str(x or "").strip()[:10]


def _max_drawdown(values: List[float]) -> float:
    if not values:
        return float("nan")
    peak = values[0]
    mdd = 0.0
    for v in values:
        if v > peak:
            peak = v
        if peak > 0:
            dd = v / peak - 1.0
            if dd < mdd:
                mdd = dd
    return float(mdd)


def _load_price_df(raw_dir: Path, ticker: str):
    import pandas as pd

    fp = raw_dir / f"{ticker}.parquet"
    if not fp.exists():
        return None

    try:
        df = pd.read_parquet(fp)
    except Exception:
        return None

    if df is None or df.empty:
        return None

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    if "close" not in df.columns:
        return None

    df = df.dropna(subset=["close"]).sort_index()
    df["close"] = df["close"].astype(float)
    return df


def _get_close_on_or_before(df, date_str: str) -> Optional[float]:
    import pandas as pd

    if df is None or df.empty:
        return None

    target = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(target):
        return None

    idx = df.index
    try:
        mask = idx <= target
        if not mask.any():
            return None
        return float(df.loc[mask, "close"].iloc[-1])
    except Exception:
        return None


def _get_close_exact(df, date_str: str) -> Optional[float]:
    import pandas as pd

    if df is None or df.empty:
        return None

    target = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(target):
        return None

    try:
        matches = df.loc[df.index.normalize() == target.normalize(), "close"]
        if matches.empty:
            return None
        return float(matches.iloc[0])
    except Exception:
        return None


@dataclass
class TradeEvent:
    date: str
    ticker: str
    action: str
    shares_delta: float
    price: float
    trade_value: float
    fee: float


@dataclass
class DailyNav:
    date: str
    cash: float
    positions_value: float
    total_value: float


def _parse_report(report_path: Path) -> Tuple[List[str], Dict[str, List[Dict[str, Any]]]]:
    obj = _read_json(report_path)
    if not isinstance(obj, dict):
        raise ValueError("report is not a dict")

    strategies = obj.get("strategies")
    if not isinstance(strategies, dict):
        raise ValueError("report missing strategies")

    tickers = obj.get("tickers")
    if not isinstance(tickers, list) or not tickers:
        tickers = []

    out: Dict[str, List[Dict[str, Any]]] = {}
    for name, content in strategies.items():
        if not isinstance(content, dict):
            continue
        trades = content.get("trades")
        if not isinstance(trades, list):
            continue
        out[str(name)] = [t for t in trades if isinstance(t, dict)]

    return [str(t).upper() for t in tickers], out


def _decisions_by_day(trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for t in trades:
        day = _fmt_date(t.get("date"))
        sym = str(t.get("ticker") or "").strip().upper()
        dec = str(t.get("decision") or "HOLD").strip().upper()
        if not day or not sym:
            continue
        if dec not in {"BUY", "SELL", "HOLD"}:
            dec = "HOLD"
        out.setdefault(day, {})[sym] = dec
    return out


def simulate_nav(
    *,
    days: List[str],
    tickers: List[str],
    decisions: Dict[str, Dict[str, str]],
    price_dfs: Dict[str, Any],
    executors: Dict[str, TickerExecutionState],
    initial_capital: float,
    trade_dollar: float,
    cost_bps: float,
    close_fallback: bool,
) -> Tuple[List[DailyNav], List[TradeEvent], Dict[str, Any]]:
    positions: Dict[str, float] = {t: 0.0 for t in tickers}
    cash = float(initial_capital)

    nav_rows: List[DailyNav] = []
    trade_rows: List[TradeEvent] = []

    gross_trade_value = 0.0

    prev_prices: Dict[str, Optional[float]] = {t: None for t in tickers}

    for day in days:
        price_today: Dict[str, Optional[float]] = {}
        for t in tickers:
            df = price_dfs.get(t)
            px = _get_close_on_or_before(df, day) if close_fallback else _get_close_exact(df, day)
            price_today[t] = px

        for t in tickers:
            sh = positions.get(t, 0.0)
            px_prev = prev_prices.get(t)
            px = price_today.get(t)
            if sh != 0.0 and px_prev is not None and px is not None:
                cash += float(sh) * (float(px) - float(px_prev))

        day_decisions = decisions.get(day, {})

        for t in tickers:
            px = price_today.get(t)
            if px is None or not math.isfinite(float(px)) or float(px) <= 0:
                continue

            sh_old = float(positions.get(t, 0.0))
            dec = str(day_decisions.get(t, "HOLD")).upper()
            raw_signal = 0
            if dec == "BUY":
                raw_signal = 1
            elif dec == "SELL":
                raw_signal = -1

            ex = executors.get(t)
            if ex is None:
                raise RuntimeError(f"Missing executor for ticker: {t}")

            target_pos = int(ex.update_signal(int(raw_signal)))
            if target_pos not in (-1, 0, 1):
                target_pos = 0

            # Determine old position direction
            old_pos = 0
            if sh_old > 0:
                old_pos = 1
            elif sh_old < 0:
                old_pos = -1

            # Skip if position direction unchanged (avoid spurious trades from price drift)
            if target_pos == old_pos:
                continue

            target_sh = 0.0 if target_pos == 0 else (float(target_pos) * float(trade_dollar) / float(px))

            delta = float(target_sh - sh_old)
            trade_value = abs(delta) * float(px)
            fee = trade_value * (float(cost_bps) / 10000.0)

            cash -= delta * float(px)
            cash -= fee

            positions[t] = target_sh

            gross_trade_value += trade_value
            trade_rows.append(
                TradeEvent(
                    date=day,
                    ticker=t,
                    action=dec,
                    shares_delta=delta,
                    price=float(px),
                    trade_value=float(trade_value),
                    fee=float(fee),
                )
            )

        positions_value = 0.0
        for t in tickers:
            sh = float(positions.get(t, 0.0))
            px = price_today.get(t)
            if px is None or not math.isfinite(float(px)):
                continue
            positions_value += sh * float(px)

        total_value = float(cash + positions_value)
        nav_rows.append(DailyNav(date=day, cash=float(cash), positions_value=float(positions_value), total_value=total_value))

        prev_prices = price_today

    values = [r.total_value for r in nav_rows]
    mdd = _max_drawdown(values)
    total_return = (values[-1] / values[0] - 1.0) if len(values) >= 2 and values[0] != 0 else float("nan")
    avg_nav = sum(values) / len(values) if values else float("nan")
    turnover = (gross_trade_value / avg_nav) if (values and math.isfinite(avg_nav) and avg_nav != 0) else float("nan")

    metrics = {
        "n_days": int(len(nav_rows)),
        "start_value": float(values[0]) if values else float("nan"),
        "end_value": float(values[-1]) if values else float("nan"),
        "total_return": float(total_return),
        "max_drawdown": float(mdd),
        "gross_trade_value": float(gross_trade_value),
        "turnover_gross_over_avg_nav": float(turnover),
        "trades": int(len(trade_rows)),
        "total_fees": float(sum(t.fee for t in trade_rows)),
    }

    return nav_rows, trade_rows, metrics


def _write_nav_csv(path: Path, rows: List[DailyNav]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("date,cash,positions_value,total_value\n")
        for r in rows:
            f.write(f"{r.date},{r.cash:.6f},{r.positions_value:.6f},{r.total_value:.6f}\n")


def _write_trades_csv(path: Path, rows: List[TradeEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("date,ticker,action,shares_delta,price,trade_value,fee\n")
        for t in rows:
            f.write(
                f"{t.date},{t.ticker},{t.action},{t.shares_delta:.10f},{t.price:.6f},{t.trade_value:.6f},{t.fee:.6f}\n"
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Portfolio NAV simulator from backtest_trader report (fixed $ sizing + transaction cost)")
    p.add_argument("--report", default="data/backtest/report_2025_12_final.json")
    p.add_argument("--out-dir", default="data/backtest/nav")
    p.add_argument("--models", default="", help="Comma-separated strategy names to run: v1_tech,v1_1_news,v1_1_ablation. Empty means all.")

    p.add_argument("--start-date", default="")
    p.add_argument("--end-date", default="")

    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--trade-dollar", type=float, default=10000.0)
    p.add_argument("--cost-bps", type=float, default=10.0)

    p.add_argument("--hold-policy", choices=["flat", "keep"], default="flat")
    p.add_argument("--min-hold-days", type=int, default=0)
    p.add_argument("--reverse-confirm-days", type=int, default=1)

    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--close-fallback", action="store_true", default=True)
    p.add_argument("--no-close-fallback", dest="close_fallback", action="store_false")

    args = p.parse_args()

    report_path = Path(args.report)
    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir)

    tickers, strat_trades = _parse_report(report_path)
    if not tickers:
        raise SystemExit("Report missing tickers")

    allowed = {"v1_tech", "v1_1_news", "v1_1_ablation"}
    requested = [m.strip() for m in str(args.models or "").split(",") if m.strip()]
    if requested:
        unknown = sorted([m for m in requested if m not in allowed])
        if unknown:
            raise SystemExit(f"Unknown --models entries: {unknown}. Allowed: {sorted(list(allowed))}")
        model_names = [m for m in requested if m in strat_trades]
    else:
        model_names = [m for m in ["v1_tech", "v1_1_news", "v1_1_ablation"] if m in strat_trades]

    price_dfs: Dict[str, Any] = {}
    for t in tickers:
        df = _load_price_df(raw_dir, t)
        if df is None:
            raise SystemExit(f"Missing or invalid parquet for ticker: {t}")
        price_dfs[t] = df

    all_days = sorted({
        _fmt_date(t.get("date"))
        for name in model_names
        for t in strat_trades.get(name, [])
        if isinstance(t, dict) and _fmt_date(t.get("date"))
    })

    start = str(args.start_date).strip()
    end = str(args.end_date).strip()
    if start:
        all_days = [d for d in all_days if d >= start]
    if end:
        all_days = [d for d in all_days if d <= end]

    if not all_days:
        raise SystemExit("No dates to simulate (after filtering)")

    summary: Dict[str, Any] = {
        "report": str(report_path),
        "out_dir": str(out_dir),
        "range": {"start": all_days[0], "end": all_days[-1]},
        "tickers": tickers,
        "initial_capital": float(args.initial_capital),
        "trade_dollar": float(args.trade_dollar),
        "cost_bps": float(args.cost_bps),
        "close_fallback": bool(args.close_fallback),
        "hold_policy": str(args.hold_policy),
        "min_hold_days": int(args.min_hold_days),
        "reverse_confirm_days": int(args.reverse_confirm_days),
        "strategies": {},
    }

    executors_by_strategy: Dict[str, Dict[str, TickerExecutionState]] = {}
    for name in model_names:
        executors_by_strategy[name] = {
            t: TickerExecutionState(
                hold_policy=str(args.hold_policy),
                min_hold_days=int(args.min_hold_days),
                reverse_confirm_days=int(args.reverse_confirm_days),
            )
            for t in tickers
        }

    for name in model_names:
        decisions = _decisions_by_day(strat_trades[name])

        nav_rows, trade_rows, metrics = simulate_nav(
            days=all_days,
            tickers=tickers,
            decisions=decisions,
            price_dfs=price_dfs,
            executors=executors_by_strategy[name],
            initial_capital=float(args.initial_capital),
            trade_dollar=float(args.trade_dollar),
            cost_bps=float(args.cost_bps),
            close_fallback=bool(args.close_fallback),
        )

        nav_path = out_dir / f"{name}_nav.csv"
        trades_path = out_dir / f"{name}_trades.csv"

        _write_nav_csv(nav_path, nav_rows)
        _write_trades_csv(trades_path, trade_rows)

        summary["strategies"][name] = {
            "metrics": metrics,
            "nav_csv": str(nav_path),
            "trades_csv": str(trades_path),
        }

        print(
            f"{name}: total_return={metrics['total_return']:.4f} max_drawdown={metrics['max_drawdown']:.4f} "
            f"turnover={metrics['turnover_gross_over_avg_nav']:.4f} trades={metrics['trades']} fees={metrics['total_fees']:.2f}"
        )

    _write_json(out_dir / "summary.json", summary)
    print(f"Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import glob
import math

import numpy as np
import pandas as pd


def _safe_div(a: float, b: float) -> float:
    if b == 0 or (isinstance(b, float) and (math.isnan(b) or math.isinf(b))):
        return float("nan")
    return a / b


def _annualize_return(daily_returns: pd.Series, periods_per_year: int) -> float:
    n = int(daily_returns.shape[0])
    if n <= 0:
        return float("nan")
    total = float((1.0 + daily_returns).prod())
    return total ** (periods_per_year / n) - 1.0


def _max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    peak = values.cummax()
    dd = values / peak - 1.0
    return float(dd.min())


def analyze_nav(nav_path: Path, rf_annual: float, periods_per_year: int) -> dict:
    df = pd.read_csv(nav_path)
    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError(f"NAV csv must contain columns date,value. got={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    values = df["value"].astype(float)
    daily = values.pct_change().dropna()

    ann_ret = _annualize_return(daily, periods_per_year=periods_per_year)
    ann_vol = float(daily.std(ddof=1)) * math.sqrt(periods_per_year)

    downside = daily[daily < 0]
    downside_vol = float(downside.std(ddof=1)) * math.sqrt(periods_per_year) if not downside.empty else float("nan")

    sharpe = _safe_div(ann_ret - rf_annual, ann_vol)
    sortino = _safe_div(ann_ret - rf_annual, downside_vol)

    mdd = _max_drawdown(values)
    calmar = _safe_div(ann_ret, abs(mdd))

    win_rate = float((daily > 0).mean()) if not daily.empty else float("nan")

    win_mean = float(daily[daily > 0].mean()) if (daily > 0).any() else float("nan")
    loss_mean = float(daily[daily < 0].mean()) if (daily < 0).any() else float("nan")
    pl_ratio = _safe_div(win_mean, abs(loss_mean))

    win_sum = float(daily[daily > 0].sum()) if (daily > 0).any() else float("nan")
    loss_sum = float(daily[daily < 0].sum()) if (daily < 0).any() else float("nan")
    profit_factor = _safe_div(win_sum, abs(loss_sum))

    best_day = float(daily.max()) if not daily.empty else float("nan")
    worst_day = float(daily.min()) if not daily.empty else float("nan")

    return {
        "start_date": str(df["date"].iloc[0].date()) if not df.empty else "",
        "end_date": str(df["date"].iloc[-1].date()) if not df.empty else "",
        "n_days": int(df.shape[0]),
        "n_return_days": int(daily.shape[0]),
        "start_value": float(values.iloc[0]) if not values.empty else float("nan"),
        "end_value": float(values.iloc[-1]) if not values.empty else float("nan"),
        "total_return": float(values.iloc[-1] / values.iloc[0] - 1.0) if values.shape[0] >= 2 else float("nan"),
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(mdd),
        "calmar": float(calmar),
        "win_rate_daily": float(win_rate),
        "pl_ratio_daily": float(pl_ratio),
        "profit_factor_daily": float(profit_factor),
        "best_day": float(best_day),
        "worst_day": float(worst_day),
        "avg_daily_return": float(daily.mean()) if not daily.empty else float("nan"),
        "std_daily_return": float(daily.std(ddof=1)) if not daily.empty else float("nan"),
        "avg_value": float(values.mean()) if not values.empty else float("nan"),
    }


def analyze_trades(trades_glob: str, avg_portfolio_value: float) -> dict:
    files = sorted(glob.glob(trades_glob))
    if not files:
        return {
            "trade_files": 0,
            "trades": 0,
        }

    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if df.empty:
            continue
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        dfs.append(df)

    if not dfs:
        return {
            "trade_files": int(len(files)),
            "trades": 0,
        }

    t = pd.concat(dfs, ignore_index=True)

    fee = t["fee"].astype(float) if "fee" in t.columns else pd.Series(dtype=float)
    trade_value = t["trade_value"].astype(float) if "trade_value" in t.columns else pd.Series(dtype=float)
    shares_delta = t["shares_delta"].astype(float) if "shares_delta" in t.columns else pd.Series(dtype=float)

    gross_trade_value = float(trade_value.abs().sum()) if not trade_value.empty else float("nan")
    total_fees = float(fee.sum()) if not fee.empty else float("nan")

    buys = int((shares_delta > 0).sum()) if not shares_delta.empty else 0
    sells = int((shares_delta < 0).sum()) if not shares_delta.empty else 0

    trades_by_day = None
    if "date" in t.columns:
        trades_by_day = t.dropna(subset=["date"]).groupby(t["date"].dt.date).size()

    turnover = _safe_div(gross_trade_value, float(avg_portfolio_value)) if not math.isnan(avg_portfolio_value) else float("nan")

    return {
        "trade_files": int(len(files)),
        "trades": int(t.shape[0]),
        "buys": buys,
        "sells": sells,
        "gross_trade_value": gross_trade_value,
        "total_fees": total_fees,
        "turnover_gross_over_avg_value": float(turnover),
        "trades_per_day_avg": float(trades_by_day.mean()) if trades_by_day is not None and not trades_by_day.empty else float("nan"),
        "trades_per_day_median": float(trades_by_day.median()) if trades_by_day is not None and not trades_by_day.empty else float("nan"),
        "trades_per_day_max": int(trades_by_day.max()) if trades_by_day is not None and not trades_by_day.empty else 0,
    }


def _fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x * 100:.2f}%"


def _fmt_f(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.{nd}f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze paper backtest NAV and trades")
    p.add_argument("--nav", default="results/backtest_2024_jan/paper_nav.csv")
    p.add_argument("--trades", default="results/backtest_2024_jan/paper_trades_*.csv")
    p.add_argument("--rf", type=float, default=0.0, help="Annual risk-free rate, e.g. 0.03")
    p.add_argument("--ppy", type=int, default=252, help="Periods per year (daily=252)")
    p.add_argument("--out", default="", help="Optional path to save JSON summary")

    args = p.parse_args()

    nav_path = Path(args.nav)
    if not nav_path.exists():
        raise SystemExit(f"NAV file not found: {nav_path}")

    nav_stats = analyze_nav(nav_path=nav_path, rf_annual=float(args.rf), periods_per_year=int(args.ppy))
    trade_stats = analyze_trades(trades_glob=str(args.trades), avg_portfolio_value=float(nav_stats.get("avg_value", float("nan"))))

    print("# Backtest Deep Check")
    print(f"- Period: {nav_stats['start_date']} -> {nav_stats['end_date']} ({nav_stats['n_days']} NAV rows)")
    print(f"- Start/End: {nav_stats['start_value']:.2f} -> {nav_stats['end_value']:.2f} ({_fmt_pct(nav_stats['total_return'])})")
    print("\n## Return / Risk")
    print(f"- Annual Return (CAGR): {_fmt_pct(nav_stats['annual_return'])}")
    print(f"- Annual Vol: {_fmt_pct(nav_stats['annual_vol'])}")
    print(f"- Sharpe (rf={args.rf}): {_fmt_f(nav_stats['sharpe'], 3)}")
    print(f"- Sortino (rf={args.rf}): {_fmt_f(nav_stats['sortino'], 3)}")
    print(f"- Max Drawdown: {_fmt_pct(nav_stats['max_drawdown'])}")
    print(f"- Calmar: {_fmt_f(nav_stats['calmar'], 3)}")

    print("\n## Hit Rate / P&L")
    print(f"- Win Rate (daily): {_fmt_pct(nav_stats['win_rate_daily'])}")
    print(f"- Avg Win / Avg Loss (daily): {_fmt_f(nav_stats['pl_ratio_daily'], 3)}")
    print(f"- Profit Factor (daily): {_fmt_f(nav_stats['profit_factor_daily'], 3)}")
    print(f"- Best Day: {_fmt_pct(nav_stats['best_day'])}")
    print(f"- Worst Day: {_fmt_pct(nav_stats['worst_day'])}")

    print("\n## Trading Activity")
    print(f"- Trade files: {trade_stats.get('trade_files', 0)}")
    print(f"- Trades: {trade_stats.get('trades', 0)} (buys={trade_stats.get('buys', 0)} sells={trade_stats.get('sells', 0)})")
    if trade_stats.get("trades", 0) > 0:
        print(f"- Gross traded value: {trade_stats.get('gross_trade_value', float('nan')):.2f}")
        print(f"- Total fees: {trade_stats.get('total_fees', float('nan')):.2f}")
        print(f"- Gross turnover / avg NAV: {_fmt_f(trade_stats.get('turnover_gross_over_avg_value', float('nan')), 3)}")
        print(
            "- Trades/day (avg/median/max): "
            f"{_fmt_f(trade_stats.get('trades_per_day_avg', float('nan')), 2)} / "
            f"{_fmt_f(trade_stats.get('trades_per_day_median', float('nan')), 2)} / "
            f"{trade_stats.get('trades_per_day_max', 0)}"
        )

    out_obj = {
        "nav": nav_stats,
        "trades": trade_stats,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            import json

            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.risk.gate import RiskGate


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _to_pct(v: Any) -> float:
    x = _to_float(v)
    if abs(x) <= 2.5:
        return x * 100.0
    return x


def load_daily_features(data_dir: Path, date_str: str) -> Optional[Dict[str, Any]]:
    fp = data_dir / f"etf_features_{date_str}.json"
    if not fp.exists():
        return None
    try:
        payload = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return payload

    if isinstance(payload, dict):
        items: List[Dict[str, Any]] = []
        for _k, v in payload.items():
            if isinstance(v, dict):
                for sym, feats in v.items():
                    if isinstance(feats, dict):
                        items.append({"symbol": sym, **feats})
        if items:
            return {"items": items}

    return None


def find_item(payload: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None
    for it in items:
        if not isinstance(it, dict):
            continue
        if str(it.get("symbol") or "").strip().upper() == str(symbol).strip().upper():
            return it
    return None


def extract_price_and_features(item: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    base = item.get("features") if isinstance(item.get("features"), dict) else item
    tech = base.get("technical") if isinstance(base.get("technical"), dict) else {}

    price = base.get("price")
    if price is None:
        price = tech.get("close")
    price_f = _to_float(price)

    change_5d = base.get("change_5d_pct")
    if change_5d is None:
        change_5d = base.get("return_5d")
    if change_5d is None:
        change_5d = tech.get("return_5d")

    vol = base.get("volatility_ann_pct")
    if vol is None:
        vol = base.get("volatility_20d")
    if vol is None:
        vol = tech.get("volatility_20d")

    dd = base.get("drawdown_20d_pct")
    if dd is None:
        dd = base.get("max_drawdown_20d")
    if dd is None:
        dd = tech.get("max_drawdown_20d")
    if dd is None:
        dd = base.get("drawdown")
    if dd is None:
        dd = tech.get("drawdown")

    merged = dict(base)
    merged["change_5d_pct"] = _to_pct(change_5d)
    merged["volatility_ann_pct"] = _to_pct(vol)
    merged["drawdown_20d_pct"] = _to_pct(dd)
    return price_f, merged


def load_news_signals(data_dir: Path, date_str: str) -> List[Dict[str, Any]]:
    fp = data_dir / f"signals_{date_str}.json"
    if not fp.exists():
        return []
    try:
        arr = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    if not isinstance(arr, list):
        return out

    for it in arr:
        if not isinstance(it, dict):
            continue
        sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
        et = str(sig.get("event_type") or "").strip()
        if not et:
            continue
        out.append({"event_type": et, "impact_equity": sig.get("impact_equity")})

    return out


def propose_policy(features: Dict[str, Any]) -> Tuple[str, float]:
    chg5d = _to_float(features.get("change_5d_pct"))
    dd20 = _to_float(features.get("drawdown_20d_pct"))

    if dd20 <= -8.0:
        return "CLEAR", 0.0

    if chg5d >= 1.0:
        return "BUY", 0.8
    if chg5d <= -1.0:
        return "REDUCE", 0.1

    return "HOLD", 0.35


class BacktestEngine:
    def __init__(self, initial_capital: float = 100000.0, rebalance_threshold: float = 0.05):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.holdings = 0.0
        self.history: List[Dict[str, Any]] = []
        self.risk_gate = RiskGate()
        self.rebalance_threshold = float(rebalance_threshold)

    def _portfolio_value(self, price: float) -> float:
        return float(self.cash + self.holdings * price)

    def _execute(self, price: float, target_pos_pct: float, date_str: str) -> None:
        if price <= 0:
            return

        total_value = self._portfolio_value(price)
        if total_value <= 0:
            return

        target_value = total_value * float(target_pos_pct)
        current_value = self.holdings * price
        diff = target_value - current_value

        if abs(diff) <= total_value * self.rebalance_threshold:
            return

        shares_to_trade = diff / price
        self.holdings += shares_to_trade
        self.cash -= shares_to_trade * price

    def run(self, symbol: str, start_date: str, end_date: str, data_dir: str = "data/daily") -> pd.DataFrame:
        data_path = Path(data_dir)
        dates = pd.date_range(start_date, end_date, freq="D")

        for d in dates:
            date_str = d.strftime("%Y-%m-%d")

            payload = load_daily_features(data_path, date_str)
            if payload is None:
                continue

            item = find_item(payload, symbol)
            if item is None:
                continue

            price, feats = extract_price_and_features(item)
            if price <= 0:
                continue

            news_signals = load_news_signals(data_path, date_str)

            proposed_action, proposed_pos = propose_policy(feats)
            final_action, final_pos, trace = self.risk_gate.adjudicate(
                feats,
                news_signals,
                proposed_action,
                proposed_pos,
            )

            self._execute(price, float(final_pos), date_str)

            total_value = self._portfolio_value(price)
            self.history.append(
                {
                    "date": date_str,
                    "price": float(price),
                    "value": float(total_value),
                    "cash": float(self.cash),
                    "holdings": float(self.holdings),
                    "proposed_action": str(proposed_action).upper(),
                    "proposed_pos": float(proposed_pos),
                    "final_action": str(final_action).upper(),
                    "final_pos": float(final_pos),
                    "risk_trace": " | ".join([str(x) for x in trace]) if trace else "",
                }
            )

        df = pd.DataFrame(self.history)
        return df


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"final_return": 0.0, "max_drawdown": 0.0}

    v = df["value"].astype(float)
    ret = (v.iloc[-1] / v.iloc[0]) - 1.0
    dd = (v / v.cummax()) - 1.0
    return {
        "final_return": float(ret),
        "max_drawdown": float(dd.min()),
    }


def save_equity_curve_plot(df: pd.DataFrame, out_png: str) -> None:
    if df.empty:
        return

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(pd.to_datetime(df["date"]), df["value"], label="Portfolio")
        plt.title("Backtest Equity Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
    except Exception:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-06-01")
    parser.add_argument("--data-dir", type=str, default="data/daily")
    parser.add_argument("--initial", type=float, default=100000.0)
    parser.add_argument("--rebalance-threshold", type=float, default=0.05)
    parser.add_argument("--out", type=str, default="backtest_result.csv")
    parser.add_argument("--plot", type=str, default="backtest_result.png")
    args = parser.parse_args()

    engine = BacktestEngine(
        initial_capital=float(args.initial),
        rebalance_threshold=float(args.rebalance_threshold),
    )

    df = engine.run(
        symbol=str(args.symbol),
        start_date=str(args.start),
        end_date=str(args.end),
        data_dir=str(args.data_dir),
    )

    metrics = compute_metrics(df)
    print("\n=== Backtest Report ===")
    print(f"Rows: {len(df)}")
    print(f"Final Return: {metrics['final_return'] * 100:.2f}%")
    print(f"Max Drawdown: {metrics['max_drawdown'] * 100:.2f}%")

    if not df.empty:
        Path(args.out).write_text(df.to_csv(index=False), encoding="utf-8")
        save_equity_curve_plot(df, str(args.plot))
        print(f"Saved CSV: {args.out}")
        print(f"Saved Chart: {args.plot}")


if __name__ == "__main__":
    main()

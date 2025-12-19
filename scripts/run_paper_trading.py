#!/usr/bin/env python

import argparse
import csv
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.strategy.execution import TickerExecutionState


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _load_json(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(raw)
    except Exception:
        s = raw.strip()
        s = s.replace("\\", "")
        import re

        s = re.sub(r'([\{,])\s*([A-Za-z0-9_\.\-]+)\s*:', r'\1"\2":', s)
        return json.loads(s)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _extract_raw_decision(item: Dict[str, Any]) -> Tuple[int, bool]:
    if not isinstance(item, dict):
        return 0, False
    final = item.get("final") if isinstance(item.get("final"), dict) else {}
    action = str(final.get("action") or "").strip().upper()
    force_flat = action == "CLEAR"
    return int(_extract_raw_signal(item)), bool(force_flat)


def _normalize_signals_payload(payload: Any, tickers: List[str]) -> Tuple[Dict[str, int], Dict[str, bool]]:
    if payload is None:
        return {}, {}

    if isinstance(payload, dict):
        if payload.get("items") is not None and isinstance(payload.get("items"), dict):
            items = payload.get("items")
            out: Dict[str, int] = {}
            ff: Dict[str, bool] = {}
            for t in tickers:
                it = items.get(t)
                if isinstance(it, dict):
                    sig, force_flat = _extract_raw_decision(it)
                    out[str(t)] = int(sig)
                    ff[str(t)] = bool(force_flat)
                else:
                    out[str(t)] = 0
                    ff[str(t)] = False
            return out, ff

        if payload and all(isinstance(v, (int, float)) for v in payload.values()):
            return {str(k).strip().upper(): int(_to_float(v)) for k, v in payload.items()}, {}

        if payload and all(isinstance(v, dict) for v in payload.values()):
            out2: Dict[str, int] = {}
            ff2: Dict[str, bool] = {}
            for k, v in payload.items():
                if not isinstance(v, dict):
                    continue
                if v.get("signal") is not None:
                    out2[str(k).strip().upper()] = int(_to_float(v.get("signal")))
                    ff2[str(k).strip().upper()] = bool(v.get("force_flat", False))
                else:
                    sig, force_flat = _extract_raw_decision(v)
                    out2[str(k).strip().upper()] = int(sig)
                    ff2[str(k).strip().upper()] = bool(force_flat)
            if out2:
                return out2, ff2

    if isinstance(payload, list):
        out3: Dict[str, int] = {}
        ff3: Dict[str, bool] = {}
        for it in payload:
            if not isinstance(it, dict):
                continue
            sym = str(it.get("ticker") or it.get("symbol") or "").strip().upper()
            if not sym:
                continue
            if it.get("signal") is not None:
                out3[sym] = int(_to_float(it.get("signal")))
                ff3[sym] = bool(it.get("force_flat", False))
            else:
                sig, force_flat = _extract_raw_decision(it)
                out3[sym] = int(sig)
                ff3[sym] = bool(force_flat)
        if out3:
            return out3, ff3

    return {}, {}


def _load_state_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = _load_json(path)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    return dict(meta)


def _ensure_dirs(paper_dir: Path) -> Dict[str, Path]:
    orders_dir = paper_dir / "orders"
    logs_dir = paper_dir / "logs"
    history_dir = paper_dir / "history"
    orders_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    return {"orders": orders_dir, "logs": logs_dir, "history": history_dir}


def _backup_if_exists(src: Path, history_dir: Path, date_str: str) -> Optional[Path]:
    if not src.exists():
        return None
    dst = history_dir / f"{src.stem}_{date_str}{src.suffix}"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def _extract_raw_signal(item: Dict[str, Any]) -> int:
    if not isinstance(item, dict):
        return 0

    final = item.get("final") if isinstance(item.get("final"), dict) else {}
    action = str(final.get("action") or "").strip().upper()
    tp = final.get("target_position")

    if tp is not None:
        tp_f = _to_float(tp)
        if tp_f < 0:
            return -1
        if tp_f > 0:
            return 1
        return 0

    if action in {"BUY", "LONG"}:
        return 1
    if action in {"SELL", "SHORT"}:
        return -1
    if action in {"HOLD", "FLAT"}:
        return 0

    parsed = item.get("parsed") if isinstance(item.get("parsed"), dict) else {}
    if isinstance(parsed.get("decision"), str):
        d = str(parsed.get("decision") or "").strip().upper()
        if d == "BUY":
            return 1
        if d == "SELL":
            return -1
        return 0

    label = parsed.get("label") if isinstance(parsed.get("label"), dict) else {}
    if label:
        act = str(label.get("action") or "").strip().upper()
        if act == "BUY":
            return 1
        if act == "SELL":
            return -1
        return 0

    return 0


def _load_close_for_date(*, parquet_path: Path, date_str: str) -> Optional[float]:
    if not parquet_path.exists():
        return None

    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return None

    if df is None or df.empty:
        return None

    try:
        if "date" in df.columns:
            s = pd.to_datetime(df["date"], errors="coerce")
        else:
            s = pd.to_datetime(df.index, errors="coerce")
        s = pd.Series(s)
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert(None)

        if "close" not in df.columns:
            return None

        m_exact = s.dt.strftime("%Y-%m-%d") == str(date_str)
        if bool(m_exact.any()):
            return float(df.loc[m_exact, "close"].iloc[-1])

        base = pd.to_datetime(date_str)
        m_prev = s <= base
        if bool(m_prev.any()):
            return float(df.loc[m_prev, "close"].iloc[-1])
    except Exception:
        return None

    return None


def _load_prices(*, data_dir: Path, tickers: List[str], date_str: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for t in tickers:
        px = _load_close_for_date(parquet_path=data_dir / "raw" / f"{t}.parquet", date_str=date_str)
        if px is None:
            continue
        if px > 0 and pd.notna(px):
            out[str(t)] = float(px)
    return out


@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, float]


def _load_portfolio(path: Path, initial_cash: float) -> PortfolioState:
    if not path.exists():
        return PortfolioState(cash=float(initial_cash), positions={})
    try:
        data = _load_json(path)
    except Exception:
        return PortfolioState(cash=float(initial_cash), positions={})

    if not isinstance(data, dict):
        return PortfolioState(cash=float(initial_cash), positions={})

    cash = _to_float(data.get("cash"))
    pos = data.get("positions") if isinstance(data.get("positions"), dict) else {}
    positions = {str(k): _to_float(v) for k, v in pos.items()}
    return PortfolioState(cash=float(cash), positions=positions)


def _save_portfolio(path: Path, st: PortfolioState, meta: Dict[str, Any]) -> None:
    payload = {"cash": float(st.cash), "positions": st.positions, **meta}
    _save_json(path, payload)


def _load_execution_state(path: Path) -> Dict[str, TickerExecutionState]:
    if not path.exists():
        return {}
    try:
        data = _load_json(path)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}

    out: Dict[str, TickerExecutionState] = {}
    items = data.get("items") if isinstance(data.get("items"), dict) else data
    if not isinstance(items, dict):
        return {}

    for sym, st in items.items():
        if not isinstance(st, dict):
            continue
        out[str(sym)] = TickerExecutionState.from_dict(st)
    return out


def _save_execution_state(path: Path, state: Dict[str, TickerExecutionState], meta: Dict[str, Any]) -> None:
    payload = {"meta": meta, "items": {k: v.to_dict() for k, v in state.items()}}
    _save_json(path, payload)


def _write_orders_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "date",
        "ticker",
        "raw_signal",
        "target_pos",
        "price",
        "shares_delta",
        "trade_value",
        "fee",
        "cash_after",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in header})


def main() -> None:
    p = argparse.ArgumentParser(description="Daily paper trading loop using TickerExecutionState (hold/confirm) with fixed-dollar sizing")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--decision", default="data/decisions_inference.json", help="Decision JSON path (from run_trading_inference.py)")
    p.add_argument("--signals", default="", help="Optional flat signals json path: {\"AAPL\": 1, ...}. If set, --decision is ignored.")
    p.add_argument("--tickers", default="", help="Comma-separated tickers. Required when using --signals.")
    p.add_argument("--paper-dir", default="data/paper")
    p.add_argument("--data-dir", default="data")

    p.add_argument("--initial-cash", type=float, default=100000.0)
    p.add_argument("--trade-dollar", type=float, default=10000.0)
    p.add_argument("--cost-bps", type=float, default=10.0)

    p.add_argument("--allow-same-day", action="store_true", default=False)

    p.add_argument("--hold-policy", choices=["flat", "keep"], default="keep")
    p.add_argument("--min-hold-days", type=int, default=0)
    p.add_argument("--reverse-confirm-days", type=int, default=2)

    args = p.parse_args()

    date_str = str(args.date).strip()
    dt.datetime.fromisoformat(date_str)

    paper_dir = Path(args.paper_dir)
    dirs = _ensure_dirs(paper_dir)

    state_path = paper_dir / "state.json"
    portfolio_path = paper_dir / "portfolio.json"

    prev_meta = _load_state_meta(state_path)
    if prev_meta.get("date") == date_str and (not bool(args.allow_same_day)):
        raise SystemExit(f"State already updated for date={date_str}. Use --allow-same-day to rerun.")

    _backup_if_exists(state_path, dirs["history"], date_str)
    _backup_if_exists(portfolio_path, dirs["history"], date_str)

    items: Dict[str, Any] = {}
    tickers: List[str] = []
    decision_path = Path(args.decision)

    signals_path = str(args.signals or "").strip()
    if signals_path:
        fp = Path(signals_path)
        if not fp.exists():
            raise SystemExit(f"Signals file not found: {fp}")
        tickers = [t.strip().upper() for t in str(args.tickers or "").split(",") if t.strip()]
        sig_data = _load_json(fp)
        if not tickers:
            if isinstance(sig_data, dict) and isinstance(sig_data.get("items"), dict):
                tickers = sorted([str(k).strip().upper() for k in sig_data.get("items").keys()])
            elif isinstance(sig_data, dict):
                tickers = sorted([str(k).strip().upper() for k in sig_data.keys()])
        signal_map, force_flat_map = _normalize_signals_payload(sig_data, tickers)
        if not signal_map:
            raise SystemExit(f"Unsupported signals json format: {fp}")
        items = signal_map
    else:
        force_flat_map = {}
        if not decision_path.exists():
            raise SystemExit(f"Decision file not found: {decision_path}")

        decision = _load_json(decision_path)
        if not isinstance(decision, dict):
            raise SystemExit(f"Invalid decision json: {decision_path}")

        items = decision.get("items") if isinstance(decision.get("items"), dict) else {}
        if not items:
            raise SystemExit("Decision JSON has no items")
        tickers = sorted([str(k).strip().upper() for k in items.keys() if str(k).strip()])

    prices = _load_prices(data_dir=Path(args.data_dir), tickers=tickers, date_str=date_str)
    missing_px = [t for t in tickers if t not in prices]
    if missing_px:
        raise SystemExit(f"Missing close prices for {len(missing_px)} tickers on/before {date_str}: {missing_px[:10]}")

    portfolio = _load_portfolio(portfolio_path, float(args.initial_cash))
    exec_state = _load_execution_state(state_path)

    for t in tickers:
        if t not in exec_state:
            exec_state[t] = TickerExecutionState(
                hold_policy=str(args.hold_policy),
                min_hold_days=int(args.min_hold_days),
                reverse_confirm_days=int(args.reverse_confirm_days),
            )
        else:
            exec_state[t].hold_policy = str(args.hold_policy)
            exec_state[t].min_hold_days = int(args.min_hold_days)
            exec_state[t].reverse_confirm_days = int(args.reverse_confirm_days)

    orders: List[Dict[str, Any]] = []

    for t in tickers:
        px = float(prices[t])
        if signals_path:
            raw_signal = int(items.get(t, 0))
            force_flat = bool(force_flat_map.get(t, False))
        else:
            raw_signal = _extract_raw_signal(items.get(t, {}))
            force_flat = False

        ex = exec_state[t]
        prev_exec_pos = int(ex.current_position)
        prev_days = int(ex.days_held)
        prev_pending = int(ex.pending_reverse_count)
        target_pos = int(ex.update_signal(int(raw_signal), force_flat=bool(force_flat)))
        post_exec_pos = int(ex.current_position)
        post_days = int(ex.days_held)
        post_pending = int(ex.pending_reverse_count)

        sh_old = float(portfolio.positions.get(t, 0.0))
        old_pos = _sign(sh_old)

        did_trade = 0

        if target_pos == old_pos:
            orders.append(
                {
                    "date": date_str,
                    "ticker": t,
                    "raw_signal": int(raw_signal),
                    "target_pos": int(target_pos),
                    "price": float(px),
                    "shares_delta": 0.0,
                    "trade_value": 0.0,
                    "fee": 0.0,
                    "cash_after": float(portfolio.cash),
                    "exec_prev_pos": prev_exec_pos,
                    "exec_post_pos": post_exec_pos,
                    "exec_prev_days": prev_days,
                    "exec_post_days": post_days,
                    "exec_prev_pending": prev_pending,
                    "exec_post_pending": post_pending,
                    "did_trade": did_trade,
                }
            )
            continue

        target_sh = 0.0 if target_pos == 0 else (float(target_pos) * float(args.trade_dollar) / px)
        delta = float(target_sh - sh_old)
        trade_value = abs(delta) * px
        fee = trade_value * (float(args.cost_bps) / 10000.0)

        portfolio.cash -= delta * px
        portfolio.cash -= fee

        if abs(target_sh) < 1e-12:
            portfolio.positions.pop(t, None)
        else:
            portfolio.positions[t] = float(target_sh)

        orders.append(
            {
                "date": date_str,
                "ticker": t,
                "raw_signal": int(raw_signal),
                "target_pos": int(target_pos),
                "price": float(px),
                "shares_delta": float(delta),
                "trade_value": float(trade_value),
                "fee": float(fee),
                "cash_after": float(portfolio.cash),
                "exec_prev_pos": prev_exec_pos,
                "exec_post_pos": post_exec_pos,
                "exec_prev_days": prev_days,
                "exec_post_days": post_days,
                "exec_prev_pending": prev_pending,
                "exec_post_pending": post_pending,
                "did_trade": 1,
            }
        )

    orders_path = dirs["orders"] / f"orders_{date_str}.csv"
    _write_orders_csv(orders_path, orders)

    equity = float(portfolio.cash)
    for t, sh in portfolio.positions.items():
        equity += float(sh) * float(prices.get(t, 0.0))

    meta = {
        "date": date_str,
        "decision": "" if signals_path else str(decision_path),
        "signals": str(signals_path),
        "trade_dollar": float(args.trade_dollar),
        "cost_bps": float(args.cost_bps),
        "hold_policy": str(args.hold_policy),
        "min_hold_days": int(args.min_hold_days),
        "reverse_confirm_days": int(args.reverse_confirm_days),
        "equity": float(equity),
        "n_orders": int(len(orders)),
    }

    _save_execution_state(state_path, exec_state, meta)
    _save_portfolio(portfolio_path, portfolio, meta)

    log_path = dirs["logs"] / f"paper_{date_str}.log"
    log_lines = [
        f"date={date_str}",
        f"decision={'' if signals_path else str(decision_path)}",
        f"signals={signals_path}",
        f"tickers={len(tickers)}",
        f"orders={len([r for r in orders if float(r.get('trade_value', 0.0)) != 0.0])}",
        f"cash={portfolio.cash:.2f}",
        f"equity={equity:.2f}",
    ]
    for r in orders:
        log_lines.append(
            "|".join(
                [
                    str(r.get("ticker")),
                    f"raw={int(r.get('raw_signal', 0))}",
                    f"exec={int(r.get('exec_prev_pos', 0))}->{int(r.get('exec_post_pos', 0))}",
                    f"days={int(r.get('exec_prev_days', 0))}->{int(r.get('exec_post_days', 0))}",
                    f"pend={int(r.get('exec_prev_pending', 0))}->{int(r.get('exec_post_pending', 0))}",
                    f"trade={int(r.get('did_trade', 0))}",
                    f"shares_delta={float(r.get('shares_delta', 0.0)):.6f}",
                    f"fee={float(r.get('fee', 0.0)):.4f}",
                ]
            )
        )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"Saved orders: {orders_path}")
    print(f"Saved state: {state_path}")
    print(f"Saved portfolio: {portfolio_path}")
    print(f"Saved log: {log_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_daily_features(daily_dir: Path, date_str: str) -> Dict[str, Any]:
    fp = daily_dir / f"etf_features_{date_str}.json"
    if not fp.exists():
        raise SystemExit(f"Missing daily features: {fp}")
    data = _load_json(fp)
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data
    raise SystemExit(f"Unsupported daily features format: {fp}")


def _iter_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            out.append(it)
    return out


def _extract_close(item: Dict[str, Any]) -> float:
    base = item.get("features") if isinstance(item.get("features"), dict) else item
    tech = base.get("technical") if isinstance(base.get("technical"), dict) else {}
    price = base.get("price")
    if price is None:
        price = tech.get("close")
    return _to_float(price)


def _load_decision(decision_path: Path) -> Dict[str, Any]:
    if not decision_path.exists():
        raise SystemExit(f"Missing trading decision: {decision_path}")
    data = _load_json(decision_path)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid decision JSON: {decision_path}")
    return data


def _apply_virtual_event(*, decision: Dict[str, Any], virtual_event_path: Optional[Path]) -> Dict[str, Any]:
    if virtual_event_path is None:
        return decision
    if not virtual_event_path.exists():
        raise SystemExit(f"Virtual event file not found: {virtual_event_path}")

    ev = _load_json(virtual_event_path)
    items = decision.get("items") if isinstance(decision.get("items"), dict) else {}

    def override_symbol(symbol: str, tp: float) -> None:
        s = str(symbol)
        if s not in items or not isinstance(items.get(s), dict):
            items[s] = {}
        it = items[s]
        final = it.get("final") if isinstance(it.get("final"), dict) else {}
        final["action"] = "VIRTUAL"
        final["target_position"] = float(tp)
        trace = final.get("trace") if isinstance(final.get("trace"), list) else []
        trace = [str(x) for x in trace]
        trace.append(f"[VIRTUAL] override target_position={float(tp):.4f}")
        final["trace"] = trace
        it["final"] = final
        items[s] = it

    if isinstance(ev, dict) and isinstance(ev.get("targets"), dict):
        for sym, tp in ev["targets"].items():
            override_symbol(str(sym), _to_float(tp))
    elif isinstance(ev, dict):
        for sym, tp in ev.items():
            override_symbol(str(sym), _to_float(tp))
    else:
        raise SystemExit("Virtual event JSON must be an object")

    decision["items"] = items
    decision["virtual_event"] = str(virtual_event_path)
    return decision


@dataclass
class AccountState:
    cash: float
    positions: Dict[str, float]


def _load_state(path: Path, initial_cash: float) -> AccountState:
    if not path.exists():
        return AccountState(cash=float(initial_cash), positions={})
    data = _load_json(path)
    if not isinstance(data, dict):
        return AccountState(cash=float(initial_cash), positions={})
    cash = _to_float(data.get("cash"))
    pos = data.get("positions") if isinstance(data.get("positions"), dict) else {}
    positions = {str(k): _to_float(v) for k, v in pos.items()}
    return AccountState(cash=float(cash), positions=positions)


def _save_state(path: Path, st: AccountState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"cash": st.cash, "positions": st.positions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _portfolio_value(*, st: AccountState, prices: Dict[str, float]) -> float:
    v = float(st.cash)
    for sym, sh in st.positions.items():
        px = _to_float(prices.get(sym))
        v += float(sh) * px
    return float(v)


def _normalize_targets(targets: Dict[str, float]) -> Dict[str, float]:
    clean = {str(k): max(0.0, float(v)) for k, v in targets.items()}
    s = sum(clean.values())
    if s <= 1.0 or s <= 0.0:
        return clean
    scale = 1.0 / s
    return {k: float(v) * scale for k, v in clean.items()}


def _write_csv_row(path: Path, header: List[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k) for k in header})


def _load_raw_close_prices_for_date(*, data_dir: Path, symbols: List[str], date_str: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for sym in symbols:
        fp = data_dir / "raw" / f"{sym}.parquet"
        if not fp.exists():
            continue
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue

        close_s: Optional[pd.Series] = None
        if "close" in df.columns:
            try:
                if "date" in df.columns:
                    dt = pd.to_datetime(df["date"], errors="coerce")
                else:
                    dt = pd.to_datetime(df.index, errors="coerce")
                dt_s = pd.Series(dt, index=df.index)
                if getattr(dt_s.dt, "tz", None) is not None:
                    dt_s = dt_s.dt.tz_convert(None)
                base = pd.to_datetime(date_str)
                m_exact = dt_s.dt.strftime("%Y-%m-%d") == date_str
                if bool(m_exact.any()):
                    close_s = df.loc[m_exact, "close"]
                else:
                    m_prev = dt_s <= base
                    if bool(m_prev.any()):
                        close_s = df.loc[m_prev, "close"].tail(1)
            except Exception:
                close_s = None

        if close_s is None or close_s.empty:
            continue
        out[str(sym)] = _to_float(close_s.iloc[-1])
    return out


def _next_trading_date_from_raw(*, data_dir: Path, symbol: str, date_str: str) -> Optional[str]:
    fp = data_dir / "raw" / f"{symbol}.parquet"
    if not fp.exists():
        return None
    try:
        df = pd.read_parquet(fp)
    except Exception:
        return None

    try:
        if "date" in df.columns:
            dt = pd.to_datetime(df["date"], errors="coerce")
        else:
            dt = pd.to_datetime(df.index, errors="coerce")
        dt = dt.dropna().sort_values()
        base = pd.to_datetime(date_str)
        nxt = dt[dt > base]
        if len(nxt) == 0:
            return None
        return nxt.iloc[0].strftime("%Y-%m-%d")
    except Exception:
        return None


def _write_meta(path: Path, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Paper trading simulation using daily trading decisions")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--daily-dir", default="data/daily")
    p.add_argument("--decision", default=None, help="Override decision json path (default: data/daily/trading_decision_DATE.json)")
    p.add_argument("--outdir", default="data/paper")
    p.add_argument("--state", default="account_state.json")
    p.add_argument("--initial-cash", type=float, default=100000.0)
    p.add_argument("--rebalance-threshold", type=float, default=0.02)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--fee-bps", type=float, default=1.0)
    p.add_argument("--min-trade-value", type=float, default=50.0)
    p.add_argument("--mtm-next-day", type=int, default=1, help="If 1, mark-to-market using next trading day close")
    p.add_argument("--data-dir", default="data", help="Root data dir containing raw/*.parquet")
    p.add_argument("--meta-out", default=None, help="Optional meta json output path")
    p.add_argument("--skip-trade", type=int, default=0, help="If 1, do not execute trades; only compute valuation/MTM")
    p.add_argument("--settle-date", default=None, help="Optional explicit settle date YYYY-MM-DD for MTM")
    p.add_argument("--virtual-event", default=None, help="Path to JSON mapping symbol->target_position (or {targets:{...}})")
    args = p.parse_args()

    date_str = str(args.date)
    daily_dir = Path(args.daily_dir)

    decision_path = Path(args.decision) if args.decision else (daily_dir / f"trading_decision_{date_str}.json")
    decision = _load_decision(decision_path)
    decision = _apply_virtual_event(decision=decision, virtual_event_path=Path(args.virtual_event) if args.virtual_event else None)

    payload = _load_daily_features(daily_dir, date_str)
    items = _iter_items(payload)

    prices: Dict[str, float] = {}
    for it in items:
        sym = str(it.get("symbol") or it.get("ticker") or "").strip()
        if not sym:
            continue
        prices[sym] = _extract_close(it)

    outdir = Path(args.outdir)
    state_path = outdir / str(args.state)
    st = _load_state(state_path, float(args.initial_cash))

    items_map = decision.get("items") if isinstance(decision.get("items"), dict) else {}
    targets_raw: Dict[str, float] = {}
    if int(args.skip_trade) != 1:
        for sym, it in items_map.items():
            if not isinstance(it, dict):
                continue
            final = it.get("final") if isinstance(it.get("final"), dict) else {}
            targets_raw[str(sym)] = _to_float(final.get("target_position"))

    targets = _normalize_targets(targets_raw)

    pv_before = _portfolio_value(st=st, prices=prices)

    symbols = sorted({*list(targets.keys()), *list(st.positions.keys())})

    trade_header = [
        "date",
        "symbol",
        "price",
        "effective_price",
        "shares_delta",
        "trade_value",
        "fee",
        "cash_after",
        "shares_after",
        "target_weight",
    ]

    if int(args.skip_trade) != 1:
        for sym, tgt_w in targets.items():
            px = _to_float(prices.get(sym))
            if px <= 0:
                continue

            pv = _portfolio_value(st=st, prices=prices)
            if pv <= 0:
                continue

            current_sh = _to_float(st.positions.get(sym))
            current_val = current_sh * px
            target_val = pv * float(tgt_w)
            diff_val = target_val - current_val

            if abs(diff_val) < pv * float(args.rebalance_threshold):
                continue
            if abs(diff_val) < float(args.min_trade_value):
                continue

            is_buy = diff_val > 0
            slip = float(args.slippage_bps) / 10000.0
            fee_rate = float(args.fee_bps) / 10000.0
            eff_px = px * (1.0 + slip) if is_buy else px * (1.0 - slip)
            if eff_px <= 0:
                continue

            shares_delta = diff_val / eff_px
            trade_value = abs(shares_delta * eff_px)
            fee = trade_value * fee_rate

            if is_buy and (trade_value + fee) > st.cash:
                affordable = max(0.0, st.cash / (1.0 + fee_rate))
                shares_delta = affordable / eff_px
                trade_value = abs(shares_delta * eff_px)
                fee = trade_value * fee_rate

            st.positions[sym] = float(current_sh + shares_delta)
            if abs(st.positions[sym]) < 1e-9:
                st.positions.pop(sym, None)

            st.cash = float(st.cash - (shares_delta * eff_px) - fee)

            _write_csv_row(
                outdir / f"paper_trades_{date_str}.csv",
                trade_header,
                {
                    "date": date_str,
                    "symbol": sym,
                    "price": px,
                    "effective_price": eff_px,
                    "shares_delta": shares_delta,
                    "trade_value": trade_value,
                    "fee": fee,
                    "cash_after": st.cash,
                    "shares_after": st.positions.get(sym, 0.0),
                    "target_weight": tgt_w,
                },
            )

    pv_after = _portfolio_value(st=st, prices=prices)

    settle_date: Optional[str] = None
    pv_mtm: Optional[float] = None
    pnl_mtm: Optional[float] = None
    if int(args.mtm_next_day) == 1:
        base_symbol = "SPY" if "SPY" in symbols else (symbols[0] if symbols else "")
        if args.settle_date and str(args.settle_date).strip():
            settle_date = str(args.settle_date).strip()
        elif base_symbol:
            settle_date = _next_trading_date_from_raw(data_dir=Path(args.data_dir), symbol=base_symbol, date_str=date_str)
        if settle_date:
            mtm_prices = _load_raw_close_prices_for_date(
                data_dir=Path(args.data_dir),
                symbols=symbols,
                date_str=str(settle_date),
            )
            pv_mtm = _portfolio_value(st=st, prices=mtm_prices)
            pnl_mtm = float(pv_mtm - pv_after)

    nav_header = ["date", "value", "cash", "note", "settle_date", "value_mtm", "pnl_mtm"]
    _write_csv_row(
        outdir / "paper_nav.csv",
        nav_header,
        {
            "date": date_str,
            "value": pv_after,
            "cash": st.cash,
            "note": "virtual_event" if args.virtual_event else "daily",
            "settle_date": settle_date or "",
            "value_mtm": "" if pv_mtm is None else pv_mtm,
            "pnl_mtm": "" if pnl_mtm is None else pnl_mtm,
        },
    )

    _save_state(state_path, st)

    if args.meta_out:
        meta_path = Path(args.meta_out)
    else:
        meta_path = outdir / f"paper_meta_{date_str}.json"
    _write_meta(
        meta_path,
        {
            "date": date_str,
            "decision": str(decision_path),
            "daily_dir": str(daily_dir),
            "data_dir": str(args.data_dir),
            "outdir": str(outdir),
            "state": str(state_path),
            "initial_cash": float(args.initial_cash),
            "rebalance_threshold": float(args.rebalance_threshold),
            "slippage_bps": float(args.slippage_bps),
            "fee_bps": float(args.fee_bps),
            "min_trade_value": float(args.min_trade_value),
            "mtm_next_day": int(args.mtm_next_day),
            "skip_trade": int(args.skip_trade),
            "portfolio_value_before": float(pv_before),
            "portfolio_value_after": float(pv_after),
            "settle_date": settle_date,
            "portfolio_value_mtm": pv_mtm,
            "pnl_mtm": pnl_mtm,
            "symbols": symbols,
            "targets": targets,
            "virtual_event": args.virtual_event or "",
        },
    )

    print(f"Decision: {decision_path}")
    print(f"Outdir: {outdir}")
    print(f"Portfolio value before: {pv_before:.2f}")
    print(f"Portfolio value after:  {pv_after:.2f}")
    if pv_mtm is not None and settle_date:
        print(f"Settle date: {settle_date}")
        print(f"Portfolio value mtm:   {pv_mtm:.2f} (pnl={pnl_mtm:+.2f})")
    print(f"State saved: {state_path}")


if __name__ == "__main__":
    main()

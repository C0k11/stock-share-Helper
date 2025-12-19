#!/usr/bin/env python

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import streamlit as st
except Exception:
    raise SystemExit(
        "Missing dependency: streamlit. Install with: .\\venv311\\Scripts\\pip install streamlit pandas plotly"
    )

import pandas as pd
import plotly.express as px


DEFAULT_PAPER_DIR = Path("data/paper_rolltest") if Path("data/paper_rolltest").exists() else Path("data/paper")


def _read_json(path: Path) -> Any:
    # Accept BOM if written by PowerShell.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_state(paper_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fp = paper_dir / "state.json"
    if not fp.exists():
        return {}, {}
    obj = _read_json(fp)
    if not isinstance(obj, dict):
        return {}, {}
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    items = obj.get("items") if isinstance(obj.get("items"), dict) else obj
    if not isinstance(items, dict):
        items = {}
    return dict(meta), dict(items)


def _load_portfolio(paper_dir: Path) -> Dict[str, Any]:
    fp = paper_dir / "portfolio.json"
    if not fp.exists():
        return {}
    obj = _read_json(fp)
    return obj if isinstance(obj, dict) else {}


def _state_to_df(items: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sym, payload in items.items():
        if not isinstance(payload, dict):
            continue
        config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        state = payload.get("state") if isinstance(payload.get("state"), dict) else payload
        pos = int(state.get("current_position") or 0)
        pending = int(state.get("pending_reverse_count") or 0)
        confirm_n = int(config.get("reverse_confirm_days") or 1)
        rows.append(
            {
                "ticker": str(sym),
                "current_position": pos,
                "days_held": int(state.get("days_held") or 0),
                "pending_reverse_count": pending,
                "reverse_confirm_days": confirm_n,
                "hold_policy": str(config.get("hold_policy") or ""),
                "min_hold_days": int(config.get("min_hold_days") or 0),
                "pending_ratio": float(pending) / float(confirm_n) if confirm_n > 0 else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["current_position", "ticker"], ascending=[False, True])
    return df


def _read_orders(paper_dir: Path) -> pd.DataFrame:
    orders_dir = paper_dir / "orders"
    if not orders_dir.exists():
        return pd.DataFrame([])
    files = sorted(orders_dir.glob("orders_*.csv"))
    if not files:
        return pd.DataFrame([])
    dfs: List[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        df["file"] = fp.name
        dfs.append(df)
    if not dfs:
        return pd.DataFrame([])
    out = pd.concat(dfs, ignore_index=True)
    if "date" in out.columns:
        out["date"] = out["date"].astype(str)
    if "trade_value" in out.columns:
        out["trade_value"] = pd.to_numeric(out["trade_value"], errors="coerce")
    if "fee" in out.columns:
        out["fee"] = pd.to_numeric(out["fee"], errors="coerce")
    return out


def _read_nav(paper_dir: Path) -> pd.DataFrame:
    history_dir = paper_dir / "history"
    rows: List[Dict[str, Any]] = []
    for fp in sorted(history_dir.glob("portfolio_*.json")) if history_dir.exists() else []:
        try:
            obj = _read_json(fp)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        date_str = str(obj.get("date") or fp.stem.replace("portfolio_", ""))
        equity = obj.get("equity")
        cash = obj.get("cash")
        rows.append({"date": date_str, "equity": equity, "cash": cash, "source": fp.name})

    cur = _load_portfolio(paper_dir)
    if cur:
        date_str = str(cur.get("date") or "")
        if date_str:
            rows.append({"date": date_str, "equity": cur.get("equity"), "cash": cur.get("cash"), "source": "portfolio.json"})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df["cash"] = pd.to_numeric(df["cash"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df


def _latest_log_tail(paper_dir: Path, n_lines: int = 50) -> str:
    log_dir = paper_dir / "logs"
    if not log_dir.exists():
        return ""
    files = glob.glob(str(log_dir / "paper_*.log"))
    if not files:
        return ""
    latest = max(files, key=os.path.getmtime)
    try:
        lines = Path(latest).read_text(encoding="utf-8").splitlines()
    except Exception:
        return ""
    tail = lines[-int(n_lines) :] if len(lines) > n_lines else lines
    return "\n".join(tail)


def _extract_clear_from_decision(dec: Any) -> List[str]:
    if not isinstance(dec, dict):
        return []
    items = dec.get("items") if isinstance(dec.get("items"), dict) else {}
    out: List[str] = []
    for sym, it in items.items():
        if not isinstance(it, dict):
            continue
        final = it.get("final") if isinstance(it.get("final"), dict) else {}
        if str(final.get("action") or "").strip().upper() == "CLEAR":
            out.append(str(sym))
    return out


def _scan_recent_clear(paper_dir: Path, max_files: int = 50) -> Tuple[str, List[str]]:
    candidates: List[Tuple[str, Path]] = []
    candidates.append(("current", paper_dir / "state.json"))
    hist = paper_dir / "history"
    if hist.exists():
        for fp in sorted(hist.glob("state_*.json"), reverse=True)[: int(max_files)]:
            candidates.append((fp.name, fp))

    for _tag, state_fp in candidates:
        if not state_fp.exists():
            continue
        try:
            st_obj = _read_json(state_fp)
        except Exception:
            continue
        if not isinstance(st_obj, dict):
            continue
        meta = st_obj.get("meta") if isinstance(st_obj.get("meta"), dict) else {}
        date_str = str(meta.get("date") or "")
        sig_path = str(meta.get("signals") or "")
        if not sig_path:
            continue
        dp = Path(sig_path)
        if not dp.exists():
            continue
        try:
            dec = _read_json(dp)
        except Exception:
            continue
        clears = _extract_clear_from_decision(dec)
        if clears:
            return date_str, sorted(list(set(clears)))

    return "", []


def _load_latest_decisions(paper_dir: Path, meta: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    sig_path = str(meta.get("signals") or "").strip() if isinstance(meta, dict) else ""
    if sig_path:
        fp = Path(sig_path)
        if fp.exists():
            try:
                obj = _read_json(fp)
                if isinstance(obj, dict):
                    return str(fp), obj
            except Exception:
                pass

    candidates: List[Path] = []
    for p in sorted(paper_dir.glob("decisions_*.json"), reverse=True)[:10]:
        candidates.append(p)
    for p in sorted(Path("data").glob("decisions*.json"), reverse=True)[:10]:
        candidates.append(p)

    best: Optional[Path] = None
    best_mtime = -1.0
    for fp in candidates:
        try:
            mt = fp.stat().st_mtime
        except Exception:
            continue
        if mt > best_mtime:
            best_mtime = mt
            best = fp

    if best and best.exists():
        try:
            obj = _read_json(best)
            if isinstance(obj, dict):
                return str(best), obj
        except Exception:
            pass

    return "", {}


def _normalize_reasoning_trace(v: Any) -> List[str]:
    if isinstance(v, list):
        out = [str(x) for x in v if str(x).strip()]
        return out
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def _derive_order_action(row: Dict[str, Any], trade_dollar: float) -> str:
    try:
        trade_value = float(row.get("trade_value") or 0.0)
    except Exception:
        trade_value = 0.0
    try:
        target_pos = int(row.get("target_pos") or 0)
    except Exception:
        target_pos = 0

    if trade_value <= 0:
        return "HOLD"

    td = float(trade_dollar) if trade_dollar and trade_dollar > 0 else 10000.0
    if target_pos == 0:
        return "CLOSE"
    if trade_value >= 1.5 * td:
        return "FLIP"
    if target_pos > 0:
        return "OPEN_LONG"
    return "OPEN_SHORT"


def main() -> None:
    st.set_page_config(layout="wide", page_title="AI Trader Cockpit")

    st.sidebar.title("Control")
    paper_dir_str = st.sidebar.text_input("Paper Dir", value=str(DEFAULT_PAPER_DIR))
    paper_dir = Path(paper_dir_str)
    if st.sidebar.button("Refresh"):
        st.rerun()

    st.title("AI Trader Cockpit")

    meta, items = _load_state(paper_dir)
    df_state = _state_to_df(items)
    df_orders = _read_orders(paper_dir)
    df_nav = _read_nav(paper_dir)
    portfolio = _load_portfolio(paper_dir)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        active = int((df_state["current_position"] != 0).sum()) if not df_state.empty else 0
        st.metric("Active Positions", active)
    with c2:
        executed = int((pd.to_numeric(df_orders.get("trade_value"), errors="coerce").fillna(0) > 0).sum()) if not df_orders.empty else 0
        st.metric("Executed Trades", executed)
    with c3:
        last_date = str(meta.get("date") or "")
        st.metric("Last Update", last_date if last_date else "N/A")
    with c4:
        equity = meta.get("equity") if isinstance(meta, dict) else None
        st.metric("Equity", f"{float(equity):.2f}" if equity is not None else "N/A")

    st.markdown("---")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("NAV")
        if df_nav.empty or df_nav["equity"].dropna().empty:
            st.info("No NAV history found yet (history/portfolio_*.json)")
        else:
            fig = px.line(df_nav, x="date", y="equity", markers=True)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Risk")
        pending_cnt = int((df_state["pending_reverse_count"] > 0).sum()) if not df_state.empty else 0
        st.metric("Pending Reverse", pending_cnt)

        last_clear_date, clear_tickers = _scan_recent_clear(paper_dir)
        if clear_tickers:
            tag = last_clear_date if last_clear_date else "(unknown date)"
            st.error(f"CLEAR @{tag}: " + ", ".join(clear_tickers))
        else:
            st.info("No CLEAR found in state history")

    st.subheader("Execution State")
    if df_state.empty:
        st.warning("state.json not found or empty")
    else:
        def _style_pending(v: Any) -> str:
            try:
                return "background-color: #ffe0b2" if float(v) > 0 else ""
            except Exception:
                return ""

        df_show = df_state.copy()
        df_show["position"] = df_show["current_position"].map({1: "LONG", 0: "FLAT", -1: "SHORT"}).fillna("FLAT")
        df_show = df_show[["ticker", "position", "days_held", "pending_reverse_count", "reverse_confirm_days", "hold_policy", "min_hold_days"]]
        st.dataframe(
            df_show.style.applymap(_style_pending, subset=["pending_reverse_count"]),
            use_container_width=True,
            height=360,
        )

    st.markdown("---")

    dec_path, decisions = _load_latest_decisions(paper_dir, meta)
    if decisions:
        st.subheader("AI Reasoning Inspector")
        if dec_path:
            st.caption(f"Decisions Source: {dec_path}")

        dec_items = decisions.get("items") if isinstance(decisions.get("items"), dict) else {}
        tickers = sorted([str(k) for k in dec_items.keys()])

        if tickers:
            selected = st.selectbox("Ticker", tickers)
            rec = dec_items.get(selected) if isinstance(dec_items.get(selected), dict) else {}
            parsed = rec.get("parsed") if isinstance(rec.get("parsed"), dict) else {}
            final = rec.get("final") if isinstance(rec.get("final"), dict) else {}

            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Final Action", str(final.get("action") or ""))
                st.write(f"Target Position: {final.get('target_position')}")
                trace = final.get("trace")
                if isinstance(trace, list) and trace:
                    st.write("Risk Trace")
                    st.code("\n".join([str(x) for x in trace[:10]]))

            with c2:
                st.write(f"Decision: {parsed.get('decision')}")
                st.write(f"Analysis: {parsed.get('analysis')}")
                rt = _normalize_reasoning_trace(parsed.get("reasoning_trace"))
                if rt:
                    st.write("Reasoning Trace")
                    for t in rt[:10]:
                        st.markdown(f"- {t}")
                else:
                    st.caption("No reasoning_trace found in parsed output")
        else:
            st.caption("No decisions.items found in decisions JSON")

    ocol, lcol = st.columns([2, 1])
    with ocol:
        st.subheader("Orders")
        if df_orders.empty:
            st.info("No orders_*.csv found")
        else:
            df = df_orders.copy()
            trade_dollar = float(meta.get("trade_dollar") or 10000.0) if isinstance(meta, dict) else 10000.0
            try:
                df["action"] = df.apply(lambda r: _derive_order_action(r.to_dict(), trade_dollar), axis=1)
            except Exception:
                df["action"] = ""
            if "trade_value" in df.columns:
                df["is_trade"] = pd.to_numeric(df["trade_value"], errors="coerce").fillna(0) > 0
                df = df.sort_values(["date", "ticker"], ascending=[False, True])
            cols = [c for c in ["date", "ticker", "action", "raw_signal", "target_pos", "trade_value", "fee", "shares_delta", "cash_after", "file"] if c in df.columns]
            st.dataframe(df[cols].head(200), use_container_width=True, height=360)

    with lcol:
        st.subheader("Latest Log")
        tail = _latest_log_tail(paper_dir, n_lines=60)
        if not tail:
            st.info("No paper_*.log found")
        else:
            st.text(tail)

    with st.expander("Raw State"):
        st.json({"meta": meta, "items": items})


if __name__ == "__main__":
    main()

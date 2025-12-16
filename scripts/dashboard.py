#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import streamlit as st
except Exception:
    raise SystemExit(
        "Missing dependency: streamlit. Install with: .\\venv311\\Scripts\\pip install streamlit pandas plotly"
    )

import pandas as pd
import plotly.express as px


DATA_DIR = Path("data/daily")


def _list_dates() -> List[str]:
    dates: List[str] = []
    for fp in DATA_DIR.glob("etf_features_*.json"):
        stem = fp.stem
        if stem.startswith("etf_features_"):
            dates.append(stem.replace("etf_features_", ""))
    return sorted(set(dates), reverse=True)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_features(date_str: str) -> Optional[Dict[str, Any]]:
    fp = DATA_DIR / f"etf_features_{date_str}.json"
    if not fp.exists():
        return None
    obj = _read_json(fp)
    return obj if isinstance(obj, dict) else None


def load_signals(date_str: str) -> List[Dict[str, Any]]:
    fp = DATA_DIR / f"signals_full_{date_str}.json"
    if not fp.exists():
        fp = DATA_DIR / f"signals_{date_str}.json"
    if not fp.exists():
        return []
    obj = _read_json(fp)
    if not isinstance(obj, list):
        return []
    return [it for it in obj if isinstance(it, dict)]


def load_trading_decision(date_str: str) -> Optional[Dict[str, Any]]:
    fp = DATA_DIR / f"trading_decision_{date_str}.json"
    if not fp.exists():
        fp = DATA_DIR / f"trading_decisions_{date_str}.json"
    if not fp.exists():
        return None
    obj = _read_json(fp)
    return obj if isinstance(obj, dict) else None


def signals_to_df(signals: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for it in signals:
        sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
        rows.append(
            {
                "market": str(it.get("market") or ""),
                "source": str(it.get("source") or ""),
                "published_at": str(it.get("published_at") or ""),
                "title": str(it.get("title") or ""),
                "event_type": str(sig.get("event_type") or ""),
                "sentiment": sig.get("sentiment"),
                "impact_equity": sig.get("impact_equity"),
                "impact_bond": sig.get("impact_bond"),
                "impact_gold": sig.get("impact_gold"),
                "summary": str(sig.get("summary") or ""),
                "url": str(it.get("url") or ""),
                "parse_ok": bool(it.get("parse_ok")),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for c in ["impact_equity", "impact_bond", "impact_gold", "sentiment"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["abs_impact_equity"] = df["impact_equity"].abs()
    return df


def features_to_df(features_obj: Dict[str, Any]) -> pd.DataFrame:
    items = features_obj.get("items")
    if not isinstance(items, list):
        return pd.DataFrame([])

    rows: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        tech = it.get("technical") if isinstance(it.get("technical"), dict) else {}
        mreg = it.get("market_regime") if isinstance(it.get("market_regime"), dict) else {}
        sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
        teacher = it.get("teacher") if isinstance(it.get("teacher"), dict) else {}
        rows.append(
            {
                "symbol": str(it.get("symbol") or ""),
                "feature_date": str(it.get("date") or ""),
                "close": tech.get("close"),
                "return_5d": tech.get("return_5d"),
                "return_21d": tech.get("return_21d"),
                "return_63d": tech.get("return_63d"),
                "volatility_20d": tech.get("volatility_20d"),
                "drawdown": tech.get("drawdown"),
                "max_drawdown_20d": tech.get("max_drawdown_20d"),
                "regime": str(mreg.get("regime") or ""),
                "regime_score": mreg.get("score"),
                "signal_strength": str(sig.get("strength") or ""),
                "signal_composite": sig.get("composite"),
                "teacher_target_position": teacher.get("target_position"),
                "teacher_target_position_profiled": teacher.get("target_position_profiled"),
                "teacher_risk_profile": str(teacher.get("risk_profile") or ""),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for c in [
        "close",
        "return_5d",
        "return_21d",
        "return_63d",
        "volatility_20d",
        "drawdown",
        "max_drawdown_20d",
        "regime_score",
        "signal_composite",
        "teacher_target_position",
        "teacher_target_position_profiled",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def decisions_to_df(decision_obj: Dict[str, Any]) -> pd.DataFrame:
    items = decision_obj.get("items") if isinstance(decision_obj.get("items"), dict) else {}
    rows: List[Dict[str, Any]] = []
    for sym, payload in items.items():
        if not isinstance(payload, dict):
            continue
        final = payload.get("final") if isinstance(payload.get("final"), dict) else {}
        rows.append(
            {
                "symbol": str(sym),
                "final_action": str(final.get("action") or ""),
                "final_target_position": final.get("target_position"),
                "parse_error": str(payload.get("parse_error") or ""),
                "parsed_ok": payload.get("parsed") is not None,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["final_target_position"] = pd.to_numeric(df["final_target_position"], errors="coerce")
    return df


def main() -> None:
    st.set_page_config(layout="wide", page_title="QuantBot Dashboard")
    st.title("QuantBot Operations Center")

    dates = _list_dates()
    if not dates:
        st.error("No etf_features_*.json found under data/daily")
        return

    selected_date = st.sidebar.selectbox("Select Date", dates)

    features = load_features(selected_date)
    signals = load_signals(selected_date)
    decision = load_trading_decision(selected_date)

    colA, colB = st.columns([1, 1])

    with colA:
        st.subheader("Market Features")
        if features is None:
            st.info("Missing etf_features file for this date")
        else:
            df_feat = features_to_df(features)
            if df_feat.empty:
                st.info("No feature items")
            else:
                st.dataframe(df_feat, use_container_width=True, height=240)

    with colB:
        st.subheader("Trading Decision")
        if decision is None:
            st.info("No trading_decision file for this date")
        else:
            df_dec = decisions_to_df(decision)
            if df_dec.empty:
                st.info("No decision items")
            else:
                st.dataframe(df_dec, use_container_width=True, height=240)

            rw = decision.get("risk_watch") if isinstance(decision.get("risk_watch"), dict) else {}
            if rw:
                with st.expander("Risk Watch Summary"):
                    st.json(rw)

    st.markdown("---")

    st.subheader("News Signals")
    df_sig = signals_to_df(signals)
    if df_sig.empty:
        st.info("No signals for this date")
        return

    fcol1, fcol2, fcol3, fcol4 = st.columns([1, 1, 1, 2])
    with fcol1:
        market_opt = ["ALL"] + sorted([x for x in df_sig["market"].dropna().unique().tolist() if str(x)])
        market_sel = st.selectbox("Market", market_opt)
    with fcol2:
        et_opt = ["ALL"] + sorted([x for x in df_sig["event_type"].dropna().unique().tolist() if str(x)])
        et_sel = st.selectbox("Event Type", et_opt)
    with fcol3:
        top_n = st.number_input("Top N (abs impact)", min_value=10, max_value=500, value=50, step=10)
    with fcol4:
        q = st.text_input("Search (title/summary)", value="")

    dff = df_sig.copy()
    if market_sel != "ALL":
        dff = dff[dff["market"] == market_sel]
    if et_sel != "ALL":
        dff = dff[dff["event_type"] == et_sel]
    if q.strip():
        ql = q.strip().lower()
        dff = dff[
            dff["title"].str.lower().str.contains(ql, na=False)
            | dff["summary"].str.lower().str.contains(ql, na=False)
        ]

    if "abs_impact_equity" in dff.columns:
        dff = dff.sort_values("abs_impact_equity", ascending=False)

    st.caption(f"Signals loaded: {len(df_sig)} | filtered: {len(dff)}")

    st.dataframe(
        dff[["published_at", "market", "event_type", "impact_equity", "title", "summary", "source", "url"]].head(int(top_n)),
        use_container_width=True,
        height=420,
    )

    st.markdown("---")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Event Type Counts")
        ct = dff["event_type"].value_counts().reset_index()
        ct.columns = ["event_type", "count"]
        fig = px.bar(ct, x="event_type", y="count")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Impact Equity Histogram")
        if dff["impact_equity"].notna().any():
            fig2 = px.histogram(dff, x="impact_equity", nbins=11)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No numeric impact_equity")


if __name__ == "__main__":
    main()

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _results_dir() -> Path:
    return _repo_root() / "results"


def _data_dir() -> Path:
    return _repo_root() / "data"


def list_runs() -> List[str]:
    rd = _results_dir()
    if not rd.exists():
        return []
    out: List[str] = []
    for p in rd.iterdir():
        if p.is_dir() and (p / "metrics.json").exists():
            out.append(p.name)
    out.sort()
    return out


def list_systems(run_id: str) -> List[str]:
    run_dir = _results_dir() / str(run_id)
    if not run_dir.exists():
        return []
    out: List[str] = []
    for p in run_dir.iterdir():
        if p.is_dir() and (p / "metrics.json").exists():
            out.append(p.name)
    out.sort()
    return out


def _iter_decision_files(system_dir: Path) -> List[Path]:
    fps = sorted(system_dir.glob("decisions_*.json"))
    return [p for p in fps if p.is_file()]


def list_dates(run_id: str) -> List[str]:
    run_dir = _results_dir() / str(run_id)
    if not run_dir.exists():
        return []
    dates: set[str] = set()
    for sys_name in list_systems(run_id):
        sys_dir = run_dir / sys_name
        for fp in _iter_decision_files(sys_dir):
            try:
                payload = _read_json(fp)
            except Exception:
                continue
            days = payload.get("days") if isinstance(payload, dict) else None
            if isinstance(days, dict):
                for d in days.keys():
                    if str(d).strip():
                        dates.add(str(d))
    return sorted(list(dates))


def _load_day(system_dir: Path, date_str: str) -> Optional[Dict[str, Any]]:
    for fp in _iter_decision_files(system_dir):
        try:
            payload = _read_json(fp)
        except Exception:
            continue
        days = payload.get("days") if isinstance(payload, dict) else None
        if isinstance(days, dict):
            day = days.get(str(date_str))
            if isinstance(day, dict):
                return day
    return None


def list_tickers(run_id: str, system: str, date_str: str) -> List[str]:
    sys_dir = _results_dir() / str(run_id) / str(system)
    day = _load_day(sys_dir, date_str)
    if not isinstance(day, dict):
        return []
    items = day.get("items") if isinstance(day.get("items"), dict) else {}
    return sorted([str(k).upper() for k in items.keys() if str(k).strip()])


def _read_daily_row(system_dir: Path, system: str, date_str: str, ticker: str) -> Optional[Dict[str, Any]]:
    fp = system_dir / "daily.csv"
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp)
    except Exception:
        return None
    if df is None or df.empty:
        return None

    t = str(ticker).upper().strip()
    s = str(system).strip()
    d = str(date_str).strip()
    try:
        sub = df[(df["date"].astype(str) == d) & (df["ticker"].astype(str).str.upper() == t) & (df["system"].astype(str) == s)]
    except Exception:
        return None
    if sub.empty:
        return None
    row = sub.iloc[0].to_dict()
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, float) and (pd.isna(v) or pd.isnull(v)):
            out[k] = None
        else:
            out[k] = v
    return out


def read_ohlc_window(ticker: str, date_str: str, lookback_days: int = 60) -> pd.DataFrame:
    fp = _data_dir() / "raw" / f"{str(ticker).upper()}.parquet"
    if not fp.exists():
        return pd.DataFrame()

    try:
        df = pd.read_parquet(fp)
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    try:
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        idx = idx.normalize()
        df = df.copy()
        df.index = idx
    except Exception:
        pass

    try:
        d = pd.to_datetime(str(date_str)).normalize()
    except Exception:
        return pd.DataFrame()

    if d not in df.index:
        return pd.DataFrame()

    try:
        n = int(lookback_days)
    except Exception:
        n = 60
    if n <= 0:
        n = 60

    df = df.sort_index()
    pos = df.index.get_loc(d)
    if isinstance(pos, slice):
        pos = pos.stop - 1
    start = max(0, int(pos) - int(n) + 1)
    win = df.iloc[start : int(pos) + 1].copy()

    def col_pick(primary: str, alts: List[str]) -> Optional[str]:
        for c in [primary] + list(alts):
            if c in win.columns:
                return c
        return None

    c_open = col_pick("open", ["Open", "o"])
    c_high = col_pick("high", ["High", "h"])
    c_low = col_pick("low", ["Low", "l"])
    c_close = col_pick("close", ["Close", "c"])
    c_vol = col_pick("volume", ["Volume", "v"])

    out = pd.DataFrame(index=win.index)
    if c_open:
        out["Open"] = win[c_open].astype(float)
    if c_high:
        out["High"] = win[c_high].astype(float)
    if c_low:
        out["Low"] = win[c_low].astype(float)
    if c_close:
        out["Close"] = win[c_close].astype(float)
    if c_vol:
        try:
            out["Volume"] = win[c_vol].astype(float)
        except Exception:
            pass

    out = out.dropna(subset=[c for c in ["Open", "High", "Low", "Close"] if c in out.columns], how="any")
    return out


def snapshot(run_id: str, system: str, date_str: str, ticker: str, lookback_days: int = 60) -> Dict[str, Any]:
    system_dir = _results_dir() / str(run_id) / str(system)
    day = _load_day(system_dir, date_str)
    if not isinstance(day, dict):
        raise KeyError(f"date not found: {date_str}")

    items = day.get("items") if isinstance(day.get("items"), dict) else {}
    rec = items.get(str(ticker).upper()) or items.get(str(ticker))
    if not isinstance(rec, dict):
        raise KeyError(f"ticker not found: {ticker}")

    daily_row = _read_daily_row(system_dir, system=system, date_str=date_str, ticker=ticker)
    ohlc_df = read_ohlc_window(ticker=ticker, date_str=date_str, lookback_days=lookback_days)

    return {
        "run_id": str(run_id),
        "system": str(system),
        "date": str(date_str),
        "ticker": str(ticker).upper(),
        "ohlc_df": ohlc_df,
        "decision": {
            "parsed": rec.get("parsed"),
            "system2": rec.get("system2"),
            "chartist": rec.get("chartist"),
            "final": rec.get("final"),
            "parse_error": rec.get("parse_error"),
            "raw": rec.get("raw"),
            "expert": rec.get("expert"),
            "router": rec.get("router"),
        },
        "risk_watch": day.get("risk_watch"),
        "macro": day.get("macro"),
        "daily": daily_row,
    }


def _run_native_gui() -> None:
    import tkinter as tk
    from tkinter import ttk

    import mplfinance as mpf
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title("AI Trading Terminal (Native)")
    root.geometry("1500x900")

    top = ttk.Frame(root)
    top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

    content = ttk.Frame(root)
    content.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

    left = ttk.Frame(content)
    left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    right = ttk.Frame(content)
    right.pack(side=tk.RIGHT, fill=tk.BOTH)

    ctrl = ttk.LabelFrame(left, text="Controls")
    ctrl.pack(side=tk.TOP, fill=tk.X)

    run_var = tk.StringVar()
    sys_var = tk.StringVar()
    date_var = tk.StringVar()
    ticker_var = tk.StringVar()
    lookback_var = tk.IntVar(value=60)

    runs = list_runs()
    if runs:
        run_var.set(runs[-1])

    ttk.Label(ctrl, text="Run").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    run_cb = ttk.Combobox(ctrl, textvariable=run_var, values=runs, width=45, state="readonly")
    run_cb.grid(row=0, column=1, sticky="we", padx=6, pady=4)

    ttk.Label(ctrl, text="System").grid(row=0, column=2, sticky="w", padx=6, pady=4)
    sys_cb = ttk.Combobox(ctrl, textvariable=sys_var, values=[], width=20, state="readonly")
    sys_cb.grid(row=0, column=3, sticky="we", padx=6, pady=4)

    ttk.Label(ctrl, text="Date").grid(row=1, column=0, sticky="w", padx=6, pady=4)
    date_cb = ttk.Combobox(ctrl, textvariable=date_var, values=[], width=45, state="readonly")
    date_cb.grid(row=1, column=1, sticky="we", padx=6, pady=4)

    ttk.Label(ctrl, text="Ticker").grid(row=1, column=2, sticky="w", padx=6, pady=4)
    ticker_cb = ttk.Combobox(ctrl, textvariable=ticker_var, values=[], width=20, state="readonly")
    ticker_cb.grid(row=1, column=3, sticky="we", padx=6, pady=4)

    ttk.Label(ctrl, text="Lookback(days)").grid(row=2, column=0, sticky="w", padx=6, pady=4)
    lookback_spin = ttk.Spinbox(ctrl, from_=20, to=200, increment=5, textvariable=lookback_var, width=10)
    lookback_spin.grid(row=2, column=1, sticky="w", padx=6, pady=4)

    ctrl.columnconfigure(1, weight=1)

    score = ttk.LabelFrame(left, text="Scoreboard")
    score.pack(side=tk.TOP, fill=tk.X, pady=8)

    action_lbl = ttk.Label(score, text="Action: ")
    action_lbl.grid(row=0, column=0, sticky="w", padx=6, pady=2)
    pnl_lbl = ttk.Label(score, text="PnL(h1_net): ")
    pnl_lbl.grid(row=0, column=1, sticky="w", padx=6, pady=2)
    exec_lbl = ttk.Label(score, text="Execution: ")
    exec_lbl.grid(row=1, column=0, sticky="w", padx=6, pady=2)
    macro_lbl = ttk.Label(score, text="Macro: ")
    macro_lbl.grid(row=1, column=1, sticky="w", padx=6, pady=2)

    chart_frame = ttk.LabelFrame(left, text="OHLC / Chartist")
    chart_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    fig = mpf.figure(style="yahoo", figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    right_nb = ttk.Notebook(right)
    right_nb.pack(fill=tk.BOTH, expand=True)

    tab_sys2 = ttk.Frame(right_nb)
    tab_trace = ttk.Frame(right_nb)
    tab_debug = ttk.Frame(right_nb)
    right_nb.add(tab_sys2, text="System2")
    right_nb.add(tab_trace, text="Trace")
    right_nb.add(tab_debug, text="Debug")

    sys2_text = tk.Text(tab_sys2, wrap="word", width=50)
    sys2_text.pack(fill=tk.BOTH, expand=True)

    trace_text = tk.Text(tab_trace, wrap="word", width=50)
    trace_text.pack(fill=tk.BOTH, expand=True)

    debug_text = tk.Text(tab_debug, wrap="none", width=50)
    debug_text.pack(fill=tk.BOTH, expand=True)

    def _set_text(w: tk.Text, s: str) -> None:
        w.configure(state="normal")
        w.delete("1.0", "end")
        w.insert("end", s)
        w.configure(state="disabled")

    def _refresh_systems(*_a) -> None:
        run_id = run_var.get().strip()
        syss = list_systems(run_id)
        sys_cb.configure(values=syss)
        if syss:
            sys_var.set(syss[0])
        else:
            sys_var.set("")
        _refresh_dates()

    def _refresh_dates(*_a) -> None:
        run_id = run_var.get().strip()
        dts = list_dates(run_id)
        date_cb.configure(values=dts)
        if dts:
            date_var.set(dts[-1])
        else:
            date_var.set("")
        _refresh_tickers()

    def _refresh_tickers(*_a) -> None:
        run_id = run_var.get().strip()
        sys_name = sys_var.get().strip()
        d = date_var.get().strip()
        if not (run_id and sys_name and d):
            ticker_cb.configure(values=[])
            ticker_var.set("")
            return
        tks = list_tickers(run_id, sys_name, d)
        ticker_cb.configure(values=tks)
        if tks:
            ticker_var.set(tks[0])
        else:
            ticker_var.set("")
        _render()

    def _render(*_a) -> None:
        run_id = run_var.get().strip()
        sys_name = sys_var.get().strip()
        d = date_var.get().strip()
        tk_sym = ticker_var.get().strip().upper()
        if not (run_id and sys_name and d and tk_sym):
            return
        try:
            snap = snapshot(run_id, sys_name, d, tk_sym, lookback_days=int(lookback_var.get()))
        except Exception as e:
            _set_text(debug_text, f"snapshot error: {e}")
            return

        final = snap.get("decision", {}).get("final") if isinstance(snap.get("decision"), dict) else {}
        action = str((final or {}).get("action") or "").upper()
        tr = (final or {}).get("trace") if isinstance((final or {}).get("trace"), list) else []

        daily = snap.get("daily") if isinstance(snap.get("daily"), dict) else {}
        pnl = daily.get("pnl_h1_net")
        exec_note = daily.get("exec_note")
        exec_edge = daily.get("exec_edge")

        macro = snap.get("macro") if isinstance(snap.get("macro"), dict) else {}
        macro_score = macro.get("risk_score")
        macro_gear = macro.get("gear")

        action_lbl.configure(text=f"Action: {action}  pos={daily.get('target_position')}")
        pnl_lbl.configure(text=f"PnL(h1_net): {pnl}")
        exec_lbl.configure(text=f"Execution: {exec_note}  edge={exec_edge}  fee={daily.get('fee')}")
        macro_lbl.configure(text=f"Macro: score={macro_score} gear={macro_gear} mult={macro.get('multiplier')}")

        df = snap.get("ohlc_df")
        if isinstance(df, pd.DataFrame) and (not df.empty):
            ax.clear()
            try:
                mpf.plot(df, type="candle", ax=ax, volume=False, style="yahoo", axtitle=f"{tk_sym} ({d})")
            except Exception:
                ax.text(0.5, 0.5, "OHLC plot error", ha="center", va="center")
            canvas.draw()
        else:
            ax.clear()
            ax.text(0.5, 0.5, "OHLC unavailable", ha="center", va="center")
            canvas.draw()

        dec = snap.get("decision") if isinstance(snap.get("decision"), dict) else {}
        chartist = dec.get("chartist") if isinstance(dec.get("chartist"), dict) else {}

        sys2 = dec.get("system2") if isinstance(dec.get("system2"), dict) else {}
        if sys2:
            judge = sys2.get("judge") if isinstance(sys2.get("judge"), dict) else {}
            critic = sys2.get("critic") if isinstance(sys2.get("critic"), dict) else {}
            proposal = sys2.get("proposal") if isinstance(sys2.get("proposal"), dict) else {}
            parts = [
                f"Judge: {judge.get('final_decision')}\n{judge.get('rationale')}\n",
                f"Proposal: {proposal.get('decision')}\n{proposal.get('analysis')}\n",
                "Critic:",
                f"  suggested_decision={critic.get('suggested_decision')}",
            ]
            rs = critic.get("reasons")
            if isinstance(rs, list):
                for r in rs:
                    parts.append(f"  - {r}")
            _set_text(sys2_text, "\n".join([str(x) for x in parts if x is not None]))
        else:
            _set_text(sys2_text, "No System-2 debate recorded.")

        _set_text(trace_text, "\n".join([str(x) for x in tr]))

        dbg = {
            "chartist": chartist,
            "parsed": dec.get("parsed"),
            "daily": daily,
            "risk_watch": snap.get("risk_watch"),
            "macro": macro,
        }
        _set_text(debug_text, json.dumps(dbg, ensure_ascii=False, indent=2, default=str))

    run_cb.bind("<<ComboboxSelected>>", _refresh_systems)
    sys_cb.bind("<<ComboboxSelected>>", _refresh_dates)
    date_cb.bind("<<ComboboxSelected>>", _refresh_tickers)
    ticker_cb.bind("<<ComboboxSelected>>", _render)

    ttk.Button(ctrl, text="Reload", command=_render).grid(row=2, column=3, sticky="e", padx=6, pady=4)

    _refresh_systems()

    root.mainloop()


if __name__ == "__main__":
    _run_native_gui()

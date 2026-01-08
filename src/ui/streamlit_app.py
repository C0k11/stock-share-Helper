import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


API_BASE_DEFAULT = "http://localhost:8000/api/v1"


st.set_page_config(page_title="交易助手玛丽", layout="wide")

st.markdown(
    """
<style>
.metric-card {
  background-color: rgba(17,24,39,0.92);
  padding: 14px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  border-left: 5px solid #4e8cff;
}
.bull-msg { background-color: #e6fffa; padding: 10px; border-radius: 5px; margin-bottom: 8px; }
.bear-msg { background-color: #fff5f5; padding: 10px; border-radius: 5px; margin-bottom: 8px; }
.judge-msg { background-color: #f7f7f7; padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-bottom: 8px; font-weight: 600; }
.small-muted { color: rgba(229,231,235,0.78); font-size: 13px; }

.metric-card b, .metric-card strong {
  color: rgba(229,231,235,0.94);
}

label, .stMarkdown, .stText, .stCaption, p, span {
  color: rgba(229,231,235,0.92);
}

.stSelectbox label, .stTextInput label, .stSlider label {
  color: rgba(229,231,235,0.92) !important;
  font-weight: 800 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def _req_json(url: str) -> dict:
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:2000]}")
    obj = resp.json()
    return obj if isinstance(obj, dict) else {}


def _repo_root() -> Path:
    # file = <repo>/src/ui/streamlit_app.py
    return Path(__file__).resolve().parents[2]


def _count_jsonl_types(jsonl_paths: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in jsonl_paths:
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    t = str(obj.get("type") or "").strip() or "unknown"
                    counts[t] = int(counts.get(t, 0)) + 1
        except Exception:
            continue
    return counts


@st.cache_data(ttl=20)
def get_ouroboros_status() -> dict:
    root = _repo_root()
    traj_dir = root / "data" / "evolution" / "trajectories"
    fin_dir = root / "data" / "finetune" / "evolution"

    traj_files = []
    try:
        if traj_dir.exists():
            traj_files = sorted([p for p in traj_dir.glob("*.jsonl") if p.is_file()])
    except Exception:
        traj_files = []

    counts = _count_jsonl_types(traj_files)
    newest = None
    try:
        if traj_files:
            newest = max(traj_files, key=lambda p: p.stat().st_mtime)
    except Exception:
        newest = None

    out_sft = fin_dir / "sft_nightly.json"
    out_dpo = fin_dir / "dpo_nightly.jsonl"
    out_alpha = fin_dir / "dpo_alpha_nightly.jsonl"

    def _size(p: Path) -> int:
        try:
            return int(p.stat().st_size)
        except Exception:
            return 0

    return {
        "traj_dir": str(traj_dir),
        "traj_files": [str(p.name) for p in traj_files],
        "newest_traj": str(newest.name) if newest else "",
        "counts": counts,
        "outputs": {
            "sft_nightly": {"path": str(out_sft), "exists": out_sft.exists(), "size": _size(out_sft)},
            "dpo_nightly": {"path": str(out_dpo), "exists": out_dpo.exists(), "size": _size(out_dpo)},
            "dpo_alpha_nightly": {"path": str(out_alpha), "exists": out_alpha.exists(), "size": _size(out_alpha)},
        },
    }


def _pick_python_for_nightly(repo_root: Path) -> str:
    # Prefer pinned venv311 python if present, otherwise use current interpreter.
    cand = repo_root / "venv311" / "Scripts" / "python.exe"
    if cand.exists():
        return str(cand)
    return sys.executable


def run_nightly_evolution_dry_run() -> dict:
    root = _repo_root()
    py = _pick_python_for_nightly(root)
    cmd = [
        str(py),
        "scripts/nightly_evolution.py",
        "--dry-run",
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    out = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return {
        "cmd": " ".join(cmd),
        "returncode": int(out.returncode),
        "stdout": str(out.stdout or ""),
        "stderr": str(out.stderr or ""),
    }


def _api_post_json(url: str, payload: dict | None = None) -> dict:
    resp = requests.post(url, json=(payload or {}), timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:2000]}")
    obj = resp.json()
    return obj if isinstance(obj, dict) else {}


def _api_get_json(url: str) -> dict:
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:2000]}")
    obj = resp.json()
    return obj if isinstance(obj, dict) else {}


def start_nightly_train(api_base: str) -> dict:
    return _api_post_json(f"{api_base}/evolution/nightly/train/start")


def stop_nightly_train(api_base: str) -> dict:
    return _api_post_json(f"{api_base}/evolution/nightly/train/stop")


def get_nightly_train_status(api_base: str) -> dict:
    return _api_get_json(f"{api_base}/evolution/nightly/train/status?tail_bytes=16000")


@st.cache_data(ttl=60)
def get_runs(api_base: str) -> list[dict]:
    obj = _req_json(f"{api_base}/runs")
    return obj.get("runs") if isinstance(obj.get("runs"), list) else []


@st.cache_data(ttl=60)
def get_dates(api_base: str, run_id: str) -> list[str]:
    obj = _req_json(f"{api_base}/runs/{run_id}/dates")
    dates = obj.get("dates") if isinstance(obj.get("dates"), list) else []
    return [str(x) for x in dates if str(x).strip()]


@st.cache_data(ttl=60)
def get_tickers(api_base: str, run_id: str, date_str: str) -> dict:
    return _req_json(f"{api_base}/runs/{run_id}/tickers/{date_str}")


@st.cache_data(ttl=60)
def get_snapshot(api_base: str, run_id: str, system: str, date_str: str, ticker: str, lookback_days: int = 60) -> dict | None:
    try:
        return _req_json(
            f"{api_base}/snapshot/{run_id}/{system}/{date_str}/{ticker}?lookback_days={int(lookback_days)}"
        )
    except Exception:
        return None


def _fmt_float(x, nd: int = 4) -> str:
    try:
        return f"{float(x):.{int(nd)}f}"
    except Exception:
        return ""


def _render_trace(lines: list[str]) -> None:
    for t in lines:
        s = str(t)
        if "[RISK]" in s or "FORCE" in s:
            st.error(s)
        elif "[MACRO]" in s:
            st.warning(s)
        elif "[CHARTIST]" in s:
            st.info(s)
        elif "[GATEKEEPER]" in s:
            st.warning(s)
        else:
            st.text(s)

st.markdown("### 控制塔")
c_api, c_sel = st.columns([1, 2])
with c_api:
    api_base = st.text_input("API 地址", value=API_BASE_DEFAULT, key="api_base")
    if st.button("刷新（清空缓存）", key="refresh_cache"):
        st.cache_data.clear()

    with st.expander("Ouroboros 素材与 Nightly Evolution", expanded=False):
        st.caption("素材来源：data/evolution/trajectories/*.jsonl；输出：data/finetune/evolution/")
        st0 = get_ouroboros_status()
        counts = st0.get("counts") if isinstance(st0.get("counts"), dict) else {}
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.write(
                {
                    "trajectory": int(counts.get("trajectory", 0)),
                    "feedback": int(counts.get("feedback", 0)),
                    "outcome": int(counts.get("outcome", 0)),
                    "newest": str(st0.get("newest_traj") or ""),
                }
            )
        with col_b:
            outs = st0.get("outputs") if isinstance(st0.get("outputs"), dict) else {}
            st.write(outs)

        if st.button("运行 nightly_evolution --dry-run", key="nightly_dry_run"):
            res = run_nightly_evolution_dry_run()
            st.session_state["nightly_last"] = res
            st.cache_data.clear()

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("一键训练（SFT+DPO）", key="nightly_train_start"):
                try:
                    res = start_nightly_train(api_base)
                    st.session_state["nightly_train_last"] = res
                except Exception as e:
                    st.session_state["nightly_train_last"] = {"ok": False, "error": str(e)}
        with c2:
            if st.button("停止训练", key="nightly_train_stop"):
                try:
                    res = stop_nightly_train(api_base)
                    st.session_state["nightly_train_last"] = res
                except Exception as e:
                    st.session_state["nightly_train_last"] = {"ok": False, "error": str(e)}
        with c3:
            if st.button("刷新训练状态", key="nightly_train_refresh"):
                try:
                    res = get_nightly_train_status(api_base)
                    st.session_state["nightly_train_status"] = res
                except Exception as e:
                    st.session_state["nightly_train_status"] = {"ok": False, "error": str(e)}

        last = st.session_state.get("nightly_last")
        if isinstance(last, dict):
            st.text_input("Command", value=str(last.get("cmd") or ""), disabled=True, key="nightly_cmd")
            st.text_input("Return code", value=str(last.get("returncode")), disabled=True, key="nightly_rc")
            st.text_area("STDOUT", value=str(last.get("stdout") or ""), height=220, key="nightly_out")
            if str(last.get("stderr") or "").strip():
                st.text_area("STDERR", value=str(last.get("stderr") or ""), height=160, key="nightly_err")

        try:
            status = get_nightly_train_status(api_base)
            st.session_state["nightly_train_status"] = status
        except Exception:
            status = st.session_state.get("nightly_train_status")

        st.subheader("Nightly Evolution 训练状态")
        if isinstance(status, dict):
            st.write({"running": bool(status.get("running")), "pid": status.get("pid"), "returncode": status.get("returncode"), "log_path": status.get("log_path")})
            meta = status.get("meta") if isinstance(status.get("meta"), dict) else {}
            out_meta = meta.get("outputs") if isinstance(meta.get("outputs"), dict) else {}
            next_adapter = str(out_meta.get("next_dpo_adapter") or "").strip()
            if next_adapter:
                st.text_input("建议切换到的 adapter", value=next_adapter, disabled=True, key="nightly_next_adapter")
            tail = str(status.get("log_tail") or "")
            st.text_area("训练日志尾部", value=tail, height=240, key="nightly_train_tail")

runs = []
try:
    runs = get_runs(api_base)
except Exception:
    runs = []

if not runs:
    st.error("无法连接到后端 API（uvicorn 是否在运行？）")
    st.stop()

run_ids = [str(r.get("run_id")) for r in runs if str(r.get("run_id") or "").strip()]
run_ids = [x for x in run_ids if x]

with c_sel:
    selected_run = st.selectbox("选择 Run", run_ids, index=0, key="selected_run")
    st.caption("Run 对应一次 Phase 运行（run_id），与工程日志中的运行编号一致。")

run_meta = next((r for r in runs if str(r.get("run_id")) == str(selected_run)), {})
systems = run_meta.get("systems") if isinstance(run_meta.get("systems"), list) else []
if not systems:
    systems = ["baseline_fast", "golden_strict"]

try:
    dates = get_dates(api_base, selected_run)
except Exception:
    dates = []

c_sel1, c_sel2, c_sel3, c_sel4 = st.columns([1, 1, 1, 1])
with c_sel1:
    selected_date = st.selectbox("选择日期", dates, index=(len(dates) - 1) if dates else 0, key="selected_date")
with c_sel2:
    selected_system = st.selectbox("系统", [str(s) for s in systems], index=0, key="selected_system")
with c_sel3:
    lookback_days = st.slider("OHLC 回看天数", min_value=20, max_value=200, value=60, step=5, key="lookback_days")
with c_sel4:
    available_tickers: list[str] = []
    try:
        tm = get_tickers(api_base, selected_run, selected_date)
        sys_map = tm.get("systems") if isinstance(tm.get("systems"), dict) else {}
        if isinstance(sys_map.get(selected_system), list):
            available_tickers = [str(x).upper() for x in sys_map.get(selected_system) if str(x).strip()]
        else:
            available_tickers = [str(x).upper() for x in (tm.get("tickers") or []) if str(x).strip()]
    except Exception:
        available_tickers = []
    selected_ticker = st.selectbox("选择标的", available_tickers, index=0, key="selected_ticker")


st.subheader("AI 交易控制塔")

snapshot = None
if selected_run and selected_date and selected_system and selected_ticker:
    snapshot = get_snapshot(api_base, selected_run, selected_system, selected_date, selected_ticker, lookback_days=lookback_days)

if not snapshot:
    st.info("请先在上方选择 Run / Date / System / Ticker")
    st.stop()


decision = snapshot.get("decision") if isinstance(snapshot.get("decision"), dict) else {}
daily = snapshot.get("daily") if isinstance(snapshot.get("daily"), dict) else {}
macro = snapshot.get("macro") if isinstance(snapshot.get("macro"), dict) else {}

final = decision.get("final") if isinstance(decision.get("final"), dict) else {}
final_action = str(final.get("action") or "UNKNOWN").upper()
final_pos = daily.get("target_position") if isinstance(daily, dict) else None
trace = final.get("trace") if isinstance(final.get("trace"), list) else []

pnl_net = daily.get("pnl_h1_net")
exec_note = daily.get("exec_note")
exec_filled = daily.get("exec_filled")
exec_price = daily.get("exec_price")
exec_edge = daily.get("exec_edge")
fee = daily.get("fee")
commission = daily.get("commission")
slippage_cost = daily.get("slippage_cost")

macro_score = macro.get("risk_score")
macro_gear = macro.get("gear")
macro_mult = macro.get("multiplier")

c1, c2, c3, c4 = st.columns(4)

macro_label = "N/A" if macro_score is None else f"{_fmt_float(macro_score, 4)}"
pnl_delta = _fmt_float(pnl_net, 4) if pnl_net is not None else ""
pos_line = f"target_position={_fmt_float(final_pos, 2) if final_pos is not None else ''}"

with c1:
    st.markdown(
        f"""
<div class='metric-card'>
  <div style='font-weight:800;font-size:16px;opacity:0.95'>Action</div>
  <div style='font-weight:900;font-size:22px;letter-spacing:0.3px'>{final_action}</div>
  <div class='small-muted'>{pos_line}</div>
  <div class='small-muted'>pnl_h1_net={pnl_delta}</div>
</div>
""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
<div class='metric-card'>
  <div style='font-weight:800;font-size:16px;opacity:0.95'>Execution</div>
  <div style='font-weight:800;font-size:18px'>{str(exec_note or 'N/A')}</div>
  <div class='small-muted'>edge={_fmt_float(exec_edge, 4)}</div>
  <div class='small-muted'>filled={exec_filled} price={_fmt_float(exec_price, 4)}</div>
</div>
""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
<div class='metric-card'>
  <div style='font-weight:800;font-size:16px;opacity:0.95'>Macro</div>
  <div style='font-weight:900;font-size:22px'>{macro_label}</div>
  <div class='small-muted'>{str(macro_gear or '')}</div>
  <div class='small-muted'>multiplier={_fmt_float(macro_mult, 4)}</div>
</div>
""",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""
<div class='metric-card'>
  <div style='font-weight:800;font-size:16px;opacity:0.95'>Run / Date</div>
  <div style='font-weight:800;font-size:16px;word-break:break-word'>{str(selected_run)}</div>
  <div class='small-muted'>{str(selected_date)}</div>
  <div class='small-muted'>system={selected_system} ticker={selected_ticker}</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.divider()

col_left, col_right = st.columns([3, 1])

with col_left:
    st.subheader("视觉 Alpha 与市场数据")

    ohlc_window = snapshot.get("ohlc_window") if isinstance(snapshot.get("ohlc_window"), dict) else {}
    rows = ohlc_window.get("rows") if isinstance(ohlc_window.get("rows"), list) else []

    if rows:
        df = pd.DataFrame(rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["date"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name=str(selected_ticker),
                )
            ]
        )

        chartist = decision.get("chartist") if isinstance(decision.get("chartist"), dict) else {}
        if chartist:
            signal = str(chartist.get("signal") or "NEUTRAL")
            color = "green" if "BULL" in signal.upper() else "red" if "BEAR" in signal.upper() else "gray"
            last_date = df["date"].iloc[-1]
            last_high = df["high"].iloc[-1]
            fig.add_annotation(
                x=last_date,
                y=last_high,
                text=f"VLM: {signal}",
                showarrow=True,
                arrowhead=2,
                bgcolor=color,
                font=dict(color="white"),
            )

        fig.update_layout(height=440, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(229,231,235,0.92)"),
        )
        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.10)",
            linecolor="rgba(255,255,255,0.18)",
            tickfont=dict(color="rgba(229,231,235,0.80)"),
        )
        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.10)",
            linecolor="rgba(255,255,255,0.18)",
            tickfont=dict(color="rgba(229,231,235,0.80)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("OHLC window unavailable for this ticker/date (missing parquet or date not in data).")

    chartist = decision.get("chartist") if isinstance(decision.get("chartist"), dict) else {}
    if chartist:
        with st.expander("Chartist Vision Analysis", expanded=True):
            st.markdown(f"**Signal:** `{chartist.get('signal')}`")
            st.markdown(f"**Confidence:** `{chartist.get('confidence')}`")
            st.write(chartist.get("reasoning") or "No reasoning provided.")

    with st.expander("Execution / Accounting (daily.csv)", expanded=False):
        st.write(
            {
                "exec_filled": exec_filled,
                "exec_note": exec_note,
                "exec_price": exec_price,
                "fee": fee,
                "commission": commission,
                "slippage_cost": slippage_cost,
                "exec_edge": exec_edge,
                "pnl_h1_net": pnl_net,
            }
        )


with col_right:
    st.subheader("系统 2 辩论大厅")

    sys2 = decision.get("system2") if isinstance(decision.get("system2"), dict) else {}
    if sys2:
        judge = sys2.get("judge") if isinstance(sys2.get("judge"), dict) else {}
        verdict = str(judge.get("final_decision") or "PENDING")
        rationale = str(judge.get("rationale") or "")

        st.markdown(
            f"""
<div class='judge-msg'>
Judge Verdict: {verdict}
<br/><span class='small-muted'>{rationale}</span>
</div>
""",
            unsafe_allow_html=True,
        )

        tab1, tab2, tab3 = st.tabs(["Bull (Proposal)", "Bear (Critic)", "Trace"])

        with tab1:
            proposal = sys2.get("proposal") if isinstance(sys2.get("proposal"), dict) else {}
            st.markdown(
                f"<div class='bull-msg'><b>SFT Proposal:</b> {proposal.get('decision')}</div>",
                unsafe_allow_html=True,
            )
            if proposal:
                st.write(proposal.get("analysis") or "")
                rt = proposal.get("reasoning_trace")
                if isinstance(rt, list) and rt:
                    st.markdown("**Reasoning Trace**")
                    for x in rt:
                        st.markdown(f"- {x}")

        with tab2:
            critic = sys2.get("critic") if isinstance(sys2.get("critic"), dict) else {}
            st.markdown(
                f"<div class='bear-msg'><b>Critic Suggestion:</b> {critic.get('suggested_decision')}</div>",
                unsafe_allow_html=True,
            )
            reasons = critic.get("reasons")
            if isinstance(reasons, list) and reasons:
                st.markdown("**Critic Reasons**")
                for r in reasons:
                    st.markdown(f"- {r}")

        with tab3:
            st.json(sys2)
            errs = sys2.get("errors") if isinstance(sys2.get("errors"), dict) else {}
            if errs:
                st.markdown("**Errors**")
                st.write(errs)

    else:
        st.info("No System-2 debate recorded for this ticker/date.")

    st.divider()

    st.subheader("守门人 / 宏观追踪")
    if isinstance(trace, list) and trace:
        _render_trace([str(x) for x in trace])
    else:
        st.caption("No trace")

    

"""
FastAPI主入口
"""

import json
import os
import subprocess
import sys
import time
import threading
import uuid
import ipaddress
import socket
import urllib.request
import urllib.parse
import urllib.error
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import re
import random
import yaml
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from loguru import logger
from pydantic import BaseModel
from openai import OpenAI

from src.analysis.narrator import narrate_trade_context
from src.memory.mari_memory import get_mari_memory, parse_memory_command
from src.learning.recorder import recorder as evolution_recorder
from src.trading.event import Event, EventType

app = FastAPI(
    title="QuantAI API",
    description="智能量化投顾助手API",
    version="0.1.0"
)


@app.middleware("http")
async def _disable_ui_cache(request: Request, call_next):
    resp = await call_next(request)
    try:
        p = str(getattr(request, "url", None).path or "")
        if p == "/dashboard.html" or p.startswith("/ui/"):
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
    except Exception:
        pass
    return resp


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"
SECRETARY_CONFIG_PATH = REPO_ROOT / "configs" / "secretary.yaml"

_DESKTOP_WEB_DIR = REPO_ROOT / "src" / "ui" / "desktop" / "web"
try:
    if _DESKTOP_WEB_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(_DESKTOP_WEB_DIR), html=True), name="ui")
except Exception:
    pass

_LIVE2D_MARI_DIR = REPO_ROOT / "玛丽偶像 _vts"
_LIVE2D_CHANGLI_DIR = REPO_ROOT / "长离带水印" / "长离带水印"
try:
    if _LIVE2D_MARI_DIR.exists():
        app.mount("/live2d_mari", StaticFiles(directory=str(_LIVE2D_MARI_DIR), html=False), name="live2d_mari")
except Exception:
    pass
try:
    if _LIVE2D_CHANGLI_DIR.exists():
        app.mount("/live2d_changli", StaticFiles(directory=str(_LIVE2D_CHANGLI_DIR), html=False), name="live2d_changli")
except Exception:
    pass

AGENT_HUB_DIR = DATA_DIR / "agent_hub"
AGENT_HUB_DIR.mkdir(parents=True, exist_ok=True)
AGENT_AUDIT_PATH = AGENT_HUB_DIR / "audit.jsonl"
CHAT_LOG_PATH = AGENT_HUB_DIR / "chat.jsonl"
TRAJECTORY_LOG_PATH = AGENT_HUB_DIR / "trajectory.jsonl"


@app.get("/")
async def root():
    try:
        fp = _DESKTOP_WEB_DIR / "dashboard.html"
        if fp.exists():
            return RedirectResponse(url="/dashboard.html")
    except Exception:
        pass
    return {"ok": True}


@app.get("/dashboard.html")
async def dashboard_html():
    fp = _DESKTOP_WEB_DIR / "dashboard.html"
    if not fp.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return FileResponse(str(fp))

_SECRETARY_CFG: Optional[Dict[str, Any]] = None
_SECRETARY_CFG_MTIME: Optional[float] = None

_AGENT_LOCK = threading.Lock()
_NEWS_JOBS: Dict[str, Dict[str, Any]] = {}
_TASKS: Dict[str, Dict[str, Any]] = {}

_EVOLUTION_TRAIN_LOCK = threading.Lock()
_EVOLUTION_TRAIN_PROC: Optional[subprocess.Popen] = None
_EVOLUTION_TRAIN_FH: Optional[Any] = None
_EVOLUTION_TRAIN_LOG: Optional[Path] = None
_EVOLUTION_TRAIN_CMD: Optional[List[str]] = None
_EVOLUTION_TRAIN_STARTED_AT: Optional[str] = None
_EVOLUTION_TRAIN_WATCHER: Optional[threading.Thread] = None

_SOVITS_LOCK = threading.Lock()
_SOVITS_RESUME_AFTER_TRAIN: bool = False
_SOVITS_RESTARTED_AFTER_TRAIN: bool = False

_RL_LOCK = threading.Lock()
_SOVITS_RESUME_AFTER_RL: bool = False
_SOVITS_RESTARTED_AFTER_RL: bool = False

_ALPHA_TRAIN_LOCK = threading.Lock()
_ALPHA_TRAIN_PROC: Optional[subprocess.Popen] = None
_ALPHA_TRAIN_FH: Optional[Any] = None
_ALPHA_TRAIN_LOG: Optional[Path] = None
_ALPHA_TRAIN_CMD: Optional[List[str]] = None
_ALPHA_TRAIN_STARTED_AT: Optional[str] = None
_ALPHA_TRAIN_WATCHER: Optional[threading.Thread] = None
_ALPHA_TRAIN_OUTPUT_DIR: Optional[str] = None
_ALPHA_TRAIN_SFT_ADAPTER: Optional[str] = None
_SOVITS_RESUME_AFTER_ALPHA: bool = False
_SOVITS_RESTARTED_AFTER_ALPHA: bool = False

# Short-lived per-session state for better conversational continuity (desktop only).
_SESSION_STATE: Dict[str, Dict[str, Any]] = {}


def _get_session_id(ctx: Dict[str, Any]) -> Optional[str]:
    try:
        if str((ctx or {}).get("client") or "").lower() != "desktop":
            return None
        sid = str((ctx or {}).get("session_id") or "").strip()
        return sid or None
    except Exception:
        return None


def _agent_logs_digest(*, text: str, ctx: Dict[str, Any]) -> Optional[str]:
    tl = str(text or "").lower()
    if _live_runner is None:
        return None

    want_any = any(
        k in tl
        for k in (
            "planner",
            "gatekeeper",
            "scalper",
            "analyst",
            "chartist",
            "macro",
            "system 2",
            "system2",
            "debate",
            "路由",
            "router",
            "moe",
        )
    )
    if not want_any:
        return None

    try:
        tk = _extract_ticker_hint(str(text or ""))
        if tk:
            tk = str(tk).upper().strip()
    except Exception:
        tk = None

    logs = getattr(_live_runner, "agent_logs", [])
    if not isinstance(logs, list) or (not logs):
        return "Sensei, 我这边还没收到 Multi-Agent 的终端日志（agent_logs 为空）。"

    tail = logs[-300:]
    groups = {
        "planner": [],
        "gatekeeper": [],
        "router": [],
        "scalper": [],
        "analyst": [],
        "chartist": [],
        "macro": [],
        "system2": [],
    }

    def _push(key: str, item: dict) -> None:
        try:
            groups[key].append(item)
            if len(groups[key]) > 6:
                groups[key] = groups[key][-6:]
        except Exception:
            return

    for it in tail:
        try:
            msg = str((it or {}).get("message") or "")
            m = msg.lower()
            if tk and (tk not in msg.upper()):
                continue
            if "planner" in m:
                _push("planner", it)
            if "gatekeeper" in m:
                _push("gatekeeper", it)
            if "moe router" in m or "router" in m:
                _push("router", it)
            if "scalper" in m:
                _push("scalper", it)
            if "analyst" in m or "dpo" in m:
                _push("analyst", it)
            if "chartist" in m or "vlm" in m:
                _push("chartist", it)
            if "macro" in m or "governor" in m or "regime" in m:
                _push("macro", it)
            if "system 2" in m or "system2" in m or "debate" in m or "judge" in m:
                _push("system2", it)
        except Exception:
            continue

    order = ["planner", "gatekeeper", "router", "scalper", "analyst", "chartist", "macro", "system2"]
    title_map = {
        "planner": "Planner",
        "gatekeeper": "Gatekeeper",
        "router": "MoE Router",
        "scalper": "Scalper",
        "analyst": "Analyst",
        "chartist": "Chartist",
        "macro": "Macro",
        "system2": "System2",
    }

    lines: list[str] = []
    head = "Sensei, 我直接从 Multi-Agent 终端里摘取了最近输出："
    if tk:
        head = head + f" (ticker={tk})"
    lines.append(head)

    any_out = False
    for k in order:
        items = groups.get(k) or []
        if not items:
            continue
        any_out = True
        lines.append(f"\n[{title_map.get(k, k)}]")
        for it in items:
            t0 = str((it or {}).get("time") or "")
            msg0 = str((it or {}).get("message") or "")
            lines.append(f"- [{t0}] {msg0}")

    if not any_out:
        try:
            t0 = str((tail[-1] or {}).get("time") or "")
            msg0 = str((tail[-1] or {}).get("message") or "")
            return "\n".join(
                [
                    "Sensei, 我这边能看到 live 终端在输出，但没有命中 planner/gatekeeper/chartist 等关键字。",
                    f"最近一条：[{t0}] {msg0}",
                ]
            )
        except Exception:
            return "Sensei, 我这边能看到 live 终端在输出，但没有命中 planner/gatekeeper/chartist 等关键字。"

    return "\n".join(lines)


def _models_status_digest(*, text: str, ctx: Dict[str, Any]) -> Optional[str]:
    t = str(text or "")
    tl = t.lower()
    want = any(
        k in tl
        for k in (
            "models",
            "model",
            "loaded",
            "load models",
            "on",
            "off",
            "模型",
            "开了吗",
            "开着",
            "关了",
            "有没有在工作",
            "有在工作",
            "在工作吗",
        )
    )
    if (not want) and ("工作" in t) and ("吗" in t or "?" in t or "？" in t):
        want = True
    if not want:
        return None
    if _live_runner is None:
        return "Sensei, live 引擎还没有启动，所以现在没有 Multi-Agent 在跑。"
    try:
        lm = bool(getattr(_live_runner, "load_models", False))
    except Exception:
        lm = False
    try:
        stg = getattr(_live_runner, "strategy", None)
        ml = bool(getattr(stg, "models_loaded", False)) if stg is not None else False
    except Exception:
        ml = False
    try:
        me = str(getattr(getattr(_live_runner, "strategy", None), "models_error", "") or "").strip()
    except Exception:
        me = ""
    infer_mode = "REAL" if (lm and ml) else "HEURISTIC"
    tail = ""
    if me:
        tail = f" (Err={me[:120]})"
    if infer_mode == "REAL":
        return f"Sensei, 现在模型是 ON 的：Models=ON | Loaded=YES | Infer=REAL{tail}"
    return (
        f"Sensei, 现在模型还没真正加载：Models={'ON' if lm else 'OFF'} | Loaded={'YES' if ml else 'NO'} | Infer=HEURISTIC{tail}\n"
        "终端仍然会有输出是因为策略会用 heuristic/规则推理兜底（不代表 GPU MoE 已经启用）。"
    )


def _portfolio_digest(*, text: str, ctx: Dict[str, Any]) -> Optional[str]:
    tl = str(text or "").lower()
    if _live_runner is None:
        return None

    want = any(
        k in tl
        for k in (
            "position",
            "positions",
            "portfolio",
            "cash",
            "trade",
            "trades",
            "holdings",
            "stock",
            "stocks",
            "持仓",
            "仓位",
            "建仓",
            "入场",
            "开仓",
            "买入",
            "卖出",
            "现金",
            "账户",
            "股票",
        )
    )
    if not want:
        return None

    try:
        cash = float(getattr(getattr(_live_runner, "broker", None), "cash", 0.0) or 0.0)
    except Exception:
        cash = 0.0

    try:
        cur = str(getattr(_live_runner, "currency", "USD") or "USD")
    except Exception:
        cur = "USD"

    positions = {}
    try:
        positions = getattr(getattr(_live_runner, "broker", None), "positions", {})
        if not isinstance(positions, dict):
            positions = {}
    except Exception:
        positions = {}

    lines: list[str] = []
    lines.append(f"Cash: {cur}${cash:,.2f}")
    lines.append(f"Positions: {len(list(positions.keys()))}")

    if positions:
        for tk, pos in list(positions.items())[:12]:
            try:
                tku = str(tk or "").upper()
                sh = float(getattr(pos, "shares", 0.0) or 0.0)
                ap = float(getattr(pos, "avg_price", 0.0) or 0.0)
                lines.append(f"- {tku}: shares={sh:g} avg=${ap:.2f}")
            except Exception:
                continue

    trades = []
    try:
        trades = getattr(_live_runner, "trade_log", [])
        if not isinstance(trades, list):
            trades = []
    except Exception:
        trades = []
    if trades:
        lines.append("Recent Trades:")
        for tr in trades[-5:]:
            try:
                lines.append(
                    f"- [{str(tr.get('time') or '')[:19].replace('T',' ')}] {str(tr.get('action') or '')} {str(tr.get('ticker') or '')} x{tr.get('shares')} @ ${float(tr.get('price') or 0.0):.2f}"
                )
            except Exception:
                continue

    if not positions and not trades:
        lines.append("No positions and no trades recorded yet.")

    return "\n".join(lines)


def _get_session_state(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sid = _get_session_id(ctx)
    if not sid:
        return None
    with _AGENT_LOCK:
        st = _SESSION_STATE.get(sid)
        if not isinstance(st, dict):
            st = {"last_task_id": None, "awaiting_detail": False, "last_detail_task_id": None}
            _SESSION_STATE[sid] = st
        return st


def _remember_last_task(ctx: Dict[str, Any], task_id: str) -> None:
    st = _get_session_state(ctx)
    if st is None:
        return
    try:
        st["last_task_id"] = str(task_id)
    except Exception:
        return


def _mark_awaiting_detail(ctx: Dict[str, Any], task_id: Optional[str]) -> None:
    st = _get_session_state(ctx)
    if st is None:
        return
    try:
        st["awaiting_detail"] = True
        st["last_detail_task_id"] = str(task_id) if task_id else None
    except Exception:
        return


def _clear_awaiting_detail(ctx: Dict[str, Any]) -> None:
    st = _get_session_state(ctx)
    if st is None:
        return
    try:
        st["awaiting_detail"] = False
    except Exception:
        return


def _is_followup_status_query(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    # If user already supplied an id, let existing logic handle it.
    if _extract_task_id(t) or _extract_news_id(t):
        return False
    # Typical short follow-ups.
    keys = [
        "现在怎么样", "现在咋样", "怎么样了", "咋样了", "进展", "进度", "好了没", "好了吗", "结果呢", "有结果了吗", "有没有结果",
        "running", "done", "status",
    ]
    tl = t.lower()
    return any((k in t) or (k in tl) for k in keys)


def _is_affirmative_short(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    # Keep it strict: only short acknowledgements.
    if len(t) > 8:
        return False
    keys = ["好的", "好", "行", "可以", "是", "嗯", "嗯嗯", "请", "麻烦", "好哒", "ok", "okay", "yes"]
    tl = t.lower()
    return any((t == k) or (tl == k) for k in keys)


def _task_status_text(task_id: str) -> str:
    with _AGENT_LOCK:
        job = _TASKS.get(str(task_id))
    if not isinstance(job, dict):
        return f"Sensei, 我没有找到这个 task_id：{task_id}"
    st = str(job.get("status") or "")
    if st == "done":
        res = job.get("result") if isinstance(job.get("result"), dict) else {}
        final = str((res or {}).get("final") or "").strip()
        if final:
            return (final + f"\n\n(task_id={task_id})").strip()
        return ("Sensei, 任务已完成，但没有生成总结文本。我把原始结果贴给您：\n\n" + json.dumps(res, ensure_ascii=False)).strip()
    if st == "error":
        return f"Sensei, 这个任务执行失败了：{job.get('error')} (task_id={task_id})"
    return f"Sensei, 这个任务还在处理中：status={st} (task_id={task_id})"


def _elaborate_task(task_id: str) -> str:
    with _AGENT_LOCK:
        job = _TASKS.get(str(task_id))
    if not isinstance(job, dict):
        return f"Sensei, 我没有找到这个 task_id：{task_id}"
    if str(job.get("status") or "") != "done":
        return _task_status_text(task_id)

    res = job.get("result") if isinstance(job.get("result"), dict) else {}
    final = str((res or {}).get("final") or "").strip()
    analyst = str((res or {}).get("analyst") or "").strip()
    trader = str((res or {}).get("trader") or "").strip()

    # Prefer deterministic expansion: show analyst+trader sections.
    parts: list[str] = []
    if final:
        parts.append(final)
    if analyst:
        parts.append("\n[分析员要点（原文）]\n" + analyst)
    if trader:
        parts.append("\n[交易员执行清单（原文）]\n" + trader)

    base = "\n\n".join([p.strip() for p in parts if p.strip()]).strip()
    if not base:
        return _task_status_text(task_id)

    # If we can call LLM, constrain it to only elaborate the given checklist.
    try:
        expand_prompt = (
            "你是 Mari（交易秘书）。用户说‘好的’，表示要你把上一条清单继续细化。\n"
            "只允许基于【输入材料】进行展开，不要引入新的事实/价格/事件。\n"
            "输出结构：\n"
            "1) 先给一个一句话总览\n"
            "2) 然后逐条细化（每条 2-4 句：为什么、怎么做、注意什么）\n"
            "要求：用中文，明确、可执行。"
        )
        # Use 'secretary' adapter for elaboration
        expanded = _call_llm_direct(system_prompt=expand_prompt, user_text=base, temperature=0.2, max_tokens=650, adapter="secretary")
        if isinstance(expanded, str) and expanded.strip():
            return (expanded.strip() + f"\n\n(task_id={task_id})").strip()
    except Exception:
        pass

    return (base + f"\n\n(task_id={task_id})").strip()

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 数据模型 ==========

class RiskProfile(BaseModel):
    """用户风险档位"""
    profile: str = "balanced"  # conservative, balanced, aggressive
    max_drawdown: float = 0.10
    target_volatility: float = 0.10


class PortfolioRequest(BaseModel):
    """组合请求"""
    symbols: List[str] = ["SPY", "TLT", "GLD"]
    risk_profile: RiskProfile = RiskProfile()


class Recommendation(BaseModel):
    """投资建议"""
    symbol: str
    action: str
    target_position: float
    current_position: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str


class RiskAlert(BaseModel):
    """风险预警"""
    type: str
    severity: str
    message: str


class DailyReport(BaseModel):
    """每日报告"""
    date: str
    regime: str
    recommendations: List[Recommendation]
    risk_alerts: List[RiskAlert]
    portfolio_value: float
    daily_return: float


class ChatRequest(BaseModel):
    message: str
    context: Dict[str, Any] = {}


class ChatResponse(BaseModel):
    reply: str
    message_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    message_id: str
    score: int
    comment: str = ""


class TtsRequest(BaseModel):
    text: str
    preset: str = "gentle"


class NewsSubmitRequest(BaseModel):
    text: str
    url: Optional[str] = None
    source: Optional[str] = None


class TaskCreateRequest(BaseModel):
    text: str
    ticker: Optional[str] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    created_at: str
    finished_at: Optional[str] = None
    input: Dict[str, Any] = {}
    result: Optional[Dict[str, Any]] = None


class NewsSubmitResponse(BaseModel):
    news_id: str
    status: str


class NewsJobStatusResponse(BaseModel):
    news_id: str
    status: str
    created_at: str
    finished_at: Optional[str] = None
    input: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None


class ActionRequest(BaseModel):
    action: str
    params: Dict[str, Any] = {}


class ActionResponse(BaseModel):
    ok: bool
    action: str
    result: Dict[str, Any] = {}


def _tool_instructions() -> str:
    return (
        "\n\n[Tools & Actions]\n"
        "You can ask the system to execute actions by outputting one or more fenced code blocks with language 'action'.\n"
        "Format exactly:\n"
        "```action\n"
        "{\"action\": \"submit_news\", \"params\": {\"text\": \"...\", \"url\": \"...\"}}\n"
        "```\n"
        "Supported actions:\n"
        "- start_rl / stop_rl\n"
        "- set_mode (params: {mode: online|offline})\n"
        "- submit_news (params: {text, url?, source?})\n"
        "- fetch_url (params: {url, timeout_sec?})\n"
        "- ui.set_live_ticker (params: {ticker})  # switch dashboard chart ticker\n"
        "- ui.refresh (params: {})  # refresh dashboard live data\n"
        "- ui.set_mode (params: {mode: online|offline})  # switch dashboard mode button\n"
        "- remember (params: {content, category?, importance?})\n"
        "If you don't have enough data, say you don't know. Never fabricate portfolio/trade numbers.\n"
        "\n[UI Buttons]\n"
        "- Online/Offline: switch trading mode\n"
        "- Refresh: refresh live status/chart/logs\n"
        "- Start RL: toggle online reinforcement learning\n"
        "\n[Rule]\n"
        "If Sensei asks to open/view a ticker chart in the desktop dashboard, use ui.set_live_ticker + ui.refresh. Do NOT use fetch_url for that.\n"
    )


def _extract_first_url(text: str) -> Optional[str]:
    t = str(text or "")
    m = re.search(r"https?://\S+", t)
    if not m:
        return None
    u = str(m.group(0)).strip().rstrip(")].,;\"")
    return u if u else None


def _extract_news_id(text: str) -> Optional[str]:
    t = str(text or "")
    m = re.search(r"\b(news_[0-9a-f]{12})\b", t, flags=re.IGNORECASE)
    if not m:
        return None
    return str(m.group(1))


def _extract_task_id(text: str) -> Optional[str]:
    t = str(text or "")
    m = re.search(r"\b(task_[0-9a-f]{12})\b", t, flags=re.IGNORECASE)
    if not m:
        return None
    return str(m.group(1))


def _extract_ticker_hint(text: str) -> Optional[str]:
    t0 = str(text or "")
    tl0 = t0.lower()
    t = t0.upper()
    candidates: list[tuple[int, str]] = []
    # Chinese company name aliases.
    try:
        zh_alias = {
            "苹果": "AAPL",
            "苹果公司": "AAPL",
            "特斯拉": "TSLA",
            "英伟达": "NVDA",
            "英伟达公司": "NVDA",
            "谷歌": "GOOGL",
            "谷歌公司": "GOOGL",
            "google": "GOOGL",
            "meta": "META",
            "脸书": "META",
            "微软": "MSFT",
            "亚马逊": "AMZN",
            "奈飞": "NFLX",
            "网飞": "NFLX",
            "amd": "AMD",
            "英特尔": "INTC",
        }
        for k, v in zh_alias.items():
            idx = tl0.rfind(str(k).lower())
            if idx >= 0:
                candidates.append((idx, v))
    except Exception:
        pass

    # Uppercase tickers, pick the last mentioned one.
    for m in re.finditer(r"\b[A-Z]{1,5}\b", t):
        cand = str(m.group(0)).strip().upper()
        # Avoid common English words that match the pattern.
        if cand in {"A", "I", "AM", "AN", "AND", "OR", "ON", "OFF", "TO", "FOR", "WITH"}:
            continue
        candidates.append((int(m.start()), cand))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], len(x[1])))
    return candidates[-1][1]


def _is_chart_switch_request(text: str) -> bool:
    t = str(text or "")
    tl = t.lower()
    keys = [
        "打开", "切换", "切到", "切去", "看看", "看下", "查看", "打开一下",
        "chart", "k线", "k 线", "图", "走势",
    ]
    return any((k in t) or (k in tl) for k in keys)


def _is_portfolio_question(text: str) -> bool:
    t = str(text or "")
    keys = [
        "持仓", "仓位", "现金", "资金", "余额", "多少钱", "总资产", "净值", "pnl", "盈亏",
        "赚", "亏", "交易", "成交", "买了", "卖了", "多少股", "持有",
    ]
    tl = t.lower()
    for k in keys:
        if k in t or k.lower() in tl:
            return True
    return False


def _is_profit_rank_question(text: str) -> bool:
    t = str(text or "")
    tl = t.lower()
    keys = [
        "谁赚钱最多",
        "谁赚得最多",
        "谁盈利最多",
        "最大盈利",
        "profit most",
    ]
    return any((k in t) or (k in tl) for k in keys)


def _is_loss_rank_question(text: str) -> bool:
    t = str(text or "")
    tl = t.lower()
    keys = [
        "谁亏最多",
        "谁亏得最多",
        "谁亏损最多",
        "最大亏损",
        "loss most",
    ]
    return any((k in t) or (k in tl) for k in keys)


def _is_biggest_position_question(text: str) -> bool:
    t = str(text or "")
    keys = [
        "最大仓位",
        "仓位最大",
        "最大持仓",
        "持仓最大",
    ]
    tl = t.lower()
    return any((k in t) or (k in tl) for k in keys)


def _live_rank_answer(user_text: str, ctx: Dict[str, Any]) -> str:
    if _live_runner is None:
        return "Sensei, 当前没有正在运行的实盘/模拟盘会话，我拿不到实时仓位数据呢…"

    positions = {}
    try:
        positions = getattr(getattr(_live_runner, "broker", None), "positions", {})
        if not isinstance(positions, dict):
            positions = {}
    except Exception:
        positions = {}
    if not positions:
        return "Sensei, 现在没有持仓，所以谈不上谁赚得最多/亏得最多。"

    rows: list[dict] = []
    for ticker, pos in positions.items():
        tk = str(ticker).upper().strip()
        if not tk:
            continue
        try:
            shares = float(getattr(pos, "shares", 0.0) or 0.0)
            avg = float(getattr(pos, "avg_price", 0.0) or 0.0)
        except Exception:
            continue
        if shares == 0:
            continue
        current = avg
        try:
            ph = getattr(_live_runner, "price_history", {})
            if isinstance(ph, dict) and tk in ph and ph[tk]:
                current = float(ph[tk][-1].get("close", avg) or avg)
        except Exception:
            current = avg
        pnl = (current - avg) * shares
        rows.append({"ticker": tk, "shares": shares, "unrealized_pnl": pnl})

    if not rows:
        return "Sensei, 现在没有持仓，所以谈不上谁赚得最多/亏得最多。"

    best = max(rows, key=lambda p: float(p.get("unrealized_pnl") or 0.0))
    worst = min(rows, key=lambda p: float(p.get("unrealized_pnl") or 0.0))
    biggest = max(rows, key=lambda p: abs(float(p.get("shares") or 0.0)))

    b_tk = str(best.get("ticker") or "")
    w_tk = str(worst.get("ticker") or "")
    g_tk = str(biggest.get("ticker") or "")

    try:
        b_pnl = float(best.get("unrealized_pnl") or 0.0)
    except Exception:
        b_pnl = 0.0
    try:
        w_pnl = float(worst.get("unrealized_pnl") or 0.0)
    except Exception:
        w_pnl = 0.0
    try:
        g_sh = int(abs(float(biggest.get("shares") or 0.0)))
    except Exception:
        g_sh = 0

    wants_best = _is_profit_rank_question(user_text)
    wants_worst = _is_loss_rank_question(user_text)
    wants_big = _is_biggest_position_question(user_text)

    if wants_best:
        primary = f"现在赚得最多的是 {b_tk}（浮动盈亏 {b_pnl:+.2f}）。"
        focus_ticker = b_tk
    elif wants_worst:
        primary = f"现在亏得最多的是 {w_tk}（浮动盈亏 {w_pnl:+.2f}）。"
        focus_ticker = w_tk
    elif wants_big:
        primary = f"当前最大仓位是 {g_tk}（持股 {g_sh}）。"
        focus_ticker = g_tk
    else:
        primary = f"现在赚得最多的是 {b_tk}（{b_pnl:+.2f}），亏得最多的是 {w_tk}（{w_pnl:+.2f}），最大仓位是 {g_tk}（{g_sh} 股）。"
        focus_ticker = b_tk

    extra = ""
    if _is_task_dispatch_request(user_text):
        task_text = (
            f"针对 {focus_ticker}：基于‘当前{('盈利领先' if focus_ticker==b_tk else '亏损较大')}'这一事实，"
            "给出今天的特别措施/风控清单（解释型，简短但有理由），目标是亏损更少、赚钱更稳。"
        )
        try:
            task_id = _enqueue_task(text=task_text, ticker=str(focus_ticker).strip() or None, ctx=ctx)
            _remember_last_task(ctx, task_id)
            extra = f"我已经把这件事交给分析员和交易员去跑了，任务编号是 {task_id}（把编号发我就能追结果）。"
        except Exception:
            extra = "我已经把这件事交给分析员和交易员去跑了（如果需要追踪编号我再补给您）。"

    return (f"Sensei，我理解您是想先要一个明确结论，然后顺手把它变成今天的执行措施。{primary}{extra}").strip()


def _is_consensus_question(text: str) -> bool:
    t = str(text or "")
    tl = t.lower()
    keys = [
        "最看好",
        "看好什么",
        "最有希望",
        "最强",
        "best idea",
        "most bullish",
        "most confident",
        "最不看好",
        "不看好什么",
        "最差",
        "最弱",
        "most bearish",
    ]
    return any((k in t) or (k in tl) for k in keys)


def _live_consensus_answer(user_text: str, ctx: Dict[str, Any]) -> str:
    if _live_runner is None:
        return "Sensei, 当前没有正在运行的实盘/模拟盘会话，我拿不到实时信号统计呢…"

    want_bear = ("不看好" in str(user_text)) or ("bear" in str(user_text).lower()) or ("最差" in str(user_text)) or ("最弱" in str(user_text))

    logs = []
    try:
        logs = list(getattr(_live_runner, "agent_logs", []) or [])
        if not isinstance(logs, list):
            logs = []
    except Exception:
        logs = []

    counts: Dict[str, Dict[str, int]] = {}
    pat = None
    try:
        pat = re.compile(r"\[Execution\]\s+SIGNAL\s+(BUY|SELL)\s+([A-Za-z\.]+)")
    except Exception:
        pat = None

    for it in list(logs)[-250:]:
        try:
            msg = str((it or {}).get("message") or "")
        except Exception:
            msg = ""
        if not msg:
            continue
        m = None
        try:
            if pat is not None:
                m = pat.search(msg)
        except Exception:
            m = None
        if not m:
            continue
        act = str(m.group(1) or "").upper()
        tk = str(m.group(2) or "").upper().strip()
        if not tk:
            continue
        if tk not in counts:
            counts[tk] = {"BUY": 0, "SELL": 0}
        if act in {"BUY", "SELL"}:
            counts[tk][act] = int(counts[tk][act]) + 1

    if not counts:
        return "Sensei, 我这边暂时没抓到足够的近期 SIGNAL 记录（可能刚启动或日志被清空）。"

    def _score(v: Dict[str, int]) -> int:
        return int(v.get("BUY", 0)) - int(v.get("SELL", 0))

    ranked = sorted(list(counts.items()), key=lambda kv: _score(kv[1]), reverse=(not want_bear))
    top = ranked[0]
    top_tk = str(top[0])
    top_score = _score(top[1])

    lines: List[str] = []
    if want_bear:
        lines.append(f"Sensei, 按最近 SIGNAL 统计，系统当前最不看好的是 {top_tk}（净 SELL 次数={abs(top_score)}）。")
    else:
        lines.append(f"Sensei, 按最近 SIGNAL 统计，系统当前最看好的是 {top_tk}（净 BUY 次数={abs(top_score)}）。")

    lines.append("Top 3：")
    for tk, v in ranked[:3]:
        sc = _score(v)
        lines.append(f"- {str(tk)}: BUY={int(v.get('BUY', 0))} SELL={int(v.get('SELL', 0))} net={sc:+d}")

    if _is_task_dispatch_request(user_text):
        task_text = f"针对 {top_tk}：解释为什么系统近期偏向{('SELL' if want_bear else 'BUY')}（引用 System2/Macro/Vol/News/Chartist 的可得信息），并给出今天的风控清单（简短可执行）。"
        try:
            task_id = _enqueue_task(text=task_text, ticker=str(top_tk).strip() or None, ctx=ctx)
            _remember_last_task(ctx, task_id)
            lines.append(f"\n我已经把这件事交给分析员和交易员去跑了，任务编号是 {task_id}（把编号发我就能追结果）。")
        except Exception:
            lines.append("\n我已经把这件事交给分析员和交易员去跑了（如果需要追踪编号我再补给您）。")

    return "\n".join(lines).strip()


def _is_task_dispatch_request(text: str) -> bool:
    t = str(text or "")
    tl = t.lower()
    # Avoid false positives: merely mentioning roles (分析员/交易员) should NOT create a task.
    strong = [
        "下达", "派单", "派发", "分派", "分配", "安排", "布置", "交给", "委派", "指派",
        "task", "任务", "指令", "command", "dispatch",
        "特别措施", "风控清单", "风控措施", "交易计划", "交易预案", "执行方案",
    ]
    if any((k in t) or (k in tl) for k in strong):
        return True

    # Phrases that imply dispatching work to a role.
    role_words = r"(分析员|分析师|研究员|交易员|交易手|操盘|交易组)"
    verb_words = r"(让|请|麻烦|帮我|交给|安排|分配|派给|委派|指派)"
    action_words = r"(去|来)?(分析|判断|生成|整理|输出|写|制定|给出|跑|处理|执行|跟进)"
    try:
        if re.search(verb_words + r".*" + role_words + r".*" + action_words, t, flags=re.IGNORECASE):
            return True
        if re.search(role_words + r".*" + verb_words + r".*" + action_words, t, flags=re.IGNORECASE):
            return True
    except Exception:
        pass

    # Catch colloquial phrasing / typos.
    patterns = [
        r"让.*(分析员|分析师|研究员).*(交易员|交易手|操盘|交易组)",
        r"(分析员|分析师|研究员).*(交易员|交易手|操盘|交易组).*(任务|指令|安排|措施)",
        r"给.*(分析员|分析师|研究员).*(任务|指令|安排)",
        r"给.*(交易员|交易手|操盘|交易组).*(任务|指令|安排)",
        r"(下达|派单|分配|安排|交给|委派|指派).*(任务|指令|安排)",
    ]
    for pat in patterns:
        try:
            if re.search(pat, t, flags=re.IGNORECASE):
                return True
        except Exception:
            continue
    return False


def _dispatch_task_answer(user_text: str, ctx: Dict[str, Any]) -> str:
    tk = _extract_ticker_from_text(user_text) or _extract_ticker_hint(user_text) or ""
    ticker = str(tk).strip() or None
    try:
        task_id = _enqueue_task(text=str(user_text or "").strip(), ticker=ticker, ctx=ctx)
        _remember_last_task(ctx, task_id)
    except Exception as e:
        return f"Sensei，我收到‘派单’指令了，但创建任务时失败：{e}"

    done = _wait_task_done(task_id, timeout_sec=8)
    if isinstance(done, dict) and done.get("status") == "done":
        result = done.get("result") if isinstance(done.get("result"), dict) else {}
        final = str((result or {}).get("final") or "").strip()
        if final:
            if ("细化" in final) or ("展开" in final):
                _mark_awaiting_detail(ctx, task_id)
            return (final + f"\n\n(task_id={task_id})").strip()
        # Fallback if merge text is missing.
        trader_out = str((result or {}).get("trader") or "").strip()
        if trader_out:
            return ("Sensei，我已经把任务交给分析员和交易员了。\n\n" + trader_out + f"\n\n(task_id={task_id})").strip()

    if isinstance(done, dict) and done.get("status") == "error":
        err = str(done.get("error") or "")
        return f"Sensei，我收到派单了，但执行失败：{err} (task_id={task_id})"

    return (
        f"Sensei，收到。我已经把任务交给分析员和交易员在跑了。"
        f"编号：{task_id}。你随时把这个编号发我，我就给你回最新结果。"
    ).strip()


def _is_datasource_question(text: str) -> bool:
    t = str(text or "")
    tl = t.lower()
    keys = [
        "数据源", "虚拟", "模拟", "真实", "实盘", "实时", "开盘", "休市", "停盘", "延迟",
        "yfinance", "simulated", "feed", "source",
    ]
    for k in keys:
        if k in t or k.lower() in tl:
            return True
    return False


def _parse_iso_time(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    try:
        if isinstance(s, datetime):
            return s
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _live_feed_status() -> Dict[str, Any]:
    if _live_runner is None:
        return {"active": False}

    data_source = str(getattr(_live_runner, "data_source", "unknown") or "unknown")
    mode = str(getattr(_live_runner, "trading_mode", "online") or "online")

    latest_time: Optional[datetime] = None
    latest_source: Optional[str] = None
    latest_ticker: Optional[str] = None
    latest_bars: Dict[str, Any] = {}

    try:
        ph = getattr(_live_runner, "price_history", {})
        if isinstance(ph, dict):
            for tk, rows in ph.items():
                if not rows:
                    continue
                last = rows[-1]
                if not isinstance(last, dict):
                    continue
                t0 = _parse_iso_time(last.get("time"))
                src0 = str(last.get("source") or "").strip() or None
                latest_bars[str(tk).upper()] = {
                    "time": last.get("time"),
                    "source": src0,
                }
                if t0 is not None and (latest_time is None or t0 > latest_time):
                    latest_time = t0
                    latest_source = src0
                    latest_ticker = str(tk).upper()
    except Exception:
        pass

    # Prefer last bar's declared source if present
    src = latest_source or data_source

    now = datetime.now(tz=latest_time.tzinfo) if latest_time and latest_time.tzinfo else datetime.now()
    age_sec: Optional[float] = None
    if latest_time is not None:
        try:
            age_sec = float((now - latest_time).total_seconds())
        except Exception:
            age_sec = None

    # Stale heuristic: if no bars or last bar older than 15 minutes
    stale = False
    stale_reason = ""
    if latest_time is None:
        stale = True
        stale_reason = "no_bar"
    elif age_sec is not None and age_sec > 15 * 60:
        stale = True
        stale_reason = "last_bar_too_old"

    return {
        "active": True,
        "trading_mode": mode,
        "data_source": src,
        "latest_ticker": latest_ticker,
        "last_bar_time": latest_time.isoformat() if latest_time else None,
        "age_sec": age_sec,
        "stale": stale,
        "stale_reason": stale_reason,
        "latest_bars": latest_bars,
    }


def _live_datasource_answer() -> str:
    st = _live_feed_status()
    if not st.get("active"):
        return "Sensei, 当前没有正在运行的行情会话（live runner 未绑定），我无法判断数据源呢…"

    src = str(st.get("data_source") or "unknown")
    mode = str(st.get("trading_mode") or "online")
    last_bar_time = str(st.get("last_bar_time") or "")
    age_sec = st.get("age_sec")
    stale = bool(st.get("stale"))

    src_label = "REAL" if "yfinance" in src.lower() else ("SIMULATED" if "sim" in src.lower() else "UNKNOWN")
    src_human = "真实行情" if src_label == "REAL" else ("模拟盘" if src_label == "SIMULATED" else "未知来源")
    mode_human = "在线模式" if str(mode).lower() == "online" else ("离线回放" if str(mode).lower() == "offline" else str(mode))

    age_desc = ""
    try:
        if isinstance(age_sec, (int, float)):
            s = float(age_sec)
            if s < 90:
                age_desc = f"大约 {int(round(max(0.0, s)))} 秒前"
            else:
                age_desc = f"大约 {s / 60.0:.1f} 分钟前"
    except Exception:
        age_desc = ""

    bt_human = ""
    try:
        dt = _parse_iso_time(last_bar_time) if last_bar_time else None
        if dt is not None:
            bt_human = f"{dt.year}年{dt.month}月{dt.day}日 {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"
    except Exception:
        bt_human = ""

    parts: List[str] = []
    parts.append(f"Sensei，现在我这边是 {mode_human}（{mode}）。")
    parts.append(f"数据源是 {src}（{src_human} / {src_label}）。")
    if bt_human:
        parts.append(f"最新一根 bar 的时间是 {bt_human}。")
    elif last_bar_time:
        parts.append(f"最新一根 bar 的时间戳是 {last_bar_time}。")
    if age_desc:
        parts.append(f"离现在 {age_desc}。")
    if stale:
        parts.append("状态：行情可能停更了（休市/延迟/网络抖动都可能）。")
    else:
        parts.append("状态：行情数据还在更新中。")
    return "".join(parts).strip()


def _format_money(x: Any) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"


def _live_portfolio_answer(user_text: str) -> str:
    if _live_runner is None:
        return "Sensei, 当前没有正在运行的实盘/模拟盘会话，我拿不到实时仓位数据呢…"

    cash = float(getattr(_live_runner.broker, "cash", 0.0) or 0.0)
    initial = float(getattr(_live_runner.broker, "initial_cash", getattr(_live_runner, "initial_cash", 0.0)) or 0.0)
    positions = getattr(_live_runner.broker, "positions", {})
    if not isinstance(positions, dict):
        positions = {}

    total_value = cash
    pos_lines: List[str] = []
    best = None  # (pnl, ticker)
    worst = None
    biggest = None  # (value, ticker)

    for ticker, pos in positions.items():
        tk = str(ticker).upper()
        try:
            shares = float(getattr(pos, "shares", 0.0) or 0.0)
            avg = float(getattr(pos, "avg_price", 0.0) or 0.0)
        except Exception:
            continue
        if shares == 0:
            continue

        current = avg
        try:
            if hasattr(_live_runner, "price_history") and tk in _live_runner.price_history and _live_runner.price_history[tk]:
                current = float(_live_runner.price_history[tk][-1].get("close", avg) or avg)
        except Exception:
            current = avg

        value = shares * current
        pnl = (current - avg) * shares
        total_value += value

        pos_lines.append(f"- {tk}: {shares:.0f} 股 @ 均价 {avg:.2f}，现价 {current:.2f}，浮动盈亏 {pnl:+.2f}")

        if biggest is None or value > biggest[0]:
            biggest = (value, tk)
        if best is None or pnl > best[0]:
            best = (pnl, tk)
        if worst is None or pnl < worst[0]:
            worst = (pnl, tk)

    total_pnl = total_value - initial
    trade_count = len(getattr(_live_runner, "trade_log", []) or [])
    mode = str(getattr(_live_runner, "trading_mode", "online") or "online")

    lines: List[str] = []
    lines.append(f"Sensei, 我按实时引擎状态给您汇报（mode={mode}）…")
    lines.append(f"现金：{_format_money(cash)}")
    lines.append(f"总资产：{_format_money(total_value)}（总盈亏 {total_pnl:+.2f}）")
    lines.append(f"交易次数：{trade_count}")

    if pos_lines:
        lines.append("当前持仓：")
        lines.extend(pos_lines[:12])
        if biggest is not None:
            lines.append(f"最大仓位：{biggest[1]}")
        if best is not None:
            lines.append(f"当前赚最多：{best[1]}（{best[0]:+.2f}）")
        if worst is not None:
            lines.append(f"当前亏最多：{worst[1]}（{worst[0]:+.2f}）")
    else:
        lines.append("当前没有持仓（positions=0）。")

    return "\n".join(lines).strip()


def _is_news_question(text: str) -> bool:
    t = str(text or "")
    return ("新闻" in t) or ("news" in t.lower())


def _wait_news_done(news_id: str, timeout_sec: int = 25) -> Optional[Dict[str, Any]]:
    t0 = time.time()
    while time.time() - t0 < float(timeout_sec):
        with _AGENT_LOCK:
            job = _NEWS_JOBS.get(str(news_id))
            if isinstance(job, dict) and job.get("status") in {"done", "error"}:
                return dict(job)
        time.sleep(0.8)
    return None


def _wait_task_done(task_id: str, timeout_sec: int = 8) -> Optional[Dict[str, Any]]:
    t0 = time.time()
    while time.time() - t0 < float(timeout_sec):
        with _AGENT_LOCK:
            job = _TASKS.get(str(task_id))
            if isinstance(job, dict) and job.get("status") in {"done", "error"}:
                return dict(job)
        time.sleep(0.5)
    return None


# ========== 路由 ==========

@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "service": "QuantAI API"}


@app.get("/api/v1/market/regime")
async def get_market_regime():
    """获取当前市场风险状态"""
    # TODO: 实际实现
    return {
        "regime": "transition",
        "score": 0,
        "vix": 18.5,
        "spy_trend": "neutral",
        "updated_at": str(date.today())
    }


@app.get("/api/v1/symbols")
async def get_symbols():
    """获取支持的标的列表"""
    return {
        "us_etf": [
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "category": "equity"},
            {"symbol": "QQQ", "name": "Invesco QQQ Trust", "category": "equity"},
            {"symbol": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "category": "bond"},
            {"symbol": "IEF", "name": "iShares 7-10 Year Treasury Bond ETF", "category": "bond"},
            {"symbol": "GLD", "name": "SPDR Gold Shares", "category": "commodity"},
            {"symbol": "SHY", "name": "iShares 1-3 Year Treasury Bond ETF", "category": "cash"}
        ]
    }


@app.post("/api/v1/recommendations")
async def get_recommendations(request: PortfolioRequest):
    """获取投资建议"""
    # TODO: 实际实现
    recommendations = []
    
    for symbol in request.symbols:
        recommendations.append(Recommendation(
            symbol=symbol,
            action="hold",
            target_position=0.2,
            current_position=0.2,
            reason="市场处于过渡期，建议维持当前仓位"
        ))
    
    return {
        "date": str(date.today()),
        "risk_profile": request.risk_profile.profile,
        "recommendations": recommendations
    }


@app.get("/api/v1/portfolio/performance")
async def get_portfolio_performance(days: int = 30):
    """获取组合历史表现"""
    # TODO: 实际实现
    return {
        "start_date": "2024-01-01",
        "end_date": str(date.today()),
        "total_return": 0.05,
        "annual_return": 0.08,
        "volatility": 0.10,
        "sharpe_ratio": 0.8,
        "max_drawdown": -0.05,
        "equity_curve": []  # 实际应返回时间序列
    }


@app.get("/api/v1/risk/alerts")
async def get_risk_alerts():
    """获取风险预警"""
    # TODO: 实际实现
    return {
        "alerts": [],
        "summary": {
            "total": 0,
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
    }


@app.get("/api/v1/news/summary")
async def get_news_summary():
    """获取新闻摘要"""
    # TODO: 实际实现
    return {
        "date": str(date.today()),
        "summary": "暂无重要新闻",
        "events": [],
        "sentiment": "neutral"
    }


def _append_audit(evt: Dict[str, Any]) -> None:
    try:
        line = json.dumps(evt, ensure_ascii=False)
        with _AGENT_LOCK:
            AGENT_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with AGENT_AUDIT_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        logger.debug(f"audit append failed: {e}")


def _append_chat_log(evt: Dict[str, Any]) -> None:
    try:
        line = json.dumps(evt, ensure_ascii=False)
        with _AGENT_LOCK:
            CHAT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with CHAT_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        logger.debug(f"chat log append failed: {e}")


def _append_trajectory_log(evt: Dict[str, Any]) -> None:
    try:
        line = json.dumps(evt, ensure_ascii=False)
        with _AGENT_LOCK:
            TRAJECTORY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with TRAJECTORY_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        logger.debug(f"trajectory log append failed: {e}")


def _load_recent_audit(limit: int = 200) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    if not AGENT_AUDIT_PATH.exists():
        return []
    try:
        lines = AGENT_AUDIT_PATH.read_text(encoding="utf-8").splitlines()
        out: List[Dict[str, Any]] = []
        for ln in lines[-limit:]:
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
        return out
    except Exception:
        return []


def _call_llm_direct(*, system_prompt: str, user_text: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None, adapter: Optional[str] = None) -> Optional[str]:
    cfg = _get_secretary_config()
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}

    try:
        t = float((llm_cfg or {}).get("temperature", 0.7)) if temperature is None else float(temperature)
    except Exception:
        t = 0.7
    try:
        mt = int((llm_cfg or {}).get("max_tokens", 256)) if max_tokens is None else int(max_tokens)
    except Exception:
        mt = 256

    # 1. Try Shared Model (A2 Architecture) - Primary Choice
    # This ensures tasks/news analysis use the loaded Qwen 7B model if available.
    if _live_runner and getattr(_live_runner, "strategy", None):
        try:
            # Check if models are actually loaded
            if _live_runner.strategy.models_loaded:
                resp = _live_runner.strategy.generic_inference(
                    user_msg=user_text,
                    system_prompt=system_prompt,
                    temperature=t,
                    max_new_tokens=mt,
                    adapter=adapter # Use specific adapter if requested (e.g., 'analyst'), else None (Base)
                )
                if resp:
                    return resp.strip()
        except Exception as e:
            logger.error(f"[LLM] shared model direct error: {e}")
            # Fallthrough to other methods

    mode = str((llm_cfg or {}).get("mode") or "api").strip().lower()
    
    if mode == "local":
        # Legacy local mode - only if shared model failed or not ready
        # In A2 architecture, we do NOT want to load a separate local model.
        # If shared model failed, we return None (fallback to rules/templates).
        return None

    api_base = str((llm_cfg or {}).get("api_base") or "").strip()
    api_key = str((llm_cfg or {}).get("api_key") or "").strip() or "local"
    model = str((llm_cfg or {}).get("model") or "").strip()
    if not api_base or not model:
        return None

    try:
        client = OpenAI(base_url=api_base, api_key=api_key)
        out = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(user_text)},
            ],
            temperature=t,
            max_tokens=mt,
        )
        content = None
        try:
            content = out.choices[0].message.content
        except Exception:
            content = None
        if isinstance(content, str) and content.strip():
            return content.strip()
        return None
    except Exception as e:
        logger.error(f"[LLM] api direct error: {e}")
        return None


def _is_blocked_ip(ip: str) -> bool:
    try:
        obj = ipaddress.ip_address(str(ip))
        return bool(
            obj.is_private
            or obj.is_loopback
            or obj.is_link_local
            or obj.is_multicast
            or obj.is_reserved
        )
    except Exception:
        return True


def _is_allowed_host(host: str) -> bool:
    h = str(host or "").strip().lower()
    if not h:
        return False
    if h in {"localhost", "localhost.localdomain"}:
        return False
    try:
        infos = socket.getaddrinfo(h, None)
    except Exception:
        return False
    for info in infos:
        try:
            ip = info[4][0]
        except Exception:
            continue
        if _is_blocked_ip(ip):
            return False
    return True


def _fetch_url_text(*, url: str, timeout_sec: int = 12, max_bytes: int = 500_000) -> Dict[str, Any]:
    cfg = _get_secretary_config()
    net_cfg = cfg.get("network") if isinstance(cfg.get("network"), dict) else {}
    allow_all = bool(net_cfg.get("allow_all", False))
    allowed = net_cfg.get("allowed_domains") if isinstance(net_cfg.get("allowed_domains"), list) else []
    block_private = bool(net_cfg.get("block_private", True))

    u = str(url or "").strip()
    if not u:
        raise ValueError("url is required")
    if not (u.startswith("https://") or u.startswith("http://")):
        raise ValueError("url must start with http(s)://")

    parts = urllib.parse.urlparse(u)
    host = (parts.hostname or "").lower().strip()
    if not host:
        raise ValueError("invalid url")

    if block_private and not _is_allowed_host(host):
        raise ValueError("host blocked")

    if (not allow_all) and allowed:
        ok = any(host == str(d).lower().strip() or host.endswith("." + str(d).lower().strip()) for d in allowed)
        if not ok:
            raise ValueError("domain not allowed")

    req0 = urllib.request.Request(u, headers={"User-Agent": "MariSecretary/0.1"})
    with urllib.request.urlopen(req0, timeout=int(timeout_sec)) as resp:
        raw = resp.read(int(max_bytes))
        ct = resp.headers.get("Content-Type", "")
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        text = str(raw)
    return {"url": u, "content_type": ct, "text": text}


def _sanitize_web_text(text: str, max_chars: int = 6_000) -> str:
    t = str(text or "")
    if not t:
        return ""
    try:
        # Strip HTML tags (very rough) and compress whitespace.
        t = re.sub(r"<script\b[^>]*>.*?</script>", " ", t, flags=re.IGNORECASE | re.DOTALL)
        t = re.sub(r"<style\b[^>]*>.*?</style>", " ", t, flags=re.IGNORECASE | re.DOTALL)
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
    except Exception:
        t = str(text or "").strip()
    if len(t) > int(max_chars):
        t = t[: int(max_chars)]
    return t


def _enqueue_news_job(*, text: str, url: Optional[str] = None, source: Optional[str] = None) -> str:
    news_id = "news_" + uuid.uuid4().hex[:12]
    job = {
        "news_id": news_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "finished_at": None,
        "input": {
            "text": str(text or "").strip(),
            "url": str(url or "").strip() or None,
            "source": str(source or "").strip() or None,
        },
        "results": None,
        "summary": None,
    }
    with _AGENT_LOCK:
        _NEWS_JOBS[news_id] = job
    t = threading.Thread(target=_run_news_analysis_job, args=(news_id,), daemon=True)
    t.start()
    return news_id


def _enqueue_task(*, text: str, ticker: Optional[str], ctx: Dict[str, Any]) -> str:
    task_id = "task_" + uuid.uuid4().hex[:12]
    job = {
        "task_id": task_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "finished_at": None,
        "input": {
            "text": str(text or "").strip(),
            "ticker": str(ticker or "").strip() or None,
            "client": (ctx or {}).get("client"),
            "session_id": (ctx or {}).get("session_id"),
        },
        "result": None,
        "error": None,
    }
    with _AGENT_LOCK:
        _TASKS[task_id] = job

    try:
        _append_trajectory_log(
            {
                "time": datetime.now().isoformat(),
                "type": "contract.task.create",
                "task_id": task_id,
                "client": (ctx or {}).get("client"),
                "session_id": (ctx or {}).get("session_id"),
                "input": {"ticker": str(ticker or "").strip() or None, "text_len": len(str(text or ""))},
            }
        )
    except Exception:
        pass

    th = threading.Thread(target=_run_task_job, args=(task_id,), daemon=True)
    th.start()
    return task_id


def _run_task_job(task_id: str) -> None:
    with _AGENT_LOCK:
        job = _TASKS.get(task_id)
        if not isinstance(job, dict):
            return
        job["status"] = "running"

    try:
        payload = job.get("input") if isinstance(job.get("input"), dict) else {}
        text = str(payload.get("text") or "").strip()
        ticker = str(payload.get("ticker") or "").strip() or None
        tk_s = str(ticker) if ticker else "该标的"

        try:
            _append_trajectory_log(
                {
                    "time": datetime.now().isoformat(),
                    "type": "contract.task.accept",
                    "task_id": task_id,
                    "ticker": ticker,
                }
            )
        except Exception:
            pass

        analyst_prompt = (
            "你是资深股票分析员（Analyst）。\n"
            "任务：根据用户的指令，为交易员生成‘特别措施/关注点’的要点清单。\n"
            "要求：不要编造价格/持仓/已发生事件；不确定的地方用条件式；输出 4-7 条 bullet。"
        )
        trader_prompt = (
            "你是严谨的交易员（Trader）。\n"
            "任务：把分析员的要点转成可执行的交易/风控措施（仓位上限、止损/止盈、事件窗口、禁开仓条件、对冲建议等）。\n"
            "要求：不要编造当前价格/仓位；用条件式表达；输出 4-7 条 bullet。"
        )

        analyst_in = f"标的：{tk_s}\n用户指令：{text}".strip()
        # Use 'analyst' adapter for deep analysis if available
        analyst_out = _call_llm_direct(system_prompt=analyst_prompt, user_text=analyst_in, temperature=0.2, max_tokens=420, adapter="analyst") or ""
        
        trader_in = f"标的：{tk_s}\n用户指令：{text}\n\n[分析员输出]\n{analyst_out}".strip()
        # Use 'scalper' adapter (or base) for trading execution plan
        trader_out = _call_llm_direct(system_prompt=trader_prompt, user_text=trader_in, temperature=0.2, max_tokens=420, adapter="scalper") or ""

        merge_prompt = (
            "你是 Mari（交易秘书）。请用自然口吻对 Sensei 输出派单结果：\n"
            "1) 先确认：我已经把任务交给分析员和交易员（必须有）。\n"
            "2) 然后给一个‘特别措施’清单（精炼、可执行）。\n"
            "3) 对交易的可执行建议(0-5条)\n"
            "要求：明确、可核查、不要编造。"
        )
        merge_in = json.dumps({"task_id": task_id, "ticker": ticker, "analyst": analyst_out, "trader": trader_out}, ensure_ascii=False)
        # Use 'secretary' adapter for the final personality wrapper
        merged = _call_llm_direct(system_prompt=merge_prompt, user_text=merge_in, temperature=0.2, max_tokens=520, adapter="secretary") or ""

        result = {
            "ticker": ticker,
            "analyst": analyst_out,
            "trader": trader_out,
            "final": merged.strip() or None,
        }

        with _AGENT_LOCK:
            job = _TASKS.get(task_id)
            if isinstance(job, dict):
                job["status"] = "done"
                job["finished_at"] = datetime.now().isoformat()
                job["result"] = result

        try:
            _append_trajectory_log(
                {
                    "time": datetime.now().isoformat(),
                    "type": "contract.task.done",
                    "task_id": task_id,
                    "ticker": ticker,
                    "output": {"analyst_len": len(analyst_out), "trader_len": len(trader_out), "final_len": len(merged or "")},
                }
            )
        except Exception:
            pass
    except Exception as e:
        with _AGENT_LOCK:
            job = _TASKS.get(task_id)
            if isinstance(job, dict):
                job["status"] = "error"
                job["finished_at"] = datetime.now().isoformat()
                job["error"] = str(e)
        try:
            _append_trajectory_log(
                {
                    "time": datetime.now().isoformat(),
                    "type": "contract.task.error",
                    "task_id": task_id,
                    "error": str(e),
                }
            )
        except Exception:
            pass


def _run_news_analysis_job(news_id: str) -> None:
    with _AGENT_LOCK:
        job = _NEWS_JOBS.get(news_id)
        if not isinstance(job, dict):
            return
        job["status"] = "running"

    try:
        payload = job.get("input") if isinstance(job.get("input"), dict) else {}
        text = str(payload.get("text") or "").strip()
        url = str(payload.get("url") or "").strip()
        source = str(payload.get("source") or "").strip()

        cfg0 = _get_secretary_config()
        trading_cfg0 = cfg0.get("trading") if isinstance(cfg0, dict) and isinstance(cfg0.get("trading"), dict) else {}
        news_cfg0 = trading_cfg0.get("news") if isinstance(trading_cfg0.get("news"), dict) else {}
        job_mode = str(news_cfg0.get("job_mode") or "simple").strip().lower() or "simple"

        # Prefer 'news' adapter if loaded; otherwise fall back to analyst.
        adapter_pref: Optional[str] = None
        try:
            stg = getattr(_live_runner, "strategy", None) if _live_runner is not None else None
            loaded = getattr(stg, "_adapters_loaded", set()) if stg is not None else set()
            if isinstance(loaded, set) and ("news" in loaded):
                adapter_pref = "news"
            elif isinstance(loaded, set) and ("analyst" in loaded):
                adapter_pref = "analyst"
        except Exception:
            adapter_pref = None

        try:
            if _live_runner is not None and isinstance(getattr(_live_runner, "agent_logs", None), list):
                _live_runner.agent_logs.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "agent",
                    "priority": 2,
                    "message": f"[News] start {news_id} source={source}" + (f" url={url}" if url else ""),
                })
        except Exception:
            pass

        _append_audit({
            "time": datetime.now().isoformat(),
            "type": "news.submit",
            "news_id": news_id,
            "source": source,
            "url": url,
        })

        base_news = text
        if url:
            base_news = base_news + "\n\n[URL] " + url
            try:
                fetched = _fetch_url_text(url=url, timeout_sec=12)
                page_txt = str(fetched.get("text") or "")
                page_txt = _sanitize_web_text(page_txt, max_chars=6_000)
                if page_txt:
                    # Keep raw (sanitized) content; let the news adapter decide what matters.
                    base_news = base_news + "\n\n[URL_CONTENT]\n" + page_txt

                    try:
                        if _live_runner is not None and isinstance(getattr(_live_runner, "agent_logs", None), list):
                            _live_runner.agent_logs.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": "agent",
                                "priority": 2,
                                "message": f"[News] fetched url content ({news_id})",
                            })
                    except Exception:
                        pass
            except Exception as e:
                _append_audit({
                    "time": datetime.now().isoformat(),
                    "type": "tool.fetch_url.error",
                    "news_id": news_id,
                    "url": url,
                    "error": str(e),
                })
        if source:
            base_news = base_news + "\n[Source] " + source

        # Simple mode: one-pass structured analysis.
        prompt_simple = (
            "你是金融新闻分析员。请根据输入的新闻文本/网页内容，输出严格JSON："
            "{verdict: ok|risky|uncertain, tickers:[string], news_sentiment: positive|neutral|negative, news_score:-1..1, confidence:0..1, key_points:[string], trade_actions:[string], summary:string}。"
            "要求：不要编造事实；不确定就uncertain；summary<=200字；只输出JSON。"
        )

        raw = _call_llm_direct(system_prompt=prompt_simple, user_text=base_news, temperature=0.2, max_tokens=420, adapter=adapter_pref)
        parsed: Any = None
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {"raw": raw}
        else:
            parsed = {"error": "no_response"}

        results: Dict[str, Any] = {"news": parsed}
        summary = ""
        try:
            if isinstance(parsed, dict):
                summary = str(parsed.get("summary") or "").strip()
        except Exception:
            summary = ""
        if not summary:
            summary = str(raw or "").strip()

        try:
            if _live_runner is not None and isinstance(getattr(_live_runner, "agent_logs", None), list):
                s0 = str(summary or "").strip().replace("\n", " ")
                if len(s0) > 260:
                    s0 = s0[:257] + "..."
                _live_runner.agent_logs.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "agent",
                    "priority": 2,
                    "message": f"[News] {news_id} summary: {s0}",
                })
        except Exception:
            pass

        try:
            mem = get_mari_memory()
            mem.remember(
                content=f"[News:{news_id}] {text}\n\n[Result]\n{json.dumps(results, ensure_ascii=False)}\n\n[Summary]\n{summary}",
                category="fact",
                importance=3,
            )
        except Exception as e:
            logger.debug(f"news memory save failed: {e}")

        with _AGENT_LOCK:
            job = _NEWS_JOBS.get(news_id)
            if isinstance(job, dict):
                job["status"] = "done"
                job["finished_at"] = datetime.now().isoformat()
                job["results"] = results
                job["summary"] = summary

        _append_audit({
            "time": datetime.now().isoformat(),
            "type": "news.done",
            "news_id": news_id,
        })
    except Exception as e:
        with _AGENT_LOCK:
            job = _NEWS_JOBS.get(news_id)
            if isinstance(job, dict):
                job["status"] = "error"
                job["finished_at"] = datetime.now().isoformat()
                job["error"] = str(e)
        _append_audit({
            "time": datetime.now().isoformat(),
            "type": "news.error",
            "news_id": news_id,
            "error": str(e),
        })


@app.post("/api/v1/news/submit", response_model=NewsSubmitResponse)
async def submit_news(req: NewsSubmitRequest):
    text = str(req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    news_id = _enqueue_news_job(text=text, url=req.url, source=req.source)
    return NewsSubmitResponse(news_id=news_id, status="queued")


@app.get("/api/v1/news/{news_id}", response_model=NewsJobStatusResponse)
async def get_news_job(news_id: str):
    with _AGENT_LOCK:
        job = _NEWS_JOBS.get(str(news_id))
        if not isinstance(job, dict):
            raise HTTPException(status_code=404, detail="news_id not found")
        return NewsJobStatusResponse(
            news_id=str(job.get("news_id")),
            status=str(job.get("status")),
            created_at=str(job.get("created_at")),
            finished_at=job.get("finished_at"),
            input=job.get("input") if isinstance(job.get("input"), dict) else {},
            results=job.get("results"),
            summary=job.get("summary"),
        )


@app.get("/api/v1/agents/audit")
async def get_agents_audit(limit: int = 200):
    return {"events": _load_recent_audit(limit=int(limit)), "count": int(limit)}


@app.post("/api/v1/tasks/create", response_model=TaskStatusResponse)
async def create_task(req: TaskCreateRequest):
    text = str(req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    ticker = str(req.ticker or "").strip() or None
    task_id = _enqueue_task(text=text, ticker=ticker, ctx={})
    with _AGENT_LOCK:
        job = _TASKS.get(task_id)
        if not isinstance(job, dict):
            raise HTTPException(status_code=500, detail="task create failed")
        return TaskStatusResponse(
            task_id=str(job.get("task_id")),
            status=str(job.get("status")),
            created_at=str(job.get("created_at")),
            finished_at=job.get("finished_at"),
            input=job.get("input") if isinstance(job.get("input"), dict) else {},
            result=job.get("result"),
        )


@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(task_id: str):
    with _AGENT_LOCK:
        job = _TASKS.get(str(task_id))
    if not isinstance(job, dict):
        raise HTTPException(status_code=404, detail="task_id not found")
    return TaskStatusResponse(
        task_id=str(job.get("task_id")),
        status=str(job.get("status")),
        created_at=str(job.get("created_at")),
        finished_at=job.get("finished_at"),
        input=job.get("input") if isinstance(job.get("input"), dict) else {},
        result=job.get("result"),
    )


@app.get("/api/v1/tools/fetch_url")
async def fetch_url(url: str, timeout_sec: int = 12):
    try:
        return _fetch_url_text(url=url, timeout_sec=int(timeout_sec))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fetch failed: {e}")


def _execute_action(*, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    a = str(action or "").strip().lower()
    p = params if isinstance(params, dict) else {}
    _append_audit({
        "time": datetime.now().isoformat(),
        "type": "action.request",
        "action": a,
        "params": p,
    })

    if a in {"start_rl", "rl_start"}:
        if _live_runner is None:
            return {"ok": False, "error": "no live session"}
        rl_manager = getattr(_live_runner, "rl_manager", None)
        if rl_manager is None:
            return {"ok": False, "error": "rl manager not initialized"}
        # Collection-only: enable experience logging, but do NOT force model loading
        # and do NOT enable online updates/training.
        rl_manager.enabled = True
        rl_manager.enable_updates = False
        rl_manager.learning_rate = 0.001
        return {"ok": True, "enabled": True}

    if a in {"stop_rl", "rl_stop"}:
        if _live_runner is None:
            return {"ok": False, "error": "no live session"}
        rl_manager = getattr(_live_runner, "rl_manager", None)
        if rl_manager is None:
            return {"ok": False, "error": "rl manager not initialized"}
        rl_manager.enabled = False
        try:
            rl_manager.enable_updates = False
        except Exception:
            pass
        return {"ok": True, "enabled": False}

    if a in {"set_mode", "trading_mode"}:
        if _live_runner is None:
            return {"ok": False, "error": "no live session"}
        mode = str(p.get("mode") or "").strip().lower()
        if mode not in {"online", "offline"}:
            return {"ok": False, "error": "mode must be online/offline"}
        _live_runner.trading_mode = mode
        if mode == "offline":
            _live_runner.start_offline_playback()
        else:
            _live_runner.stop_offline_playback()
        return {"ok": True, "mode": mode}

    if a in {"submit_news", "news"}:
        text = str(p.get("text") or "").strip()
        if not text:
            return {"ok": False, "error": "text required"}
        url = str(p.get("url") or "").strip() or None
        source = str(p.get("source") or "chat").strip() or "chat"
        news_id = _enqueue_news_job(text=text, url=url, source=source)
        return {"ok": True, "news_id": news_id}

    if a in {"fetch_url"}:
        u = str(p.get("url") or "").strip()
        if not u:
            return {"ok": False, "error": "url required"}
        try:
            res = _fetch_url_text(url=u, timeout_sec=int(p.get("timeout_sec") or 12))
            return {"ok": True, "url": res.get("url"), "content_type": res.get("content_type"), "text": res.get("text")}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    if a in {"remember"}:
        mem_text = str(p.get("content") or "").strip()
        if not mem_text:
            return {"ok": False, "error": "content required"}
        try:
            memory = get_mari_memory()
            mid = memory.remember(mem_text, category=str(p.get("category") or "general"), importance=int(p.get("importance") or 2))
            return {"ok": True, "memory_id": mid}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    return {"ok": False, "error": f"unknown action: {a}"}


def _live_last_price(ticker: str) -> float:
    try:
        tk = str(ticker or "").upper().strip()
    except Exception:
        tk = ""
    if not tk or _live_runner is None:
        return 0.0
    try:
        ph = getattr(_live_runner, "price_history", {})
        if isinstance(ph, dict) and tk in ph and ph[tk]:
            return float(ph[tk][-1].get("close") or 0.0)
    except Exception:
        return 0.0
    return 0.0


def _parse_trade_command(text: str) -> Optional[Dict[str, Any]]:
    s = str(text or "").strip()
    if not s:
        return None
    tl = s.lower()

    m = re.search(r"\b(buy|sell|short|cover|close)\s+([a-z]{1,6})\b", tl)
    if m:
        verb = str(m.group(1) or "").strip().lower()
        tk = str(m.group(2) or "").strip().upper()
        act = "BUY" if verb in {"buy", "cover"} else "SELL" if verb in {"sell", "short"} else "CLOSE"
        return {"action": act, "ticker": tk, "raw": s}

    m = re.search(r"(买入|买|做多|多)\s*([A-Za-z]{1,6})", s)
    if m:
        return {"action": "BUY", "ticker": str(m.group(2) or "").strip().upper(), "raw": s}
    m = re.search(r"(卖出|卖|做空|空)\s*([A-Za-z]{1,6})", s)
    if m:
        return {"action": "SELL", "ticker": str(m.group(2) or "").strip().upper(), "raw": s}
    m = re.search(r"(平仓|平掉|平)\s*([A-Za-z]{1,6})", s)
    if m:
        return {"action": "CLOSE", "ticker": str(m.group(2) or "").strip().upper(), "raw": s}

    return None


def _execute_live_order_from_chat(cmd: Dict[str, Any]) -> Dict[str, Any]:
    if _live_runner is None:
        return {"ok": False, "error": "no live session"}

    tk = str(cmd.get("ticker") or "").strip().upper()
    act = str(cmd.get("action") or "").strip().upper()
    if not tk or act not in {"BUY", "SELL", "CLOSE"}:
        return {"ok": False, "error": "invalid command"}

    price = _live_last_price(tk)
    if not (price > 0.0):
        return {"ok": False, "error": f"no price for {tk}"}

    shares = 75.0
    try:
        raw = str(cmd.get("raw") or "")
        m = re.search(r"x\s*(\d+)", raw, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"(\d+)\s*(股|shares)?", raw)
        if m:
            shares = float(m.group(1))
    except Exception:
        shares = 75.0
    shares = float(max(1.0, min(shares, 100000.0)))

    if act == "CLOSE":
        pos = None
        try:
            pos = getattr(getattr(_live_runner, "broker", None), "positions", {}).get(tk)
        except Exception:
            pos = None
        if pos is not None:
            try:
                sh = float(getattr(pos, "shares", 0.0) or 0.0)
            except Exception:
                sh = 0.0
            if sh < 0:
                act = "BUY"
                shares = abs(float(sh))
            elif sh > 0:
                act = "SELL"
                shares = abs(float(sh))
            else:
                return {"ok": False, "error": "no position to close"}
        else:
            return {"ok": False, "error": "no position to close"}

    payload = {
        "ticker": tk,
        "action": act,
        "price": float(price),
        "shares": float(shares),
        "expert": "chat",
        "analysis": f"chat_command: {str(cmd.get('raw') or '').strip()}",
        "timestamp": datetime.now().isoformat(),
    }

    eng = getattr(_live_runner, "engine", None)
    if eng is None or not hasattr(eng, "push_event"):
        return {"ok": False, "error": "live engine unavailable"}

    try:
        eng.push_event(Event(type=EventType.ORDER, timestamp=datetime.now(), payload=payload, priority=2))
    except Exception as e:
        return {"ok": False, "error": f"order push failed: {e}"}

    try:
        logs = getattr(_live_runner, "agent_logs", None)
        if isinstance(logs, list):
            logs.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": "fill",
                "priority": 2,
                "message": f"[Execution] ORDER {act} {tk} x{float(shares):g} @ ${float(price):.2f} (expert=chat)",
            })
    except Exception:
        pass

    return {"ok": True, "ticker": tk, "action": act, "shares": float(shares), "price": float(price)}


@app.post("/api/v1/actions/execute", response_model=ActionResponse)
async def execute_action(req: ActionRequest):
    res = _execute_action(action=req.action, params=req.params)
    return ActionResponse(ok=bool(res.get("ok")), action=str(req.action), result=res)


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    text = str(req.message or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="message is required")
    ctx = req.context if isinstance(req.context, dict) else {}

    trade_res: Optional[Dict[str, Any]] = None
    try:
        cmd = _parse_trade_command(text)
        if isinstance(cmd, dict):
            trade_res = _execute_live_order_from_chat(cmd)
    except Exception:
        trade_res = None

    # If Ouroboros training is running, do not start new local inference.
    try:
        with _EVOLUTION_TRAIN_LOCK:
            p = _EVOLUTION_TRAIN_PROC
        if p is not None and p.poll() is None:
            return JSONResponse(
                content={
                    "reply": "Sensei, Ouroboros 正在训练中（会占满 GPU）。训练完成后我再继续回答。",
                    "message_id": None,
                },
                media_type="application/json; charset=utf-8",
            )
    except Exception:
        pass

    t0 = time.perf_counter()
    reply = await run_in_threadpool(_secretary_reply, text, ctx)
    dt_ms = int((time.perf_counter() - t0) * 1000.0)

    try:
        if isinstance(trade_res, dict) and bool(trade_res.get("ok")):
            r0 = f"(已提交交易指令：{trade_res.get('action')} {trade_res.get('ticker')} x{float(trade_res.get('shares') or 0):g} @ ${float(trade_res.get('price') or 0.0):.2f})"
            reply = (str(reply or "").rstrip() + "\n\n" + r0).strip()
        elif isinstance(trade_res, dict) and trade_res.get("error"):
            r0 = f"(交易指令未执行：{str(trade_res.get('error') or '').strip()})"
            reply = (str(reply or "").rstrip() + "\n\n" + r0).strip()
    except Exception:
        pass

    # Deterministic UI chart switching: do not rely on LLM to emit actions.
    try:
        if str(ctx.get("client") or "").lower() == "desktop" and _is_chart_switch_request(text):
            tk = _extract_ticker_hint(text)
            if tk:
                # Common typos / aliases.
                alias = {
                    "APPL": "AAPL",
                }
                tk = alias.get(tk, tk)

                # If live runner is available, correct to the closest known ticker.
                try:
                    st0 = _live_feed_status()
                    tickers0 = st0.get("tickers") if isinstance(st0.get("tickers"), list) else []
                    tickers = [str(x).strip().upper() for x in tickers0 if str(x).strip()]
                except Exception:
                    tickers = []

                if tickers and (tk not in tickers):
                    def _dist(a: str, b: str) -> int:
                        # Small edit-distance for short tickers.
                        la, lb = len(a), len(b)
                        if abs(la - lb) > 2:
                            return 999
                        dp = list(range(lb + 1))
                        for i, ca in enumerate(a, start=1):
                            prev = dp[0]
                            dp[0] = i
                            for j, cb in enumerate(b, start=1):
                                cur = dp[j]
                                cost = 0 if ca == cb else 1
                                dp[j] = min(
                                    dp[j] + 1,
                                    dp[j - 1] + 1,
                                    prev + cost,
                                )
                                prev = cur
                        return dp[-1]

                    best = min(tickers, key=lambda x: _dist(tk, x))
                    if _dist(tk, best) <= 1:
                        tk = best
            if tk:
                action_blocks = (
                    "```action\n" + json.dumps({"action": "ui.set_live_ticker", "params": {"ticker": tk}}, ensure_ascii=False) + "\n```\n"
                    "```action\n" + json.dumps({"action": "ui.refresh", "params": {}}, ensure_ascii=False) + "\n```"
                )
                reply = (str(reply or "").rstrip() + "\n\n" + action_blocks).strip()
    except Exception:
        pass
    try:
        snap = _build_secretary_context(ctx)
        keep: Dict[str, Any] = {}
        if isinstance(snap.get("live_trading"), dict):
            lt = snap.get("live_trading")
            keep["live_trading"] = {
                "mode": lt.get("mode"),
                "trading_mode": lt.get("trading_mode"),
                "data_source": lt.get("data_source"),
                "cash": lt.get("cash"),
                "total_value": lt.get("total_value"),
                "total_pnl": lt.get("total_pnl"),
                "positions": lt.get("positions"),
                "latest_bars": lt.get("latest_bars"),
            }
        if isinstance(snap.get("status"), dict):
            st = snap.get("status")
            keep["status"] = {
                "trading": st.get("trading"),
                "voice_training": st.get("voice_training"),
            }

        try:
            prof = _classify_secretary_profile(text)
        except Exception:
            prof = "work"

        _append_trajectory_log(
            {
                "time": datetime.now().isoformat(),
                "type": "trajectory.chat.turn",
                "client": ctx.get("client"),
                "session_id": ctx.get("session_id"),
                "profile": prof,
                "latency_ms": dt_ms,
                "message": text,
                "reply_len": len(str(reply or "")),
            }
        )

        _append_chat_log(
            {
                "time": datetime.now().isoformat(),
                "type": "chat.turn",
                "client": ctx.get("client"),
                "session_id": ctx.get("session_id"),
                "message": text,
                "reply": reply,
                "context": ctx,
                "server_context": keep,
            }
        )
    except Exception:
        pass

    msg_id: Optional[str] = None
    try:
        tid = _extract_task_id(text) or _extract_task_id(str(reply or ""))
        shared_urls = []
        try:
            url_in_msg = _extract_first_url(text)
            if url_in_msg:
                shared_urls.append(url_in_msg)
        except Exception:
            pass

        msg_id = evolution_recorder.record(
            agent_id="secretary",
            context=json.dumps(
                {
                    "client": str(ctx.get("client") or ""),
                    "session_id": str(ctx.get("session_id") or ""),
                    "message": text,
                    "task_id": tid,
                    "trainer": "secretary_chat",
                    "shared_urls": shared_urls,
                },
                ensure_ascii=False,
            ),
            action=str(reply or ""),
            outcome=None,
            feedback="wait_for_user",
        )
    except Exception:
        pass
    return JSONResponse(content={"reply": reply, "message_id": msg_id}, media_type="application/json; charset=utf-8")


@app.post("/api/v1/feedback")
async def feedback(req: FeedbackRequest):
    score = int(req.score)
    if score not in (1, -1):
        raise HTTPException(status_code=400, detail="score must be 1 or -1")
    try:
        logger.info(f"[Feedback] ref_id={str(req.message_id)[:12]} score={score} comment_len={len(str(req.comment or ''))}")
    except Exception:
        pass
    try:
        evolution_recorder.log_feedback(ref_id=str(req.message_id), score=score, comment=str(req.comment or ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to record feedback: {e}")
    return {"ok": True}


@app.post("/api/v1/llm/unload")
async def unload_local_llm():
    if _live_runner and getattr(_live_runner, "strategy", None):
        try:
            stg = _live_runner.strategy
            # Manually unload to free memory
            stg.model = None
            stg.tokenizer = None
            stg.models_loaded = False
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return {"ok": True, "message": "Shared models unloaded"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"unload failed: {e}")
    
    return {"ok": False, "message": "No active strategy to unload"}


@app.post("/api/v1/llm/warmup")
async def warmup_local_llm():
    cfg = _get_secretary_config()
    trading_cfg = cfg.get("trading") if isinstance(cfg.get("trading"), dict) else {}
    
    # Check if we are in local/shared mode logic?
    # Actually, we should just ensure the strategy models are loaded.
    
    if _live_runner and getattr(_live_runner, "strategy", None):
        try:
            stg = _live_runner.strategy
            if not stg.models_loaded:
                # Trigger load in threadpool to avoid blocking
                await run_in_threadpool(stg.load_models)
            
            return {
                "ok": True, 
                "warmed": True, 
                "base_model": stg.base_model,
                "adapters": list(stg._adapters_loaded)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"warmup failed: {e}")

    return {"ok": False, "message": "Live runner not ready"}


_REF_WAV_CACHE: Dict[str, Dict[str, Any]] = {}

_SOVITS_WEIGHTS_CACHE: Dict[str, Any] = {"gpt": "", "sovits": "", "ts": 0.0}


def _read_ref_prompt_text(path: Path) -> str:
    try:
        s = path.read_text(encoding="utf-8", errors="replace")
        s = str(s or "").strip()
        return s[:200]
    except Exception:
        return ""


def _pick_ref_audio_from_dirs(ref_dirs: List[str], *, max_files: int = 1800, ttl_sec: float = 60.0) -> Optional[Dict[str, str]]:
    try:
        dirs = [str(d or "").strip() for d in (ref_dirs or []) if str(d or "").strip()]
        if not dirs:
            return None
        key = "|".join(dirs)
        now = time.time()
        cached = _REF_WAV_CACHE.get(key) if isinstance(_REF_WAV_CACHE.get(key), dict) else None
        if cached and (now - float(cached.get("t") or 0.0) < float(ttl_sec)):
            files = cached.get("files")
            if isinstance(files, list) and files:
                picked = random.choice(files)
                return dict(picked) if isinstance(picked, dict) else None

        audio_exts = {".mp3", ".ogg", ".wav", ".flac", ".m4a"}

        files: List[Dict[str, str]] = []
        for d in dirs:
            try:
                p = Path(d)
                if not p.exists():
                    continue
                for root, _, fns in os.walk(str(p)):
                    for fn in fns:
                        if not isinstance(fn, str):
                            continue
                        fp = Path(root) / fn
                        if fp.suffix.lower() not in audio_exts:
                            continue
                        prompt = ""
                        try:
                            # GPT-SoVITS datasets often store transcripts in .lab files.
                            # Support both xxx.wav.txt / xxx.txt and xxx.wav.lab / xxx.lab.
                            cand = [
                                fp.with_name(fp.name + ".txt"),
                                fp.with_suffix(".txt"),
                                fp.with_name(fp.name + ".lab"),
                                fp.with_suffix(".lab"),
                            ]
                            for tp in cand:
                                if tp.exists() and tp.is_file():
                                    prompt = _read_ref_prompt_text(tp)
                                    if prompt:
                                        break
                        except Exception:
                            prompt = ""

                        # Only keep refs that have a transcript prompt.
                        # Using audio refs without prompt_text frequently causes garbled output.
                        if not str(prompt or "").strip():
                            continue

                        files.append({"audio": str(fp), "prompt_text": str(prompt or "")})
                        if len(files) >= int(max_files):
                            break
                    if len(files) >= int(max_files):
                        break
            except Exception:
                continue

        if files:
            _REF_WAV_CACHE[key] = {"t": now, "files": files}
            picked = random.choice(files)
            return dict(picked) if isinstance(picked, dict) else None
        _REF_WAV_CACHE[key] = {"t": now, "files": []}
        return None
    except Exception:
        return None


def _pick_tts_preset(text: str) -> str:
    t = str(text or "")
    tl = t.lower()
    if any(k in t for k in ["亏", "回撤", "止损", "爆仓", "试炼", "下跌", "跌破"]) or any(k in tl for k in ["drawdown", "stop", "loss", "risk", "crash"]):
        return "worry"
    if any(k in t for k in ["盈利", "赚钱", "福报", "上涨", "恭喜", "祝福"]) or any(k in tl for k in ["profit", "pnl", "gain", "win"]):
        return "happy"
    return "gentle"


@app.post("/api/v1/voice/tts")
async def voice_tts(req: TtsRequest):
    """Generate Mari voice audio (web dashboard).

    Uses secretary.yaml voice.gpt_sovits preset configuration.
    Returns audio bytes (wav or other audio/*).
    """
    raise HTTPException(status_code=404, detail="TTS disabled")
    text = str(req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    cfg = _get_secretary_config()
    voice_cfg = cfg.get("voice") if isinstance(cfg.get("voice"), dict) else {}
    try:
        voice_enabled = bool((voice_cfg or {}).get("enabled", True))
    except Exception:
        voice_enabled = True
    if not voice_enabled:
        raise HTTPException(status_code=400, detail="voice disabled")
    backend = str((voice_cfg or {}).get("backend") or "").strip().lower() or "gpt_sovits"
    if backend != "gpt_sovits":
        raise HTTPException(status_code=400, detail=f"unsupported voice backend: {backend}")

    gs = (voice_cfg or {}).get("gpt_sovits") if isinstance((voice_cfg or {}).get("gpt_sovits"), dict) else {}
    api_base = str((gs or {}).get("api_base") or "").strip()
    endpoint = str((gs or {}).get("endpoint") or "/tts").strip() or "/tts"
    if not api_base:
        raise HTTPException(status_code=500, detail="gpt_sovits api_base missing")

    preset = str(req.preset or "").strip() or _pick_tts_preset(text)
    presets = gs.get("presets") if isinstance(gs.get("presets"), dict) else {}
    p0 = presets.get(preset) if isinstance(presets.get(preset), dict) else {}

    ref_dirs = (p0 or {}).get("ref_dirs") if isinstance((p0 or {}).get("ref_dirs"), list) else []
    try:
        random_ref = bool((p0 or {}).get("random_ref")) if (p0 or {}).get("random_ref") is not None else bool((gs or {}).get("random_ref", True))
    except Exception:
        random_ref = True
    try:
        max_ref_tries = int((gs or {}).get("max_ref_tries") or 6)
    except Exception:
        max_ref_tries = 6
    max_ref_tries = max(1, min(max_ref_tries, 20))

    refer_wav_path = str((p0 or {}).get("refer_wav_path") or "").strip()
    picked_audio = None
    picked_prompt = None
    if bool(random_ref):
        for _ in range(int(max_ref_tries)):
            picked = _pick_ref_audio_from_dirs(ref_dirs)
            if isinstance(picked, dict):
                a0 = str(picked.get("audio") or "").strip()
                p0t = str(picked.get("prompt_text") or "").strip()
                if a0 and Path(a0).exists():
                    # Only use random refs when we also have a transcript prompt.
                    # Using an audio ref without prompt_text often causes garbled output.
                    if not p0t:
                        try:
                            logger.info(f"[TTS] skip_ref_no_prompt audio={Path(a0).name}")
                        except Exception:
                            pass
                        continue
                    picked_audio = a0
                    picked_prompt = p0t
                    break
                continue
    if picked_audio:
        refer_wav_path = str(picked_audio)
    prompt_text = str((p0 or {}).get("prompt_text") or "").strip()
    if picked_prompt:
        prompt_text = picked_prompt
    try:
        logger.info(
            f"[TTS] preset={preset} random_ref={bool(random_ref)} ref={refer_wav_path} prompt_from_file={bool(picked_prompt)}"
        )
    except Exception:
        pass
    text_lang = str((p0 or {}).get("text_language") or (gs or {}).get("text_language") or "zh").strip() or "zh"
    prompt_lang = str((p0 or {}).get("prompt_language") or (gs or {}).get("prompt_language") or "ja").strip() or "ja"
    gpt_path = str((gs or {}).get("gpt_path") or "").strip()
    sovits_path = str((gs or {}).get("sovits_path") or "").strip()
    timeout_sec = float((gs or {}).get("timeout_sec") or 120)

    # Ensure correct voice weights are loaded (avoids using default voice -> unclear speech)
    try:
        now = time.time()
        cache_ttl = 30.0
        need_reload = False
        try:
            if (gpt_path and (gpt_path != str(_SOVITS_WEIGHTS_CACHE.get("gpt") or ""))) or (
                sovits_path and (sovits_path != str(_SOVITS_WEIGHTS_CACHE.get("sovits") or ""))
            ):
                need_reload = True
            if (now - float(_SOVITS_WEIGHTS_CACHE.get("ts") or 0.0)) > cache_ttl:
                need_reload = True
        except Exception:
            need_reload = True

        if need_reload and api_base:
            # Use dedicated endpoints if available
            try:
                if gpt_path:
                    ok0 = False
                    u0 = api_base.rstrip("/") + "/set_gpt_weights?" + urllib.parse.urlencode({"weights_path": gpt_path})
                    try:
                        with urllib.request.urlopen(u0, timeout=float(timeout_sec)) as _:
                            ok0 = True
                    except Exception:
                        ok0 = False
                    if not ok0:
                        try:
                            raw = json.dumps({"weights_path": gpt_path}, ensure_ascii=False).encode("utf-8")
                            req0 = urllib.request.Request(
                                api_base.rstrip("/") + "/set_gpt_weights",
                                data=raw,
                                headers={"Content-Type": "application/json"},
                                method="POST",
                            )
                            with urllib.request.urlopen(req0, timeout=float(timeout_sec)) as _:
                                ok0 = True
                        except Exception:
                            ok0 = False
                if sovits_path:
                    ok1 = False
                    u1 = api_base.rstrip("/") + "/set_sovits_weights?" + urllib.parse.urlencode({"weights_path": sovits_path})
                    try:
                        with urllib.request.urlopen(u1, timeout=float(timeout_sec)) as _:
                            ok1 = True
                    except Exception:
                        ok1 = False
                    if not ok1:
                        try:
                            raw = json.dumps({"weights_path": sovits_path}, ensure_ascii=False).encode("utf-8")
                            req1 = urllib.request.Request(
                                api_base.rstrip("/") + "/set_sovits_weights",
                                data=raw,
                                headers={"Content-Type": "application/json"},
                                method="POST",
                            )
                            with urllib.request.urlopen(req1, timeout=float(timeout_sec)) as _:
                                ok1 = True
                        except Exception:
                            ok1 = False

                _SOVITS_WEIGHTS_CACHE["gpt"] = gpt_path
                _SOVITS_WEIGHTS_CACHE["sovits"] = sovits_path
                _SOVITS_WEIGHTS_CACHE["ts"] = now
                try:
                    logger.info(f"[TTS] loaded weights gpt={Path(gpt_path).name if gpt_path else ''} sovits={Path(sovits_path).name if sovits_path else ''}")
                except Exception:
                    pass
            except Exception as e:
                try:
                    logger.warning(f"[TTS] set weights failed: {e}")
                except Exception:
                    pass
    except Exception:
        pass

    if not refer_wav_path:
        raise HTTPException(status_code=500, detail=f"gpt_sovits preset '{preset}' refer_wav_path missing")

    url = api_base.rstrip("/") + "/" + endpoint.lstrip("/")
    # Send both key variants to maximize compatibility across GPT-SoVITS forks.
    # Some servers expect *_lang, others expect *_language, and ref key names also vary.
    payload_primary: Dict[str, Any] = {
        "text": text,
        "prompt_text": prompt_text,
        "text_lang": text_lang,
        "prompt_lang": prompt_lang,
        "ref_audio_path": refer_wav_path,
        "text_language": text_lang,
        "prompt_language": prompt_lang,
        "refer_wav_path": refer_wav_path,
    }
    # Legacy payload variants (used as fallback when server 400s).
    payload_legacy: Dict[str, Any] = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": refer_wav_path,
        "prompt_lang": prompt_lang,
        "prompt_text": prompt_text,
    }

    def _post_tts(_payload: Dict[str, Any]) -> Tuple[str, bytes]:
        raw0 = json.dumps(_payload, ensure_ascii=False).encode("utf-8")
        ureq0 = urllib.request.Request(url, data=raw0, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(ureq0, timeout=timeout_sec) as resp0:
            ct0 = str(resp0.headers.get("content-type") or "").lower()
            data0 = resp0.read()
        return ct0, data0

    try:
        try:
            logger.info(f"[TTS] POST {url}")
        except Exception:
            pass
        ct, data = _post_tts(payload_primary)
    except Exception as e:
        # Surface response body for HTTP 400 debugging.
        try:
            if isinstance(e, urllib.error.HTTPError):
                body = b""
                try:
                    body = e.read() or b""
                except Exception:
                    body = b""
                detail = body.decode("utf-8", errors="replace") if body else ""

                # Some GPT-SoVITS variants accept `ref_wav_path` or `refer_wav_path` instead.
                if int(getattr(e, "code", 0) or 0) == 400:
                    # First fallback: try legacy field names.
                    try:
                        ct, data = _post_tts(payload_legacy)
                    except Exception:
                        # Second fallback: try alternative ref key names.
                        alt = dict(payload_legacy)
                        alt.pop("ref_audio_path", None)
                        alt["ref_wav_path"] = refer_wav_path
                        try:
                            ct, data = _post_tts(alt)
                        except Exception:
                            alt2 = dict(payload_legacy)
                            alt2.pop("ref_audio_path", None)
                            alt2["refer_wav_path"] = refer_wav_path
                            try:
                                ct, data = _post_tts(alt2)
                            except Exception:
                                raise HTTPException(
                                    status_code=500,
                                    detail=f"gpt_sovits request failed: HTTP {int(e.code)} {detail}".strip(),
                                )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"gpt_sovits request failed: HTTP {int(e.code)} {detail}".strip(),
                    )
            else:
                raise HTTPException(status_code=500, detail=f"gpt_sovits request failed: {e}")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=500, detail=f"gpt_sovits request failed: {e}")

    try:
        logger.info(f"[TTS] response content-type={ct} bytes={len(data) if data else 0}")
    except Exception:
        pass

    if ct.startswith("audio/"):
        return Response(content=data, media_type=ct.split(";")[0] or "audio/wav")

    try:
        obj = json.loads(data.decode("utf-8", errors="replace"))
    except Exception:
        raise HTTPException(status_code=500, detail="gpt_sovits bad response")

    if isinstance(obj, dict):
        b64 = obj.get("audio") or obj.get("data") or obj.get("wav")
        if isinstance(b64, str) and b64.strip():
            try:
                import base64

                wav = base64.b64decode(b64)
                return Response(content=wav, media_type="audio/wav")
            except Exception:
                pass
        msg = obj.get("message") or obj.get("detail") or obj.get("error")
        raise HTTPException(status_code=500, detail=f"gpt_sovits error: {msg}")

    raise HTTPException(status_code=500, detail="gpt_sovits bad response")


def _pick_python_for_repo() -> str:
    try:
        py = REPO_ROOT / "venv311" / "Scripts" / "python.exe"
        return str(py) if py.exists() else sys.executable
    except Exception:
        return sys.executable


def _tail_text(path: Path, max_bytes: int = 12000) -> str:
    try:
        if not path.exists():
            return ""
        bs = int(max_bytes)
        with path.open("rb") as f:
            try:
                f.seek(0, os.SEEK_END)
                n = f.tell()
                f.seek(max(0, n - bs), os.SEEK_SET)
            except Exception:
                pass
            raw = f.read(bs)
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _read_sovits_state() -> Dict[str, Any]:
    try:
        p = REPO_ROOT / "logs" / "gpt_sovits.state.json"
        if not p.exists():
            return {}
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_sovits_state(obj: Dict[str, Any]) -> None:
    try:
        p = REPO_ROOT / "logs" / "gpt_sovits.state.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _tcp_listening(host: str, port: int, timeout_s: float = 0.3) -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(float(timeout_s))
        try:
            return s.connect_ex((str(host), int(port))) == 0
        finally:
            try:
                s.close()
            except Exception:
                pass
    except Exception:
        return False


def _stop_gpt_sovits_from_state() -> Dict[str, Any]:
    st = _read_sovits_state()
    host = str(st.get("host") or "127.0.0.1")
    try:
        port = int(st.get("port") or 9880)
    except Exception:
        port = 9880

    listening = _tcp_listening(host, int(port))
    pid = st.get("pid")
    try:
        pid_i = int(pid) if pid is not None else 0
    except Exception:
        pid_i = 0

    if (not listening) and pid_i <= 0:
        return {"ok": True, "running": False, "stopped": False, "pid": 0, "host": host, "port": int(port)}

    if pid_i > 0:
        try:
            subprocess.run(["taskkill", "/PID", str(pid_i), "/T", "/F"], capture_output=True, text=True)
        except Exception as e:
            return {"ok": False, "running": bool(listening), "stopped": False, "pid": pid_i, "error": str(e)}

        t0 = time.time()
        while time.time() - t0 < 8.0:
            if not _tcp_listening(host, int(port)):
                break
            time.sleep(0.25)

        return {"ok": True, "running": False, "stopped": True, "pid": pid_i, "host": host, "port": int(port)}

    return {"ok": True, "running": bool(listening), "stopped": False, "pid": 0, "host": host, "port": int(port)}


def _start_gpt_sovits_from_state() -> Dict[str, Any]:
    st = _read_sovits_state()
    host = str(st.get("host") or "127.0.0.1")
    try:
        port = int(st.get("port") or 9880)
    except Exception:
        port = 9880

    if _tcp_listening(host, int(port)):
        return {"ok": True, "started": False, "running": True, "host": host, "port": int(port), "pid": int(st.get("pid") or 0)}

    py = str(st.get("py") or "").strip()
    root = str(st.get("root") or "").strip()
    args = st.get("args") if isinstance(st.get("args"), list) else []
    outp = str(st.get("out") or str(REPO_ROOT / "logs" / "gpt_sovits.out.log"))
    errp = str(st.get("err") or str(REPO_ROOT / "logs" / "gpt_sovits.err.log"))

    if (not py) or (not Path(py).exists()) or (not root) or (not Path(root).exists()):
        return {"ok": False, "started": False, "error": "missing sovits py/root", "py": py, "root": root}

    if not args:
        args = ["api_v2.py", "-a", host, "-p", str(int(port))]

    try:
        Path(outp).parent.mkdir(parents=True, exist_ok=True)
        Path(errp).parent.mkdir(parents=True, exist_ok=True)
        fout = open(outp, "a", encoding="utf-8", errors="replace")
        ferr = open(errp, "a", encoding="utf-8", errors="replace")
    except Exception as e:
        return {"ok": False, "started": False, "error": f"log open failed: {e}"}

    try:
        proc = subprocess.Popen([py, *[str(x) for x in args]], cwd=str(root), stdout=fout, stderr=ferr)
    except Exception as e:
        try:
            fout.close()
        except Exception:
            pass
        try:
            ferr.close()
        except Exception:
            pass
        return {"ok": False, "started": False, "error": str(e)}

    t0 = time.time()
    while time.time() - t0 < 12.0:
        if _tcp_listening(host, int(port)):
            break
        time.sleep(0.25)

    st2 = dict(st)
    st2["pid"] = int(getattr(proc, "pid", 0) or 0)
    st2["updated_at"] = datetime.now().isoformat()
    st2["note"] = "restarted"
    _write_sovits_state(st2)
    return {"ok": True, "started": True, "running": _tcp_listening(host, int(port)), "host": host, "port": int(port), "pid": int(getattr(proc, "pid", 0) or 0)}


def _spawn_train_watcher(proc: subprocess.Popen) -> None:
    global _EVOLUTION_TRAIN_WATCHER
    global _SOVITS_RESUME_AFTER_TRAIN, _SOVITS_RESTARTED_AFTER_TRAIN

    def _run() -> None:
        global _EVOLUTION_TRAIN_FH
        try:
            proc.wait()
        except Exception:
            return

        with _EVOLUTION_TRAIN_LOCK:
            fh = _EVOLUTION_TRAIN_FH
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
            with _EVOLUTION_TRAIN_LOCK:
                _EVOLUTION_TRAIN_FH = None

        with _SOVITS_LOCK:
            if _SOVITS_RESUME_AFTER_TRAIN and (not _SOVITS_RESTARTED_AFTER_TRAIN):
                _start_gpt_sovits_from_state()
                _SOVITS_RESTARTED_AFTER_TRAIN = True
                _SOVITS_RESUME_AFTER_TRAIN = False

    try:
        th = threading.Thread(target=_run, daemon=True)
        _EVOLUTION_TRAIN_WATCHER = th
        th.start()
    except Exception:
        _EVOLUTION_TRAIN_WATCHER = None


def _spawn_alpha_train_watcher(proc: subprocess.Popen) -> None:
    global _ALPHA_TRAIN_WATCHER
    global _SOVITS_RESUME_AFTER_ALPHA, _SOVITS_RESTARTED_AFTER_ALPHA

    def _run() -> None:
        global _ALPHA_TRAIN_FH
        global _ALPHA_TRAIN_OUTPUT_DIR, _ALPHA_TRAIN_SFT_ADAPTER
        try:
            proc.wait()
        except Exception:
            return

        try:
            rc = proc.returncode
        except Exception:
            rc = None

        try:
            if rc == 0:
                out_dir = str(_ALPHA_TRAIN_OUTPUT_DIR or "").strip()
                sft_adapter = str(_ALPHA_TRAIN_SFT_ADAPTER or "").strip()
                if out_dir:
                    out_path = Path(out_dir)
                    if not out_path.is_absolute():
                        out_path = (REPO_ROOT / out_path).resolve()
                    ok = out_path.exists() and (
                        (out_path / "adapter_config.json").exists()
                        or (out_path / "adapter_model.safetensors").exists()
                        or (out_path / "adapter_model.bin").exists()
                    )
                    if ok:
                        scalper_path = ""
                        try:
                            cfg = _get_secretary_config()
                            trading_cfg = cfg.get("trading") if isinstance(cfg, dict) and isinstance(cfg.get("trading"), dict) else {}
                            scalper_path = str(trading_cfg.get("moe_scalper") or "").strip()
                        except Exception:
                            scalper_path = ""

                        try:
                            payload = {
                                "time": datetime.now().isoformat(),
                                "source": "alpha_evolution",
                                "base_model": "Qwen/Qwen2.5-7B-Instruct",
                                "active_moe_scalper": scalper_path,
                                "active_moe_analyst": str(out_path),
                                "alpha_sft_adapter": sft_adapter,
                            }
                            p = REPO_ROOT / "data" / "finetune" / "evolution" / "active_trading_models.json"
                            p.parent.mkdir(parents=True, exist_ok=True)
                            p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                        except Exception:
                            pass
        except Exception:
            pass

        with _ALPHA_TRAIN_LOCK:
            fh = _ALPHA_TRAIN_FH
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
            with _ALPHA_TRAIN_LOCK:
                _ALPHA_TRAIN_FH = None

        with _SOVITS_LOCK:
            if _SOVITS_RESUME_AFTER_ALPHA and (not _SOVITS_RESTARTED_AFTER_ALPHA):
                _start_gpt_sovits_from_state()
                _SOVITS_RESTARTED_AFTER_ALPHA = True
                _SOVITS_RESUME_AFTER_ALPHA = False

    try:
        th = threading.Thread(target=_run, daemon=True)
        _ALPHA_TRAIN_WATCHER = th
        th.start()
    except Exception:
        _ALPHA_TRAIN_WATCHER = None


def _read_last_adapter_meta() -> dict:
    try:
        p = REPO_ROOT / "data" / "finetune" / "evolution" / "last_adapter.json"
        if not p.exists():
            return {}
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


@app.post("/api/v1/evolution/nightly/train/start")
async def start_nightly_evolution_train():
    """Start Ouroboros nightly_evolution training (SFT + DPO) in background."""
    global _EVOLUTION_TRAIN_PROC, _EVOLUTION_TRAIN_FH, _EVOLUTION_TRAIN_LOG, _EVOLUTION_TRAIN_CMD, _EVOLUTION_TRAIN_STARTED_AT
    global _EVOLUTION_TRAIN_WATCHER
    global _SOVITS_RESUME_AFTER_TRAIN, _SOVITS_RESTARTED_AFTER_TRAIN
    with _EVOLUTION_TRAIN_LOCK:
        p = _EVOLUTION_TRAIN_PROC
        if p is not None and p.poll() is None:
            return {
                "ok": True,
                "running": True,
                "pid": int(getattr(p, "pid", 0) or 0),
                "log_path": str(_EVOLUTION_TRAIN_LOG) if _EVOLUTION_TRAIN_LOG else "",
                "cmd": " ".join(_EVOLUTION_TRAIN_CMD or []),
            }

    with _SOVITS_LOCK:
        if not _SOVITS_RESUME_AFTER_TRAIN and not _SOVITS_RESTARTED_AFTER_TRAIN:
            res = _stop_gpt_sovits_from_state()
            if bool(res.get("stopped")):
                _SOVITS_RESUME_AFTER_TRAIN = True
                _SOVITS_RESTARTED_AFTER_TRAIN = False

    try:
        from src.llm.local_chat import unload_model
        await run_in_threadpool(unload_model)
    except Exception:
        pass

    try:
        setattr(_live_runner, "load_models", True)
    except Exception:
        pass

    try:
        stg = getattr(_live_runner, "strategy", None)
        if stg is not None and (not bool(getattr(stg, "models_loaded", False))):
            fn = getattr(stg, "load_models", None)
            if callable(fn):
                fn()
    except Exception:
        pass

    out_dir = REPO_ROOT / "data" / "finetune" / "evolution"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"nightly_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    python_exe = _pick_python_for_repo()
    cmd = [str(python_exe), "scripts/nightly_evolution.py"]
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    fh: Any = None
    proc: Optional[subprocess.Popen] = None
    try:
        fh = log_path.open("w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
    except Exception as e:
        try:
            if fh is not None:
                fh.close()
        except Exception:
            pass
        return {"ok": False, "running": False, "error": str(e)}

    with _EVOLUTION_TRAIN_LOCK:
        _EVOLUTION_TRAIN_PROC = proc
        _EVOLUTION_TRAIN_FH = fh
        _EVOLUTION_TRAIN_LOG = log_path
        _EVOLUTION_TRAIN_CMD = list(cmd)
        _EVOLUTION_TRAIN_STARTED_AT = datetime.now().isoformat()

    try:
        _spawn_train_watcher(proc)
    except Exception:
        pass

    return {
        "ok": True,
        "running": True,
        "pid": int(getattr(proc, "pid", 0) or 0),
        "log_path": str(log_path),
        "cmd": " ".join(cmd),
    }


@app.get("/api/v1/evolution/nightly/train/status")
async def get_nightly_evolution_train_status(tail_bytes: int = 12000):
    """Get status and log tail for Ouroboros nightly_evolution training."""
    global _EVOLUTION_TRAIN_FH
    global _SOVITS_RESUME_AFTER_TRAIN, _SOVITS_RESTARTED_AFTER_TRAIN
    with _EVOLUTION_TRAIN_LOCK:
        p = _EVOLUTION_TRAIN_PROC
        logp = _EVOLUTION_TRAIN_LOG
        cmd = list(_EVOLUTION_TRAIN_CMD or [])
        started = _EVOLUTION_TRAIN_STARTED_AT
        fh = _EVOLUTION_TRAIN_FH

    if p is None:
        meta = _read_last_adapter_meta()
        return {"ok": True, "running": False, "started_at": None, "returncode": None, "cmd": "", "log_path": "", "log_tail": "", "meta": meta}

    rc = p.poll()
    running = rc is None
    if (not running) and fh is not None:
        try:
            fh.close()
        except Exception:
            pass
        with _EVOLUTION_TRAIN_LOCK:
            _EVOLUTION_TRAIN_FH = None

    if (not running):
        with _SOVITS_LOCK:
            if _SOVITS_RESUME_AFTER_TRAIN and (not _SOVITS_RESTARTED_AFTER_TRAIN):
                _start_gpt_sovits_from_state()
                _SOVITS_RESTARTED_AFTER_TRAIN = True
                _SOVITS_RESUME_AFTER_TRAIN = False

    meta = _read_last_adapter_meta()
    tail = _tail_text(logp, max_bytes=int(tail_bytes)) if logp is not None else ""
    return {
        "ok": True,
        "running": bool(running),
        "pid": int(getattr(p, "pid", 0) or 0),
        "started_at": started,
        "returncode": None if running else int(rc if rc is not None else -1),
        "cmd": " ".join(cmd),
        "log_path": str(logp) if logp is not None else "",
        "log_tail": str(tail or ""),
        "meta": meta,
    }


@app.post("/api/v1/evolution/nightly/train/stop")
async def stop_nightly_evolution_train():
    """Stop Ouroboros nightly_evolution training process."""
    global _EVOLUTION_TRAIN_PROC, _EVOLUTION_TRAIN_FH
    global _SOVITS_RESUME_AFTER_TRAIN, _SOVITS_RESTARTED_AFTER_TRAIN
    with _EVOLUTION_TRAIN_LOCK:
        p = _EVOLUTION_TRAIN_PROC
        fh = _EVOLUTION_TRAIN_FH
        logp = _EVOLUTION_TRAIN_LOG

    if p is None or p.poll() is not None:
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
        with _EVOLUTION_TRAIN_LOCK:
            _EVOLUTION_TRAIN_PROC = None
            _EVOLUTION_TRAIN_FH = None

        with _SOVITS_LOCK:
            if _SOVITS_RESUME_AFTER_TRAIN and (not _SOVITS_RESTARTED_AFTER_TRAIN):
                _start_gpt_sovits_from_state()
                _SOVITS_RESTARTED_AFTER_TRAIN = True
                _SOVITS_RESUME_AFTER_TRAIN = False

        return {"ok": True, "stopped": False, "message": "not running", "log_path": str(logp) if logp else ""}

    try:
        p.terminate()
    except Exception as e:
        return {"ok": False, "stopped": False, "error": str(e), "log_path": str(logp) if logp else ""}

    with _SOVITS_LOCK:
        if _SOVITS_RESUME_AFTER_TRAIN:
            _SOVITS_RESTARTED_AFTER_TRAIN = False

    return {"ok": True, "stopped": True, "pid": int(getattr(p, "pid", 0) or 0), "log_path": str(logp) if logp else ""}


@app.post("/api/v1/evolution/alpha/train/start")
async def start_alpha_evolution_train(
    reward_thr: float = 50.0,
    punish_thr: float = -20.0,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    sft_adapter: str = "models/trader_stock_v1_1_tech_plus_news/lora_weights",
    output_dir: str = "",
):
    global _ALPHA_TRAIN_PROC, _ALPHA_TRAIN_FH, _ALPHA_TRAIN_LOG, _ALPHA_TRAIN_CMD, _ALPHA_TRAIN_STARTED_AT
    global _SOVITS_RESUME_AFTER_ALPHA, _SOVITS_RESTARTED_AFTER_ALPHA

    with _ALPHA_TRAIN_LOCK:
        p = _ALPHA_TRAIN_PROC
        if p is not None and p.poll() is None:
            return {
                "ok": True,
                "running": True,
                "pid": int(getattr(p, "pid", 0) or 0),
                "log_path": str(_ALPHA_TRAIN_LOG) if _ALPHA_TRAIN_LOG else "",
                "cmd": " ".join(_ALPHA_TRAIN_CMD or []),
            }

    try:
        with _EVOLUTION_TRAIN_LOCK:
            tp = _EVOLUTION_TRAIN_PROC
        if tp is not None and tp.poll() is None:
            return {"ok": False, "running": False, "error": "nightly training is running"}
    except Exception:
        pass

    with _SOVITS_LOCK:
        if not _SOVITS_RESUME_AFTER_ALPHA and not _SOVITS_RESTARTED_AFTER_ALPHA:
            res = _stop_gpt_sovits_from_state()
            if bool(res.get("stopped")):
                _SOVITS_RESUME_AFTER_ALPHA = True
                _SOVITS_RESTARTED_AFTER_ALPHA = False

    try:
        from src.llm.local_chat import unload_model
        await run_in_threadpool(unload_model)
    except Exception:
        pass

    try:
        if _live_runner is not None:
            rm = getattr(_live_runner, "rl_manager", None)
            if rm is not None:
                try:
                    setattr(rm, "enabled", False)
                except Exception:
                    pass
    except Exception:
        pass

    python_exe = _pick_python_for_repo()
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    evo_out_dir = REPO_ROOT / "data" / "finetune" / "evolution"
    evo_out_dir.mkdir(parents=True, exist_ok=True)

    alpha_data_path = evo_out_dir / "dpo_alpha_nightly.jsonl"
    gen_cmd = [str(python_exe), "scripts/generate_alpha_dataset.py", "--reward-thr", str(float(reward_thr)), "--punish-thr", str(float(punish_thr))]
    try:
        subprocess.run(gen_cmd, cwd=str(REPO_ROOT), check=False, env=env, capture_output=True, text=True, timeout=180)
    except Exception as e:
        return {"ok": False, "running": False, "error": f"alpha dataset generation failed: {e}"}

    pairs_count = 0
    try:
        if alpha_data_path.exists():
            with alpha_data_path.open("r", encoding="utf-8") as f:
                for _ in f:
                    pairs_count += 1
    except Exception:
        pairs_count = 0

    if pairs_count <= 0:
        with _SOVITS_LOCK:
            if _SOVITS_RESUME_AFTER_ALPHA and (not _SOVITS_RESTARTED_AFTER_ALPHA):
                _start_gpt_sovits_from_state()
                _SOVITS_RESTARTED_AFTER_ALPHA = True
                _SOVITS_RESUME_AFTER_ALPHA = False
        return {"ok": False, "running": False, "error": "no alpha dpo pairs", "pairs": int(pairs_count), "data_path": str(alpha_data_path)}

    out_dir = str(output_dir or "").strip()
    if not out_dir:
        out_dir = str(REPO_ROOT / "models" / f"alpha_dpo_from_pnl_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    log_path = evo_out_dir / f"alpha_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    cmd = [
        str(python_exe),
        "scripts/train_dpo.py",
        "--base-model",
        str(base_model),
        "--sft-adapter",
        str(sft_adapter),
        "--data-path",
        str(alpha_data_path.as_posix()),
        "--output-dir",
        str(out_dir),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--grad-accum",
        "8",
        "--reference-free",
    ]

    fh: Any = None
    proc: Optional[subprocess.Popen] = None
    try:
        fh = log_path.open("w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
    except Exception as e:
        try:
            if fh is not None:
                fh.close()
        except Exception:
            pass
        return {"ok": False, "running": False, "error": str(e)}

    with _ALPHA_TRAIN_LOCK:
        _ALPHA_TRAIN_PROC = proc
        _ALPHA_TRAIN_FH = fh
        _ALPHA_TRAIN_LOG = log_path
        _ALPHA_TRAIN_CMD = list(cmd)
        _ALPHA_TRAIN_STARTED_AT = datetime.now().isoformat()
        _ALPHA_TRAIN_OUTPUT_DIR = str(out_dir)
        _ALPHA_TRAIN_SFT_ADAPTER = str(sft_adapter)

    try:
        _spawn_alpha_train_watcher(proc)
    except Exception:
        pass

    return {
        "ok": True,
        "running": True,
        "pid": int(getattr(proc, "pid", 0) or 0),
        "pairs": int(pairs_count),
        "data_path": str(alpha_data_path),
        "output_dir": str(out_dir),
        "log_path": str(log_path),
        "cmd": " ".join(cmd),
    }


@app.get("/api/v1/evolution/alpha/train/status")
async def get_alpha_evolution_train_status(tail_bytes: int = 12000):
    global _ALPHA_TRAIN_FH
    with _ALPHA_TRAIN_LOCK:
        p = _ALPHA_TRAIN_PROC
        logp = _ALPHA_TRAIN_LOG
        cmd = list(_ALPHA_TRAIN_CMD or [])
        started = _ALPHA_TRAIN_STARTED_AT
        fh = _ALPHA_TRAIN_FH

    if p is None:
        return {"ok": True, "running": False, "started_at": None, "returncode": None, "cmd": "", "log_path": "", "log_tail": "", "voice_stopped": bool(_SOVITS_RESUME_AFTER_ALPHA)}

    rc = p.poll()
    running = rc is None
    if (not running) and fh is not None:
        try:
            fh.close()
        except Exception:
            pass
        with _ALPHA_TRAIN_LOCK:
            _ALPHA_TRAIN_FH = None

    tail = _tail_text(logp, max_bytes=int(tail_bytes)) if logp is not None else ""
    return {
        "ok": True,
        "running": bool(running),
        "pid": int(getattr(p, "pid", 0) or 0),
        "started_at": started,
        "returncode": None if running else int(rc if rc is not None else -1),
        "cmd": " ".join(cmd),
        "log_path": str(logp) if logp is not None else "",
        "log_tail": str(tail or ""),
        "voice_stopped": bool(_SOVITS_RESUME_AFTER_ALPHA),
    }


@app.post("/api/v1/evolution/alpha/train/stop")
async def stop_alpha_evolution_train():
    global _ALPHA_TRAIN_PROC, _ALPHA_TRAIN_FH
    global _SOVITS_RESUME_AFTER_ALPHA, _SOVITS_RESTARTED_AFTER_ALPHA
    with _ALPHA_TRAIN_LOCK:
        p = _ALPHA_TRAIN_PROC
        fh = _ALPHA_TRAIN_FH
        logp = _ALPHA_TRAIN_LOG

    if p is None or p.poll() is not None:
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
        with _ALPHA_TRAIN_LOCK:
            _ALPHA_TRAIN_PROC = None
            _ALPHA_TRAIN_FH = None

        with _SOVITS_LOCK:
            if _SOVITS_RESUME_AFTER_ALPHA and (not _SOVITS_RESTARTED_AFTER_ALPHA):
                _start_gpt_sovits_from_state()
                _SOVITS_RESTARTED_AFTER_ALPHA = True
                _SOVITS_RESUME_AFTER_ALPHA = False

        return {"ok": True, "stopped": False, "message": "not running", "log_path": str(logp) if logp else ""}

    try:
        p.terminate()
    except Exception as e:
        return {"ok": False, "stopped": False, "error": str(e), "log_path": str(logp) if logp else ""}

    with _SOVITS_LOCK:
        if _SOVITS_RESUME_AFTER_ALPHA:
            _SOVITS_RESTARTED_AFTER_ALPHA = False

    return {"ok": True, "stopped": True, "pid": int(getattr(p, "pid", 0) or 0), "log_path": str(logp) if logp else ""}


@app.get("/api/v1/status")
async def status():
    """Unified status snapshot for Secretary (trading + voice training)."""
    try:
        return _build_status_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"status failed: {e}")


def _get_secretary_config() -> Dict[str, Any]:
    global _SECRETARY_CFG, _SECRETARY_CFG_MTIME

    try:
        fp = SECRETARY_CONFIG_PATH
        if not fp.exists():
            _SECRETARY_CFG = {}
            _SECRETARY_CFG_MTIME = None
            return {}
        mtime = float(fp.stat().st_mtime)
        if _SECRETARY_CFG is not None and _SECRETARY_CFG_MTIME is not None and mtime == _SECRETARY_CFG_MTIME:
            return dict(_SECRETARY_CFG)
        cfg = yaml.safe_load(fp.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            cfg = {}

        try:
            p = REPO_ROOT / "data" / "finetune" / "evolution" / "active_secretary_adapter.json"
            if p.exists():
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    ap = str(obj.get("active_secretary_adapter") or "").strip()
                    if ap:
                        cand = Path(ap)
                        if not cand.is_absolute():
                            cand = (REPO_ROOT / cand).resolve()
                        if cand.exists():
                            llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
                            llm_cfg = dict(llm_cfg)
                            llm_cfg["local_adapter"] = str(cand)
                            cfg["llm"] = llm_cfg
        except Exception:
            pass

        _SECRETARY_CFG = dict(cfg)
        _SECRETARY_CFG_MTIME = mtime
        return dict(_SECRETARY_CFG)
    except Exception:
        _SECRETARY_CFG = {}
        _SECRETARY_CFG_MTIME = None
        return {}


def _build_secretary_context(extra: Dict[str, Any]) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    
    # Add live trading data if available
    if _live_runner is not None:
        try:
            live_ctx: Dict[str, Any] = {
                "mode": "live_paper_trading",
                "cash": _live_runner.broker.cash,
                "initial_cash": getattr(_live_runner.broker, "initial_cash", 500000.0),
                "data_source": getattr(_live_runner, "data_source", "unknown"),
                "trading_mode": getattr(_live_runner, "trading_mode", "online"),
            }
            # Calculate PnL
            positions = getattr(_live_runner.broker, "positions", {})
            total_value = _live_runner.broker.cash
            position_details = []
            for ticker, pos in positions.items():
                if hasattr(pos, "shares") and hasattr(pos, "avg_price"):
                    # Get current price
                    current_price = pos.avg_price  # fallback
                    if hasattr(_live_runner, "price_history") and ticker in _live_runner.price_history:
                        prices = _live_runner.price_history[ticker]
                        if prices:
                            current_price = prices[-1].get("close", pos.avg_price)
                    position_value = pos.shares * current_price
                    total_value += position_value
                    unrealized_pnl = (current_price - pos.avg_price) * pos.shares
                    position_details.append({
                        "ticker": ticker,
                        "shares": pos.shares,
                        "avg_price": round(pos.avg_price, 2),
                        "current_price": round(current_price, 2),
                        "unrealized_pnl": round(unrealized_pnl, 2),
                    })
            
            initial = getattr(_live_runner.broker, "initial_cash", 500000.0)
            live_ctx["total_value"] = round(total_value, 2)
            live_ctx["total_pnl"] = round(total_value - initial, 2)
            live_ctx["total_pnl_pct"] = round((total_value - initial) / initial * 100, 2)
            live_ctx["positions"] = position_details
            live_ctx["trade_count"] = len(_live_runner.trade_log)

            # Latest bar time (per ticker, best-effort)
            try:
                latest_bars: Dict[str, str] = {}
                if hasattr(_live_runner, "price_history") and isinstance(_live_runner.price_history, dict):
                    for tk, rows in _live_runner.price_history.items():
                        if rows:
                            t0 = rows[-1].get("time")
                            if t0:
                                latest_bars[str(tk)] = str(t0)
                if latest_bars:
                    live_ctx["latest_bars"] = latest_bars
            except Exception:
                pass
            
            # Recent trades (last 5)
            if _live_runner.trade_log:
                recent = _live_runner.trade_log[-5:]
                live_ctx["recent_trades"] = [
                    {"action": t.get("action"), "ticker": t.get("ticker"), 
                     "price": t.get("price"), "shares": t.get("shares")}
                    for t in recent
                ]
            
            # Recent agent logs (last 10)
            if hasattr(_live_runner, "agent_logs") and _live_runner.agent_logs:
                live_ctx["recent_agent_logs"] = _live_runner.agent_logs[-10:]
            
            ctx["live_trading"] = live_ctx
        except Exception as e:
            ctx["live_trading"] = {"error": str(e)}
    
    try:
        runs = _list_run_dirs()
        if runs:
            run_dir = runs[0]
            ctx["latest_run_id"] = run_dir.name
            sys_dirs = _list_system_dirs(run_dir)
            ctx["systems"] = [p.name for p in sys_dirs]
            dates = _list_dates_for_run(run_dir)
            if dates:
                ctx["latest_date"] = dates[-1]

            run_metrics_fp = run_dir / "metrics.json"
            if run_metrics_fp.exists():
                try:
                    rm = _read_json(run_metrics_fp)
                    if isinstance(rm, dict):
                        ctx["run_metrics"] = rm
                except Exception:
                    pass
    except Exception:
        pass

    try:
        ctx["status"] = _build_status_snapshot()
    except Exception:
        pass

    if isinstance(extra, dict) and extra:
        ctx["client_context"] = extra
    return ctx


def _tail_text_lines(fp: Path, *, max_lines: int = 80) -> List[str]:
    raw_text = ""
    try:
        data = fp.read_bytes()
    except Exception:
        return []

    # Heuristic: some Windows logs are UTF-16LE (lots of NUL bytes).
    try:
        if b"\x00" in data[:512]:
            raw_text = data.decode("utf-16-le", errors="replace")
        else:
            raw_text = data.decode("utf-8", errors="replace")
    except Exception:
        try:
            raw_text = data.decode("utf-8", errors="replace")
        except Exception:
            return []

    lines = [str(x) for x in raw_text.splitlines() if str(x).strip()]
    if max_lines <= 0:
        return lines
    return lines[-int(max_lines) :]


def _detect_latest_matching(dir_path: Path, *, pattern: str) -> Optional[Dict[str, Any]]:
    try:
        if not dir_path.exists() or not dir_path.is_dir():
            return None
        best: Optional[Path] = None
        for p in dir_path.glob(str(pattern)):
            if not p.is_file():
                continue
            if best is None or float(p.stat().st_mtime) > float(best.stat().st_mtime):
                best = p
        if best is None:
            return None
        return {
            "path": str(best),
            "name": best.name,
            "size": int(best.stat().st_size),
            "mtime": float(best.stat().st_mtime),
        }
    except Exception:
        return None


def _read_json_safe(fp: Path) -> Any:
    try:
        return _read_json(fp)
    except Exception:
        return None


def _build_trading_status() -> Dict[str, Any]:
    run_id = ""
    state: Dict[str, Any] = {"available": False}
    events_tail: List[Dict[str, Any]] = []

    runs = _list_run_dirs()
    if not runs:
        return {"available": False}
    run_dir = runs[0]
    run_id = run_dir.name

    state_fp = run_dir / "engine_state.json"
    if state_fp.exists():
        payload = _read_json_safe(state_fp)
        if isinstance(payload, dict):
            state = {"available": True, **payload}
        else:
            state = {"available": True, "run_id": run_id, "status": "unknown"}

    events_fp = run_dir / "events.jsonl"
    if events_fp.exists():
        tail_lines = _tail_text_lines(events_fp, max_lines=40)
        for s in tail_lines:
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                events_tail.append(obj)

    return {
        "available": True,
        "latest_run_id": run_id,
        "run_dir": str(run_dir),
        "engine_state": state,
        "events_tail": events_tail,
    }


def _build_voice_status() -> Dict[str, Any]:
    cfg = _get_secretary_config()
    vt = cfg.get("voice_training") if isinstance(cfg.get("voice_training"), dict) else {}
    exp_dir_s = str((vt or {}).get("exp_dir") or "").strip()
    if not exp_dir_s:
        return {"available": False}

    exp_dir = Path(exp_dir_s)
    out: Dict[str, Any] = {"available": True, "exp_dir": str(exp_dir)}

    s1_log = str((vt or {}).get("s1_log") or "").strip()
    if s1_log:
        s1_fp = Path(s1_log)
        if s1_fp.exists():
            out["s1_log_tail"] = _tail_text_lines(s1_fp, max_lines=30)

    s2_log = str((vt or {}).get("s2_log") or "").strip()
    if s2_log:
        s2_fp = Path(s2_log)
        if s2_fp.exists():
            out["s2_log_tail"] = _tail_text_lines(s2_fp, max_lines=30)

    s1_ckpt_dir = Path(str((vt or {}).get("s1_ckpt_dir") or exp_dir / "logs_s1_v2" / "ckpt"))
    s2_ckpt_dir = Path(str((vt or {}).get("s2_ckpt_dir") or exp_dir / "logs_s2_v2"))
    out["s1_latest_ckpt"] = _detect_latest_matching(s1_ckpt_dir, pattern="*.ckpt")
    out["s2_latest_G"] = _detect_latest_matching(s2_ckpt_dir, pattern="G_*.pth")
    out["s2_latest_D"] = _detect_latest_matching(s2_ckpt_dir, pattern="D_*.pth")

    return out


def _build_status_snapshot() -> Dict[str, Any]:
    return {
        "updated_at": float(time.time()),
        "trading": _build_trading_status(),
        "voice": _build_voice_status(),
    }


def _extract_ticker_from_text(text: str) -> Optional[str]:
    t = str(text or "").upper()
    m = re.search(r"\b[A-Z]{2,6}\b", t)
    if not m:
        return None
    cand = str(m.group(0)).upper().strip()
    if cand in {"THE", "AND", "FOR", "WITH", "THIS", "THAT"}:
        return None
    return cand


def _extract_date_from_text(text: str) -> Optional[str]:
    t = str(text or "")
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", t)
    if not m:
        return None
    return str(m.group(1))


def _extract_run_from_text(text: str) -> Optional[str]:
    t = str(text or "")
    m = re.search(r"\b(phase\d+[_\-][A-Za-z0-9_\-]+)\b", t)
    if not m:
        return None
    return str(m.group(1))


def _maybe_build_trade_rag(*, user_text: str, ctx: Dict[str, Any]) -> str:
    """Return RAG context string or empty."""
    # Skip historical RAG if live trading is active - use real-time data instead
    if _live_runner is not None:
        return ""  # Live trading context is already in _build_secretary_context
    
    ticker = _extract_ticker_from_text(user_text)
    if not ticker:
        return ""

    merged = _build_secretary_context(ctx)
    latest_run = _safe_str(merged.get("latest_run_id") or "")
    latest_date = _safe_str(merged.get("latest_date") or "")

    run_id = _extract_run_from_text(user_text) or latest_run
    date_str = _extract_date_from_text(user_text) or latest_date

    if not run_id or not date_str:
        return ""

    try:
        res = narrate_trade_context(run_id=run_id, date=date_str, ticker=ticker)
    except Exception as e:
        logger.warning(f"narrator failed: {e}")
        return ""

    if not isinstance(res.narrative, str) or not res.narrative.strip():
        return ""

    return "\n\n[交易档案检索结果（事实依据，请以此回答）]\n" + res.narrative.strip() + "\n"


def _classify_secretary_profile(text: str) -> str:
    t = str(text or "").strip()
    if not t:
        return "chat"

    tl = t.lower()
    if tl.startswith("/"):
        return "work"
    if tl.startswith("news:") or tl.startswith("新闻:"):
        return "work"

    if _extract_ticker_from_text(t):
        return "work"

    if len(t) >= 180:
        return "work"

    heavy = [
        "监控",
        "状态",
        "复盘",
        "策略",
        "回测",
        "实盘",
        "模拟盘",
        "执行",
        "信号",
        "订单",
        "成交",
        "风控",
        "波动",
        "波动率",
        "仓位",
        "持仓",
        "现金",
        "盈亏",
        "股票",
        "美股",
        "买",
        "买入",
        "卖",
        "卖出",
        "入场",
        "开仓",
        "平仓",
        "加仓",
        "减仓",
        "持有",
        "pnl",
        "portfolio",
        "position",
        "positions",
        "holding",
        "holdings",
        "share",
        "shares",
        "gatekeeper",
        "planner",
        "router",
        "moe",
        "system 2",
        "debate",
        "macro",
        "chart",
        "news",
        "训练",
        "online rl",
        "rl",
    ]
    for k in heavy:
        if k in t or k in tl:
            return "work"

    return "chat"


def _call_llm(*, text: str, ctx: Dict[str, Any], profile: str = "work") -> Optional[str]:
    cfg = _get_secretary_config()
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    sec_cfg = cfg.get("secretary") if isinstance(cfg.get("secretary"), dict) else {}

    # Check mode: local or api
    mode = str((llm_cfg or {}).get("mode") or "api").strip().lower()
    
    try:
        temperature = float((llm_cfg or {}).get("temperature", 0.7))
    except Exception:
        temperature = 0.7
    try:
        max_tokens = int((llm_cfg or {}).get("max_tokens", 256))
    except Exception:
        max_tokens = 256

    prof = str(profile or "work").strip().lower()
    system_prompt = str((sec_cfg or {}).get("system_prompt") or "You are a helpful assistant.").strip()

    if prof == "chat":
        try:
            temperature = max(float(temperature), 0.85)
        except Exception:
            temperature = 0.85
        try:
            max_tokens = min(int(max_tokens), 160)
        except Exception:
            max_tokens = 160

        try:
            memory = get_mari_memory()
            memory_ctx = memory.get_context_for_llm(limit=6)
            if memory_ctx:
                system_prompt = system_prompt + "\n\n" + memory_ctx
        except Exception as e:
            logger.debug(f"Memory context error: {e}")

    else:
        merged_ctx = _build_secretary_context(ctx)
        rag = _maybe_build_trade_rag(user_text=str(text), ctx=ctx)

        try:
            memory = get_mari_memory()
            memory_ctx = memory.get_context_for_llm(limit=20)
            if memory_ctx:
                system_prompt = system_prompt + "\n\n" + memory_ctx
        except Exception as e:
            logger.debug(f"Memory context error: {e}")

        if merged_ctx:
            try:
                ctx_yaml = yaml.safe_dump(merged_ctx, allow_unicode=True, sort_keys=False)
            except Exception:
                ctx_yaml = ""
            if ctx_yaml:
                system_prompt = system_prompt + "\n\n[Context]\n" + ctx_yaml

        if rag:
            system_prompt = system_prompt + rag

        system_prompt = system_prompt + _tool_instructions()

    # === LOCAL MODE: Direct model inference ===
    if mode == "local":
        try:
            from src.llm.local_chat import chat as local_chat
            local_model = str((llm_cfg or {}).get("local_model") or "Qwen/Qwen3-8B")
            local_adapter = str((llm_cfg or {}).get("local_adapter") or "").strip() or None
            use_4bit = bool((llm_cfg or {}).get("use_4bit", False))
            use_8bit = bool((llm_cfg or {}).get("use_8bit", True))
            
            quant = "8bit" if use_8bit else ("4bit" if use_4bit else "fp16")
            logger.info(f"[LLM] Local mode: {local_model} ({quant}) profile={prof}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(text)},
            ]
            response = local_chat(
                messages,
                model_name=local_model,
                temperature=temperature,
                max_new_tokens=max_tokens,
                use_4bit=use_4bit,
                use_8bit=(not use_4bit),
                adapter_path=local_adapter,
            )
            if response:
                return response.strip()
            return None
        except Exception as e:
            logger.error(f"[LLM] Local mode error: {e}")
            return None

    # === API MODE: Ollama/OpenAI compatible ===
    api_base = str((llm_cfg or {}).get("api_base") or "").strip()
    api_key = str((llm_cfg or {}).get("api_key") or "").strip() or "local"
    model = str((llm_cfg or {}).get("model") or "").strip()
    if not api_base or not model:
        return None

    try:
        timeout_sec = float((llm_cfg or {}).get("timeout_sec", 35.0))
    except Exception:
        timeout_sec = 35.0
    timeout_sec = max(5.0, min(timeout_sec, 180.0))

    try:
        client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout_sec)
    except Exception:
        client = OpenAI(base_url=api_base, api_key=api_key)
    logger.info(f"[LLM] API mode: {api_base} model={model} profile={prof}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(text)},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.warning(f"[LLM] API call failed: {e}")
        return None
    logger.info(f"[LLM] Response received, choices={bool(getattr(resp, 'choices', None))}")
    if not getattr(resp, "choices", None):
        return None
    msg = resp.choices[0].message
    content = getattr(msg, "content", None)
    logger.info(f"[LLM] Raw content type={type(content)}, len={len(content) if content else 0}")
    
    # Qwen3 may return content with <think>...</think> tags, strip them
    if isinstance(content, str):
        import re
        # Remove thinking tags if present
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        logger.info(f"[LLM] Cleaned content len={len(content)}")
        if content:
            return content
    return None


def _secretary_reply(text: str, ctx: Dict[str, Any]) -> str:
    t = str(text or "").strip()
    tl = t.lower()

    try:
        d0 = _portfolio_digest(text=t, ctx=ctx)
        if isinstance(d0, str) and d0.strip():
            return d0.strip()
    except Exception:
        pass

    try:
        dms = _models_status_digest(text=t, ctx=ctx)
        if isinstance(dms, str) and dms.strip():
            return dms.strip()
    except Exception:
        pass

    try:
        d = _agent_logs_digest(text=t, ctx=ctx)
        if isinstance(d, str) and d.strip():
            return d.strip()
    except Exception:
        pass

    # 0) Follow-up handling for desktop sessions (no explicit id).
    try:
        st = _get_session_state(ctx)
        if st is not None:
            last_tid = str(st.get("last_task_id") or "").strip() or None
            if bool(st.get("awaiting_detail")) and _is_affirmative_short(t):
                _clear_awaiting_detail(ctx)
                if last_tid:
                    return _elaborate_task(last_tid)
            if last_tid and _is_followup_status_query(t):
                return _task_status_text(last_tid)
    except Exception:
        pass

    tid = _extract_task_id(t)
    if tid:
        # Remember for future follow-ups.
        _remember_last_task(ctx, tid)
        return _task_status_text(tid)

    # News job status query by id
    nid = _extract_news_id(t)
    if nid:
        with _AGENT_LOCK:
            job = _NEWS_JOBS.get(str(nid))
        if not isinstance(job, dict):
            return f"Sensei, 我没有找到这个 news_id：{nid}"
        st = str(job.get("status") or "")
        if st == "done":
            summary = str(job.get("summary") or "").strip()
            return (summary + f"\n\n(news_id={nid})").strip()
        if st == "error":
            return f"Sensei, 这条新闻分析失败了：{job.get('error')} (news_id={nid})"
        return f"Sensei, 这条新闻还在处理中：status={st} (news_id={nid})"

    # Direct news submission from chat
    if tl.startswith("/news") or tl.startswith("news:") or tl.startswith("新闻:"):
        body = t.split("\n", 1)[-1]
        body = re.sub(r"^(/news\s*|news:\s*|新闻:\s*)", "", body, flags=re.IGNORECASE).strip()
        if not body:
            return "Sensei, 请把新闻正文或链接贴给我。"
        url = _extract_first_url(body)
        news_id = _enqueue_news_job(text=body, url=url, source="chat")
        try:
            _append_trajectory_log(
                {
                    "time": datetime.now().isoformat(),
                    "type": "contract.news.submit",
                    "news_id": news_id,
                    "client": (ctx or {}).get("client"),
                    "session_id": (ctx or {}).get("session_id"),
                    "input": {"url": url, "text_len": len(body)},
                }
            )
        except Exception:
            pass
        done = _wait_news_done(news_id, timeout_sec=6)
        if isinstance(done, dict) and done.get("status") == "done":
            summary = str(done.get("summary") or "").strip()
            return (summary + f"\n\n(news_id={news_id})").strip()
        return f"Sensei, 我已经把新闻交给分析员们处理了。稍后你可以再问我这个编号：{news_id}"

    # Auto-submit short URL messages as news (no need to prefix /news)
    try:
        url0 = _extract_first_url(t)
    except Exception:
        url0 = None
    if url0:
        s0 = t.strip()
        is_bare_url = s0.startswith("http://") or s0.startswith("https://") or s0.startswith("/http://") or s0.startswith("/https://")
        if is_bare_url or _is_news_question(t):
            news_id = _enqueue_news_job(text=t, url=url0, source="chat_auto_url")
            try:
                _append_trajectory_log(
                    {
                        "time": datetime.now().isoformat(),
                        "type": "contract.news.submit",
                        "news_id": news_id,
                        "client": (ctx or {}).get("client"),
                        "session_id": (ctx or {}).get("session_id"),
                        "input": {"url": url0, "text_len": len(t)},
                    }
                )
            except Exception:
                pass
            done = _wait_news_done(news_id, timeout_sec=6)
            if isinstance(done, dict) and done.get("status") == "done":
                summary = str(done.get("summary") or "").strip()
                return (summary + f"\n\n(news_id={news_id})").strip()
            return f"Sensei, 我已经把新闻交给分析员们处理了。稍后你可以再问我这个编号：{news_id}"

    # Heuristic: long pasted text with URL -> treat as news
    if len(t) >= 350 and ("\n" in t or _extract_first_url(t)):
        url = _extract_first_url(t)
        news_id = _enqueue_news_job(text=t, url=url, source="chat")
        try:
            _append_trajectory_log(
                {
                    "time": datetime.now().isoformat(),
                    "type": "contract.news.submit",
                    "news_id": news_id,
                    "client": (ctx or {}).get("client"),
                    "session_id": (ctx or {}).get("session_id"),
                    "input": {"url": url, "text_len": len(t)},
                }
            )
        except Exception:
            pass
        return f"Sensei, 我收到新闻了，已分发给分析员。编号：{news_id}（你可以直接把这个编号发给我获取总结）"

    # Deterministic rank queries to avoid keyword-grab / verbosity.
    if _is_profit_rank_question(t) or _is_loss_rank_question(t) or _is_biggest_position_question(t):
        return _live_rank_answer(t, ctx)

    if _is_task_dispatch_request(t):
        return _dispatch_task_answer(t, ctx)

    # Data source / market open questions must be answered from feed status (no LLM guessing)
    if _is_datasource_question(t):
        return _live_datasource_answer()

    # Portfolio state questions must be answered from live engine state (no LLM guessing)
    if _is_portfolio_question(t):
        return _live_portfolio_answer(t)

    # News questions without concrete input -> ask user to submit news to avoid fabrication
    if _is_news_question(t) and (not _extract_news_id(t)):
        return "Sensei, 如果要我让分析员评估‘新闻是否影响股价’，请您用 `/news` 把新闻正文或链接贴给我；否则我这边不会凭空编造结论呢…"

    # Handle memory commands first
    mem_cmd = parse_memory_command(t)
    if mem_cmd:
        memory = get_mari_memory()
        action = mem_cmd.get("action")
        
        if action == "remember":
            content = mem_cmd.get("content", "")
            memory.remember(content, category="instruction", importance=3)
            return f"Sensei, 我已经记住了: {content}"
        
        elif action == "forget":
            query = mem_cmd.get("query", "")
            memories = memory.recall(query=query, limit=1)
            if memories:
                memory.forget(memories[0].get("id", ""))
                return f"Sensei, 我已经忘记了关于'{query}'的记忆。"
            return f"Sensei, 我没有找到关于'{query}'的记忆。"
        
        elif action == "list":
            all_mems = memory.get_all_memories()
            if not all_mems:
                return "Sensei, 我还没有任何记忆呢。"
            lines = ["Sensei, 这是我记住的内容:"]
            for m in all_mems[-10:]:
                lines.append(f"- {m.get('content', '')}")
            return "\n".join(lines)

    is_status = tl.startswith("/status") or ("监控" in t) or ("在干什么" in t) or ("应该做什么" in t) or ("状态" in t)
    is_help = tl.startswith("/help") or ("怎么用" in t) or ("找不到" in t) or ("在哪" in t) or ("位置" in t) or ("向导" in t)

    if is_status:
        runs = _list_run_dirs()
        if not runs:
            return "当前没有可用的 results 运行结果。先在本地跑一次 pipeline/回测/执行，然后再让我汇总。"

        run_dir = runs[0]
        run_id = run_dir.name
        sys_dirs = _list_system_dirs(run_dir)
        systems = [p.name for p in sys_dirs]
        dates = _list_dates_for_run(run_dir)
        d0 = dates[-1] if dates else ""

        lines = []
        lines.append(f"监控摘要（最近一次 Run）：{run_id}")
        if systems:
            lines.append(f"系统：{', '.join(systems[:6])}{'...' if len(systems) > 6 else ''}")
        else:
            lines.append("系统：未发现 metrics.json 子目录")
        if dates:
            lines.append(f"可用日期：{len(dates)} 天（最近：{d0}）")
        else:
            lines.append("可用日期：未发现 decisions_*.json")

        lines.append("下一步建议：在控制塔顶部依次选择 Run / 日期 / 系统 / 标的，然后查看指标与图表；需要我解释某个信号就直接问。")
        lines.append("你也可以输入：/help 获取界面向导。")
        return "\n".join(lines)

    if is_help:
        return "\n".join(
            [
                "界面向导（控制塔）：",
                "1) 顶部第一行：API 地址、选择 Run（运行ID）。",
                "2) 第二行：选择日期、选择系统、OHLC 窗口天数、选择标的。",
                "3) 中间卡片：显示 Action/Execution/Macro 等关键结论。",
                "4) 下方图表：OHLC 与分析内容。",
                "如果你说‘找不到XX按钮/选项’，把你当前截图发我，我会告诉你点击路径。",
            ]
        )

    try:
        r = _planner_mari_reply(t, ctx)
        if isinstance(r, str) and r.strip():
            return r.strip()
    except Exception:
        pass

    # Fallback to shared-base LLM chat (A2 Architecture)
    # Planner strategy holds the shared model and can hot-swap to 'secretary' adapter.
    try:
        if _live_runner is not None and getattr(_live_runner, "strategy", None):
            # System prompt injection (context + personality)
            sys_p = _build_secretary_system_prompt(ctx)
            reply = _live_runner.strategy.generate_reply(t, system_prompt=sys_p)
            if reply:
                return reply
    except Exception:
        pass

    return "Sensei，我这边暂时没有新的可播报结论。"

def _build_secretary_system_prompt(ctx: Dict[str, Any]) -> str:
    """Build the system prompt for Secretary LLM from config and context"""
    cfg = _get_secretary_config()
    base_prompt = str((cfg.get("secretary") or {}).get("system_prompt") or "").strip()
    
    # Add context (time, user role, etc.)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    context_lines = [
        f"Current Time: {now_str}",
        f"User Role: {str((ctx or {}).get('user_role') or 'Sensei')}",
    ]
    
    # Add memory context if available
    try:
        mem = get_mari_memory()
        mem_ctx = mem.get_context_for_llm(limit=5)
        if mem_ctx:
            context_lines.append("\n[Memory]\n" + mem_ctx)
    except Exception:
        pass

    # Add Live Agent Logs (System Status) for situational awareness
    if _live_runner is not None:
        try:
            logs = getattr(_live_runner, "agent_logs", [])
            if logs:
                # Take last 15 logs for context to understand current system state
                tail = logs[-15:]
                log_lines = []
                for it in tail:
                    t = str((it or {}).get("time") or "")
                    m = str((it or {}).get("message") or "")
                    if t and m:
                        log_lines.append(f"[{t}] {m}")
                if log_lines:
                    context_lines.append("\n[System Live Logs]\n" + "\n".join(log_lines))
        except Exception:
            pass

    return base_prompt + "\n\n" + "\n".join(context_lines)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _list_run_dirs() -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    out: List[Path] = []
    for p in RESULTS_DIR.iterdir():
        if not p.is_dir():
            continue
        if (p / "metrics.json").exists() or (p / "engine_state.json").exists():
            out.append(p)
    out.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return out


def _list_system_dirs(run_dir: Path) -> List[Path]:
    out: List[Path] = []
    if not run_dir.exists():
        return out
    for p in run_dir.iterdir():
        if p.is_dir() and (p / "metrics.json").exists():
            out.append(p)
    out.sort(key=lambda x: x.name)
    return out


def _load_decisions_day(*, system_dir: Path, date_str: str) -> Optional[Dict[str, Any]]:
    for fp in sorted(system_dir.glob("decisions_*.json")):
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


def _list_dates_for_run(run_dir: Path) -> List[str]:
    dates: set[str] = set()
    for sys_dir in _list_system_dirs(run_dir):
        for fp in sorted(sys_dir.glob("decisions_*.json")):
            try:
                payload = _read_json(fp)
            except Exception:
                continue
            days = payload.get("days") if isinstance(payload, dict) else None
            if isinstance(days, dict):
                for k in days.keys():
                    if k:
                        dates.add(str(k))
    return sorted(list(dates))


def _read_ohlc_from_raw(*, ticker: str, date_str: str) -> Dict[str, Any]:
    raw_dir = DATA_DIR / "raw"
    fp = raw_dir / f"{str(ticker).upper()}.parquet"
    if not fp.exists():
        return {"available": False, "source": str(fp)}

    try:
        df = pd.read_parquet(fp)
    except Exception as e:
        return {"available": False, "source": str(fp), "error": str(e)}

    if df is None or df.empty:
        return {"available": False, "source": str(fp)}

    idx = df.index
    try:
        idx = pd.to_datetime(idx)
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
        return {"available": False, "source": str(fp), "error": f"bad date: {date_str}"}

    if d not in df.index:
        return {"available": False, "source": str(fp), "date": str(date_str)}

    row = df.loc[d]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    def pick(keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in row.index:
                v = _safe_float(row.get(k))
                if v is not None:
                    return v
        return None

    o = pick(["open", "Open", "o"])
    h = pick(["high", "High", "h"])
    l = pick(["low", "Low", "l"])
    c = pick(["close", "Close", "c"])
    v = pick(["volume", "Volume", "v"])
    return {
        "available": True,
        "source": str(fp),
        "date": str(date_str),
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
    }


def _read_ohlc_window_from_raw(*, ticker: str, date_str: str, lookback_days: int = 60) -> Dict[str, Any]:
    raw_dir = DATA_DIR / "raw"
    fp = raw_dir / f"{str(ticker).upper()}.parquet"
    if not fp.exists():
        return {"available": False, "source": str(fp), "rows": []}

    try:
        df = pd.read_parquet(fp)
    except Exception as e:
        return {"available": False, "source": str(fp), "error": str(e), "rows": []}

    if df is None or df.empty:
        return {"available": False, "source": str(fp), "rows": []}

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
        return {"available": False, "source": str(fp), "error": f"bad date: {date_str}", "rows": []}

    if d not in df.index:
        return {"available": False, "source": str(fp), "date": str(date_str), "rows": []}

    try:
        lookback_n = int(lookback_days)
    except Exception:
        lookback_n = 60
    if lookback_n <= 0:
        lookback_n = 60

    df2 = df.sort_index()
    pos = df2.index.get_loc(d)
    if isinstance(pos, slice):
        pos = pos.stop - 1
    start = max(0, int(pos) - int(lookback_n) + 1)
    win = df2.iloc[start : int(pos) + 1]

    def col_pick(col: str, alts: List[str]) -> Optional[str]:
        for c in [col] + list(alts):
            if c in win.columns:
                return c
        return None

    c_open = col_pick("open", ["Open", "o"])
    c_high = col_pick("high", ["High", "h"])
    c_low = col_pick("low", ["Low", "l"])
    c_close = col_pick("close", ["Close", "c"])
    c_vol = col_pick("volume", ["Volume", "v"])

    rows: List[Dict[str, Any]] = []
    for ts, r in win.iterrows():
        rows.append(
            {
                "date": ts.strftime("%Y-%m-%d"),
                "open": _safe_float(r.get(c_open)) if c_open else None,
                "high": _safe_float(r.get(c_high)) if c_high else None,
                "low": _safe_float(r.get(c_low)) if c_low else None,
                "close": _safe_float(r.get(c_close)) if c_close else None,
                "volume": _safe_float(r.get(c_vol)) if c_vol else None,
            }
        )

    return {
        "available": True,
        "source": str(fp),
        "date": str(date_str),
        "lookback_days": int(lookback_n),
        "rows": rows,
    }


def _read_daily_row(*, system_dir: Path, system: str, date_str: str, ticker: str) -> Optional[Dict[str, Any]]:
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


@app.get("/api/v1/runs")
async def list_runs():
    runs: List[Dict[str, Any]] = []
    for run_dir in _list_run_dirs():
        run_id = run_dir.name
        meta: Dict[str, Any] = {
            "run_id": run_id,
            "path": str(run_dir),
            "updated_at": float(run_dir.stat().st_mtime),
        }

        metrics_fp = run_dir / "metrics.json"
        if metrics_fp.exists():
            try:
                payload = _read_json(metrics_fp)
                if isinstance(payload, dict):
                    meta["protocol"] = payload.get("protocol")
                    meta["outputs"] = payload.get("outputs")
            except Exception:
                pass

        systems = []
        for sys_dir in _list_system_dirs(run_dir):
            systems.append(sys_dir.name)
        meta["systems"] = systems

        summaries: Dict[str, Any] = {}
        for sys_dir in _list_system_dirs(run_dir):
            sys_fp = sys_dir / "metrics.json"
            try:
                sp = _read_json(sys_fp)
            except Exception:
                continue
            if not isinstance(sp, dict):
                continue

            stats = (((sp.get("execution") or {}).get("stats")) if isinstance(sp.get("execution"), dict) else {})
            if not isinstance(stats, dict):
                stats = {}
            total = _safe_float(stats.get("total_orders"))
            missed = _safe_float(stats.get("missed_orders"))
            miss_rate = None
            if total is not None and total > 0 and missed is not None:
                miss_rate = float(missed) / float(total)

            summaries[sys_dir.name] = {
                "range": sp.get("range"),
                "trade_count": sp.get("trade_count"),
                "fees_total": sp.get("fees_total"),
                "exec_edge_total": sp.get("exec_edge_total"),
                "pnl_sum_net": (sp.get("pnl_sum_net") or {}).get("1") if isinstance(sp.get("pnl_sum_net"), dict) else None,
                "miss_rate": miss_rate,
                "execution": sp.get("execution"),
            }
        if summaries:
            meta["summary"] = summaries

        runs.append(meta)

    return {"runs": runs}


@app.get("/api/v1/runs/{run_id}/dates")
async def list_run_dates(run_id: str):
    run_dir = RESULTS_DIR / str(run_id)
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"run_id not found: {run_id}")
    return {"run_id": str(run_id), "dates": _list_dates_for_run(run_dir)}


@app.get("/api/v1/runs/{run_id}/tickers/{date_str}")
async def list_tickers_for_date(run_id: str, date_str: str):
    run_dir = RESULTS_DIR / str(run_id)
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"run_id not found: {run_id}")

    systems: Dict[str, List[str]] = {}
    all_tickers: set[str] = set()
    for sys_dir in _list_system_dirs(run_dir):
        day = _load_decisions_day(system_dir=sys_dir, date_str=str(date_str))
        if not isinstance(day, dict):
            continue
        items = day.get("items") if isinstance(day.get("items"), dict) else {}
        tickers = sorted([str(k).upper() for k in items.keys() if str(k).strip()])
        if tickers:
            systems[sys_dir.name] = tickers
            for t in tickers:
                all_tickers.add(t)

    if not systems:
        raise HTTPException(status_code=404, detail=f"no decisions for run_id={run_id} date={date_str}")

    return {
        "run_id": str(run_id),
        "date": str(date_str),
        "tickers": sorted(list(all_tickers)),
        "systems": systems,
    }


@app.get("/api/v1/snapshot/{run_id}/{system}/{date_str}/{ticker}")
async def snapshot(run_id: str, system: str, date_str: str, ticker: str, lookback_days: int = 60):
    run_dir = RESULTS_DIR / str(run_id)
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"run_id not found: {run_id}")

    system_dir = run_dir / str(system)
    if not system_dir.exists() or not system_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"system not found in run: {system}")

    day = _load_decisions_day(system_dir=system_dir, date_str=str(date_str))
    if not isinstance(day, dict):
        raise HTTPException(status_code=404, detail=f"date not found in decisions: {date_str}")

    items = day.get("items") if isinstance(day.get("items"), dict) else {}
    rec = items.get(str(ticker).upper()) or items.get(str(ticker))
    if not isinstance(rec, dict):
        raise HTTPException(status_code=404, detail=f"ticker not found in decisions: {ticker}")

    daily_row = _read_daily_row(system_dir=system_dir, system=str(system), date_str=str(date_str), ticker=str(ticker))
    ohlc = _read_ohlc_from_raw(ticker=str(ticker), date_str=str(date_str))
    ohlc_window = _read_ohlc_window_from_raw(ticker=str(ticker), date_str=str(date_str), lookback_days=int(lookback_days))

    sys_metrics_fp = system_dir / "metrics.json"
    sys_metrics = None
    if sys_metrics_fp.exists():
        try:
            sys_metrics = _read_json(sys_metrics_fp)
        except Exception:
            sys_metrics = None

    run_metrics_fp = run_dir / "metrics.json"
    run_metrics = None
    if run_metrics_fp.exists():
        try:
            run_metrics = _read_json(run_metrics_fp)
        except Exception:
            run_metrics = None

    return {
        "run_id": str(run_id),
        "system": str(system),
        "date": str(date_str),
        "ticker": str(ticker).upper(),
        "ohlc": ohlc,
        "ohlc_window": ohlc_window,
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
        "metrics": {
            "run": run_metrics,
            "system": sys_metrics,
        },
    }


# ========== Phase 3.4: Live Paper Trading API ==========

# Global reference to live trading runner (set by external script)
_live_runner = None

_markets_tiles_cache: Dict[str, Any] = {"ts": 0.0, "data": None}


def set_live_runner(runner):
    """Set the live trading runner for API access"""
    global _live_runner
    _live_runner = runner

    try:
        cfg = _get_secretary_config()
        trading_cfg = cfg.get("trading") if isinstance(cfg.get("trading"), dict) else {}
        desired_load_models = trading_cfg.get("load_models")
        if desired_load_models is None:
            desired_load_models = True
        desired_load_models = bool(desired_load_models)
        try:
            setattr(_live_runner, "load_models", desired_load_models)
        except Exception:
            pass

        try:
            stg = getattr(_live_runner, "strategy", None)
            if stg is not None and desired_load_models and (not bool(getattr(stg, "models_loaded", False))):
                fn = getattr(stg, "load_models", None)
                if callable(fn):
                    fn()
        except Exception:
            pass
    except Exception:
        pass

    try:
        cfg = _get_secretary_config()
        rl_cfg = cfg.get("rl") if isinstance(cfg.get("rl"), dict) else {}
        if bool(rl_cfg.get("auto_start", False)):
            rl_manager = getattr(_live_runner, "rl_manager", None)
            if rl_manager is not None:
                rl_manager.enabled = True
                _append_audit({
                    "time": datetime.now().isoformat(),
                    "type": "rl.auto_start",
                    "enabled": True,
                })
    except Exception as e:
        logger.debug(f"rl auto_start failed: {e}")


@app.get("/api/v1/live/status")
async def get_live_status():
    """Get live paper trading engine status"""
    if _live_runner is None:
        return {"active": False, "message": "No live trading session"}
    
    # Calculate total value and P&L
    cash = _live_runner.broker.cash
    positions = getattr(_live_runner.broker, "positions", {})
    total_value = cash
    
    for ticker, pos in positions.items():
        if hasattr(pos, "shares") and hasattr(pos, "avg_price"):
            # Get current price from price history
            current_price = pos.avg_price
            if hasattr(_live_runner, "price_history") and ticker in _live_runner.price_history:
                prices = _live_runner.price_history[ticker]
                if prices:
                    current_price = prices[-1].get("close", pos.avg_price)
            total_value += pos.shares * current_price
    
    initial_cash = getattr(_live_runner, "initial_cash", 500000.0)
    total_pnl = total_value - initial_cash

    currency = str(getattr(_live_runner, "currency", "USD") or "USD")

    feed = _live_feed_status()
    
    infer_mode = "REAL" if (bool(getattr(_live_runner, "load_models", False)) and bool(getattr(getattr(_live_runner, "strategy", None), "models_loaded", False))) else "HEURISTIC"

    return {
        "active": True,
        "tickers": _live_runner.strategy.tickers,
        "cash": cash,
        "initial_cash": initial_cash,
        "currency": currency,
        "total_value": total_value,
        "total_pnl": total_pnl,
        "positions": positions,
        "trade_count": len(_live_runner.trade_log),
        "mode": getattr(_live_runner, "trading_mode", "online"),

        "load_models": bool(getattr(_live_runner, "load_models", False)),
        "models_loaded": bool(getattr(getattr(_live_runner, "strategy", None), "models_loaded", False)),
        "models_error": str(getattr(getattr(_live_runner, "strategy", None), "models_error", "") or ""),
        "infer_mode": infer_mode,

        # Feed/source diagnostics
        "data_source": feed.get("data_source"),
        "last_bar_time": feed.get("last_bar_time"),
        "age_sec": feed.get("age_sec"),
        "stale": feed.get("stale"),
        "stale_reason": feed.get("stale_reason"),
    }


def _markets_tiles_get_cached(*, ttl_sec: float = 25.0) -> Dict[str, Any]:
    import time

    now = float(time.time())
    try:
        ts = float((_markets_tiles_cache or {}).get("ts") or 0.0)
    except Exception:
        ts = 0.0
    if (_markets_tiles_cache.get("data") is not None) and (now - ts) < ttl_sec:
        try:
            return dict(_markets_tiles_cache.get("data") or {})
        except Exception:
            return _markets_tiles_cache.get("data") or {}

    # Approximate Yahoo homepage tiles with curated symbol sets.
    lists: Dict[str, list[str]] = {
        "trending": ["NVDA", "TSLA", "AAPL", "MSFT", "AMD"],
        "futures": ["ES=F", "NQ=F", "YM=F", "RTY=F"],
        "currencies": ["EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X"],
        "crypto": ["BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "SOL-USD"],
        "etfs": ["SPY", "QQQ", "IWM", "DIA", "XLF"],
        # Movers are placeholders (true screeners require paid endpoints); we still show a few symbols.
        "stocks_gainers": ["NVDA", "AMD", "TSLA", "AAPL", "MSFT"],
        "stocks_losers": ["INTC", "PYPL", "NKE", "DIS", "BABA"],
        "stocks_most_actives": ["NVDA", "AAPL", "TSLA", "AMD", "INTC"],
    }

    def _quote_one(sym: str) -> Optional[Dict[str, Any]]:
        try:
            import yfinance as yf

            tk = yf.Ticker(sym)
            fi = getattr(tk, "fast_info", None)
            last = None
            prev = None
            if isinstance(fi, dict):
                last = fi.get("last_price")
                prev = fi.get("previous_close")
            if last is None or prev is None:
                info = getattr(tk, "info", None)
                if isinstance(info, dict):
                    if last is None:
                        last = info.get("regularMarketPrice")
                    if prev is None:
                        prev = info.get("regularMarketPreviousClose")
            last_f = float(last) if last is not None else None
            prev_f = float(prev) if prev is not None else None
            if last_f is None:
                return None
            chg = None
            pct = None
            if prev_f not in (None, 0.0):
                chg = last_f - prev_f
                pct = (chg / prev_f) * 100.0
            return {
                "symbol": str(sym),
                "last": last_f,
                "change": chg,
                "pct": pct,
            }
        except Exception:
            return None

    out: Dict[str, Any] = {"asof": datetime.now().isoformat(), "tiles": {}}
    tiles: Dict[str, Any] = {}
    for k, syms in lists.items():
        rows: list[Dict[str, Any]] = []
        for s in list(syms)[:10]:
            q = _quote_one(s)
            if q is not None:
                rows.append(q)
        tiles[k] = rows
    out["tiles"] = tiles

    _markets_tiles_cache["ts"] = now
    _markets_tiles_cache["data"] = out
    return out


@app.get("/api/v1/markets/tiles")
async def get_markets_tiles():
    try:
        return _markets_tiles_get_cached(ttl_sec=25.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tiles error: {e}")


class SetModeRequest(BaseModel):
    mode: str


class LiveRestartRequest(BaseModel):
    data_source: str = ""
    load_models: Optional[bool] = None


@app.post("/api/v1/live/reload_models")
async def reload_live_models():
    if _live_runner is None:
        raise HTTPException(status_code=404, detail="No live trading session")

    stg = getattr(_live_runner, "strategy", None)
    if stg is None:
        raise HTTPException(status_code=404, detail="No strategy")

    changed = False
    err = ""
    need_reload = False
    msg = ""
    try:
        p = REPO_ROOT / "data" / "finetune" / "evolution" / "active_trading_models.json"
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                s = str(obj.get("active_moe_scalper") or "").strip()
                a = str(obj.get("active_moe_analyst") or "").strip()
                if s:
                    try:
                        if str(getattr(stg, "moe_scalper", "") or "") != s:
                            setattr(stg, "moe_scalper", s)
                            changed = True
                    except Exception:
                        pass
                if a:
                    try:
                        if str(getattr(stg, "moe_analyst", "") or "") != a:
                            setattr(stg, "moe_analyst", a)
                            changed = True
                    except Exception:
                        pass

        try:
            need_reload = bool(changed) or (not bool(getattr(stg, "models_loaded", False)))
        except Exception:
            need_reload = True

        if need_reload:
            try:
                setattr(_live_runner, "load_models", True)
            except Exception:
                pass

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            acquired = False
            try:
                lk = getattr(stg, "_inference_lock", None)
                if lk is not None:
                    acquired = bool(lk.acquire(timeout=3.0))
            except Exception:
                acquired = False
            if not acquired:
                err = "inference_lock_busy"
                msg = "busy"
            else:
                try:
                    try:
                        setattr(stg, "model", None)
                    except Exception:
                        pass
                    try:
                        setattr(stg, "tokenizer", None)
                    except Exception:
                        pass
                    try:
                        setattr(stg, "models_loaded", False)
                    except Exception:
                        pass
                finally:
                    try:
                        lk.release()
                    except Exception:
                        pass

                fn = getattr(stg, "load_models", None)
                if callable(fn):
                    fn()
                msg = "reloaded"
        else:
            msg = "noop"

        try:
            logs = getattr(_live_runner, "agent_logs", None)
            if isinstance(logs, list):
                t_str = datetime.now().strftime("%H:%M:%S")
                logs.append(
                    {
                        "time": t_str,
                        "type": "agent",
                        "priority": 2,
                        "message": f"[ReloadModels] changed={changed} | scalper={getattr(stg, 'moe_scalper', '')} | analyst={getattr(stg, 'moe_analyst', '')}",
                    }
                )
        except Exception:
            pass
    except Exception as e:
        err = str(e)

    if err:
        return {
            "ok": False,
            "changed": bool(changed),
            "error": err,
            "moe_scalper": str(getattr(stg, "moe_scalper", "") or ""),
            "moe_analyst": str(getattr(stg, "moe_analyst", "") or ""),
            "models_loaded": bool(getattr(stg, "models_loaded", False)),
            "message": msg,
            "need_reload": bool(need_reload),
        }

    return {
        "ok": True,
        "changed": bool(changed),
        "moe_scalper": str(getattr(stg, "moe_scalper", "") or ""),
        "moe_analyst": str(getattr(stg, "moe_analyst", "") or ""),
        "models_loaded": bool(getattr(stg, "models_loaded", False)),
        "message": msg,
        "need_reload": bool(need_reload),
    }


@app.post("/api/v1/live/set_mode")
async def set_live_mode(req: SetModeRequest):
    """Switch between online and offline trading modes"""
    if _live_runner is None:
        raise HTTPException(status_code=404, detail="No live trading session")
    
    mode = req.mode.lower()
    if mode not in ("online", "offline"):
        raise HTTPException(status_code=400, detail="Mode must be 'online' or 'offline'")
    
    _live_runner.trading_mode = mode
    
    # If switching to offline, start backtest playback
    if mode == "offline":
        _live_runner.start_offline_playback()
    else:
        _live_runner.stop_offline_playback()
    
    return {"mode": mode, "message": f"Switched to {mode} mode"}


@app.post("/api/v1/live/restart")
async def restart_live(req: LiveRestartRequest):
    if _live_runner is None:
        raise HTTPException(status_code=404, detail="No live trading session")

    src = str(req.data_source or "").strip().lower()
    if src and src not in {"auto", "yfinance", "simulated"}:
        raise HTTPException(status_code=400, detail="data_source must be auto/yfinance/simulated")

    try:
        _live_runner.stop_offline_playback()
    except Exception:
        pass

    try:
        df = getattr(_live_runner, "data_feed", None)
        if df is not None:
            try:
                df.stop()
            except Exception:
                pass
    except Exception:
        pass

    prev_src = None
    try:
        prev_src = str(getattr(_live_runner, "data_source", "auto") or "auto")
    except Exception:
        prev_src = None

    if src:
        try:
            setattr(_live_runner, "data_source", src)
        except Exception:
            pass

    try:
        cur_src = str(getattr(_live_runner, "data_source", "auto") or "auto")
        if prev_src is not None and cur_src != prev_src:
            fn = getattr(_live_runner, "reset_price_history", None)
            if callable(fn):
                fn(None)
    except Exception:
        pass

    lm_req = req.load_models
    if lm_req is not None:
        try:
            setattr(_live_runner, "load_models", bool(lm_req))
        except Exception:
            pass

        try:
            stg = getattr(_live_runner, "strategy", None)
            if stg is not None:
                if bool(lm_req) and (not bool(getattr(stg, "models_loaded", False))):
                    fn = getattr(stg, "load_models", None)
                    if callable(fn):
                        fn()
                if (not bool(lm_req)) and bool(getattr(stg, "models_loaded", False)):
                    acquired = False
                    try:
                        lk = getattr(stg, "_inference_lock", None)
                        if lk is not None:
                            acquired = bool(lk.acquire(timeout=3.0))
                    except Exception:
                        acquired = False
                    if acquired:
                        try:
                            try:
                                setattr(stg, "model", None)
                            except Exception:
                                pass
                            try:
                                setattr(stg, "tokenizer", None)
                            except Exception:
                                pass
                            try:
                                setattr(stg, "models_loaded", False)
                            except Exception:
                                pass
                        finally:
                            try:
                                lk.release()
                            except Exception:
                                pass
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
        except Exception:
            pass

    # Hot-swap MoE adapters from pointer file (if present) and reload models
    try:
        stg = getattr(_live_runner, "strategy", None)
        if stg is not None:
            p = REPO_ROOT / "data" / "finetune" / "evolution" / "active_trading_models.json"
            if p.exists():
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    s = str(obj.get("active_moe_scalper") or "").strip()
                    a = str(obj.get("active_moe_analyst") or "").strip()
                    changed = False
                    if s:
                        try:
                            if str(getattr(stg, "moe_scalper", "") or "") != s:
                                setattr(stg, "moe_scalper", s)
                                changed = True
                        except Exception:
                            pass
                    if a:
                        try:
                            if str(getattr(stg, "moe_analyst", "") or "") != a:
                                setattr(stg, "moe_analyst", a)
                                changed = True
                        except Exception:
                            pass

                    if changed:
                        try:
                            setattr(_live_runner, "load_models", True)
                        except Exception:
                            pass

                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass

                        acquired = False
                        try:
                            lk = getattr(stg, "_inference_lock", None)
                            if lk is not None:
                                acquired = bool(lk.acquire(timeout=3.0))
                        except Exception:
                            acquired = False
                        if acquired:
                            try:
                                try:
                                    setattr(stg, "model", None)
                                except Exception:
                                    pass
                                try:
                                    setattr(stg, "tokenizer", None)
                                except Exception:
                                    pass
                                try:
                                    setattr(stg, "models_loaded", False)
                                except Exception:
                                    pass
                            finally:
                                try:
                                    lk.release()
                                except Exception:
                                    pass

                            fn = getattr(stg, "load_models", None)
                            if callable(fn):
                                fn()
                        else:
                            try:
                                logs = getattr(_live_runner, "agent_logs", None)
                                if isinstance(logs, list):
                                    import datetime as _dt
                                    t_str = _dt.datetime.now().strftime("%H:%M:%S")
                                    logs.append(
                                        {
                                            "time": t_str,
                                            "type": "agent",
                                            "priority": 2,
                                            "message": "[HotSwap] skipped reload (inference_lock_busy)",
                                        }
                                    )
                            except Exception:
                                pass

                    try:
                        logs = getattr(_live_runner, "agent_logs", None)
                        if isinstance(logs, list):
                            import datetime as _dt
                            t_str = _dt.datetime.now().strftime("%H:%M:%S")
                            logs.append(
                                {
                                    "time": t_str,
                                    "type": "agent",
                                    "priority": 2,
                                    "message": f"[HotSwap] scalper={getattr(stg, 'moe_scalper', '')} | analyst={getattr(stg, 'moe_analyst', '')}",
                                }
                            )
                    except Exception:
                        pass
    except Exception:
        pass

    try:
        stg = getattr(_live_runner, "strategy", None)
        if stg is not None and bool(getattr(_live_runner, "load_models", False)) and bool(getattr(stg, "models_loaded", False)):
            fn2 = getattr(stg, "_warmup_kv_cache", None)
            if callable(fn2):
                try:
                    logs = getattr(_live_runner, "agent_logs", None)
                    if isinstance(logs, list):
                        import datetime as _dt
                        t_str = _dt.datetime.now().strftime("%H:%M:%S")
                        logs.append({
                            "time": t_str,
                            "type": "agent",
                            "priority": 2,
                            "message": "[Warmup] start",
                        })
                except Exception:
                    pass

                fn2()

                try:
                    logs = getattr(_live_runner, "agent_logs", None)
                    if isinstance(logs, list):
                        import datetime as _dt
                        t_str = _dt.datetime.now().strftime("%H:%M:%S")
                        logs.append({
                            "time": t_str,
                            "type": "agent",
                            "priority": 2,
                            "message": "[Warmup] done",
                        })
                except Exception:
                    pass
    except Exception:
        pass

    try:
        from src.trading.data_feed import create_data_feed

        tickers = []
        try:
            tickers = list(getattr(_live_runner.strategy, "tickers", []) or [])
        except Exception:
            tickers = []

        new_src = str(getattr(_live_runner, "data_source", "auto") or "auto")
        interval_sec = 5.0
        try:
            interval_sec = float(getattr(_live_runner, "_data_feed_interval_sec", 5.0) or 5.0)
        except Exception:
            interval_sec = 5.0

        spt = 0
        try:
            spt = int(getattr(_live_runner, "_md_symbols_per_tick", 0) or 0)
        except Exception:
            spt = 0
        if spt <= 0:
            try:
                cfg = _get_secretary_config()
                trading_cfg = cfg.get("trading") if isinstance(cfg.get("trading"), dict) else {}
                spt = int(trading_cfg.get("md_symbols_per_tick") or 0)
            except Exception:
                spt = 0

        new_df = create_data_feed(tickers, source=new_src, interval_sec=float(interval_sec), symbols_per_tick=int(spt))
        actual_src = None
        try:
            actual_src = str(getattr(new_df, "source", "") or "").strip().lower() or None
        except Exception:
            actual_src = None
        try:
            new_df.subscribe(_live_runner._on_market_data)
        except Exception:
            pass
        new_df.start()
        try:
            setattr(_live_runner, "data_feed", new_df)
        except Exception:
            pass

        try:
            if actual_src:
                setattr(_live_runner, "data_source", actual_src)
        except Exception:
            pass

        if (actual_src or str(new_src)).lower() == "yfinance":
            try:
                fn = getattr(_live_runner, "backfill_intraday", None)
                if callable(fn):
                    ok = await run_in_threadpool(lambda: fn(max_bars=600))
                    try:
                        logs = getattr(_live_runner, "agent_logs", None)
                        if isinstance(logs, list):
                            import datetime as _dt
                            t_str = _dt.datetime.now().strftime("%H:%M:%S")
                            bars_n = 0
                            try:
                                ph = getattr(_live_runner, "price_history", {})
                                if isinstance(ph, dict):
                                    for _tk, _bars in ph.items():
                                        if isinstance(_bars, list):
                                            bars_n += len(_bars)
                            except Exception:
                                bars_n = -1
                            logs.append({
                                "time": t_str,
                                "type": "agent",
                                "priority": 2,
                                "message": f"[Backfill] yfinance intraday {'OK' if ok else 'FAIL'} | bars={bars_n}",
                            })
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"restart failed: {e}")

    try:
        st = await get_live_status()
        return {"ok": True, "status": st}
    except Exception:
        return {"ok": True}


@app.get("/api/v1/live/chart/{ticker}")
async def get_live_chart(ticker: str, limit: int = 100):
    """Get price history for live chart rendering"""
    if _live_runner is None:
        raise HTTPException(status_code=404, detail="No live trading session")
    
    ticker = ticker.upper()
    prices = _live_runner.get_chart_data(ticker)

    cleaned: list = []
    try:
        items = prices if isinstance(prices, list) else []
        tmp = []
        for p in items:
            if not isinstance(p, dict):
                continue
            t = p.get("time")
            if not t:
                continue
            try:
                c = float(p.get("close") or 0.0)
            except Exception:
                c = 0.0
            if c <= 0:
                continue
            tmp.append(p)
        tmp.sort(key=lambda x: str(x.get("time") or ""))
        seen = set()
        for p in reversed(tmp):
            ts = str(p.get("time") or "")
            if not ts or ts in seen:
                continue
            seen.add(ts)
            cleaned.append(p)
        cleaned.reverse()
    except Exception:
        cleaned = prices if isinstance(prices, list) else []

    if limit > 0:
        cleaned = cleaned[-limit:]
    
    return {
        "ticker": ticker,
        "prices": cleaned,
        "count": len(cleaned),
    }


@app.get("/api/v1/live/trades")
async def get_live_trades():
    """Get trade markers for chart overlay (buy/sell points)"""
    if _live_runner is None:
        raise HTTPException(status_code=404, detail="No live trading session")
    
    trades = _live_runner.get_trade_markers()
    
    # Format for chart markers
    markers = []
    for t in trades:
        markers.append({
            "time": t.get("time"),
            "ticker": t.get("ticker"),
            "action": t.get("action"),
            "price": t.get("price"),
            "shares": t.get("shares"),
            "marker_type": "buy" if t.get("action") == "BUY" else "sell",
        })
    
    return {"trades": markers, "count": len(markers)}


@app.get("/api/v1/live/chat")
async def get_live_chat(limit: int = 50):
    """Get Mari's chat history"""
    if _live_runner is None:
        raise HTTPException(status_code=404, detail="No live trading session")
    
    chat = _live_runner.mari.chat_log[-limit:] if _live_runner.mari.chat_log else []
    return {"messages": chat, "count": len(chat)}


@app.get("/api/v1/live/agent_logs")
async def get_agent_logs(limit: int = 100):
    """Get Multi-Agent system logs for terminal display"""
    if _live_runner is None:
        raise HTTPException(status_code=404, detail="No live trading session")
    
    logs = getattr(_live_runner, "agent_logs", [])
    return {"logs": logs[-limit:] if logs else [], "count": len(logs)}


@app.post("/api/v1/evolution/nightly/dry-run")
async def run_nightly_evolution_dry_run():
    """Run Ouroboros nightly_evolution in dry-run mode and return its output."""
    py = REPO_ROOT / "venv311" / "Scripts" / "python.exe"
    python_exe = str(py) if py.exists() else sys.executable

    cmd = [
        str(python_exe),
        "scripts/nightly_evolution.py",
        "--dry-run",
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        out = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=120,
        )
        return {
            "ok": int(out.returncode) == 0,
            "cmd": " ".join(cmd),
            "returncode": int(out.returncode),
            "stdout": str(out.stdout or ""),
            "stderr": str(out.stderr or ""),
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "cmd": " ".join(cmd),
            "returncode": -1,
            "stdout": "",
            "stderr": "timeout",
        }


# ========== Online RL API ==========

@app.post("/api/v1/live/rl/start")
async def start_online_rl():
    """Start online reinforcement learning"""
    if _live_runner is None:
        raise HTTPException(status_code=404, detail="No live trading session")
    
    rl_manager = getattr(_live_runner, "rl_manager", None)
    if rl_manager is None:
        raise HTTPException(status_code=500, detail="RL manager not initialized")

    try:
        with _EVOLUTION_TRAIN_LOCK:
            tp = _EVOLUTION_TRAIN_PROC
        if tp is not None and tp.poll() is None:
            raise HTTPException(status_code=409, detail="Ouroboros training is running")
    except HTTPException:
        raise
    except Exception:
        pass

    # Keep GPT-SoVITS and chat online during RL; RL should not interrupt voice/chat UX.
    global _SOVITS_RESUME_AFTER_RL, _SOVITS_RESTARTED_AFTER_RL
    with _SOVITS_LOCK:
        _SOVITS_RESUME_AFTER_RL = False
        _SOVITS_RESTARTED_AFTER_RL = False

    # Collection-only: enable experience logging, but do NOT force model loading
    # and do NOT enable online updates/training.

    try:
        setattr(rl_manager, "enabled", True)
    except Exception:
        pass

    try:
        setattr(rl_manager, "enable_updates", False)
    except Exception:
        pass

    try:
        setattr(rl_manager, "learning_rate", 0.001)
    except Exception:
        pass

    buf_size = 0
    try:
        b = getattr(rl_manager, "buffer", None)
        buf_size = len(b) if b is not None else 0
    except Exception:
        buf_size = 0

    metrics: Dict[str, Any] = {}
    try:
        m = rl_manager.get_metrics(window=100)
        if isinstance(m, dict):
            metrics = m
    except Exception:
        metrics = {}

    return {
        "status": "started",
        "enabled": True,
        "buffer_size": int(buf_size),
        "updates": int(getattr(rl_manager, "update_count", 0) or 0),
        "metrics": metrics,
        "voice_stopped": False,
    }


@app.post("/api/v1/live/rl/stop")
async def stop_online_rl():
    """Stop online reinforcement learning"""
    if _live_runner is None:
        raise HTTPException(status_code=404, detail="No live trading session")
    
    rl_manager = getattr(_live_runner, "rl_manager", None)
    if rl_manager:
        try:
            setattr(rl_manager, "enabled", False)
        except Exception:
            pass
        try:
            setattr(rl_manager, "enable_updates", False)
        except Exception:
            pass

    global _SOVITS_RESUME_AFTER_RL, _SOVITS_RESTARTED_AFTER_RL
    with _SOVITS_LOCK:
        if _SOVITS_RESUME_AFTER_RL and (not _SOVITS_RESTARTED_AFTER_RL):
            _start_gpt_sovits_from_state()
            _SOVITS_RESTARTED_AFTER_RL = True
            _SOVITS_RESUME_AFTER_RL = False

    return {"status": "stopped", "enabled": False}


@app.get("/api/v1/live/rl/status")
async def get_rl_status():
    """Get online RL status"""
    if _live_runner is None:
        return {"active": False}
    
    rl_manager = getattr(_live_runner, "rl_manager", None)
    if rl_manager is None:
        return {"active": False, "message": "RL manager not initialized"}

    buf_size = 0
    try:
        b = getattr(rl_manager, "buffer", None)
        buf_size = len(b) if b is not None else 0
    except Exception:
        buf_size = 0

    metrics: Dict[str, Any] = {}
    try:
        m = rl_manager.get_metrics(window=100)
        if isinstance(m, dict):
            metrics = m
    except Exception:
        metrics = {}

    return {
        "active": True,
        "enabled": bool(getattr(rl_manager, "enabled", False)),
        "buffer_size": int(buf_size),
        "updates": int(getattr(rl_manager, "update_count", 0) or 0),
        "metrics": metrics,
        "voice_stopped": bool(_SOVITS_RESUME_AFTER_RL),
    }


# ========== 启动 ==========

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """启动服务器"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()

"""
FastAPI主入口
"""

import json
import time
import threading
import uuid
import ipaddress
import socket
import urllib.request
import urllib.parse
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import re
import yaml
from openai import OpenAI

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from src.analysis.narrator import narrate_trade_context
from src.memory.mari_memory import get_mari_memory, parse_memory_command

app = FastAPI(
    title="QuantAI API",
    description="智能量化投顾助手API",
    version="0.1.0"
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"
SECRETARY_CONFIG_PATH = REPO_ROOT / "configs" / "secretary.yaml"

AGENT_HUB_DIR = DATA_DIR / "agent_hub"
AGENT_HUB_DIR.mkdir(parents=True, exist_ok=True)
AGENT_AUDIT_PATH = AGENT_HUB_DIR / "audit.jsonl"

_SECRETARY_CFG: Optional[Dict[str, Any]] = None
_SECRETARY_CFG_MTIME: Optional[float] = None

_AGENT_LOCK = threading.Lock()
_NEWS_JOBS: Dict[str, Dict[str, Any]] = {}

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


class NewsSubmitRequest(BaseModel):
    text: str
    url: Optional[str] = None
    source: Optional[str] = None


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


def _is_portfolio_question(text: str) -> bool:
    t = str(text or "")
    keys = [
        "持仓", "仓位", "现金", "资金", "余额", "多少钱", "总资产", "净值", "pnl", "盈亏",
        "赚", "亏", "交易", "成交", "买了", "卖了", "多少股", "持有",
        "我们现在", "目前",
    ]
    tl = t.lower()
    for k in keys:
        if k in t or k.lower() in tl:
            return True
    return False


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


def _call_llm_direct(*, system_prompt: str, user_text: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Optional[str]:
    cfg = _get_secretary_config()
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}

    mode = str((llm_cfg or {}).get("mode") or "api").strip().lower()
    try:
        t = float((llm_cfg or {}).get("temperature", 0.7)) if temperature is None else float(temperature)
    except Exception:
        t = 0.7
    try:
        mt = int((llm_cfg or {}).get("max_tokens", 256)) if max_tokens is None else int(max_tokens)
    except Exception:
        mt = 256

    if mode == "local":
        try:
            from src.llm.local_chat import chat as local_chat
            local_model = str((llm_cfg or {}).get("local_model") or "Qwen/Qwen3-8B")
            use_4bit = bool((llm_cfg or {}).get("use_4bit", False))
            use_8bit = bool((llm_cfg or {}).get("use_8bit", True))
            messages = [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(user_text)},
            ]
            resp = local_chat(
                messages,
                model_name=local_model,
                temperature=t,
                max_new_tokens=mt,
                use_4bit=use_4bit,
                use_8bit=use_8bit,
            )
            return resp.strip() if isinstance(resp, str) and resp.strip() else None
        except Exception as e:
            logger.error(f"[LLM] local direct error: {e}")
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
                page_txt = page_txt.strip()
                if page_txt:
                    base_news = base_news + "\n\n[URL_CONTENT]\n" + page_txt[:20_000]
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

        roles: Dict[str, str] = {
            "macro": "你是宏观分析员。任务：判断该新闻对宏观/行业/风险偏好的影响。输出JSON：{verdict: ok|risky|uncertain, impact: string, reasons: [string], tickers: [string], actions: [string]}。只输出JSON。",
            "risk": "你是风控分析员。任务：找出潜在风险、误读点、黑天鹅、确认信息缺口。输出JSON：{verdict: ok|risky|uncertain, risk_level: 1-5, reasons: [string], missing_info: [string], actions: [string]}。只输出JSON。",
            "factcheck": "你是事实核查分析员。任务：识别可疑点、需要验证的事实、可能的谣言/误传。输出JSON：{verdict: ok|risky|uncertain, suspicious: [string], verification_steps: [string], confidence: 0-1}。只输出JSON。",
            "sentiment": "你是情绪/舆情分析员。任务：判断市场情绪与可能的短期交易反应。输出JSON：{verdict: ok|risky|uncertain, sentiment: positive|neutral|negative, reasons: [string], actions: [string]}。只输出JSON。",
        }

        results: Dict[str, Any] = {}
        for role, sp in roles.items():
            raw = _call_llm_direct(system_prompt=sp, user_text=base_news, temperature=0.2, max_tokens=512)
            parsed: Any = None
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = {"raw": raw}
            else:
                parsed = {"error": "no_response"}
            results[role] = parsed
            _append_audit({
                "time": datetime.now().isoformat(),
                "type": "agent.result",
                "news_id": news_id,
                "agent": role,
                "result": parsed,
            })

        merge_prompt = (
            "你是交易秘书的总结官。请基于4个分析员的JSON输出，给Sensei一段总结：\n"
            "1) 结论(是否采纳/是否需要更多核查)\n"
            "2) 关键理由(3-6条)\n"
            "3) 对交易的可执行建议(0-5条)\n"
            "要求：明确、可核查、不要编造。"
        )
        merge_input = json.dumps(results, ensure_ascii=False)
        summary = _call_llm_direct(system_prompt=merge_prompt, user_text=merge_input, temperature=0.2, max_tokens=512) or ""

        try:
            mem = get_mari_memory()
            mem.remember(
                content=f"[News:{news_id}] {text}\n\n[Agents]\n{json.dumps(results, ensure_ascii=False)}\n\n[Summary]\n{summary}",
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
        rl_manager.enabled = True
        return {"ok": True, "enabled": True}

    if a in {"stop_rl", "rl_stop"}:
        if _live_runner is None:
            return {"ok": False, "error": "no live session"}
        rl_manager = getattr(_live_runner, "rl_manager", None)
        if rl_manager is None:
            return {"ok": False, "error": "rl manager not initialized"}
        rl_manager.enabled = False
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
    return ChatResponse(reply=_secretary_reply(text, ctx))


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


def _call_llm(*, text: str, ctx: Dict[str, Any]) -> Optional[str]:
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

    system_prompt = str((sec_cfg or {}).get("system_prompt") or "You are a helpful assistant.").strip()
    merged_ctx = _build_secretary_context(ctx)
    rag = _maybe_build_trade_rag(user_text=str(text), ctx=ctx)

    # Add Mari's long-term memory context
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
            use_4bit = bool((llm_cfg or {}).get("use_4bit", False))
            use_8bit = bool((llm_cfg or {}).get("use_8bit", True))
            
            quant = "8bit" if use_8bit else ("4bit" if use_4bit else "fp16")
            logger.info(f"[LLM] Local mode: {local_model} ({quant})")
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
                use_8bit=use_8bit,
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

    client = OpenAI(base_url=api_base, api_key=api_key)
    logger.info(f"[LLM] API mode: {api_base} model={model}")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(text)},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
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
        done = _wait_news_done(news_id, timeout_sec=25)
        if isinstance(done, dict) and done.get("status") == "done":
            summary = str(done.get("summary") or "").strip()
            return (summary + f"\n\n(news_id={news_id})").strip()
        return f"Sensei, 我已经把新闻交给分析员们处理了。稍后你可以再问我这个编号：{news_id}"

    # Heuristic: long pasted text with URL -> treat as news
    if len(t) >= 350 and ("\n" in t or _extract_first_url(t)):
        url = _extract_first_url(t)
        news_id = _enqueue_news_job(text=t, url=url, source="chat")
        return f"Sensei, 我收到新闻了，已分发给分析员。编号：{news_id}（你可以直接把这个编号发给我获取总结）"

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
        r = _call_llm(text=t, ctx=ctx)
        if isinstance(r, str) and r.strip():
            return r.strip()
    except Exception as e:
        logger.warning(f"secretary llm call failed: {e}")
        return (
            "我这边暂时连不上本地 LLM（OpenAI-compatible）服务。\n"
            f"错误：{e}\n"
            "请检查：\n"
            "1) LM Studio / vLLM / Ollama 是否在运行（例如 http://localhost:1234/v1）\n"
            "2) configs/secretary.yaml 里的 llm.api_base 和 llm.model 是否和服务端一致"
        )

    return (
        "本地 LLM 没有返回有效内容。\n"
        "请检查：configs/secretary.yaml 的 llm.api_base / llm.model 是否正确，"
        "以及本地 LLM 服务是否支持 /v1/chat/completions。"
    )


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


def set_live_runner(runner):
    """Set the live trading runner for API access"""
    global _live_runner
    _live_runner = runner

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
    
    return {
        "active": True,
        "tickers": _live_runner.strategy.tickers,
        "cash": cash,
        "total_value": total_value,
        "total_pnl": total_pnl,
        "positions": positions,
        "trade_count": len(_live_runner.trade_log),
        "mode": getattr(_live_runner, "trading_mode", "online"),
    }


class SetModeRequest(BaseModel):
    mode: str


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


@app.get("/api/v1/live/chart/{ticker}")
async def get_live_chart(ticker: str, limit: int = 100):
    """Get price history for live chart rendering"""
    if _live_runner is None:
        raise HTTPException(status_code=404, detail="No live trading session")
    
    ticker = ticker.upper()
    prices = _live_runner.get_chart_data(ticker)
    
    if limit > 0:
        prices = prices[-limit:]
    
    return {
        "ticker": ticker,
        "prices": prices,
        "count": len(prices),
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


# ========== Online RL API ==========

@app.post("/api/v1/live/rl/start")
async def start_online_rl():
    """Start online reinforcement learning"""
    if _live_runner is None:
        raise HTTPException(status_code=404, detail="No live trading session")
    
    rl_manager = getattr(_live_runner, "rl_manager", None)
    if rl_manager is None:
        raise HTTPException(status_code=500, detail="RL manager not initialized")
    
    # Enable online learning
    rl_manager.enabled = True
    rl_manager.learning_rate = 0.001
    
    return {
        "status": "started",
        "message": "Online RL enabled - learning from trades",
        "buffer_size": len(rl_manager.buffer) if hasattr(rl_manager, "buffer") else 0,
    }


@app.post("/api/v1/live/rl/stop")
async def stop_online_rl():
    """Stop online reinforcement learning"""
    if _live_runner is None:
        raise HTTPException(status_code=404, detail="No live trading session")
    
    rl_manager = getattr(_live_runner, "rl_manager", None)
    if rl_manager:
        rl_manager.enabled = False
    
    return {"status": "stopped", "message": "Online RL disabled"}


@app.get("/api/v1/live/rl/status")
async def get_rl_status():
    """Get online RL status"""
    if _live_runner is None:
        return {"active": False}
    
    rl_manager = getattr(_live_runner, "rl_manager", None)
    if rl_manager is None:
        return {"active": False, "message": "RL manager not initialized"}
    
    return {
        "active": True,
        "enabled": getattr(rl_manager, "enabled", False),
        "buffer_size": len(rl_manager.buffer) if hasattr(rl_manager, "buffer") else 0,
        "updates": getattr(rl_manager, "update_count", 0),
    }


# ========== 启动 ==========

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """启动服务器"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()

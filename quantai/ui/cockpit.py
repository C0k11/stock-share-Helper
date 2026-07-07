"""工作台后台分析守护：模型加载与 LLM 推理在守护线程完成，UI 只读状态。

为什么不在 st.fragment 定时器里跑推理：streamlit 同一会话同一时刻只执行一个
script run，fragment 里 60-90 秒的生成会把所有交互排队卡死（审查轮实锤）。
守护线程随 server 进程常驻（UI 用 st.cache_resource 拿单例）：打开工作台即
开始加载分析员（本地 GPU 模型或外接 API，按 config）并进入分析循环——
UI 的任何刷新/交互只读 `state` 字典，毫秒级返回。

`collect_advices` 是 UI 规则卡与守护线程共用的纯逻辑（fetch 注入：UI 给带
st.cache 的取数、守护线程给直拉），保证两边操作卡口径永远一致。
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Callable, Optional


def direct_fetch(symbol: str, period: str, interval: str):
    """守护线程用的直拉取数（与 UI _fetch_hist 同口径：纽约墙钟 naive 索引）。"""
    import yfinance as yf

    raw = yf.Ticker(symbol).history(period=period, interval=interval)
    raw.columns = [str(c).lower() for c in raw.columns]
    if getattr(raw.index, "tz", None) is not None:
        raw = raw.tz_convert("America/New_York").tz_localize(None)
    return raw


def bar_complete(symbol: str, last_ts) -> bool:
    """尾根日线 bar 是否已走完（未走完的半根 bar 绝不进日线因子）。

    索引为纽约墙钟 naive、机器在美东时区（诚实假设）。美股 bar 在其日期的
    16:10 后完整；加密（-USD，UTC 切日、时间戳=起始）在起始 +24h 后完整。
    """
    import pandas as pd

    if str(symbol).upper().endswith("-USD"):
        end = last_ts + pd.Timedelta(hours=24)
    else:
        end = last_ts.replace(hour=16, minute=10)
    return datetime.now() >= end.to_pydatetime()


def collect_advices(
    symbols: list[str],
    held_cost: dict[str, float],
    fetch: Callable,
    lang: str = "zh",
) -> list[dict]:
    """拉数据 → 日线五因子 + 盘中卖压 → 操作卡列表（持仓优先、|计分| 降序）。

    held_cost: {symbol: 成本均价}（键集合即持仓集合）。单标的失败静默跳过
    （数据源抖动不该放倒整个作战台）。
    """
    from datetime import date

    from quantai.agents.analyst import intraday_stats
    from quantai.agents.tactician import advise
    from quantai.distill.scenarios import build_indicator_brief

    advs: list[dict] = []
    for s in symbols:
        try:
            daily = fetch(s, "2y", "1d")
            if daily is None or daily.empty or "close" not in daily.columns:
                continue
            d = daily
            if not bar_complete(s, d.index[-1]):
                d = d.iloc[:-1]
            # 门槛 15 根（两周）而非 60：新上市持仓（如 SPCX，IPO 三周）也要有卡——
            # 算不出的因子在 build_indicator_brief 里诚实为 NaN，advise 自动跳过
            # 不计分，理由只引用真实存在的数字；盘中卖压层不受影响（1m 数据全量）。
            if len(d) < 15:
                continue
            vals = build_indicator_brief(d, s)[1]
            ist = None
            m1 = fetch(s, "1d", "1m")
            if m1 is not None and not m1.empty and m1.index[-1].date() == date.today():
                prev_close = float(d["close"].dropna().iloc[-1])
                avg_vol = float(d["volume"].dropna().tail(20).mean()) if "volume" in d.columns else 0.0
                ist = intraday_stats(m1, prev_close, avg_vol) or None
            advs.append(
                advise(s, vals, ist, held=s in held_cost, avg_cost=held_cost.get(s), lang=lang)
            )
        except Exception:  # noqa: BLE001 - 单标的失败不炸整批
            continue
    advs.sort(key=lambda a: (not a["held"], -abs(a["score"])))
    return advs


class CockpitDaemon:
    """常驻分析循环：加载分析员 → 每 interval 秒出一轮战术综述 + 新闻情绪入库。

    `state` 是唯一对外面（UI 直接渲染）：
    {status, model, out, ts, scored, usage, error, round}
    """

    def __init__(self, llm_factory: Callable, interval_sec: int = 300) -> None:
        self._llm_factory = llm_factory
        self.interval_sec = int(interval_sec)
        self.state: dict = {
            "status": "starting", "model": "", "out": "", "ts": 0.0,
            "scored": 0, "error": "", "round": 0,
        }
        self._symbols: list[str] = []
        self._held_cost: dict[str, float] = {}
        self._held_shares: dict[str, float] = {}
        self._lang = "zh"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._kick = threading.Event()  # 提前唤醒（语言切换等）：打断轮间休眠立即重出一轮
        self._thread: Optional[threading.Thread] = None

    def configure(
        self, symbols: list[str], held_cost: dict[str, float], lang: str = "zh",
        held_shares: Optional[dict[str, float]] = None,
    ) -> None:
        """UI 每次渲染时同步标的池与语言（线程安全）。语言变化立即唤醒重出一轮——
        否则用户切了语言还要盯着旧语言的综述最多 5 分钟。"""
        with self._lock:
            lang_changed = lang != self._lang
            self._symbols = list(symbols)
            self._held_cost = dict(held_cost)
            self._held_shares = dict(held_shares or {})
            self._lang = lang
        if lang_changed and self.state.get("round", 0) > 0:
            self._kick.set()

    def start(self) -> "CockpitDaemon":
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name="cockpit-analyst", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._kick.set()

    # ----------------------------------------------------------------- #
    def _loop(self) -> None:
        import time

        try:
            self.state["status"] = "loading"
            llm = self._llm_factory()
            self.state["model"] = str(getattr(llm, "model_name", "local"))
        except Exception as exc:  # noqa: BLE001 - 加载失败要让 UI 看见原因
            self.state.update(status="load_failed", error=f"{type(exc).__name__}: {exc}")
            return

        from quantai.agents.news_scorer import score_news
        from quantai.agents.reporting import _persist_scores
        from quantai.agents.tactician import build_tactical_brief, tactical_system_prompt
        from quantai.data.news import NewsFetcher

        while not self._stop.is_set():
            try:
                with self._lock:
                    symbols = list(self._symbols)
                    held_cost = dict(self._held_cost)
                    held_shares = dict(self._held_shares)
                    lang = self._lang
                if not symbols:
                    self.state["status"] = "waiting_symbols"
                    self._stop.wait(5)
                    continue
                self.state["status"] = "collecting"
                advs = collect_advices(symbols, held_cost, direct_fetch, lang=lang)
                if not advs:
                    self.state["status"] = "no_data"
                    self._stop.wait(self.interval_sec)
                    continue
                self.state["status"] = "scoring"
                news = NewsFetcher().fetch_all([a["symbol"] for a in advs[:6]], limit_per_symbol=5)
                scored = score_news(news, llm) if news else []
                if scored:
                    _persist_scores(scored, model=self.state["model"])
                # 期权对冲层（持仓首标的）：BS 引擎算好整句给 LLM 转述；
                # 无期权链/抓取失败静默跳过，绝不影响主综述
                hedges: list[str] = []
                try:
                    held_syms = [a["symbol"] for a in advs if a.get("held")]
                    if held_syms:
                        from quantai.agents.tactician import hedge_lines
                        from quantai.analysis.options import (
                            chain_stats, covered_call_plan, protective_put_plan,
                        )
                        from quantai.data.options_chain import OptionChainFetcher

                        hsym = held_syms[0]
                        ch = OptionChainFetcher().fetch(hsym)
                        spot = next((a.get("last") for a in advs if a["symbol"] == hsym), None)
                        if ch and spot:
                            sh = float(held_shares.get(hsym, 0))
                            stats = chain_stats(ch["calls"], ch["puts"], spot)
                            pp = protective_put_plan(sh, spot, ch["puts"], ch["days_to_expiry"])
                            cc = covered_call_plan(sh, spot, ch["calls"], ch["days_to_expiry"])
                            hedges = hedge_lines(pp, cc, stats, ch["expiry"], shares=sh, lang=lang)
                except Exception:  # noqa: BLE001
                    hedges = []

                # Polymarket 事件概率（公开免 key；失败静默跳过，不影响主综述）
                events: list[str] = []
                try:
                    from quantai.config import load_config
                    from quantai.data.events import EventsFetcher

                    tags = load_config().data.event_tags
                    if tags:
                        odds = EventsFetcher().fetch_all(tags, limit_per_tag=10)
                        top = sorted(odds, key=lambda o: -(o.volume_24h or 0))[:5]
                        events = [
                            f"[{o.category}] {o.question} - Yes {o.yes_price * 100:.0f}%"
                            for o in top
                        ]
                except Exception:  # noqa: BLE001
                    events = []

                self.state["status"] = "summarizing"
                brief = build_tactical_brief(
                    advs, scored, as_of=datetime.now().strftime("%H:%M"),
                    hedges=hedges, events=events,
                )
                out = llm.generate(brief, system=tactical_system_prompt(lang))
                self.state.update(
                    out=out, out_lang=lang, ts=datetime.now().timestamp(), scored=len(scored),
                    error="", status="running", round=self.state["round"] + 1,
                )
            except Exception as exc:  # noqa: BLE001 - 单轮失败下一轮重试，但要留痕
                self.state.update(error=f"{type(exc).__name__}: {exc}", status="round_failed")
            self._kick.clear()
            self._kick.wait(self.interval_sec)  # 等满一轮，或被语言切换/stop 提前唤醒

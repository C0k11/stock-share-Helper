"""多页共享件：格式化/时间档/驻留 LLM 单例/全局 chrome（纯平移自 streamlit_app.py）。"""
from __future__ import annotations

from typing import Any

#: 时间档 -> (yfinance period, interval)。1D/1W 是盘中分钟级，3M 起为日线/周线。
TIMEFRAMES = {
    "1D": ("1d", "1m"),
    "1W": ("5d", "15m"),
    "1M": ("1mo", "1h"),
    "3M": ("3mo", "1d"),
    "1Y": ("1y", "1d"),
    "5Y": ("5y", "1wk"),
}


def fmt_money(x: Any) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)


def fmt_pct(x: Any) -> str:
    try:
        v = float(x)
        return f"{v * 100:+.2f}%" if v == v else "n/a"
    except Exception:
        return "n/a"


def _load_llm_impl():
    """驻留 LLM 单例底层（模块级唯一函数 -> st.cache_resource 全站共享一份模型）。"""
    from quantai.agents.reporting import load_report_llm

    return load_report_llm()


def resident_llm(st):
    """工作台/AI 分析页共用的驻留模型（学生 adapter，若已挂载）。"""
    return st.cache_resource(show_spinner="首次加载本地 LLM（Qwen3-8B + 学生 adapter，约 40 秒）…")(
        _load_llm_impl
    )()


def inject_chrome(st) -> None:
    """全局視覺微调：藏 streamlit 默认头/脚、收紧上边距（券商终端式信息密度）。"""
    st.markdown(
        """<style>
  header[data-testid="stHeader"] {background: transparent; height: 0.8rem;}
  .block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1500px;}
  div[data-testid="stMetricValue"] {font-size: 1.35rem;}
  #MainMenu, footer, div[data-testid="stToolbar"] {visibility: hidden;}
</style>""",
        unsafe_allow_html=True,
    )

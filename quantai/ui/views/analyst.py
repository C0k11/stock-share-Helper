"""AI 分析页：AI 怎么读今天的市场？（LLM 日报生成与浏览，纯平移自 streamlit_app.py）"""
from __future__ import annotations

from quantai.ui.common import resident_llm as _resident_llm


def render() -> None:  # pragma: no cover - 需 streamlit 运行时
    from pathlib import Path

    import streamlit as st

    from quantai.ui.i18n import tr

    lang = st.session_state.get("ui_lang", "zh")
    st.title(tr(lang, "ai.title"))
    st.markdown(tr(lang, "ai.desc"))

    use_llm = st.toggle(tr(lang, "ai.toggle"), value=False)
    llm = _resident_llm(st) if use_llm else None

    from quantai.agents.reporting import make_daily_report

    if st.button(tr(lang, "ai.gen_llm" if use_llm else "ai.gen_plain")):
        with st.spinner(tr(lang, "ai.working")):
            make_daily_report(llm, log=lambda m: None)
        st.rerun()

    # 仓库根锚定（与 app.py 的 icon 同套写法）：根外启动 streamlit 也能列出报告
    reports_dir = Path(__file__).resolve().parents[3] / "data" / "reports"
    # 按修改时间倒序：新报告永远排第一个（旧版按文件名排，刚生成的盘中报告会被埋进列表）
    reports = sorted(reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if reports:
        pick = st.selectbox(tr(lang, "ai.history"), [p.name for p in reports])
        st.markdown((reports_dir / pick).read_text(encoding="utf-8"))
    else:
        st.caption(tr(lang, "ai.none"))

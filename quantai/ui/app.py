"""QuantAI 仪表盘入口：st.navigation 五页装配（一页一个问题）。

页面本体在 `quantai/ui/views/`（每页一个模块，从旧单文件纯平移）；本文件只做
set_page_config / 全局 CSS / 语言选择 / 导航装配。旧入口 `streamlit_app.py`
保留为薄壳（README/启动脚本的 `streamlit run` 命令不变）。
"""
from __future__ import annotations


def main() -> None:  # pragma: no cover - 需 streamlit 运行时
    import os
    from pathlib import Path

    import streamlit as st

    from quantai.ui.common import inject_chrome
    from quantai.ui.i18n import LANGS, tr
    from quantai.ui.views import analyst, portfolio, sim, watchlist, workstation

    st.set_page_config(page_title="QuantAI Dashboard", layout="wide")
    inject_chrome(st)

    lang_label = st.sidebar.radio(
        "语言 / Language", list(LANGS), index=0, horizontal=True, key="ui_lang_label"
    )
    lang = LANGS[lang_label]
    st.session_state["ui_lang"] = lang

    # Tableau 一键打开（找 tableau/ 下的 .twb/.twbx；没有就开导出目录）
    tb_files = list(Path("tableau").glob("*.twb*"))
    label = (tr(lang, "nav.tableau_open", name=tb_files[0].name) if tb_files
             else tr(lang, "nav.tableau_dir"))
    if st.sidebar.button(label):
        os.startfile(str(tb_files[0]) if tb_files else str(Path("tableau/exports").resolve()))

    pages = [
        st.Page(workstation.render, title=tr(lang, "nav.workstation"), url_path="workstation", default=True),
        st.Page(portfolio.render, title=tr(lang, "nav.portfolio"), url_path="portfolio"),
        st.Page(watchlist.render, title=tr(lang, "nav.watchlist"), url_path="watchlist"),
        st.Page(analyst.render, title=tr(lang, "nav.analyst"), url_path="analyst"),
        st.Page(sim.render, title=tr(lang, "nav.sim"), url_path="sim"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()

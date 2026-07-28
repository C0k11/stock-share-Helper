"""Streamlit 仪表盘入口薄壳（兼容旧命令行）。

真正的入口在 `quantai/ui/app.py`（st.navigation 五页装配），页面本体在
`quantai/ui/views/`。保留本文件是为了 README / 启动脚本里的
    venv311\\Scripts\\python.exe -m streamlit run quantai/ui/streamlit_app.py
命令不变；`streamlit run quantai/ui/app.py` 等价。
"""
from __future__ import annotations

from quantai.ui.app import main

if __name__ == "__main__":
    main()

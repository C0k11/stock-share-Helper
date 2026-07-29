"""跨平台一键启动仪表盘：`quantai-dashboard`（等价于 streamlit run quantai/ui/app.py）。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    app = Path(__file__).with_name("app.py")
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]
    )

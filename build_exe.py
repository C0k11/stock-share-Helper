#!/usr/bin/env python
"""
Build standalone EXE launcher for AI Trading Terminal
使用 PyInstaller 打包一键启动程序

Usage:
    pip install pyinstaller
    python build_exe.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def build():
    print("Building AI Trading Terminal Launcher...")
    
    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name", "AI_Trading_Terminal",
        "--icon", "NONE",  # Add icon path if you have one
        "--add-data", f"scripts;scripts",
        "--add-data", f"configs;configs",
        str(PROJECT_ROOT / "launch.py"),
    ]
    
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    print("\n" + "=" * 60)
    print("Build complete!")
    print(f"EXE location: {PROJECT_ROOT / 'dist' / 'AI_Trading_Terminal.exe'}")
    print("=" * 60)


if __name__ == "__main__":
    build()

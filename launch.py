#!/usr/bin/env python
"""
AI Trading Terminal - One-Click Launcher
一键启动量化交易终端

Double-click this file or run:
    python launch.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent

def main():
    print("=" * 60)
    print("  AI Trading Terminal - Starting...")
    print("=" * 60)
    
    # Change to project directory
    os.chdir(PROJECT_ROOT)
    
    # Add to path
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    # Run the PowerShell launcher
    ps_script = PROJECT_ROOT / "scripts" / "launch_desktop.ps1"
    
    if ps_script.exists():
        print("\nLaunching via PowerShell script...")
        subprocess.Popen(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps_script)],
            cwd=str(PROJECT_ROOT),
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        # Fallback: direct Python launch
        print("\nLaunching directly...")
        
        # Start API + Trading Engine
        trading_script = PROJECT_ROOT / "scripts" / "run_live_paper_trading.py"
        subprocess.Popen(
            [sys.executable, str(trading_script), "--with-api"],
            cwd=str(PROJECT_ROOT),
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        
        time.sleep(3)
        
        # Start Desktop UI
        desktop_script = PROJECT_ROOT / "src" / "ui" / "desktop" / "main.py"
        subprocess.Popen(
            [sys.executable, str(desktop_script)],
            cwd=str(PROJECT_ROOT),
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    
    print("\nTerminal launched! You can close this window.")


if __name__ == "__main__":
    main()

@echo off
chcp 65001 >nul
title AI Trading Terminal
cd /d "%~dp0"
echo ============================================================
echo   AI Trading Terminal - 启动中...
echo ============================================================
powershell -ExecutionPolicy Bypass -File "scripts\launch_desktop.ps1"
pause

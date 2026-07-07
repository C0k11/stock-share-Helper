@echo off
rem QuantAI 仪表盘一键启动：先拉起 v2 API（实盘会话页用，:8000，纸面交易零真钱），
rem 再起 Streamlit（浏览器自动打开 http://localhost:8501）。API 日志在 logs\api.log
cd /d "%~dp0"
netstat -ano | findstr "LISTENING" | findstr ":8000" >nul
if errorlevel 1 (
    echo [dashboard] starting QuantAI API on :8000 ...
    start "QuantAI API" /min cmd /c "venv311\Scripts\python.exe scripts\serve.py --host 127.0.0.1 >> logs\api.log 2>&1"
)
venv311\Scripts\python.exe -m streamlit run quantai\ui\streamlit_app.py
pause

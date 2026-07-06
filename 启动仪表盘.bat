@echo off
rem QuantAI 仪表盘一键启动（Streamlit，浏览器自动打开 http://localhost:8501）
cd /d "%~dp0"
venv311\Scripts\python.exe -m streamlit run quantai\ui\streamlit_app.py
pause

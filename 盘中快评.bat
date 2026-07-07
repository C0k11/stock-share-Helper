@echo off
rem QuantAI 盘中快评：当日 1 分钟会话 + 卖压指标 + LLM 快评 + 新闻情绪入库
rem 由任务计划在开市时段每 30 分钟触发（也可手动双击）。日志 logs\intraday.log
cd /d "%~dp0"
echo ==== %date% %time% ==== >> logs\intraday.log
venv311\Scripts\python.exe scripts\report.py --intraday --llm >> logs\intraday.log 2>&1

@echo off
rem QuantAI 盘中快评（纯数据档案）：当日 1 分钟会话 + 卖压代理指标，30 分钟一份入档。
rem 不带 --llm：盘中的 LLM 实时分析/新闻打分由行情工作台「实时作战台」驻留模型承担——
rem 定时任务再各自加载 10GB 模型会和驻留模型撞显存（24GB 卡放不下两份 + 桌面占用）。
rem 日志 logs\intraday.log
cd /d "%~dp0"
echo ==== %date% %time% ==== >> logs\intraday.log
venv311\Scripts\python.exe scripts\report.py --intraday >> logs\intraday.log 2>&1

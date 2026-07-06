@echo off
rem QuantAI 每日数据刷新：行情+新闻 -> DuckDB -> dbt -> Tableau CSV + 数据简报
rem 由 Windows 任务计划每日自动执行（也可手动双击）。日志在 logs\daily_refresh.log
cd /d "%~dp0"
echo ==== %date% %time% ==== >> logs\daily_refresh.log
venv311\Scripts\python.exe scripts\warehouse.py --full >> logs\daily_refresh.log 2>&1
venv311\Scripts\python.exe scripts\report.py >> logs\daily_refresh.log 2>&1

@echo off
rem QuantAI 每日数据刷新：行情+新闻 -> DuckDB -> dbt -> Tableau CSV + 数据简报
rem 由 Windows 任务计划每日自动执行（也可手动双击）。日志在 logs\daily_refresh.log
rem 任一步失败立即以非零码退出——否则 warehouse 挂了、report 正常退出，
rem Task Scheduler 的 Last Result 会把整体伪装成成功（审查实锤）。
cd /d "%~dp0"
echo ==== %date% %time% ==== >> logs\daily_refresh.log
venv311\Scripts\python.exe scripts\warehouse.py --full >> logs\daily_refresh.log 2>&1
if errorlevel 1 (
    echo [FAIL] warehouse step failed >> logs\daily_refresh.log
    exit /b 1
)
venv311\Scripts\python.exe scripts\report.py >> logs\daily_refresh.log 2>&1
if errorlevel 1 (
    echo [FAIL] report step failed >> logs\daily_refresh.log
    exit /b 1
)

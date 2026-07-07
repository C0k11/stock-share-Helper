@echo off
rem QuantAI 每日决策日志：行情快照 -> 规则学生判断 -> DeepSeek 教师作答+打分 -> 训练资产
rem 由 Windows 任务计划工作日 17:30 自动执行（也可手动双击）。日志在 logs\journal.log
rem 幂等：同一交易日已有产物直接跳过（周末/节假日照跑也零花费）；先回填已成熟的结局收益。
cd /d "%~dp0"
echo ==== %date% %time% ==== >> logs\journal.log
venv311\Scripts\python.exe scripts\journal.py --backfill-outcomes --run --confirm-spend --workers 6 >> logs\journal.log 2>&1

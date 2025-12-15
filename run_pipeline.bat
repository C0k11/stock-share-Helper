@echo off
setlocal enabledelayedexpansion

set VENV=.\venv311\Scripts\python.exe

for /f %%i in ('%VENV% -c "import datetime as dt; print(dt.date.today().strftime(\"%%Y-%%m-%%d\"))"') do set TODAY=%%i

echo [1/3] Fetching RSS for %TODAY%...
%VENV% scripts\fetch_daily_rss.py --date %TODAY%
if errorlevel 1 goto :error

echo [2/3] Running Inference for %TODAY%...
%VENV% scripts\run_daily_inference.py ^
  --date %TODAY% ^
  --use-lora --load-in-4bit ^
  --batch-size 4 ^
  --max-input-chars 6000 ^
  --save-every 20
if errorlevel 1 goto :error

echo [3/3] Generating Report for %TODAY%...
%VENV% scripts\generate_daily_report.py --date %TODAY%
if errorlevel 1 goto :error

echo [Done] Pipeline finished for %TODAY%
pause
exit /b 0

:error
echo [Error] Pipeline failed for %TODAY% (exit code %errorlevel%)
pause
exit /b %errorlevel%

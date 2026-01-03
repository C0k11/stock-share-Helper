param (
    [switch]$NoVoice
    ,[switch]$NoTrading
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  AI Trading Terminal - Full Stack Launcher" -ForegroundColor Cyan
Write-Host "  GPT-SoVITS + Paper Trading Engine + Desktop App" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Ensure we're at repo root when running
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$processes = @()

# 1) Start GPT-SoVITS (Mari's Voice) - default ON
$sovitsProcess = $null
if (-not $NoVoice) {
    Write-Host "[1/3] Starting GPT-SoVITS (Mari Voice)..." -ForegroundColor Yellow
    $sovitsRoot = "D:\Project\ml_cache\GPT-SoVITS"
    $sovitsPy = "D:\Project\ml_cache\venvs\gpt_sovits\Scripts\python.exe"
    
    if (Test-Path $sovitsPy) {
        $sovitsArgs = @(
            "api_v2.py",
            "-a", "127.0.0.1",
            "-p", "9880"
        )
        $sovitsProcess = Start-Process -FilePath $sovitsPy -WorkingDirectory $sovitsRoot -ArgumentList $sovitsArgs -PassThru -WindowStyle Hidden
        $processes += $sovitsProcess
        Write-Host "      -> GPT-SoVITS starting on port 9880" -ForegroundColor Gray
        Start-Sleep -Seconds 3
    } else {
        Write-Host "      -> GPT-SoVITS not found, skipping voice" -ForegroundColor DarkYellow
    }
} else {
    Write-Host "[1/3] GPT-SoVITS skipped (--NoVoice)" -ForegroundColor DarkGray
}

# 2) Start Paper Trading Engine + API
$tradingProcess = $null
if (-not $NoTrading) {
    Write-Host "[2/3] Starting Paper Trading Engine + API..." -ForegroundColor Yellow
    $tradingArgs = @(
        "scripts/run_live_paper_trading.py",
        "--data-source", "simulated",
        "--with-api",
        "--api-port", "8000"
    )
    $tradingProcess = Start-Process -FilePath ".\venv311\Scripts\python.exe" -ArgumentList $tradingArgs -PassThru -WindowStyle Hidden
    $processes += $tradingProcess
    Write-Host "      -> Trading Engine + API starting on port 8000" -ForegroundColor Gray
    Start-Sleep -Seconds 3
} else {
    Write-Host "[2/3] Paper Trading skipped (--NoTrading)" -ForegroundColor DarkGray
}

# 3) Start Desktop UI (foreground, blocking)
Write-Host "[3/3] Starting Desktop UI..." -ForegroundColor Green
Write-Host "      -> Mari is waking up..." -ForegroundColor Gray
Write-Host ""

try {
    & ".\venv311\Scripts\python.exe" "src/ui/desktop/main.py"
} finally {
    Write-Host ""
    Write-Host "Shutting down all services..." -ForegroundColor Red
    foreach ($proc in $processes) {
        if ($proc -and -not $proc.HasExited) {
            try {
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            } catch {}
        }
    }
    Write-Host "All services stopped." -ForegroundColor Gray
}

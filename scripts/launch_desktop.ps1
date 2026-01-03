param (
    [switch]$NoVoice
    ,[switch]$NoTrading
    ,[switch]$NoLLM
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  AI 玛丽交易秘书 - Full Stack Launcher" -ForegroundColor Cyan
Write-Host "  Mari Voice + Paper Trading Engine + Desktop App" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Ensure we're at repo root when running
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$processes = @()

# 0) Start Ollama LLM Server
if (-not $NoLLM) {
    Write-Host "[0/4] Starting Ollama LLM Server..." -ForegroundColor Yellow
    
    # Check if Ollama is already running
    $ollamaRunning = Get-Process -Name "ollama*" -ErrorAction SilentlyContinue
    if ($ollamaRunning) {
        Write-Host "      -> Ollama already running" -ForegroundColor Gray
    } else {
        # Start Ollama serve
        $ollamaPath = "ollama"
        try {
            $ollamaProcess = Start-Process -FilePath $ollamaPath -ArgumentList "serve" -PassThru -WindowStyle Hidden
            $processes += $ollamaProcess
            Write-Host "      -> Ollama starting on port 11434" -ForegroundColor Gray
            Start-Sleep -Seconds 3
        } catch {
            Write-Host "      -> Ollama not found in PATH, trying default location..." -ForegroundColor DarkYellow
            $ollamaDefault = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
            if (Test-Path $ollamaDefault) {
                $ollamaProcess = Start-Process -FilePath $ollamaDefault -ArgumentList "serve" -PassThru -WindowStyle Hidden
                $processes += $ollamaProcess
                Write-Host "      -> Ollama starting on port 11434" -ForegroundColor Gray
                Start-Sleep -Seconds 3
            } else {
                Write-Host "      -> Ollama not found, Mari chat will not work" -ForegroundColor Red
            }
        }
    }
} else {
    Write-Host "[0/4] Ollama skipped (--NoLLM)" -ForegroundColor DarkGray
}

# 1) Start GPT-SoVITS (Mari's Voice) - default ON
$sovitsProcess = $null
if (-not $NoVoice) {
    Write-Host "[1/4] Starting GPT-SoVITS (Mari Voice)..." -ForegroundColor Yellow
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
    Write-Host "[1/4] GPT-SoVITS skipped (--NoVoice)" -ForegroundColor DarkGray
}

# 2) Start Paper Trading Engine + API
$tradingProcess = $null
if (-not $NoTrading) {
    Write-Host "[2/4] Starting Paper Trading Engine + API..." -ForegroundColor Yellow
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
    Write-Host "[2/4] Paper Trading skipped (--NoTrading)" -ForegroundColor DarkGray
}

# 3) Start Desktop UI (foreground, blocking)
Write-Host "[3/4] Starting Desktop UI..." -ForegroundColor Green
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

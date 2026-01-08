param (
    [switch]$NoVoice
    ,[switch]$NoTrading
    ,[switch]$NoLLM
    ,[switch]$ForceRestart
    ,[ValidateSet("simulated","yfinance","auto")] [string]$DataSource = "simulated"
    ,[switch]$NoModels
    ,[double]$SecretaryVramFrac = 0.55
    ,[double]$TradingVramFrac = 0.45
    ,[string]$ApiHost = "127.0.0.1"
    ,[int]$ApiPort = 8000
    ,[string]$SovitsHost = "127.0.0.1"
    ,[int]$SovitsPort = 9880
)

try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
}

function Show-LogTail {
    param(
        [string]$Path,
        [int]$Lines = 120
    )
    try {
        if ($Path -and (Test-Path $Path)) {
            Write-Host "------ tail: $Path ------" -ForegroundColor DarkGray
            Get-Content -Path $Path -Tail $Lines -ErrorAction SilentlyContinue
        }
    } catch {
    }
}

try {
    chcp 65001 > $null
} catch {
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  AI 玛丽交易秘书 - Full Stack Launcher" -ForegroundColor Cyan
Write-Host "  Mari Voice + Paper Trading Engine + Desktop App" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Ensure we're at repo root when running
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$processes = @()

$py311 = Join-Path $repo "venv311\Scripts\python.exe"
if (-not (Test-Path $py311)) {
    Write-Host "[Error] python not found: $py311" -ForegroundColor Red
    Write-Host "Please create venv311 or fix the path in scripts\\launch_desktop.ps1" -ForegroundColor Red
    exit 1
}

$logDir = Join-Path $repo "logs"
try {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
} catch {
}

function Test-TcpPort {
    param(
        [string]$TargetHost,
        [int]$Port
    )
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar = $client.BeginConnect($TargetHost, $Port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(600)
        if (-not $ok) {
            try { $client.Close() } catch {}
            return $false
        }
        $client.EndConnect($iar)
        try { $client.Close() } catch {}
        return $true
    } catch {
        return $false
    }
}

function Wait-TcpPort {
    param(
        [string]$TargetHost,
        [int]$Port,
        [int]$TimeoutSec = 30
    )
    $t0 = Get-Date
    while (((Get-Date) - $t0).TotalSeconds -lt $TimeoutSec) {
        if (Test-TcpPort -TargetHost $TargetHost -Port $Port) {
            return $true
        }
        Start-Sleep -Milliseconds 300
    }
    return $false
}

function Stop-ProcessByPort {
    param(
        [int]$Port
    )
    try {
        $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        foreach ($c in $conns) {
            $procId = $c.OwningProcess
            if ($procId -and $procId -gt 0) {
                try {
                    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                } catch {
                }
            }
        }
    } catch {
    }
}

$autoNoLLM = $false
try {
    $cfgPath = Join-Path $repo "configs\secretary.yaml"
    if (Test-Path $cfgPath) {
        $raw = Get-Content $cfgPath -Raw
        $m = [regex]::Match($raw, '(?ms)^\s*llm\s*:\s*.*?^\s*mode\s*:\s*"?([A-Za-z0-9_\-]+)"?')
        if ($m.Success) {
            $mode = ($m.Groups[1].Value | ForEach-Object { $_.ToString().Trim().ToLower() })
            if ($mode -eq "local") {
                $autoNoLLM = $true
            }
        }
    }
} catch {
}

if ($autoNoLLM -and (-not $NoLLM)) {
    Write-Host "[Auto] secretary.yaml llm.mode=local -> skip Ollama" -ForegroundColor DarkGray
    $NoLLM = $true
}

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
    $sovitsHost = $SovitsHost
    $sovitsPort = [int]$SovitsPort
    $sovitsStatePath = Join-Path $logDir "gpt_sovits.state.json"
    
    if (Test-Path $sovitsPy) {
        if ($ForceRestart) {
            Stop-ProcessByPort -Port $sovitsPort
            Start-Sleep -Milliseconds 400
        }

        if (Test-TcpPort -TargetHost $sovitsHost -Port $sovitsPort) {
            Write-Host "      -> GPT-SoVITS already running on port $sovitsPort" -ForegroundColor Gray
            try {
                $sovitsPid = $null
                try {
                    $conns = Get-NetTCPConnection -LocalPort $sovitsPort -State Listen -ErrorAction SilentlyContinue
                    if ($conns -and $conns.Count -gt 0) {
                        $sovitsPid = $conns[0].OwningProcess
                    }
                } catch {}

                $sovitsOut = Join-Path $logDir "gpt_sovits.out.log"
                $sovitsErr = Join-Path $logDir "gpt_sovits.err.log"
                $state = @{
                    pid = $sovitsPid
                    host = "$sovitsHost"
                    port = $sovitsPort
                    py = "$sovitsPy"
                    root = "$sovitsRoot"
                    args = @("api_v2.py", "-a", "$sovitsHost", "-p", "$sovitsPort")
                    out = "$sovitsOut"
                    err = "$sovitsErr"
                    updated_at = (Get-Date).ToString("o")
                    note = "already_running"
                }
                ($state | ConvertTo-Json -Depth 5) | Set-Content -Path $sovitsStatePath -Encoding UTF8
            } catch {}
        } else {
        $sovitsArgs = @(
            "api_v2.py",
            "-a", "$sovitsHost",
            "-p", "$sovitsPort"
        )
        $sovitsOut = Join-Path $logDir "gpt_sovits.out.log"
        $sovitsErr = Join-Path $logDir "gpt_sovits.err.log"
        $sovitsProcess = Start-Process -FilePath $sovitsPy -WorkingDirectory $sovitsRoot -ArgumentList $sovitsArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $sovitsOut -RedirectStandardError $sovitsErr
        $processes += $sovitsProcess
        Write-Host "      -> GPT-SoVITS starting on port $sovitsPort" -ForegroundColor Gray
        try {
            $sovitsPid = $null
            try { $sovitsPid = $sovitsProcess.Id } catch {}
            $state = @{
                pid = $sovitsPid
                host = "$sovitsHost"
                port = $sovitsPort
                py = "$sovitsPy"
                root = "$sovitsRoot"
                args = $sovitsArgs
                out = "$sovitsOut"
                err = "$sovitsErr"
                updated_at = (Get-Date).ToString("o")
                note = "started"
            }
            ($state | ConvertTo-Json -Depth 5) | Set-Content -Path $sovitsStatePath -Encoding UTF8
        } catch {}
        if (-not (Wait-TcpPort -TargetHost $sovitsHost -Port $sovitsPort -TimeoutSec 12)) {
            Write-Host "      -> GPT-SoVITS failed to open port $sovitsPort (see logs\gpt_sovits.*.log)" -ForegroundColor DarkYellow
        }
        }
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
    $apiHost = $ApiHost
    $apiPort = [int]$ApiPort

    try {
        $env:SECRETARY_MAX_MEMORY_FRAC = [string]$SecretaryVramFrac
        $env:TRADING_MAX_MEMORY_FRAC = [string]$TradingVramFrac
    } catch {
    }

    if ($ForceRestart) {
        Stop-ProcessByPort -Port $apiPort
        try { Stop-ProcessByPort -Port $sovitsPort } catch {}
        Start-Sleep -Milliseconds 500
    }

    if (Test-TcpPort -TargetHost $apiHost -Port $apiPort) {
        Write-Host "      -> API already running on port $apiPort (NOTE: if you updated code, use -ForceRestart)" -ForegroundColor Gray
    } else {
    $tradingArgs = @(
        "scripts/run_live_paper_trading.py",
        "--data-source", "$DataSource",
        "--with-api",
        "--api-host", "$apiHost",
        "--api-port", "$apiPort"
    )
    if (-not $NoModels) {
        $tradingArgs += "--load-models"
    }
    $tradingOut = Join-Path $logDir "trading_api.out.log"
    $tradingErr = Join-Path $logDir "trading_api.err.log"
    Write-Host "      -> Trading+API logs: logs\trading_api.out.log / logs\trading_api.err.log" -ForegroundColor DarkGray
    $tradingProcess = Start-Process -FilePath $py311 -WorkingDirectory $repo -ArgumentList $tradingArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $tradingOut -RedirectStandardError $tradingErr
    $processes += $tradingProcess
    Write-Host "      -> Trading Engine + API starting on port $apiPort" -ForegroundColor Gray
    Write-Host "      -> Waiting for API to be ready (up to 60s)..." -ForegroundColor DarkGray -NoNewline
    $waited = 0
    while ($waited -lt 60) {
        try {
            if ($tradingProcess -and $tradingProcess.HasExited) {
                Write-Host " EXIT" -ForegroundColor Red
                Write-Host "      -> Trading+API process exited early. See logs below." -ForegroundColor Red
                Show-LogTail -Path $tradingErr -Lines 120
                Show-LogTail -Path $tradingOut -Lines 120
                throw "API process exited"
            }
        } catch {
            throw
        }
        if (Test-TcpPort -TargetHost $apiHost -Port $apiPort) {
            Write-Host " OK" -ForegroundColor Green
            break
        }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
        $waited += 2
    }
    if ($waited -ge 60) {
        Write-Host " TIMEOUT" -ForegroundColor Red
        Write-Host "      -> API failed to open port $apiPort within 60s. See logs below." -ForegroundColor Red
        Show-LogTail -Path $tradingErr -Lines 120
        Show-LogTail -Path $tradingOut -Lines 120
        throw "API did not start"
    }
    }
} else {
    Write-Host "[2/4] Paper Trading skipped (--NoTrading)" -ForegroundColor DarkGray
}

# 3) Start Desktop UI (foreground, blocking)
Write-Host "[3/4] Starting Desktop UI..." -ForegroundColor Green
Write-Host "      -> Mari is waking up..." -ForegroundColor Gray
Write-Host ""

try {
    $uiOut = Join-Path $logDir "desktop_ui.out.log"
    $uiErr = Join-Path $logDir "desktop_ui.err.log"
    Write-Host "      -> Desktop UI logs: logs\desktop_ui.out.log / logs\desktop_ui.err.log" -ForegroundColor DarkGray

    $oldQtOpenGl = $env:QT_OPENGL
    $oldQtFlags = $env:QTWEBENGINE_CHROMIUM_FLAGS
    $oldQtDisable = $env:QTWEBENGINE_DISABLE_SANDBOX
    $env:QT_OPENGL = "angle"
    if (-not $env:QTWEBENGINE_CHROMIUM_FLAGS) {
        $env:QTWEBENGINE_CHROMIUM_FLAGS = "--use-angle=d3d11 --ignore-gpu-blocklist --enable-webgl --enable-accelerated-2d-canvas"
    }
    if (-not $env:QTWEBENGINE_DISABLE_SANDBOX) {
        $env:QTWEBENGINE_DISABLE_SANDBOX = "1"
    }

    $uiProcess = Start-Process -FilePath $py311 -WorkingDirectory $repo -ArgumentList @("src/ui/desktop/main.py") -PassThru -WindowStyle Normal -RedirectStandardOutput $uiOut -RedirectStandardError $uiErr
    Wait-Process -Id $uiProcess.Id

    $env:QT_OPENGL = $oldQtOpenGl
    $env:QTWEBENGINE_CHROMIUM_FLAGS = $oldQtFlags
    $env:QTWEBENGINE_DISABLE_SANDBOX = $oldQtDisable
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

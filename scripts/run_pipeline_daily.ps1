$ErrorActionPreference = 'Stop'

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $ProjectRoot

$VenvPy = Join-Path $ProjectRoot 'venv311\Scripts\python.exe'
if (!(Test-Path $VenvPy)) {
  throw "Python venv not found: $VenvPy"
}

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot 'logs') | Out-Null

$today = if ($env:DATE_OVERRIDE) { $env:DATE_OVERRIDE } else { (Get-Date).ToString('yyyy-MM-dd') }
$log = Join-Path $ProjectRoot ("logs\news_ops_{0}.log" -f $today)
$mutexName = "Global\Stock-NewsOps-7Days"

$NewsModel = if ($env:NEWS_MODEL) { $env:NEWS_MODEL } else { "Qwen/Qwen2.5-14B-Instruct" }
$NewsLora = if ($env:NEWS_LORA) { $env:NEWS_LORA } else { "models\llm_qwen14b_lora_c_hybrid\lora_weights" }
$NewsResume = if ($env:NEWS_RESUME) { $env:NEWS_RESUME } else { "1" }

$TradingBase = if ($env:TRADING_BASE) { $env:TRADING_BASE } else { "Qwen/Qwen2.5-14B-Instruct" }
$TradingAdapter = if ($env:TRADING_ADAPTER) { $env:TRADING_ADAPTER } else { "models\llm_etf_trading_qwen25_14b\lora_weights" }
$TradingRiskWatchMarket = if ($env:TRADING_RISK_WATCH_MARKET) { $env:TRADING_RISK_WATCH_MARKET } else { "BOTH" }
$TradingRiskWatchTop = if ($env:TRADING_RISK_WATCH_TOP) { $env:TRADING_RISK_WATCH_TOP } else { "3" }

$RunPaper = if ($env:RUN_PAPER) { $env:RUN_PAPER } else { "0" }
$PaperOutdir = if ($env:PAPER_OUTDIR) { $env:PAPER_OUTDIR } else { "data\paper" }
$PaperState = if ($env:PAPER_STATE) { $env:PAPER_STATE } else { "account_state.json" }
$PaperInitialCash = if ($env:PAPER_INITIAL_CASH) { $env:PAPER_INITIAL_CASH } else { "100000" }
$PaperRebalanceThreshold = if ($env:PAPER_REBALANCE_THRESHOLD) { $env:PAPER_REBALANCE_THRESHOLD } else { "0.02" }
$PaperSlippageBps = if ($env:PAPER_SLIPPAGE_BPS) { $env:PAPER_SLIPPAGE_BPS } else { "2" }
$PaperFeeBps = if ($env:PAPER_FEE_BPS) { $env:PAPER_FEE_BPS } else { "1" }
$PaperMtmNextDay = if ($env:PAPER_MTM_NEXT_DAY) { $env:PAPER_MTM_NEXT_DAY } else { "1" }
$PaperDataDir = if ($env:PAPER_DATA_DIR) { $env:PAPER_DATA_DIR } else { "data" }
$PaperMetaOut = if ($env:PAPER_META_OUT) { $env:PAPER_META_OUT } else { "" }
$PaperVirtualEvent = if ($env:PAPER_VIRTUAL_EVENT) { $env:PAPER_VIRTUAL_EVENT } else { "" }

$SkipFetch = if ($env:SKIP_FETCH) { $env:SKIP_FETCH } else { "0" }
$SkipFeatures = if ($env:SKIP_FEATURES) { $env:SKIP_FEATURES } else { "0" }

$ExperimentOutdir = if ($env:EXP_OUTDIR) { $env:EXP_OUTDIR } else { "" }

function Snapshot-Experiment([string]$expDir, [string]$dateStr) {
  if (-not $expDir -or $expDir.Trim().Length -eq 0) { return }
  New-Item -ItemType Directory -Force -Path $expDir | Out-Null

  $cfg = @{
    date = $dateStr
    log = $log
    NEWS_MODEL = $NewsModel
    NEWS_LORA = $NewsLora
    NEWS_RESUME = $NewsResume
    TRADING_BASE = $TradingBase
    TRADING_ADAPTER = $TradingAdapter
    TRADING_RISK_WATCH_MARKET = $TradingRiskWatchMarket
    TRADING_RISK_WATCH_TOP = $TradingRiskWatchTop
    RUN_PAPER = $RunPaper
    PAPER_OUTDIR = $PaperOutdir
    PAPER_STATE = $PaperState
    PAPER_MTM_NEXT_DAY = $PaperMtmNextDay
    PAPER_DATA_DIR = $PaperDataDir
    SKIP_FETCH = $SkipFetch
    SKIP_FEATURES = $SkipFeatures
  }
  ($cfg | ConvertTo-Json -Depth 5) | Out-File -FilePath (Join-Path $expDir 'run_config.json') -Encoding utf8

  $daily = Join-Path $ProjectRoot 'data\daily'
  $paths = @(
    (Join-Path $daily ("news_{0}.json" -f $dateStr)),
    (Join-Path $daily ("health_{0}.json" -f $dateStr)),
    (Join-Path $daily ("etf_features_{0}.json" -f $dateStr)),
    (Join-Path $daily ("signals_{0}.json" -f $dateStr)),
    (Join-Path $daily ("report_{0}.md" -f $dateStr)),
    (Join-Path $daily ("trading_decision_{0}.json" -f $dateStr))
  )

  foreach ($p in $paths) {
    if (Test-Path $p) {
      Copy-Item $p $expDir -Force
    }
  }

  if (Test-Path $log) {
    Copy-Item $log (Join-Path $expDir (Split-Path $log -Leaf)) -Force
  }
}

function Write-Log([string]$text) {
  $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
  "[$ts] $text" | Out-File -FilePath $log -Append -Encoding utf8
}

function ConvertTo-SafeName([string]$name) {
  return ($name -replace "[^A-Za-z0-9_-]+", "_")
}

function Invoke-Step([string]$name, [string[]]$argList) {
  "`n" | Out-File -FilePath $log -Append -Encoding utf8
  Write-Log "=== $name ==="
  Write-Log "CMD: $VenvPy $($argList -join ' ')"

  Write-Host "=== $name ==="

  $safe = ConvertTo-SafeName $name
  $runId = "{0}_{1}" -f $PID, (Get-Date).ToString('HHmmssfff')
  $tmpOut = Join-Path $ProjectRoot ("logs\tmp_{0}_{1}_{2}.out" -f $safe, $today, $runId)
  $tmpErr = Join-Path $ProjectRoot ("logs\tmp_{0}_{1}_{2}.err" -f $safe, $today, $runId)

  if (Test-Path $tmpOut) { Remove-Item -Force $tmpOut -ErrorAction SilentlyContinue }
  if (Test-Path $tmpErr) { Remove-Item -Force $tmpErr -ErrorAction SilentlyContinue }

  $p = Start-Process -FilePath $VenvPy -ArgumentList $argList -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tmpOut -RedirectStandardError $tmpErr

  if (Test-Path $tmpOut) {
    $outText = Get-Content -Path $tmpOut -Raw
    if ($outText) {
      $outText | Out-File -FilePath $log -Append -Encoding utf8
      Write-Host $outText
    }
  }
  if (Test-Path $tmpErr) {
    $errText = Get-Content -Path $tmpErr -Raw
    if ($errText) {
      $errText | Out-File -FilePath $log -Append -Encoding utf8
      Write-Host $errText
    }
  }

  if (Test-Path $tmpOut) { Remove-Item -Force $tmpOut -ErrorAction SilentlyContinue }
  if (Test-Path $tmpErr) { Remove-Item -Force $tmpErr -ErrorAction SilentlyContinue }

  $code = [int]$p.ExitCode
  Write-Log "EXIT: $code"
  Write-Host "EXIT: $code"
  if ($code -ne 0) {
    throw "$name failed with exit code $code (see $log)"
  }
}

$mutex = $null
$hasMutex = $false
try {
  Write-Log "START (root=$ProjectRoot)"
  Write-Log "VENV_PY=$VenvPy"
  Write-Log "NEWS_MODEL=$NewsModel"
  Write-Log "NEWS_LORA=$NewsLora"
  Write-Log "NEWS_RESUME=$NewsResume"
  Write-Log "TRADING_BASE=$TradingBase"
  Write-Log "TRADING_ADAPTER=$TradingAdapter"
  Write-Log "TRADING_RISK_WATCH_MARKET=$TradingRiskWatchMarket"
  Write-Log "TRADING_RISK_WATCH_TOP=$TradingRiskWatchTop"
  Write-Log "RUN_PAPER=$RunPaper"
  Write-Log "PAPER_OUTDIR=$PaperOutdir"
  Write-Log "PAPER_STATE=$PaperState"
  Write-Log "PAPER_MTM_NEXT_DAY=$PaperMtmNextDay"
  Write-Log "PAPER_DATA_DIR=$PaperDataDir"
  Write-Log "SKIP_FETCH=$SkipFetch"
  Write-Log "SKIP_FEATURES=$SkipFeatures"
  Write-Log "EXP_OUTDIR=$ExperimentOutdir"

  Write-Host "START: $today"
  Write-Host "LOG: $log"

  # Single-instance lock (prevents overlapping manual + scheduled runs)
  $mutex = New-Object System.Threading.Mutex($false, $mutexName)
  $hasMutex = $mutex.WaitOne(0)
  if (-not $hasMutex) {
    Write-Log "SKIP: another run is active (mutex=$mutexName)"
    Write-Host "SKIP: another run is active (mutex=$mutexName)"
    exit 0
  }

  if ($SkipFetch -ne "1") {
    Invoke-Step "Fetch RSS" @('scripts\fetch_daily_rss.py','--date',$today,'--health-out','auto')
  } else {
    Write-Log "SKIP: Fetch RSS"
    Write-Host "SKIP: Fetch RSS"
  }

  if ($SkipFeatures -ne "1") {
    Invoke-Step "Build ETF Features" @('scripts\build_daily_etf_features.py','--date',$today)
  } else {
    Write-Log "SKIP: Build ETF Features"
    Write-Host "SKIP: Build ETF Features"
  }
  $newsArgs = @('scripts\run_daily_inference.py','--date',$today,'--model',$NewsModel,'--lora',$NewsLora,'--use-lora','--load-in-4bit','--batch-size','4','--max-input-chars','6000','--save-every','20')
  if ($NewsResume -eq "1") { $newsArgs += @('--resume') }
  Invoke-Step "Run News Inference" $newsArgs
  Invoke-Step "Generate Report" @('scripts\generate_daily_report.py','--date',$today)

  $tradingOut = ("data\\daily\\trading_decision_{0}.json" -f $today)
  Invoke-Step "Run Trading Inference" @(
    'scripts\\run_trading_inference.py',
    '--date',$today,
    '--daily-dir','data/daily',
    '--base',$TradingBase,
    '--adapter',$TradingAdapter,
    '--risk-watch-market',$TradingRiskWatchMarket,
    '--risk-watch-top',$TradingRiskWatchTop,
    '--out',$tradingOut,
    '--load-4bit'
  )

  if ($RunPaper -eq "1") {
    $paperArgs = @(
      'scripts\\paper_trade_sim.py',
      '--date',$today,
      '--daily-dir','data/daily',
      '--decision',$tradingOut,
      '--outdir',$PaperOutdir,
      '--state',$PaperState,
      '--initial-cash',$PaperInitialCash,
      '--rebalance-threshold',$PaperRebalanceThreshold,
      '--slippage-bps',$PaperSlippageBps,
      '--fee-bps',$PaperFeeBps,
      '--mtm-next-day',$PaperMtmNextDay,
      '--data-dir',$PaperDataDir
    )
    if ($PaperMetaOut -and $PaperMetaOut.Trim().Length -gt 0) {
      $paperArgs += @('--meta-out',$PaperMetaOut)
    }
    if ($PaperVirtualEvent -and $PaperVirtualEvent.Trim().Length -gt 0) {
      $paperArgs += @('--virtual-event',$PaperVirtualEvent)
    }
    Invoke-Step "Paper Trade Sim" $paperArgs
  }

  Snapshot-Experiment $ExperimentOutdir $today

  Write-Log "DONE"
  Write-Host "DONE"
  Write-Host "OK: pipeline finished for $today (log: $log)"
  exit 0
} catch {
  Write-Log "ERROR: $($_.Exception.Message)"
  Write-Log "TRACE: $($_ | Out-String)"
  Write-Host "ERROR: $($_.Exception.Message)"
  Write-Host "FAILED (see log): $log"
  Write-Error "Pipeline failed. See log: $log"
  exit 1
} finally {
  if ($mutex -and $hasMutex) {
    $mutex.ReleaseMutex() | Out-Null
  }
  if ($mutex) {
    $mutex.Dispose()
  }
}

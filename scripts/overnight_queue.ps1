$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$pattern = '--outdir models/llm_qwen14b_overnight_v2'
Write-Host "[QUEUE] Waiting for current training to finish..." 
Write-Host "[QUEUE] Match pattern: $pattern"

while ($true) {
  $p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*$pattern*" }
  if (-not $p) { break }
  Write-Host "[QUEUE] Still running... $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
  Start-Sleep -Seconds 60
}

Write-Host "[QUEUE] Current training finished. $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

function Run-Step([string]$label, [string[]]$args) {
  Write-Host "[QUEUE] RUN: $label"
  & .\venv311\Scripts\python.exe @args
  if ($LASTEXITCODE -ne 0) {
    throw "[QUEUE] Step failed ($label) with exit code $LASTEXITCODE"
  }
}

Run-Step 'Refresh dataset (append/dedup/split-val)' @(
  'scripts/build_finetune_dataset.py',
  '--limit','800',
  '--add-explain',
  '--append','--dedup',
  '--split-val','--val-ratio','0.05'
)

Run-Step 'Train B: seq512 lr5e-5 e20' @(
  'scripts/finetune_llm.py',
  '--model','Qwen/Qwen2.5-14B-Instruct',
  '--data','data/finetune/train.json',
  '--qlora','--grad-ckpt',
  '--max-seq-len','512',
  '--batch-size','1','--grad-acc','8',
  '--lr','5e-5',
  '--epochs','20',
  '--save-steps','20','--save-total-limit','10',
  '--outdir','models/llm_qwen14b_overnight_seq512',
  '--resume','auto'
)

Run-Step 'Train C: seq768 lr1e-4 e20' @(
  'scripts/finetune_llm.py',
  '--model','Qwen/Qwen2.5-14B-Instruct',
  '--data','data/finetune/train.json',
  '--qlora','--grad-ckpt',
  '--max-seq-len','768',
  '--batch-size','1','--grad-acc','8',
  '--lr','1e-4',
  '--epochs','20',
  '--save-steps','20','--save-total-limit','10',
  '--outdir','models/llm_qwen14b_overnight_lr1e4',
  '--resume','auto'
)

Run-Step 'Train D: seq512 lr5e-5 e30' @(
  'scripts/finetune_llm.py',
  '--model','Qwen/Qwen2.5-14B-Instruct',
  '--data','data/finetune/train.json',
  '--qlora','--grad-ckpt',
  '--max-seq-len','512',
  '--batch-size','1','--grad-acc','8',
  '--lr','5e-5',
  '--epochs','30',
  '--save-steps','20','--save-total-limit','10',
  '--outdir','models/llm_qwen14b_overnight_seq512_e30',
  '--resume','auto'
)

Run-Step 'Train E: seq768 lr2e-5 e20' @(
  'scripts/finetune_llm.py',
  '--model','Qwen/Qwen2.5-14B-Instruct',
  '--data','data/finetune/train.json',
  '--qlora','--grad-ckpt',
  '--max-seq-len','768',
  '--batch-size','1','--grad-acc','8',
  '--lr','2e-5',
  '--epochs','20',
  '--save-steps','20','--save-total-limit','10',
  '--outdir','models/llm_qwen14b_overnight_lr2e5',
  '--resume','auto'
)

Write-Host "[QUEUE] ALL DONE. $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

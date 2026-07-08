"""v3 夜训编排：基线检查 → smoke 自检 → 阶梯降级 → 全量发射。

背景（2026-07-07 实测）：v3 语料含期权/市场报告长样本（p99≈3000 token，
市况任务 p99≈1000），而 bs1/seq3584 与 bs1/seq3072 在 3.9GB 桌面基线下
均 OOM（训练本体需求 >18.6GB）。故夜训流程固化为：

1. 基线检查：GPU 已用 < --max-baseline-mb（默认 2500MB）才准起飞——
   睡前需关游戏/浏览器/模拟器等占卡进程；不达标列数字拒绝启动。
2. smoke 自检：用最长的 24 条样本在候选 seq 上跑 1 epoch（~5 分钟），
   OOM 就降档重试（3072 → 2560 → 2048），首个存活的 seq 胜出。
3. 全量：胜出 seq + bs1/accum16（等效 batch 16 与 v2 一致）/ckpt/4bit/
   lr1e-4/2ep。seq2048 兜底档会截断约六成期权答案——若落到这档，
   日志明说，盲评时重点核期权任务。

用法（用户说"开训"后手动执行；仪表盘等 GPU 驻留进程必须先停）：
    venv311\\Scripts\\python.exe scripts\\night_train_v3.py
    # 跳过基线检查（明知故犯时）：--force
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
PY = str(ROOT / "venv311" / "Scripts" / "python.exe")
SEQ_LADDER = (3072, 2560, 2048)

TRAIN_JSONL = ROOT / "data" / "distill" / "sft_v3_train.jsonl"
EVAL_JSONL = ROOT / "data" / "distill" / "sft_v3_eval.jsonl"
SMOKE_TRAIN = ROOT / "data" / "distill" / "smoke_v3_train.jsonl"
SMOKE_EVAL = ROOT / "data" / "distill" / "smoke_v3_eval.jsonl"
OUT_DIR = ROOT / "models" / "llm" / "finance_v3_sft"
LOG = ROOT / "logs" / "night_train_v3.log"


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def gpu_used_mb() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return int(out.strip().splitlines()[0])


def build_smoke_sets() -> None:
    """train 集中最长 24+4 条（一次性，已存在则跳过）。"""
    if SMOKE_TRAIN.exists() and SMOKE_EVAL.exists():
        return
    log("building smoke sets (tokenizing full train set once)...")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    rows = [json.loads(l) for l in TRAIN_JSONL.read_text(encoding="utf-8").splitlines()]
    rows.sort(key=lambda r: -len(tok(tok.apply_chat_template(r["conversations"], tokenize=False)).input_ids))
    for path, recs in ((SMOKE_TRAIN, rows[:24]), (SMOKE_EVAL, rows[24:28])):
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_train(sft: Path, eval_: Path, out: Path, seq: int, epochs: int) -> int:
    env = os.environ.copy()
    env.update({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "QUANTAI__llm__finetune__load_in_4bit": "true",
        "QUANTAI__llm__finetune__gradient_checkpointing": "true",
        "QUANTAI__llm__finetune__learning_rate": "0.0001",
        "QUANTAI__llm__finetune__num_epochs": str(epochs),
        "QUANTAI__llm__finetune__max_seq_length": str(seq),
        "QUANTAI__llm__finetune__batch_size": "1",
        "QUANTAI__llm__finetune__gradient_accumulation_steps": "16",
    })
    cmd = [PY, str(ROOT / "scripts" / "train.py"), "--sft", str(sft), "--eval", str(eval_),
           "--confirm-compute", "--output-dir", str(out)]
    with LOG.open("a", encoding="utf-8") as f:
        return subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="跳过 GPU 基线检查")
    p.add_argument("--max-baseline-mb", type=int, default=2500)
    args = p.parse_args()

    used = gpu_used_mb()
    if used > args.max_baseline_mb and not args.force:
        log(f"ABORT: GPU baseline {used}MB > {args.max_baseline_mb}MB —— "
            f"先关掉占卡进程（游戏/浏览器/模拟器/仪表盘）再来，或 --force")
        return 2
    log(f"GPU baseline {used}MB OK")

    build_smoke_sets()

    chosen = None
    for seq in SEQ_LADDER:
        log(f"smoke test @ seq={seq} ...")
        rc = run_train(SMOKE_TRAIN, SMOKE_EVAL, ROOT / "logs" / f"smoke_out_{seq}", seq, epochs=1)
        if rc == 0:
            chosen = seq
            log(f"smoke PASSED @ seq={seq}")
            break
        log(f"smoke FAILED @ seq={seq} (rc={rc})，降档")
    if chosen is None:
        log("ABORT: 全部 seq 档位 smoke 失败，不发射全量。查 logs/night_train_v3.log")
        return 3
    if chosen == 2048:
        log("WARNING: 兜底档 seq2048 —— 约六成期权答案会被截断，盲评必须重点核期权任务")

    log(f"FULL RUN: seq={chosen} bs1/accum16 2ep -> {OUT_DIR}")
    rc = run_train(TRAIN_JSONL, EVAL_JSONL, OUT_DIR, chosen, epochs=2)
    log(f"full run finished rc={rc}" + ("" if rc == 0 else "（失败，查日志）"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

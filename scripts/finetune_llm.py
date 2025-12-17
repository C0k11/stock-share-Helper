#!/usr/bin/env python
"""
一键微调脚本

建议先用小模型跑通流程（默认），再切换到Qwen2.5-7B正式训练。

示例：
  .\\venv311\\Scripts\\python.exe scripts\\finetune_llm.py --smoke
  .\\venv311\\Scripts\\python.exe scripts\\finetune_llm.py --model Qwen/Qwen2.5-7B-Instruct --epochs 1
"""

import sys
from pathlib import Path
import argparse
import re
from typing import Optional
from loguru import logger


# 添加项目根目录到路径（确保可直接运行脚本）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="LLM LoRA fine-tuning runner")
    parser.add_argument("--data", default="data/finetune/train.json", help="训练数据JSON路径")
    parser.add_argument("--eval-data", default=None, help="验证数据JSON路径（可选）")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="基础模型名称")
    parser.add_argument("--outdir", default="models/llm", help="输出目录（含checkpoints与lora权重）")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=0, help="验证 batch size（0 表示与训练一致）")
    parser.add_argument("--grad-acc", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=0, help="验证间隔 steps（0 表示与 save-steps 一致；<0 表示禁用 eval）")
    parser.add_argument("--eval-max-samples", type=int, default=0, help="验证集最多取前 N 条（0 表示全量）")
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=1024, help="最大序列长度（7B建议 512/1024）")
    parser.add_argument("--grad-ckpt", action="store_true", help="启用梯度检查点以节省显存")
    parser.add_argument("--qlora", action="store_true", help="启用4bit量化训练（14B建议开启）")
    parser.add_argument(
        "--resume",
        default=None,
        help="断点续训：填checkpoint路径，或填 auto 自动选择最新 checkpoint",
    )
    parser.add_argument("--smoke", action="store_true", help="冒烟测试：更小batch+更少步数")

    args = parser.parse_args()

    if isinstance(args.model, str):
        args.model = args.model.replace("\\", "/")

    if args.smoke:
        args.epochs = 1
        args.batch_size = 1
        args.grad_acc = 4
        args.save_steps = 20
        logger.info("Running SMOKE fine-tune to validate pipeline")

    def resolve_resume_path(resume_arg: Optional[str], outdir: str) -> Optional[str]:
        if not resume_arg:
            return None
        if resume_arg != "auto":
            return resume_arg

        ckpt_root = Path(outdir) / "checkpoints"
        if not ckpt_root.exists():
            return None
        candidates = []
        for p in ckpt_root.glob("checkpoint-*"):
            m = re.match(r"checkpoint-(\d+)$", p.name)
            if m:
                candidates.append((int(m.group(1)), p))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        latest = candidates[-1][1]
        return str(latest)

    from src.llm.finetune.train import FineTuner

    trainer = FineTuner(
        model_name=args.model,
        output_dir=args.outdir,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        max_seq_length=args.max_seq_len,
        gradient_checkpointing=args.grad_ckpt,
        load_in_4bit=args.qlora,
    )

    trainer.train(
        train_data_path=args.data,
        eval_data_path=args.eval_data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_acc,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_max_samples=args.eval_max_samples,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=resolve_resume_path(args.resume, args.outdir),
    )


if __name__ == "__main__":
    main()

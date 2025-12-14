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
from loguru import logger


# 添加项目根目录到路径（确保可直接运行脚本）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="LLM LoRA fine-tuning runner")
    parser.add_argument("--data", default="data/finetune/train.json", help="训练数据JSON路径")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="基础模型名称")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-acc", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-seq-len", type=int, default=1024, help="最大序列长度（7B建议 512/1024）")
    parser.add_argument("--grad-ckpt", action="store_true", help="启用梯度检查点以节省显存")
    parser.add_argument("--smoke", action="store_true", help="冒烟测试：更小batch+更少步数")

    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1
        args.batch_size = 1
        args.grad_acc = 4
        args.save_steps = 20
        logger.info("Running SMOKE fine-tune to validate pipeline")

    from src.llm.finetune.train import FineTuner

    trainer = FineTuner(
        model_name=args.model,
        output_dir="models/llm",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        max_seq_length=args.max_seq_len,
        gradient_checkpointing=args.grad_ckpt,
    )

    trainer.train(
        train_data_path=args.data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_acc,
        save_steps=args.save_steps,
    )


if __name__ == "__main__":
    main()

"""薄 CLI：本地学生模型训练（QLoRA SFT / DPO）——蒸馏产物 → LoRA adapter。

⛔ 算力闸（硬约定，与 distill.py 的 --confirm-spend 同款）：真训练必须显式给
`--confirm-compute`，否则只打印计划就退出；本命令**只由用户手动执行**（需 GPU +
`pip install -e .[llm]`），任何自动化流程不得调用。无 GPU 时在加载模型前明确报错。

链路位置（见 README 飞轮图）：
    scripts/distill.py  → data/distill/sft_*.jsonl + dpo_*.jsonl
    scripts/train.py    → models/llm/<run>/lora_weights（本文件）
    scripts/evolve.py   → 飞轮偏好对的离线 DPO（另一条数据来源）
    回测对比            → quantai.backtest.run_backtest / scripts/lookahead_compare.py

示例：
    # SFT：蒸馏 conversations JSONL -> QLoRA adapter（真跑要 --confirm-compute）
    python scripts/train.py --sft data/distill/sft_20260701_192218.jsonl --confirm-compute

    # DPO：偏好对 JSONL（在 SFT adapter 之上继续对齐）
    python scripts/train.py --dpo data/distill/dpo_20260701_192218.jsonl --sft-adapter models/llm/lora_weights --confirm-compute

产物：<output-dir>/lora_weights（SFT）或 <output-dir>（DPO），可直接填进
`LLMConfig.adapters` 或经 `EvolutionTrainer.set_active_adapter` 热切换。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantai.config import load_config


def _gpu_or_exit() -> None:
    try:
        import torch
    except ImportError:
        print("⛔ 缺 torch：pip install -e .[llm]（需 GPU 环境）", file=sys.stderr)
        raise SystemExit(2)
    if not torch.cuda.is_available():
        print("⛔ 未检测到 CUDA GPU：QLoRA/DPO 训练需要 GPU。", file=sys.stderr)
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="QuantAI student training (QLoRA SFT / DPO)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sft", metavar="JSONL", help="SFT 训练数据（conversations 格式）")
    g.add_argument("--dpo", metavar="JSONL", help="DPO 偏好对数据（prompt/chosen/rejected）")
    p.add_argument("--sft-adapter", default="", help="DPO 的初始 SFT adapter 路径（可选）")
    p.add_argument("--eval", default=None, help="SFT 验证集 JSONL（可选）")
    p.add_argument("--resume", default=None, help="SFT 断点续训 checkpoint（可选）")
    p.add_argument("--output-dir", default=None, help="输出目录（默认 models/llm/<mode>_<时间戳>）")
    p.add_argument("--confirm-compute", action="store_true", help="确认占用 GPU 真跑训练（不给则只打印计划）")
    args = p.parse_args(argv)

    cfg = load_config()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "sft" if args.sft else "dpo"
    out_dir = args.output_dir or f"models/llm/{mode}_{stamp}"

    data_path = args.sft or args.dpo
    if not Path(data_path).exists():
        print(f"⛔ 数据文件不存在：{data_path}", file=sys.stderr)
        return 1
    if args.dpo and not args.sft_adapter:
        print("⛔ DPO 按设计在 SFT adapter 之上继续对齐：请给 --sft-adapter <路径>。", file=sys.stderr)
        print("   （先跑 --sft 产出 lora_weights，再拿它做 DPO 初始化。）", file=sys.stderr)
        return 1

    print(f"[train] mode={mode} base={cfg.llm.model_name} data={data_path} out={out_dir}")
    if not args.confirm_compute:
        print("⛔ 未给 --confirm-compute：只打印计划，不加载模型不训练（防自动化误触）。", file=sys.stderr)
        return 2
    _gpu_or_exit()

    if args.sft:
        from quantai.llm import QLoRAFineTuner

        tuner = QLoRAFineTuner.from_config(cfg.llm, output_dir=out_dir)
        saved = tuner.train(data_path, eval_data_path=args.eval, resume_from_checkpoint=args.resume)
    else:
        from quantai.llm import DPORunner

        # from_config 读的是 LLMConfig 字段（model_name/dpo/cache_dir）——传 AppConfig
        # 根对象会 AttributeError（审查实锤，DPO CLI 路径此前必炸）
        runner = DPORunner.from_config(cfg.llm, sft_adapter=args.sft_adapter, output_dir=out_dir)
        saved = runner.train(data_path)

    print(f"[train] done -> {saved}")
    print("[train] 挂载：configs/default.yaml llm.adapters 加一行，或 EvolutionTrainer.set_active_adapter 热切换。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

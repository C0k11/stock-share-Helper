"""薄 CLI：数据飞轮离线训练编排（quantai.evolution.EvolutionTrainer）。

诚实说明：本项目**没有在线实时梯度**（B-1）。"进化"= 用采集到的决策/盈亏建 DPO 偏好对，
跑**离线** DPO 训练出新 adapter，再写 active 指针供推理热切换。

示例：
    # 只建数据集（CPU 即可，看能配出多少偏好对）
    python scripts/evolve.py --build-dataset

    # 跑离线 DPO 训练（需 GPU + trl），并把新 adapter 设为 scalper 的 active
    python scripts/evolve.py --train --role scalper
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantai.config import load_config
from quantai.evolution import EvolutionTrainer


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QuantAI evolution (offline DPO data flywheel)")
    p.add_argument("--build-dataset", action="store_true", help="只构建偏好对数据集（CPU）")
    p.add_argument("--train", action="store_true", help="跑离线 DPO 训练（需 GPU + trl）")
    p.add_argument("--role", default="scalper", help="要训练/热切换的专家角色")
    p.add_argument("--no-config", action="store_true", help="跳过 load_config，用纯默认配置")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    from quantai.config.schema import AppConfig

    cfg = AppConfig() if args.no_config else load_config()
    trainer = EvolutionTrainer.from_config(cfg)

    if args.train:
        print(f"[evolve] offline DPO training role={args.role} ...")
        res = trainer.train_offline(role=args.role)
        print(f"[evolve] done: {res}")
        return

    # 默认：只建数据集
    path = trainer.build_dataset()
    pairs = trainer.to_dpo_records(_read_pairs(path))
    print(f"[evolve] dataset built: {path}")
    print(f"[evolve] preference records (DPO-ready): {len(pairs)}")
    if not pairs:
        print("[evolve] (no pairs yet — run live with --collect to accumulate decisions/outcomes first)")


def _read_pairs(path) -> list:
    import json

    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


if __name__ == "__main__":
    main()

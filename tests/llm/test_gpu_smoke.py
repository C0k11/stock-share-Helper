"""GPU 集成冒烟测（env-gated）。

默认 **skip**；仅当 `QUANTAI_GPU_TEST=1` 且本机有空闲显存时运行：真加载 Qwan2.5-7B
跑一次推理，并链式跑「一小步 QLoRA -> 一小步 DPO（reference_free）」端到端。

运行：
    $env:QUANTAI_GPU_TEST="1"; venv311\\Scripts\\python.exe -m pytest tests/llm/test_gpu_smoke.py -q -s
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTAI_GPU_TEST") != "1",
    reason="设 QUANTAI_GPU_TEST=1 运行 GPU 冒烟（会真加载 7B，需空闲显存）",
)


def _cuda_or_skip() -> None:
    try:
        import torch
    except Exception:
        pytest.skip("torch 不可用")
    if not torch.cuda.is_available():
        pytest.skip("无可用 CUDA 设备")


def test_inference_smoke() -> None:
    _cuda_or_skip()
    from quantai.config import load_config
    from quantai.llm import LocalLLM

    cfg = load_config()
    llm = LocalLLM.from_config(cfg.llm)
    try:
        out = llm.generate("Reply with exactly the word: OK", max_new_tokens=8, temperature=0.0)
        assert isinstance(out, str) and out != ""
    finally:
        llm.unload()


def test_qlora_then_dpo_one_step(tmp_path: Path) -> None:
    _cuda_or_skip()
    from quantai.config import load_config
    from quantai.llm import DPORunner, QLoRAFineTuner

    cfg = load_config()

    # --- 一小步 QLoRA：2 条 conversations，4bit 省显存 ---
    sft_data = tmp_path / "sft.jsonl"
    rows = [
        {"conversations": [{"role": "user", "content": "Say BUY"}, {"role": "assistant", "content": "BUY"}]},
        {"conversations": [{"role": "user", "content": "Say HOLD"}, {"role": "assistant", "content": "HOLD"}]},
    ]
    sft_data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    cfg.llm.finetune.load_in_4bit = True
    cfg.llm.finetune.gradient_checkpointing = True
    cfg.llm.finetune.num_epochs = 1
    cfg.llm.finetune.batch_size = 1
    cfg.llm.finetune.save_steps = 1

    ft = QLoRAFineTuner.from_config(cfg.llm, output_dir=str(tmp_path / "sft_out"))
    adapter_dir = ft.train(str(sft_data))
    assert Path(adapter_dir).exists()

    # --- 一小步 DPO（reference_free）on 上一步的 adapter ---
    dpo_data = tmp_path / "dpo.jsonl"
    pair = {
        "prompt": [{"role": "user", "content": "Decide for NVDA"}],
        "chosen": "BUY",
        "rejected": "SELL",
    }
    dpo_data.write_text(json.dumps(pair) + "\n", encoding="utf-8")

    cfg.llm.dpo.reference_free = True
    cfg.llm.dpo.num_epochs = 1
    cfg.llm.dpo.batch_size = 1
    cfg.llm.dpo.save_steps = 1

    runner = DPORunner.from_config(cfg.llm, sft_adapter=adapter_dir, output_dir=str(tmp_path / "dpo_out"))
    out_dir = runner.train(str(dpo_data))
    assert Path(out_dir).exists()

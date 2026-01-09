#!/usr/bin/env python

import argparse
import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


class JsonlVlmDataset(Dataset):
    def __init__(self, path: str, *, limit: int = 0):
        p = Path(str(path))
        if not p.exists():
            raise SystemExit(f"dataset not found: {p}")
        rows: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = (line or "").strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
                if int(limit) > 0 and len(rows) >= int(limit):
                    break
        if not rows:
            raise SystemExit(f"empty dataset: {p}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.rows[int(i)]


def _decode_image_b64_to_pil(b64: str):
    try:
        from PIL import Image
    except Exception as e:
        raise SystemExit("PIL is required (pip install pillow)") from e

    raw = base64.b64decode(str(b64 or "").strip(), validate=False)
    img = Image.open(io.BytesIO(raw))  # type: ignore[name-defined]
    return img.convert("RGB")


@dataclass
class VlmSftCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = []
        prompt_texts: List[str] = []
        full_texts: List[str] = []

        for rec in features:
            sp = str(rec.get("system_prompt") or "").strip()
            up = str(rec.get("user_prompt") or "").strip()
            ans = str(rec.get("assistant") or "").strip()
            b64 = str(rec.get("image_base64") or "").strip()
            if (not sp) or (not up) or (not ans) or (not b64):
                raise ValueError("bad sample: missing system_prompt/user_prompt/assistant/image_base64")

            try:
                from PIL import Image

                raw = base64.b64decode(b64, validate=False)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as e:
                raise ValueError(f"failed to decode image_base64: {e}") from e

            images.append(img)

            prompt_messages = [
                {"role": "system", "content": sp},
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": up}]},
            ]
            full_messages = [
                {"role": "system", "content": sp},
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": up}]},
                {"role": "assistant", "content": ans},
            ]

            if hasattr(self.processor, "apply_chat_template"):
                prompt_text = self.processor.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            else:
                prompt_text = f"{sp}\n{up}\n"
                full_text = f"{sp}\n{up}\n{ans}\n"

            prompt_texts.append(str(prompt_text))
            full_texts.append(str(full_text))

        prompt_inputs = self.processor(text=prompt_texts, images=images, padding=True, return_tensors="pt")
        full_inputs = self.processor(text=full_texts, images=images, padding=True, return_tensors="pt")

        try:
            prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        except Exception:
            prompt_lens = [int(x.shape[0]) for x in prompt_inputs["input_ids"]]

        labels = full_inputs["input_ids"].clone()
        attn = full_inputs.get("attention_mask")
        for i, pl in enumerate(prompt_lens):
            try:
                labels[i, : int(pl)] = -100
            except Exception:
                pass
        if attn is not None:
            labels[attn == 0] = -100

        out = dict(full_inputs)
        out["labels"] = labels
        return out


def _pick_torch_dtype(s: str) -> torch.dtype:
    v = str(s or "").strip().lower()
    if v == "float16":
        return torch.float16
    if v == "bfloat16":
        return torch.bfloat16
    return torch.float16


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", default="data/finetune/vlm/chartist_sft.jsonl")
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"))
    ap.add_argument("--outdir", default="models/vlm/chartist_lora")

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)

    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])

    ap.add_argument("--qlora", action="store_true", default=True)
    ap.add_argument("--no-qlora", dest="qlora", action="store_false")

    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    ap.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA",
    )

    args = ap.parse_args()

    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from transformers import Trainer, TrainingArguments

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as e:
        raise SystemExit(f"peft missing: {e}") from e

    dtype = _pick_torch_dtype(str(args.dtype))

    quant_cfg = None
    model_kwargs: Dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
    if bool(args.qlora):
        try:
            quant_cfg = BitsAndBytesConfig(load_in_4bit=True)
        except Exception:
            quant_cfg = None
        if quant_cfg is not None:
            model_kwargs["quantization_config"] = quant_cfg
        else:
            model_kwargs["torch_dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype

    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.model), **model_kwargs)

    if bool(args.qlora):
        model = prepare_model_for_kbit_training(model)

    target_modules = [s.strip() for s in str(args.target_modules).split(",") if s.strip()]
    lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    train_ds = JsonlVlmDataset(str(args.data), limit=int(args.limit))
    data_collator = VlmSftCollator(processor=processor)

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(outdir),
        num_train_epochs=float(args.epochs),
        per_device_train_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        learning_rate=float(args.lr),
        warmup_ratio=float(args.warmup_ratio),
        max_steps=int(args.max_steps) if int(args.max_steps) > 0 else -1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to=[],
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(str(outdir))
    try:
        processor.save_pretrained(str(outdir))
    except Exception:
        pass

    print(json.dumps({"outdir": str(outdir.as_posix())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

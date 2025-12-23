import argparse
import json
import re

import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_signal(text: str, *, binary: bool = False) -> str:
    raw = str(text)
    up = raw.upper()

    # Prefer explicit training-template markers.
    if "SIGNAL: BUY" in up:
        return "BUY"
    if "SIGNAL: CLEAR" in up:
        return "CLEAR"
    if not binary:
        if "SIGNAL: SELL" in up:
            return "SELL"
        if "SIGNAL: HOLD" in up:
            return "HOLD"

    # Heuristic: many instruct models put the decision at the end.
    tail = up.strip().splitlines()[-8:]
    tail_text = "\n".join(tail)
    for key in ["FINAL", "DECISION", "RECOMMENDATION", "ACTION", "SIGNAL"]:
        if key in tail_text:
            if "BUY" in tail_text:
                return "BUY"
            if "CLEAR" in tail_text:
                return "CLEAR"
            if not binary:
                if "SELL" in tail_text:
                    return "SELL"
                if "HOLD" in tail_text:
                    return "HOLD"

    # Regex fallback on the tail region, e.g. "Decision: BUY" / "Recommendation - BUY".
    tail_region = "\n".join(up.strip().splitlines()[-30:])
    if binary:
        m = re.search(r"(?:SIGNAL|DECISION|RECOMMENDATION|ACTION)\s*[:\-]\s*(BUY|CLEAR)\b", tail_region)
    else:
        m = re.search(r"(?:SIGNAL|DECISION|RECOMMENDATION|ACTION)\s*[:\-]\s*(BUY|SELL|CLEAR|HOLD)\b", tail_region)
    if m:
        return str(m.group(1)).upper()

    return "UNKNOWN"


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 15.3g: Verify V4 Surgical DPO Adapter (Analyst-only unit test)")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter", default="models/trader_v4_dpo_analyst_alpha")
    p.add_argument(
        "--adapter-path",
        dest="adapter",
        default="models/trader_v4_dpo_analyst_alpha",
        help="Alias for --adapter (kept for convenience).",
    )
    p.add_argument("--data", required=True, help="Path to a jsonl dataset (prompt/chosen/rejected). Required for safety.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument(
        "--merge-adapter",
        default="",
        help="Optional prerequisite adapter to merge into the base before loading the target adapter.",
    )
    p.add_argument(
        "--append-format-hint",
        action="store_true",
        help="Append a strict output-format hint to the prompt to force a 'Signal:' line.",
    )
    p.add_argument(
        "--binary-signals",
        action="store_true",
        help="Only count/parse BUY vs CLEAR (recommended for Phase 15 surgical verification).",
    )
    p.add_argument(
        "--save-jsonl",
        default="",
        help="Optional path to save per-sample outputs (jsonl).",
    )
    p.add_argument(
        "--success-threshold",
        type=float,
        default=0.8,
        help="SUCCESS threshold on BUY conversion rate (default: 0.8).",
    )
    args = p.parse_args()

    base_model_name = str(args.base_model)
    adapter_path = str(args.adapter)
    data_path = str(args.data)

    print(f"Using data: {data_path}")

    merge_adapter = str(args.merge_adapter).strip()
    if merge_adapter:
        print(f"Loading Base (bf16) for merge: {base_model_name}")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print(f"Loading Base (4-bit): {base_model_name}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if merge_adapter:
        print(f"Merging prerequisite adapter into base: {merge_adapter}")
        model = PeftModel.from_pretrained(model, merge_adapter)
        model = model.merge_and_unload()
        print("Merge complete.")

    print(f"Loading V4 Adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    prompts = []
    metadata = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"])
            metadata.append(obj.get("metadata", {}))

    print(f"Verifying {len(prompts)} surgical samples...")
    print("-" * 60)

    results = {"BUY": 0, "CLEAR": 0, "OTHER": 0}

    save_f = None
    if str(args.save_jsonl).strip():
        save_f = open(str(args.save_jsonl), "w", encoding="utf-8")

    if args.binary_signals:
        format_hint_block = (
            "Please output exactly two lines:\n"
            "Signal: <BUY/CLEAR>\n"
            "Confidence: <0-1>\n"
        )
    else:
        format_hint_block = (
            "Please output exactly two lines:\n"
            "Signal: <BUY/SELL/CLEAR/HOLD>\n"
            "Confidence: <0-1>\n"
        )

    def _inject_format_hint(prompt_text: str) -> str:
        # If the prompt already contains Qwen chat markers, inject a *new user turn*
        # before the final assistant marker.
        marker = "<|im_start|>assistant"
        if marker in prompt_text:
            idx = prompt_text.rfind(marker)
            if idx >= 0:
                return (
                    prompt_text[:idx]
                    + "<|im_start|>user\n"
                    + format_hint_block
                    + "<|im_end|>\n"
                    + prompt_text[idx:]
                )
        # Fallback: plain-text prompts.
        return prompt_text + "\n\n" + format_hint_block

    for i, prompt in enumerate(tqdm(prompts)):
        if args.append_format_hint:
            prompt = _inject_format_hint(str(prompt))
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_ids = out[0][inputs["input_ids"].shape[-1] :]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        signal = parse_signal(response, binary=bool(args.binary_signals))

        if save_f is not None:
            rec = {
                "i": int(i),
                "metadata": metadata[i],
                "signal": signal,
                "response": response,
            }
            save_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if signal == "BUY":
            results["BUY"] += 1
        elif signal == "CLEAR":
            results["CLEAR"] += 1
        else:
            results["OTHER"] += 1

        if i < 3:
            tic = str(metadata[i].get("ticker", ""))
            date = str(metadata[i].get("date", ""))
            print(f"\n[Sample {i + 1}] {tic} on {date}")
            tail = response[-300:] if len(response) > 300 else response
            print(f"V4 Output (tail): ...{tail}")
            print(f"Signal: {signal}")

    total = max(1, len(prompts))
    print("-" * 60)
    print("Verification Results (V4 Analyst on Alpha Days):")
    print(f"Total Samples: {len(prompts)}")
    print(f"BUY Signals:   {results['BUY']} (Conversion Rate: {results['BUY'] / total:.1%})")
    print(f"CLEAR Signals: {results['CLEAR']}")
    print(f"Other:         {results['OTHER']}")
    print("-" * 60)

    thr = float(args.success_threshold)
    if results["BUY"] >= len(prompts) * thr:
        print("SUCCESS: V4 Analyst has successfully learned to hunt Alpha!")
    else:
        print("WARNING: V4 Analyst is still too conservative. More training needed.")

    if save_f is not None:
        save_f.close()


if __name__ == "__main__":
    main()

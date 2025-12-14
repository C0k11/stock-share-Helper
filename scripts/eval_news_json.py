#!/usr/bin/env python
"""Quick eval for news->strict JSON.

Example:
  .\\venv311\\Scripts\\python.exe scripts\\eval_news_json.py --model Qwen/Qwen2.5-14B-Instruct --use-lora --lora models\\llm_qwen14b_overnight_v2\\lora_weights --load-in-4bit \
    --title "Fed signals rates may stay higher" --content "In the latest minutes..."

Or read from stdin:
  type news.txt | .\\venv311\\Scripts\\python.exe scripts\\eval_news_json.py --stdin
"""

import sys
import argparse
import json
from pathlib import Path
from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def build_messages(title: str, content: str, pod_style: bool) -> list:
    if pod_style:
        system = (
            "You are Pod 042 from NieR:Automata: calm, concise, logically rigorous. "
            "You must output STRICT JSON only (no markdown, no prose outside JSON). "
            "The JSON values must still be plain text."
        )
    else:
        system = (
            "You are a professional financial news analyst. "
            "You must output STRICT JSON only (no markdown, no prose outside JSON)."
        )

    schema = (
        "Output exactly ONE JSON object with these fields: "
        "event_type (string), sentiment (string), impact_equity (-1/0/1), impact_bond (-1/0/1), impact_gold (-1/0/1), summary (string)."
    )

    user = (
        f"Title: {title}\n"
        f"Content: {content}\n\n"
        f"{schema}\n"
        "Do not include any additional keys."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def extract_first_json(text: str) -> str:
    """Extract first top-level JSON object from model output."""
    if not text:
        raise ValueError("Empty model output")

    start = text.find("{")
    if start < 0:
        raise ValueError("No '{' found in output")

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Unbalanced JSON braces in output")


def main():
    parser = argparse.ArgumentParser(description="Evaluate news->strict JSON output")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model")
    parser.add_argument("--lora", default="models/llm/lora_weights", help="LoRA weights path")
    parser.add_argument("--use-lora", action="store_true", help="Load LoRA weights")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit")
    parser.add_argument("--title", default="", help="News title")
    parser.add_argument("--content", default="", help="News content")
    parser.add_argument("--stdin", action="store_true", help="Read title/content from stdin")
    parser.add_argument("--pod", action="store_true", help="Use Pod 042 style system prompt (still JSON-only)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)

    args = parser.parse_args()

    if args.stdin:
        raw = sys.stdin.read().strip()
        if not raw:
            raise SystemExit("--stdin provided but stdin is empty")
        # simple split: first line title, rest content
        lines = raw.splitlines()
        title = lines[0].strip() if lines else ""
        content = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    else:
        title = (args.title or "").strip()
        content = (args.content or "").strip()

    if not title and not content:
        raise SystemExit("Provide --title/--content or use --stdin")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if args.use_lora:
        from peft import PeftModel

        lora_path = Path(args.lora)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
        logger.info(f"Loading LoRA weights: {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path))

    messages = build_messages(title=title, content=content, pod_style=args.pod)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # decoded contains prompt + output in some templates; extract tail by removing prompt if present
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt) :]

    decoded = decoded.strip()

    try:
        json_text = extract_first_json(decoded)
        obj = json.loads(json_text)
    except Exception as e:
        print("RAW_OUTPUT_START")
        print(decoded)
        print("RAW_OUTPUT_END")
        raise SystemExit(f"Failed to parse JSON: {e}")

    required = ["event_type", "sentiment", "impact_equity", "impact_bond", "impact_gold", "summary"]
    missing = [k for k in required if k not in obj]
    extra = [k for k in obj.keys() if k not in required]

    if missing or extra:
        logger.warning(f"Schema mismatch. missing={missing} extra={extra}")

    print(json.dumps(obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import argparse
import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _load_prompt_yaml(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise SystemExit(f"Prompt yaml not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read prompt yaml: {path}") from e

    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid prompt yaml format: {path}")

    sp = payload.get("system_prompt")
    up = payload.get("user_prompt")
    if not isinstance(sp, str) or not sp.strip():
        raise SystemExit(f"Missing system_prompt in: {path}")
    if not isinstance(up, str) or not up.strip():
        raise SystemExit(f"Missing user_prompt in: {path}")

    return {"system_prompt": sp, "user_prompt": up}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Input jsonl not found: {path}")
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _extract_image_data_url(rec: Dict[str, Any]) -> str:
    b64 = rec.get("image_base64")
    if not isinstance(b64, str) or not b64.strip():
        return ""
    return f"data:image/png;base64,{b64.strip()}"


def _decode_image_pil(rec: Dict[str, Any]) -> Any:
    b64 = rec.get("image_base64")
    if not isinstance(b64, str) or not b64.strip():
        return None
    try:
        from PIL import Image
    except Exception as e:
        raise SystemExit("PIL is required for --local-vlm-model. Install pillow.") from e

    try:
        raw = base64.b64decode(b64.strip(), validate=False)
        img = Image.open(io.BytesIO(raw))
        return img.convert("RGB")
    except Exception:
        return None


def _format_user_prompt(template: str, *, ticker: str, asof: str) -> str:
    try:
        return template.format(ticker=str(ticker), asof=str(asof))
    except Exception:
        return template


def _call_vlm(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_data_url: str,
    max_tokens: int,
    temperature: float,
) -> str:
    # Uses OpenAI-compatible Chat Completions with image_url content.
    resp = client.chat.completions.create(
        model=str(model),
        messages=[
            {"role": "system", "content": str(system_prompt)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": str(user_prompt)},
                    {"type": "image_url", "image_url": {"url": str(image_data_url)}},
                ],
            },
        ],
        max_tokens=int(max_tokens),
        temperature=float(temperature),
    )

    try:
        return str(resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _call_local_vlm(
    *,
    model_obj: Any,
    processor: Any,
    system_prompt: str,
    user_prompt: str,
    image: Any,
    max_tokens: int,
    temperature: float,
) -> str:
    import torch

    prompt = ""

    messages = [
        {"role": "system", "content": str(system_prompt)},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": str(user_prompt)},
            ],
        },
    ]

    # Qwen2.5-VL official utility (optional) to process vision info.
    # If not installed, we fall back to passing PIL images directly.
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore
    except Exception:
        process_vision_info = None

    if hasattr(processor, "apply_chat_template"):
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        else:
            inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
    else:
        inputs = processor(text=[str(user_prompt)], images=[image], padding=True, return_tensors="pt")

    dev = getattr(model_obj, "device", None)
    if dev is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}

    gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(max_tokens)}
    if float(temperature) > 0:
        gen_kwargs.update({"do_sample": True, "temperature": float(temperature)})
    else:
        gen_kwargs.update({"do_sample": False})

    with torch.inference_mode():
        out_ids = model_obj.generate(**inputs, **gen_kwargs)

    txt = processor.batch_decode(out_ids, skip_special_tokens=True)
    if not txt:
        return ""

    out = str(txt[0] or "")
    if isinstance(prompt, str) and prompt and out.startswith(prompt):
        out = out[len(prompt) :]
    return out.strip()


def _dry_run_response(*, ticker: str, asof: str) -> Dict[str, Any]:
    return {
        "signal": "NEUTRAL",
        "confidence": 0.0,
        "reasoning": f"dry_run: no VLM call executed for {ticker} {asof}",
    }


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None

    s = text.strip()

    # Try direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m_code = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.IGNORECASE | re.DOTALL)
    if m_code:
        try:
            obj = json.loads(m_code.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    for m in re.finditer(r"\{.*?\}", s, flags=re.DOTALL):
        sub = m.group(0)
        try:
            obj = json.loads(sub)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    m_signal = re.search(r"\bsignal\b\s*[:：]\s*[*_`~\s]*([A-Za-z_]+)", s, flags=re.IGNORECASE)
    m_conf = re.search(
        r"\bconfidence\b\s*[:：]\s*[*_`~\s]*([0-9]+(?:\.[0-9]+)?)",
        s,
        flags=re.IGNORECASE,
    )
    m_reason = re.search(r"\breasoning\b\s*[:：]\s*(.+)", s, flags=re.IGNORECASE | re.DOTALL)
    if m_signal:
        signal = str(m_signal.group(1) or "").strip().upper()
        out: Dict[str, Any] = {"signal": signal}
        if m_conf:
            try:
                out["confidence"] = float(m_conf.group(1))
            except Exception:
                pass
        if m_reason:
            out["reasoning"] = str(m_reason.group(1) or "").strip()
        return out

    return None


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input-jsonl",
        default="",
        help="charts_base64.jsonl path (default: data/charts/<asof>/charts_base64.jsonl if --asof provided)",
    )
    ap.add_argument(
        "--input-file",
        dest="input_jsonl",
        default=None,
        help="Alias of --input-jsonl",
    )
    ap.add_argument("--asof", default="", help="Optional as-of date YYYY-MM-DD used to build default input path")

    ap.add_argument("--out-jsonl", default="results/phase21_chartist/chart_signals.jsonl")

    ap.add_argument("--prompt-yaml", default="configs/prompts/chartist_prompt.yaml")

    ap.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", "Qwen2-VL-7B-Instruct"))

    ap.add_argument(
        "--local-vlm-model",
        default="",
        help="If set, run fully local Transformers VLM (e.g. Qwen/Qwen2.5-VL-7B-Instruct) instead of calling --api-base",
    )
    ap.add_argument(
        "--local-device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for local VLM backend",
    )
    ap.add_argument(
        "--local-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="dtype for local VLM backend",
    )
    ap.add_argument(
        "--local-min-image-pixels",
        type=int,
        default=0,
        help="Optional min_pixels for Qwen VL processor (0 = leave default)",
    )
    ap.add_argument(
        "--local-max-image-pixels",
        type=int,
        default=0,
        help="Optional max_pixels for Qwen VL processor (0 = leave default)",
    )
    ap.add_argument("--progress-every", type=int, default=50)

    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)

    ap.add_argument("--dry-run", action="store_true", default=False)

    ap.add_argument("--limit", type=int, default=0, help="Optional limit N records")
    ap.add_argument("--verbose", action="store_true", default=False)

    args = ap.parse_args()

    input_path = str(args.input_jsonl or "").strip()
    if not input_path and str(args.asof or "").strip():
        input_path = str(Path("data") / "charts" / str(args.asof).strip() / "charts_base64.jsonl")
    if not input_path:
        raise SystemExit("Missing --input-jsonl (or provide --asof to use default path)")

    prompt_cfg = _load_prompt_yaml(Path(str(args.prompt_yaml)))
    system_prompt = prompt_cfg["system_prompt"]
    user_template = prompt_cfg["user_prompt"]

    items = _read_jsonl(Path(input_path))
    if int(args.limit) > 0:
        items = items[: int(args.limit)]

    client: Any = None
    local_model_obj: Any = None
    local_processor: Any = None

    if not bool(args.dry_run) and str(args.local_vlm_model or "").strip():
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except Exception:
            Qwen2_5_VLForConditionalGeneration = None

        device = str(args.local_device)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype_s = str(args.local_dtype)
        dtype = None
        if dtype_s == "float16":
            dtype = torch.float16
        elif dtype_s == "bfloat16":
            dtype = torch.bfloat16
        elif dtype_s == "float32":
            dtype = torch.float32

        mpath = str(args.local_vlm_model).strip()
        local_processor = AutoProcessor.from_pretrained(mpath, trust_remote_code=True)
        if int(args.local_min_image_pixels) > 0 and hasattr(local_processor, "min_pixels"):
            try:
                local_processor.min_pixels = int(args.local_min_image_pixels)
            except Exception:
                pass
        if int(args.local_max_image_pixels) > 0 and hasattr(local_processor, "max_pixels"):
            try:
                local_processor.max_pixels = int(args.local_max_image_pixels)
            except Exception:
                pass
        if Qwen2_5_VLForConditionalGeneration is None:
            raise SystemExit("Your transformers build does not include Qwen2_5_VLForConditionalGeneration")

        load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype

        if device == "cuda":
            load_kwargs["device_map"] = "auto"

        local_model_obj = Qwen2_5_VLForConditionalGeneration.from_pretrained(mpath, **load_kwargs)
        if device != "cuda":
            local_model_obj = local_model_obj.to(device)
        local_model_obj.eval()

    elif not bool(args.dry_run):
        try:
            from openai import OpenAI
        except Exception as e:
            raise SystemExit("openai python client is required. Install it via requirements.txt") from e

        client = OpenAI(base_url=str(args.api_base), api_key=str(args.api_key))

    out_path = Path(str(args.out_jsonl))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = 0
    failed = 0

    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(items):
            ticker = str(rec.get("ticker") or "").upper().strip()
            asof = str(rec.get("asof") or str(args.asof or "")).strip()
            img_url = _extract_image_data_url(rec)

            if not ticker or not img_url:
                failed += 1
                continue

            user_prompt = _format_user_prompt(user_template, ticker=ticker, asof=asof)

            if bool(args.dry_run):
                parsed = _dry_run_response(ticker=ticker, asof=asof)
                text = json.dumps(parsed, ensure_ascii=False)
            else:
                try:
                    if local_model_obj is not None and local_processor is not None:
                        img = _decode_image_pil(rec)
                        if img is None:
                            raise RuntimeError("failed to decode image_base64")
                        text = _call_local_vlm(
                            model_obj=local_model_obj,
                            processor=local_processor,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            image=img,
                            max_tokens=int(args.max_tokens),
                            temperature=float(args.temperature),
                        )
                    else:
                        text = _call_vlm(
                            client=client,
                            model=str(args.model),
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            image_data_url=img_url,
                            max_tokens=int(args.max_tokens),
                            temperature=float(args.temperature),
                        )
                except Exception as e:
                    failed += 1
                    if bool(args.verbose):
                        f.write(
                            json.dumps(
                                {
                                    "ticker": ticker,
                                    "asof": asof,
                                    "ok": False,
                                    "error": repr(e),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    continue

                parsed = _try_parse_json(text)
                if parsed is None:
                    failed += 1
                    if bool(args.verbose):
                        f.write(
                            json.dumps(
                                {
                                    "ticker": ticker,
                                    "asof": asof,
                                    "ok": False,
                                    "raw": text,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    continue

            out_rec: Dict[str, Any] = {
                "ticker": ticker,
                "asof": asof,
                "ok": True,
                "model": str(args.local_vlm_model).strip() if local_model_obj is not None else str(args.model),
                "dry_run": bool(args.dry_run),
                "signal": parsed.get("signal"),
                "confidence": parsed.get("confidence"),
                "reasoning": parsed.get("reasoning"),
                "raw": text,
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            f.flush()
            ok += 1

            pe = int(args.progress_every)
            if pe > 0 and ((i + 1) % pe == 0):
                dt = max(1e-6, time.time() - t0)
                rate = (ok + failed) / dt
                print(json.dumps({"i": i + 1, "ok": ok, "failed": failed, "rate": rate}, ensure_ascii=False))

    print(json.dumps({"input": input_path, "out": str(out_path.as_posix()), "ok": ok, "failed": failed}, ensure_ascii=False))


if __name__ == "__main__":
    main()

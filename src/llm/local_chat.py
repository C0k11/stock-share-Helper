"""
Local LLM Chat - Direct inference without Ollama
Uses transformers + bitsandbytes for efficient local inference
"""

import threading
from typing import Optional, List, Dict, Any
from pathlib import Path

# Lazy loading to avoid slow startup
_model = None
_tokenizer = None
_lock = threading.Lock()
_gen_lock = threading.Lock()
_loaded_signature = None

# Default model (can be changed in config)
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_CACHE_DIR = Path("D:/Project/ml_cache/models")


def _load_model(
    model_name: str = DEFAULT_MODEL,
    use_4bit: bool = False,
    use_8bit: bool = True,
    adapter_path: Optional[str] = None,
):
    """Load model with optional 4-bit or 8-bit quantization"""
    global _model, _tokenizer, _loaded_signature

    if bool(use_4bit) and bool(use_8bit):
        use_8bit = False

    ap = str(adapter_path or "").strip()
    if ap:
        try:
            ap = str(Path(ap).resolve())
        except Exception:
            ap = str(adapter_path or "").strip()

    sig = (str(model_name or ""), bool(use_4bit), bool(use_8bit), ap)
    
    if _model is not None and _loaded_signature == sig:
        return _model, _tokenizer
    
    with _lock:
        if _model is not None and _loaded_signature == sig:
            return _model, _tokenizer

        if _model is not None and _loaded_signature != sig:
            try:
                del _model
            except Exception:
                pass
            try:
                del _tokenizer
            except Exception:
                pass
            _model = None
            _tokenizer = None
            _loaded_signature = None
        
        quant_mode = "4-bit" if use_4bit else ("8-bit" if use_8bit else "fp16")
        print(f"[LocalLLM] Loading {model_name} ({quant_mode})...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
        except ImportError:
            raise ImportError(
                "需要安装: pip install transformers torch bitsandbytes accelerate"
            )
        
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(MODEL_CACHE_DIR),
            trust_remote_code=True,
        )
        
        max_memory = None
        try:
            frac_raw = str(__import__("os").environ.get("SECRETARY_MAX_MEMORY_FRAC", "") or "").strip()
            if frac_raw:
                frac = float(frac_raw)
                if frac > 0 and frac <= 1.0 and torch.cuda.is_available():
                    total = int(torch.cuda.get_device_properties(0).total_memory)
                    gib = int((total * frac) / (1024 ** 3))
                    gib = max(1, gib - 1)
                    max_memory = {0: f"{gib}GiB"}
        except Exception:
            max_memory = None

        # Model with quantization options
        if use_4bit:
            # 4-bit quantization - saves more VRAM
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            _model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(MODEL_CACHE_DIR),
                quantization_config=bnb_config,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
            )
        elif use_8bit:
            # 8-bit quantization - good balance of speed and quality
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            _model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(MODEL_CACHE_DIR),
                quantization_config=bnb_config,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
            )
        else:
            # FP16 - full quality
            _model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(MODEL_CACHE_DIR),
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
            )
        
        print(f"[LocalLLM] Model loaded: {model_name} ({quant_mode})")

        if ap:
            try:
                from peft import PeftModel

                p = Path(ap)
                if p.exists():
                    print(f"[LocalLLM] Loading LoRA from: {p} ...")
                    _model = PeftModel.from_pretrained(_model, str(p), is_trainable=False)
                    print(f"[LocalLLM] LoRA model active: {type(_model)}")
                    try:
                        _model.eval()
                    except Exception:
                        pass
                    print(f"[LocalLLM] LoRA loaded: {p}")
                else:
                    print(f"[LocalLLM] Warning: LoRA path does not exist: {p}")
            except Exception as e:
                import traceback
                print(f"[LocalLLM] LoRA load CRITICAL ERROR: {e}")
                traceback.print_exc()

        _loaded_signature = sig
        return _model, _tokenizer


def chat(
    messages: List[Dict[str, str]],
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    use_4bit: bool = False,
    use_8bit: bool = True,
    adapter_path: Optional[str] = None,
) -> str:
    """
    Generate chat response using local model
    
    Args:
        messages: List of {"role": "system/user/assistant", "content": "..."}
        model_name: HuggingFace model name
        temperature: Sampling temperature
        max_new_tokens: Max tokens to generate
        use_4bit: Use 4-bit quantization (saves VRAM)
        use_8bit: Use 8-bit quantization (balance speed/quality)
    
    Returns:
        Generated response text
    """
    import torch
    import re
    
    model, tokenizer = _load_model(model_name, use_4bit=use_4bit, use_8bit=use_8bit, adapter_path=adapter_path)
    
    # Format messages for Qwen chat template
    # Disable Qwen3 thinking mode by adding /no_think suffix
    if messages and messages[-1].get("role") == "user":
        messages[-1]["content"] = messages[-1]["content"] + " /no_think"
    
    # NOTE: GPU VRAM can spike badly if multiple threads call generate concurrently.
    # Serialize local inference at process-level to keep the desktop UX stable.
    with _gen_lock:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

    # Strip Qwen3 thinking tags if present (handle incomplete tags too)
    raw = str(response or "")
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)  # Handle unclosed tag
    cleaned = cleaned.replace("</think>", "").strip()

    # If model output is only thinking (or becomes empty after cleaning), keep the raw text
    # so callers won't treat it as "no valid content".
    if not cleaned:
        raw2 = raw.strip()
        if not raw2:
            print("[LocalLLM] Warning: empty decoded response")
        return raw2

    return cleaned


def simple_chat(
    user_message: str,
    system_prompt: str = "You are a helpful assistant.",
    **kwargs
) -> str:
    """Simple wrapper for single-turn chat"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return chat(messages, **kwargs)


def unload_model():
    """Unload model to free VRAM"""
    global _model, _tokenizer
    with _lock:
        if _model is not None:
            del _model
            _model = None
        if _tokenizer is not None:
            del _tokenizer
            _tokenizer = None

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        print("[LocalLLM] Model unloaded")
    except Exception:
        pass


def status() -> Dict[str, Any]:
    global _model, _tokenizer, _loaded_signature
    out: Dict[str, Any] = {
        "loaded": bool(_model is not None),
        "signature": _loaded_signature,
    }
    try:
        out["device"] = str(getattr(_model, "device", "")) if _model is not None else ""
    except Exception:
        out["device"] = ""
    try:
        out["model_type"] = str(type(_model)) if _model is not None else ""
    except Exception:
        out["model_type"] = ""
    return out


# Singleton instance for easy access
class LocalChatModel:
    """Singleton wrapper for local chat model"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL, use_4bit: bool = True):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self._loaded = False
        self.adapter_path: str = ""
    
    def load(self):
        """Pre-load model"""
        if not self._loaded:
            _load_model(self.model_name, self.use_4bit, adapter_path=self.adapter_path)
            self._loaded = True
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ) -> str:
        return chat(
            messages,
            model_name=self.model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_4bit=self.use_4bit,
            adapter_path=self.adapter_path,
        )
    
    def simple_chat(self, user_message: str, system_prompt: str = "", **kwargs) -> str:
        return simple_chat(user_message, system_prompt, **kwargs)
    
    def unload(self):
        unload_model()
        self._loaded = False


if __name__ == "__main__":
    # Test
    print("Testing local chat...")
    response = simple_chat(
        "你好，介绍一下自己",
        system_prompt="你是Mari，一个可爱的AI助手。",
        max_new_tokens=100,
    )
    print(f"Response: {response}")

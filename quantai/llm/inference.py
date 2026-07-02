"""quantai.llm.inference —— 本地 LLM 推理（Qwen3-8B，lazy torch）。

忠实迁移 `src/llm/local_chat.py`：
- 量化加载（4bit/8bit/fp16）+ 可选 LoRA adapter；
- chat：套用 tokenizer chat 模板 -> 生成 -> 切片解码新 token -> 清洗 Qwen3 `<think>`；
- 进程级 `_gen_lock` 串行化生成（多线程并发 generate 会让显存暴涨）。

设计：`torch/transformers/peft` 全部在方法内**懒导入**，因此 `import quantai.llm.inference`
不需要 GPU 依赖。编排逻辑（adapter 热切换、解码切片、max_time 回退）抽成可注入/可单测的接缝：
- `LocalLLM.attach(model, tokenizer)`：依赖注入，绕过 torch 加载，供测试与高级用法；
- `decode_new_tokens(...)` / `_safe_generate(...)`：模块级纯函数，可用假 model/tokenizer 单测。
"""

from __future__ import annotations

import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .prompts import build_messages, strip_think_tags, with_no_think_suffix

Message = Dict[str, str]


# --------------------------------------------------------------------------- #
# 模块级纯函数（可用假 tokenizer/model 单测，不需要 torch）
# --------------------------------------------------------------------------- #
def decode_new_tokens(tokenizer: Any, output_ids: Any, input_len: int) -> str:
    """只解码新生成的 token（切掉 prompt 部分），忠实迁移旧码的解码切片。

    旧码：`tok.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)`。
    """
    new_ids = output_ids[0][input_len:]
    return str(tokenizer.decode(new_ids, skip_special_tokens=True) or "").strip()


def _safe_generate(model: Any, inputs: Any, gen_kwargs: Dict[str, Any]) -> Any:
    """调用 model.generate；若该版本 transformers 不支持 `max_time` 则去掉重试。

    旧码在 `TypeError` 含 "max_time" 时回退；这里抽成可单测的函数。
    """
    try:
        return model.generate(**inputs, **gen_kwargs)
    except TypeError as exc:
        if "max_time" in str(exc):
            retry = dict(gen_kwargs)
            retry.pop("max_time", None)
            return model.generate(**inputs, **retry)
        raise


class LocalLLM:
    """本地 Qwen 推理封装。重活懒加载；多线程生成由 `_gen_lock` 串行化。"""

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-8B",
        cache_dir: str = "models/hf_cache",
        quantization: str = "8bit",
        device: str = "cuda",
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        max_context: int = 0,
        gen_max_time_sec: float = 12.0,
        default_adapter: str = "scalper",
    ) -> None:
        self.model_name = str(model_name)
        self.cache_dir = Path(cache_dir)
        self.quantization = str(quantization)
        self.device = str(device)
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.max_context = int(max_context)
        self.gen_max_time_sec = float(gen_max_time_sec)
        self.default_adapter = str(default_adapter)

        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded_signature: Optional[tuple] = None
        self._adapters_loaded: set[str] = set()
        self._lock = threading.Lock()       # 加载锁
        self._gen_lock = threading.Lock()   # 生成串行锁

    @classmethod
    def from_config(cls, cfg: Any) -> "LocalLLM":
        """从 `quantai.config.LLMConfig` 构造。"""
        inf = cfg.inference
        return cls(
            model_name=cfg.model_name,
            cache_dir=cfg.cache_dir,
            quantization=inf.quantization,
            device=inf.device,
            temperature=inf.temperature,
            max_new_tokens=inf.max_new_tokens,
            max_context=inf.max_context,
            gen_max_time_sec=inf.gen_max_time_sec,
            default_adapter=inf.default_adapter,
        )

    # ----------------------------------------------------------------- #
    # 状态 / 依赖注入
    # ----------------------------------------------------------------- #
    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def attach(self, model: Any, tokenizer: Any, adapters: Iterable[str] = ()) -> "LocalLLM":
        """注入已构造好的 model/tokenizer（绕过 torch 加载）。供测试与高级用法。"""
        self._model = model
        self._tokenizer = tokenizer
        self._adapters_loaded = set(adapters)
        self._loaded_signature = ("attached",)
        return self

    def status(self) -> Dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "model_name": self.model_name,
            "quantization": self.quantization,
            "adapters": sorted(self._adapters_loaded),
            "device": str(getattr(self._model, "device", "")) if self.is_loaded else "",
        }

    # ----------------------------------------------------------------- #
    # 重活：加载 / 卸载（lazy torch）
    # ----------------------------------------------------------------- #
    def load(self, adapter_path: Optional[str] = None) -> None:
        """加载基座模型（按 quantization）+ 可选单个 LoRA adapter。幂等：同签名不重载。"""
        ap = str(adapter_path or "").strip()
        if ap:
            try:
                ap = str(Path(ap).resolve())
            except Exception:
                pass
        signature = (self.model_name, self.quantization, ap)
        if self.is_loaded and self._loaded_signature == signature:
            return

        with self._lock:
            if self.is_loaded and self._loaded_signature == signature:
                return

            import torch  # noqa: F401  (lazy)
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache = str(self.cache_dir)

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=cache, trust_remote_code=True
            )

            model_kwargs: Dict[str, Any] = {
                "cache_dir": cache,
                "device_map": "auto",
                "trust_remote_code": True,
            }
            model_kwargs.update(self._quantization_kwargs())
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

            if ap and Path(ap).exists():
                from peft import PeftModel

                self._model = PeftModel.from_pretrained(self._model, ap, is_trainable=False)
                try:
                    self._model.eval()
                except Exception:
                    pass

            self._loaded_signature = signature

    def _quantization_kwargs(self) -> Dict[str, Any]:
        """按 quantization 返回 from_pretrained 的量化相关 kwargs（lazy torch/bnb）。"""
        import torch
        from transformers import BitsAndBytesConfig

        if self.quantization == "4bit":
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            }
        if self.quantization == "8bit":
            return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
        return {"torch_dtype": torch.float16}  # fp16

    def unload(self) -> None:
        """释放模型与显存。"""
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._loaded_signature = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ----------------------------------------------------------------- #
    # 推理
    # ----------------------------------------------------------------- #
    def chat(
        self,
        messages: List[Message],
        *,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        adapter: Optional[str] = None,
        no_think: bool = True,
    ) -> str:
        """多轮 chat：套模板 -> 生成 -> 切片解码 -> 清洗 `<think>`。返回纯文本。

        `adapter` 命中已加载的 MoE 专家时热切换；生成后复位到 `default_adapter`。
        生成全程持 `_gen_lock`，避免并发 generate 显存暴涨。
        """
        if not self.is_loaded:
            self.load()
        model, tok = self._model, self._tokenizer
        if model is None or tok is None:
            return ""

        msgs = with_no_think_suffix(messages) if no_think else [dict(m) for m in messages]
        target = self._resolve_adapter(adapter)

        with self._gen_lock:
            if target:
                model.set_adapter(target)

            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            tk_kwargs: Dict[str, Any] = {"return_tensors": "pt"}
            if self.max_context > 0:
                tk_kwargs.update(truncation=True, max_length=self.max_context)
            inputs = tok(text, **tk_kwargs).to(getattr(model, "device", self.device))
            input_len = int(inputs["input_ids"].shape[1])

            gen_kwargs = self._generation_kwargs(temperature, max_new_tokens, tok)
            try:
                import torch

                ctx = torch.no_grad()
            except Exception:
                ctx = nullcontext()
            with ctx:
                output_ids = _safe_generate(model, inputs, gen_kwargs)

            response = decode_new_tokens(tok, output_ids, input_len)
            self._restore_default_adapter(model)

        return strip_think_tags(response)

    def generate(self, user: str, system: Optional[str] = None, **kwargs: Any) -> str:
        """单轮便捷封装：build_messages + chat。"""
        return self.chat(build_messages(user, system), **kwargs)

    # ----------------------------------------------------------------- #
    # 内部
    # ----------------------------------------------------------------- #
    def _resolve_adapter(self, adapter: Optional[str]) -> str:
        a = str(adapter or "").strip()
        return a if a and a in self._adapters_loaded else ""

    def _restore_default_adapter(self, model: Any) -> None:
        if not self._adapters_loaded:
            return
        default = (
            self.default_adapter
            if self.default_adapter in self._adapters_loaded
            else next(iter(self._adapters_loaded))
        )
        try:
            model.set_adapter(default)
        except Exception:
            pass

    def _generation_kwargs(
        self, temperature: Optional[float], max_new_tokens: Optional[int], tok: Any
    ) -> Dict[str, Any]:
        t = self.temperature if temperature is None else float(temperature)
        mn = int(max_new_tokens) if max_new_tokens else self.max_new_tokens
        kwargs: Dict[str, Any] = {
            "max_new_tokens": int(mn),
            "temperature": float(t),
            "do_sample": float(t) > 0,
            "top_p": 0.9,
            "pad_token_id": getattr(tok, "eos_token_id", None),
        }
        if self.gen_max_time_sec > 0:
            kwargs["max_time"] = float(self.gen_max_time_sec)
        return kwargs

"""Chartist VLM 后端 —— mplfinance 渲染 K 线图 + Qwen2.5-VL 推理（B-3）。

忠实迁移自 `strategy.py::_render_chartist_image` / `_maybe_load_chartist_vlm` /
`_chartist_overlay` 的推理段。与旧版的差异（让它能复用 + 可测）：

- **数据解耦**：渲染不再读 `self.price_history`，而是通过注入的 `bars_provider(ticker)`
  取 OHLCV bars（list[dict] 或 DataFrame）。集成层（live/backtest）负责喂价。
- **重活全懒加载**：torch / transformers / peft / mplfinance / PIL / qwen_vl_utils 全在方法内
  懒导入；`import` 本模块零重依赖。缺权重 / 缺 GPU / 缺库 -> `analyze()` 返回 None ->
  `ChartistOverlay` 打 0 分（不影响主管线）。
- **可注入接缝**：`attach(model, processor)` 绕过加载供测试；`render_candles` / `build_messages`
  / `parse` 是可单测的纯函数级接缝。

打分逻辑不在这里（在 `chartist.ChartistOverlay.score_from_vlm`）。本类只产出
`{"signal","confidence","reasoning"}`，符合 `ChartistOverlay.assess(analyzer=...)` 的协议。
"""
from __future__ import annotations

import contextlib
import io
from typing import Any, Callable, Dict, List, Optional, Union

from quantai.config.schema import ChartistConfig
from quantai.llm.json_utils import extract_json_text, repair_and_parse_json

BarsProvider = Callable[[str], Union[List[Dict[str, Any]], Any]]

_DEFAULT_SYSTEM = (
    "Analyze the candlestick chart image. Return only JSON {signal,confidence,reasoning} "
    "where signal is one of BULLISH/BEARISH/NEUTRAL and confidence is a float in [0,1]."
)
_DEFAULT_USER = "Analyze this chart for ticker={ticker} asof={asof}. Return only the JSON object."


def render_candles(bars: Union[List[Dict[str, Any]], Any], lookback: int = 60) -> Any:
    """把 OHLCV bars 渲染成蜡烛图 PIL.Image（忠实迁移 `_render_chartist_image`）。

    缺 mplfinance/PIL/pandas、数据不足或渲染失败 -> None。
    """
    try:
        import mplfinance as mpf
        import pandas as pd
        from PIL import Image
    except Exception:
        return None

    try:
        if isinstance(bars, list):
            rows = list(bars)[-int(lookback):]
            if len(rows) < 5:
                return None
            df = pd.DataFrame(rows)
        else:
            df = bars.tail(int(lookback)) if hasattr(bars, "tail") else pd.DataFrame(bars)
    except Exception:
        return None

    if df is None or df.empty or "time" not in df.columns:
        return None

    try:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).set_index("time")
    except Exception:
        return None

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return None

    try:
        df = df[~df.index.duplicated(keep="last")].sort_index()
    except Exception:
        pass

    n = int(len(df))
    if n >= 200:
        mav: Optional[tuple] = (20, 50, 200)
    elif n >= 50:
        mav = (20, 50)
    elif n >= 20:
        mav = (20,)
    else:
        mav = None

    buf = io.BytesIO()
    try:
        plot_kwargs: Dict[str, Any] = {
            "type": "candle",
            "volume": "volume" in df.columns,
            "style": "yahoo",
            "savefig": dict(fname=buf, dpi=110, bbox_inches="tight"),
        }
        if mav is not None:
            plot_kwargs["mav"] = mav
        mpf.plot(df, **plot_kwargs)
    except Exception:
        return None

    try:
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    except Exception:
        return None


class QwenVLChartist:
    """Qwen2.5-VL chartist：渲染 K 线 -> VLM 推理 -> {signal,confidence,reasoning}。"""

    def __init__(
        self,
        *,
        vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        load_4bit: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        min_image_pixels: int = 0,
        max_image_pixels: int = 0,
        adapter: str = "",
        lookback: int = 60,
        bars_provider: Optional[BarsProvider] = None,
        system_prompt: str = "",
        user_prompt_template: str = "",
    ) -> None:
        self.vlm_model = str(vlm_model)
        self.load_4bit = bool(load_4bit)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.min_image_pixels = int(min_image_pixels)
        self.max_image_pixels = int(max_image_pixels)
        self.adapter = str(adapter or "")
        self.lookback = int(lookback)
        self.bars_provider = bars_provider
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM
        self.user_prompt_template = user_prompt_template or _DEFAULT_USER

        self._model: Any = None
        self._processor: Any = None
        self._error: str = ""

    @classmethod
    def from_config(
        cls, cfg: ChartistConfig, *, bars_provider: Optional[BarsProvider] = None
    ) -> "QwenVLChartist":
        return cls(
            vlm_model=cfg.vlm_model,
            load_4bit=cfg.load_4bit,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            min_image_pixels=cfg.min_image_pixels,
            max_image_pixels=cfg.max_image_pixels,
            adapter=cfg.adapter,
            lookback=cfg.lookback,
            bars_provider=bars_provider,
        )

    # --- 加载（lazy torch/transformers） --- #
    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def attach(self, model: Any, processor: Any) -> "QwenVLChartist":
        """注入已构造的 model/processor（绕过加载）。供测试与高级用法。"""
        self._model = model
        self._processor = processor
        self._error = ""
        return self

    def load(self) -> None:
        """忠实迁移 `_maybe_load_chartist_vlm`：AutoProcessor + Qwen2.5-VL(可 4bit) + 可选 LoRA。"""
        if self.is_loaded:
            return
        if not self.vlm_model:
            self._error = "missing vlm_model"
            return
        try:
            from transformers import AutoProcessor
        except Exception as exc:  # pragma: no cover - 依赖缺失路径
            self._error = f"transformers missing: {exc}"
            return
        try:
            import torch
            from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

            qcfg = None
            if self.load_4bit:
                with contextlib.suppress(Exception):
                    qcfg = BitsAndBytesConfig(load_in_4bit=True)

            proc = AutoProcessor.from_pretrained(self.vlm_model, trust_remote_code=True)
            if self.min_image_pixels > 0 and hasattr(proc, "min_pixels"):
                proc.min_pixels = self.min_image_pixels
            if self.max_image_pixels > 0 and hasattr(proc, "max_pixels"):
                proc.max_pixels = self.max_image_pixels

            load_kwargs: Dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
            if qcfg is not None:
                load_kwargs["quantization_config"] = qcfg
            else:
                load_kwargs["torch_dtype"] = (
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                )

            model_obj = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.vlm_model, **load_kwargs
            )
            model_obj.eval()

            if self.adapter:
                with contextlib.suppress(Exception):
                    from peft import PeftModel

                    model_obj = PeftModel.from_pretrained(
                        model_obj, self.adapter, is_trainable=False
                    )
                    model_obj.eval()

            self._processor = proc
            self._model = model_obj
            self._error = ""
        except Exception as exc:  # pragma: no cover - 重加载路径
            self._error = str(exc)
            self._model = None
            self._processor = None

    # --- 纯逻辑接缝 --- #
    def build_messages(self, image: Any, ticker: str, asof: str = "") -> List[Dict[str, Any]]:
        try:
            user = self.user_prompt_template.format(
                ticker=str(ticker).upper(), asof=str(asof)
            )
        except Exception:
            user = self.user_prompt_template
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user},
                ],
            },
        ]

    def parse(self, raw: str) -> Optional[Dict[str, Any]]:
        text = extract_json_text(str(raw or ""))
        if text is None:
            return None
        obj = repair_and_parse_json(text)
        if not isinstance(obj, dict):
            return None
        try:
            conf = float(obj.get("confidence") or 0.0)
        except Exception:
            conf = 0.0
        return {
            "signal": str(obj.get("signal") or "").strip().upper(),
            "confidence": conf,
            "reasoning": str(
                obj.get("reasoning") or obj.get("analysis") or obj.get("reason") or ""
            ).strip(),
        }

    # --- 推理 --- #
    def render(self, ticker: str) -> Any:
        if self.bars_provider is None:
            return None
        try:
            bars = self.bars_provider(str(ticker).upper())
        except Exception:
            return None
        return render_candles(bars, lookback=self.lookback)

    def infer_from_image(self, image: Any, ticker: str = "", asof: str = "") -> str:
        """忠实迁移 `_chartist_overlay` 推理段：套 VLM 模板 -> generate -> 解码。"""
        if not self.is_loaded or image is None:
            return ""
        proc = self._processor
        model_obj = self._model
        messages = self.build_messages(image, ticker, asof)

        try:
            from qwen_vl_utils import process_vision_info
        except Exception:
            process_vision_info = None

        try:
            prompt = ""
            if hasattr(proc, "apply_chat_template"):
                prompt = proc.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if process_vision_info is not None:
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = proc(
                        text=[prompt],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                else:
                    inputs = proc(
                        text=[prompt], images=[image], padding=True, return_tensors="pt"
                    )
            else:
                user = self.user_prompt_template.format(
                    ticker=str(ticker).upper(), asof=str(asof)
                )
                inputs = proc(text=[user], images=[image], padding=True, return_tensors="pt")

            dev = getattr(model_obj, "device", None)
            if dev is not None and isinstance(inputs, dict):
                inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}

            gen_kwargs: Dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
            if self.temperature > 0:
                gen_kwargs.update({"do_sample": True, "temperature": self.temperature})
            else:
                gen_kwargs["do_sample"] = False

            try:
                import torch

                gen_ctx: Any = torch.inference_mode()
            except Exception:
                gen_ctx = contextlib.nullcontext()

            with gen_ctx:
                out_ids = model_obj.generate(**inputs, **gen_kwargs)

            txts = proc.batch_decode(out_ids, skip_special_tokens=True)
            out = str(txts[0] or "").strip() if txts else ""
            if prompt and out.startswith(prompt):
                out = out[len(prompt):].strip()
            return out
        except Exception:
            return ""

    def analyze(self, ticker: str, asof: str = "") -> Optional[Dict[str, Any]]:
        """渲染 -> 推理 -> 解析；任何环节失败/未就绪 -> None（上层打 0 分）。"""
        if not self.is_loaded:
            with contextlib.suppress(Exception):
                self.load()
        if not self.is_loaded:
            return None
        image = self.render(ticker)
        if image is None:
            return None
        raw = self.infer_from_image(image, ticker, asof)
        return self.parse(raw)

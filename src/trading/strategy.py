# src/trading/strategy.py
"""
Phase 3.4: Multi-Agent Strategy with Real Model Integration

Architecture:
- Core Engine: Planner (SFT) + RL Gatekeeper
- Expert System: MoE Router -> {Scalper | Analyst (DPO)}
- Overlays: Chartist (VLM) + Macro Governor + System 2 Debate
- Execution: Simulator (Passive 40bps)
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .event import Event, EventType
from src.learning.recorder import recorder as evolution_recorder
from src.agent.gatekeeper import Gatekeeper
from src.agent.planner import Planner
from src.utils.llm_tools import extract_json_text, repair_and_parse_json

# Default model paths
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SCALPER_ADAPTER = "models/trader_stock_v1_1_tech_plus_news/lora_weights"
DEFAULT_ANALYST_ADAPTER = "models/trader_v3_dpo_analyst"
DEFAULT_CHARTIST_VLM_MODEL = "Qwen2.5-VL-3B-Instruct-4bit"

class MultiAgentStrategy:
    """
    Multi-Agent Strategy with real model inference
    
    Components:
    - Planner: Decides market regime (aggressive/defensive/cash_preservation)
    - Gatekeeper: RL-based trade filtering (Q-value threshold)
    - MoE Router: Routes to Scalper or Analyst based on market features
    - System 2 Debate: Critic + Judge for high-stakes decisions
    - Chartist Overlay: VLM-based chart pattern analysis
    - Macro Governor: Global risk scoring
    """

    def __init__(
        self,
        engine: Any,
        tickers: List[str] = ["NVDA"],
        *,
        load_models: bool = False,
        base_model: str = DEFAULT_BASE_MODEL,
        moe_scalper: str = DEFAULT_SCALPER_ADAPTER,
        moe_analyst: str = DEFAULT_ANALYST_ADAPTER,
        moe_secretary: str = "",
        moe_system2: str = "",
        moe_news: str = "",
        load_4bit: bool = True,
        llm_max_context: Optional[int] = None,
        llm_max_new_tokens: int = 256,
        chartist_vlm_cfg: Optional[Dict[str, Any]] = None,
        news_cfg: Optional[Dict[str, Any]] = None,
        perf_cfg: Optional[Dict[str, Any]] = None,
        planner_policy: str = "rule",
        planner_sft_model_path: str = "models/planner_sft_v1.pt",
        gatekeeper_model_path: str = "models/rl_gatekeeper_v2.pt",
        gatekeeper_threshold: float = 0.0,
        system2_lenient: bool = False,
        sim_aggressive_entry: bool = False,
    ) -> None:
        self.engine = engine
        try:
            tks = [str(x or "").upper().strip() for x in (tickers or [])]
            tks = [x for x in tks if x]
        except Exception:
            tks = []
        self.tickers = tks if tks else ["NVDA", "TSLA", "AAPL", "MSFT", "AMD"]
        
        # Model state
        self.models_loaded = False
        self.model = None
        self.tokenizer = None
        self.models_error = ""
        self._adapters_loaded: set[str] = set()
        self.planner = None
        self.gatekeeper = None
        self._inference_lock = threading.Lock()
        self._inference_lock_owner: str = ""
        self._inference_lock_hold_since_ts: float = 0.0

        self.planner_policy = str(planner_policy or "rule").strip().lower()
        self.planner_sft_model_path = str(planner_sft_model_path or "").strip()
        self.gatekeeper_model_path = str(gatekeeper_model_path or "").strip()
        try:
            self.gatekeeper_threshold = float(gatekeeper_threshold)
        except Exception:
            self.gatekeeper_threshold = 0.0

        self._planner_model: Planner | None = None
        self._gatekeeper_model: Gatekeeper | None = None
        self._force_real_gatekeeper: bool = True
        
        # Config
        self.base_model = base_model
        self.moe_scalper = moe_scalper
        try:
            if moe_analyst == DEFAULT_ANALYST_ADAPTER:
                p4 = Path("models/trader_v4_dpo_analyst_alpha_max_v3")
                if p4.exists():
                    moe_analyst = str(p4)
        except Exception:
            pass
        self.moe_analyst = moe_analyst
        self.moe_secretary = str(moe_secretary or "").strip()
        self.moe_system2 = str(moe_system2 or "").strip()
        self.moe_news = str(moe_news or "").strip()
        self.load_4bit = load_4bit

        try:
            self.llm_max_context = int(llm_max_context) if llm_max_context is not None else 0
        except Exception:
            self.llm_max_context = 0
        try:
            self.llm_max_new_tokens = int(llm_max_new_tokens)
        except Exception:
            self.llm_max_new_tokens = 256
        self.llm_max_context = max(0, int(self.llm_max_context or 0))
        self.llm_max_new_tokens = max(8, int(self.llm_max_new_tokens or 256))

        self._chartist_vlm_cfg: Dict[str, Any] = dict(chartist_vlm_cfg or {})
        self._chartist_vlm_model: Any = None
        self._chartist_vlm_processor: Any = None
        self._chartist_vlm_error: str = ""

        self._news_cfg: Dict[str, Any] = dict(news_cfg or {})
        self._news_cache: Dict[str, Dict[str, Any]] = {}
        self._news_last_fetch_ts: Dict[str, float] = {}
        self._news_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self._news_inflight: Dict[str, Any] = {}

        self._perf_cfg: Dict[str, Any] = dict(perf_cfg or {})
        try:
            self._perf_backlog_degrade_threshold = int(self._perf_cfg.get("backlog_degrade_threshold", 40))
        except Exception:
            self._perf_backlog_degrade_threshold = 40
        try:
            self._perf_backlog_skip_system2_vlm_threshold = int(self._perf_cfg.get("backlog_skip_system2_vlm_threshold", 40))
        except Exception:
            self._perf_backlog_skip_system2_vlm_threshold = 40
        try:
            self._perf_backlog_drop_logs_threshold = int(self._perf_cfg.get("backlog_drop_logs_threshold", 40))
        except Exception:
            self._perf_backlog_drop_logs_threshold = 40

        self.all_agents_mode: bool = False
        self.committee_policy: str = "conservative"
        
        # MoE routing thresholds
        self.moe_any_news = True
        self.moe_news_threshold = 0.8
        self.moe_vol_threshold = 60.0  # annualized vol %
        try:
            self.moe_any_news = bool(self._news_cfg.get("moe_any_news", self.moe_any_news))
        except Exception:
            pass
        try:
            self.moe_news_threshold = float(self._news_cfg.get("moe_news_threshold", self.moe_news_threshold))
        except Exception:
            pass
        try:
            self.moe_vol_threshold = float(self._news_cfg.get("moe_vol_threshold", self.moe_vol_threshold))
        except Exception:
            pass
        
        # Risk parameters
        self.max_drawdown_pct = -8.0
        self.vol_trigger_ann_pct = 120.0
        
        # System 2 debate settings
        self.system2_enabled = True
        self.system2_buy_only = True

        self.system2_lenient = bool(system2_lenient)
        self.sim_aggressive_entry = bool(sim_aggressive_entry)

        self._heuristic_only_until_ts: float = 0.0
        self._slow_infer_warn_sec: float = 25.0
        self._slow_infer_fuse_sec: float = 90.0
        self._gen_max_time_sec: float = 12.0
        self._tick_infer_budget_sec: float = 1.2

        self._infer_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self._infer_future: Any = None
        self._infer_inflight_since_ts: float = 0.0
        
        # Chartist overlay
        self.chart_confidence_threshold = 0.7
        
        # Macro governor
        self.macro_risk_map: Dict[str, float] = {}
        
        # Price history for technical analysis
        self.price_history: Dict[str, List[Dict]] = {}

        self._hold_diag: Dict[str, int] = {}
        
        # Load models if requested
        if load_models:
            self.load_models()

        try:
            if self.planner_policy in {"rule", "sft"}:
                self._planner_model = Planner(policy=self.planner_policy, sft_model_path=self.planner_sft_model_path)
        except Exception:
            self._planner_model = None

        try:
            if str(self.gatekeeper_model_path or "").strip():
                self._gatekeeper_model = Gatekeeper(model_path=str(self.gatekeeper_model_path), threshold=float(self.gatekeeper_threshold))
        except Exception:
            self._gatekeeper_model = None
        
        print("Multi-Agent Strategy Initialized.")

    def load_models(self) -> None:
        """Load MoE models (Scalper + Analyst adapters)"""
        if self.models_loaded:
            return

        acquired = False
        try:
            lt = float(getattr(self, "_inference_lock_timeout_sec", 1.0) or 1.0)
            lt = max(0.1, min(lt, 10.0))
            acquired = bool(self._inference_lock.acquire(timeout=float(lt)))
        except Exception:
            acquired = False
        if not acquired:
            try:
                owner = str(getattr(self, "_inference_lock_owner", "") or "")
                held = 0.0
                try:
                    ts0 = float(getattr(self, "_inference_lock_hold_since_ts", 0.0) or 0.0)
                    if ts0 > 0:
                        held = float(time.time() - ts0)
                except Exception:
                    held = 0.0
                self._log(f"[LoadModels] Inference lock busy: owner={owner} held={held:.2f}s", priority=2)
            except Exception:
                pass
            return

        try:
            self._inference_lock_owner = "load_models"
            self._inference_lock_hold_since_ts = time.time()
        except Exception:
            pass

        self._log("Loading Multi-Agent models...", priority=1)

        try:
            # Import model loading functions
            project_root = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(project_root / "scripts"))
            import importlib
            import run_trading_inference as _rti
            try:
                _rti = importlib.reload(_rti)
            except Exception:
                pass
            load_model_moe = getattr(_rti, "load_model_moe")
            
            def _resolve_adapter_path(raw: str) -> str:
                p = Path(str(raw or "").strip())
                if not p.is_absolute():
                    p = (project_root / p).resolve()
                if p.exists():
                    return str(p)
                # Backward compat: some configs used /lora_weights suffix but the adapter lives in the parent dir
                if p.name.lower() == "lora_weights" and p.parent.exists():
                    return str(p.parent)
                return str(p)

            scalper_path = _resolve_adapter_path(self.moe_scalper)
            analyst_path = _resolve_adapter_path(self.moe_analyst)
            secretary_path = _resolve_adapter_path(self.moe_secretary) if self.moe_secretary else ""
            system2_path = _resolve_adapter_path(self.moe_system2) if self.moe_system2 else ""
            news_path = _resolve_adapter_path(self.moe_news) if self.moe_news else ""

            adapters: Dict[str, str] = {}
            if Path(scalper_path).exists():
                adapters["scalper"] = scalper_path
            if Path(analyst_path).exists():
                adapters["analyst"] = analyst_path
            if secretary_path and Path(secretary_path).exists():
                adapters["secretary"] = secretary_path
            if system2_path and Path(system2_path).exists():
                adapters["system2"] = system2_path
            if news_path and Path(news_path).exists():
                adapters["news"] = news_path

            try:
                self._adapters_loaded = set(adapters.keys())
            except Exception:
                self._adapters_loaded = set()

            if not adapters:
                self._log(f"No MoE adapters found", priority=1)
                self.models_error = f"No MoE adapters found"
                self.models_loaded = False
                return

            warn = ""
            if "analyst" not in adapters:
                warn = f"Analyst adapter not found: {analyst_path} (running scalper-only)"
                self._log(warn, priority=1)
            
            self.model, self.tokenizer = load_model_moe(
                self.base_model,
                adapters,
                self.load_4bit,
                default_adapter="scalper",
            )
            self.models_loaded = True
            self.models_error = warn
            self._log(f"MoE models loaded: {list(adapters.keys())}", priority=2)

            try:
                self._log_vram(prefix="[VRAM] after_moe_load")
            except Exception:
                pass

            try:
                self._warmup_kv_cache()
            except Exception:
                pass

            try:
                self._maybe_load_chartist_vlm()
            except Exception:
                pass
            
        except Exception as e:
            self._log(f"Model loading failed: {e}", priority=2)
            self.models_loaded = False
            self.models_error = str(e)
            try:
                self.model = None
            except Exception:
                pass
            try:
                self.tokenizer = None
            except Exception:
                pass
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        finally:
            try:
                self._inference_lock_owner = ""
                self._inference_lock_hold_since_ts = 0.0
            except Exception:
                pass
            try:
                self._inference_lock.release()
            except Exception:
                pass

    def generic_inference(
        self,
        user_msg: str,
        system_prompt: str = "",
        adapter: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generic thread-safe inference using the shared model.
        Supports hot-swapping adapters (or using base model if adapter is None).
        """
        if not self.models_loaded:
            self._log("Lazy loading models for inference...", priority=1)
            self.load_models()

        if (not self.models_loaded) or (self.model is None) or (self.tokenizer is None):
            return ""

        # Determine target adapter
        target_adapter = str(adapter or "").strip()
        if target_adapter and target_adapter not in self._adapters_loaded:
            # Fallback: if requested adapter missing, try secretary, then base
            if "secretary" in self._adapters_loaded:
                target_adapter = "secretary"
            else:
                target_adapter = ""

        acquired = False
        try:
            lt = float(getattr(self, "_inference_lock_timeout_sec", 3.0) or 3.0)
            # Chat/generic requests should wait for the lock instead of failing immediately.
            wait_time = max(10.0, min(20.0, float(lt) * 2.0))
            acquired = bool(self._inference_lock.acquire(timeout=float(wait_time)))
        except Exception:
            acquired = False
        if not acquired:
            try:
                self._log("Inference busy: skip generic_inference (lock held)", priority=2)
            except Exception:
                pass
            return "(model busy)"

        try:
            try:
                self._inference_lock_owner = "generic_inference"
                self._inference_lock_hold_since_ts = time.time()
            except Exception:
                pass
            try:
                import torch
                from contextlib import nullcontext

                # Context manager for adapter usage
                adapter_ctx = nullcontext()

                model0 = getattr(self, "model", None)
                tok0 = getattr(self, "tokenizer", None)
                if model0 is None or tok0 is None:
                    return ""
                if target_adapter:
                    model0.set_adapter(target_adapter)
                else:
                    # Use base model
                    if hasattr(model0, "disable_adapter"):
                        adapter_ctx = model0.disable_adapter()

                with adapter_ctx:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": user_msg})

                    text = tok0.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    tk_kwargs: Dict[str, Any] = {"return_tensors": "pt"}
                    if int(getattr(self, "llm_max_context", 0) or 0) > 0:
                        tk_kwargs.update({"truncation": True, "max_length": int(self.llm_max_context)})
                    inputs = tok0([text], **tk_kwargs).to(model0.device)

                    mn = int(max_new_tokens or 0)
                    if mn <= 0:
                        mn = int(getattr(self, "llm_max_new_tokens", 256) or 256)

                    with torch.no_grad():
                        try:
                            mt = float(getattr(self, "_gen_max_time_sec", 12.0) or 12.0)
                            try:
                                mt = min(mt, 2.5)
                            except Exception:
                                pass
                            gen_ids = model0.generate(
                                **inputs,
                                max_new_tokens=mn,
                                temperature=temperature,
                                do_sample=(temperature > 0),
                                top_p=0.9,
                                max_time=mt,
                                pad_token_id=tok0.eos_token_id,
                            )
                        except TypeError as e:
                            if "max_time" in str(e):
                                gen_ids = model0.generate(
                                    **inputs,
                                    max_new_tokens=mn,
                                    temperature=temperature,
                                    do_sample=(temperature > 0),
                                    top_p=0.9,
                                    pad_token_id=tok0.eos_token_id,
                                )
                            else:
                                raise
                    
                    output = tok0.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

                # Restore default adapter (scalper) to minimize impact on next trading tick
                # (Scalper is usually the default active one)
                default = "scalper" if "scalper" in self._adapters_loaded else (list(self._adapters_loaded)[0] if self._adapters_loaded else None)
                if default:
                    model0.set_adapter(default)

                return str(output or "").strip()

            except Exception as e:
                self._log(f"Generic inference failed: {e}", priority=2)
                # Try to restore default
                try:
                    default = "scalper" if "scalper" in self._adapters_loaded else (list(self._adapters_loaded)[0] if self._adapters_loaded else None)
                    if default:
                        model0.set_adapter(default)
                except Exception:
                    pass
                return ""

        finally:
            try:
                self._inference_lock_owner = ""
                self._inference_lock_hold_since_ts = 0.0
            except Exception:
                pass
            try:
                self._inference_lock.release()
            except Exception:
                pass

    def _log_vram(self, *, prefix: str) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                self._log(f"{prefix} cuda=unavailable", priority=2)
                return
            free, total = torch.cuda.mem_get_info()
            alloc = torch.cuda.memory_allocated()
            reserv = torch.cuda.memory_reserved()
            gb = 1024 ** 3
            self._log(
                f"{prefix} free={free / gb:.2f}GB total={total / gb:.2f}GB alloc={alloc / gb:.2f}GB reserved={reserv / gb:.2f}GB",
                priority=2,
            )
        except Exception as e:
            self._log(f"{prefix} vram_log_failed: {e}", priority=2)

    def _warmup_kv_cache(self) -> None:
        try:
            if (not self.models_loaded) or (self.model is None) or (self.tokenizer is None):
                return
            adapters = list(getattr(self, "_adapters_loaded", []) or [])
            if not adapters:
                return
            for ad in adapters:
                try:
                    self.generic_inference(
                        user_msg="warmup",
                        system_prompt="You are warming up the model. Reply with a single token.",
                        adapter=str(ad),
                        temperature=0.0,
                        max_new_tokens=4,
                    )
                except Exception:
                    continue
            try:
                self._log_vram(prefix="[VRAM] after_warmup")
            except Exception:
                pass
        except Exception:
            return

    def _maybe_load_chartist_vlm(self) -> None:
        cfg = self._chartist_vlm_cfg if isinstance(self._chartist_vlm_cfg, dict) else {}
        if not bool(cfg.get("enabled", False)):
            return
        if self._chartist_vlm_model is not None and self._chartist_vlm_processor is not None:
            return
        mname = str(cfg.get("local_model") or "").strip()
        if not mname:
            self._chartist_vlm_error = "missing chartist_vlm.local_model"
            return

        try:
            from transformers import AutoProcessor
        except Exception as e:
            self._chartist_vlm_error = f"transformers missing: {e}"
            self._log(f"Chartist VLM unavailable: {self._chartist_vlm_error}", priority=2)
            return

        try:
            import torch
            from transformers import BitsAndBytesConfig
            from transformers import Qwen2_5_VLForConditionalGeneration

            load_4bit = bool(cfg.get("load_4bit", True))
            qcfg = None
            if load_4bit:
                try:
                    qcfg = BitsAndBytesConfig(load_in_4bit=True)
                except Exception:
                    qcfg = None

            proc = AutoProcessor.from_pretrained(mname, trust_remote_code=True)
            try:
                if int(cfg.get("min_image_pixels") or 0) > 0 and hasattr(proc, "min_pixels"):
                    proc.min_pixels = int(cfg.get("min_image_pixels"))
            except Exception:
                pass
            try:
                if int(cfg.get("max_image_pixels") or 0) > 0 and hasattr(proc, "max_pixels"):
                    proc.max_pixels = int(cfg.get("max_image_pixels"))
            except Exception:
                pass

            load_kwargs: Dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
            if qcfg is not None:
                load_kwargs["quantization_config"] = qcfg
            else:
                load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            model_obj = Qwen2_5_VLForConditionalGeneration.from_pretrained(mname, **load_kwargs)
            model_obj.eval()

            adapter_path = str(cfg.get("adapter") or cfg.get("adapter_path") or "").strip()
            if adapter_path:
                try:
                    from peft import PeftModel

                    model_obj = PeftModel.from_pretrained(model_obj, adapter_path, is_trainable=False)
                    model_obj.eval()
                    self._log(f"Chartist VLM adapter loaded: {Path(adapter_path).name}", priority=2)
                except Exception as e:
                    self._log(f"Chartist VLM adapter load failed: {e}", priority=2)

            self._chartist_vlm_processor = proc
            self._chartist_vlm_model = model_obj
            self._chartist_vlm_error = ""
            self._log(f"Chartist VLM loaded: {mname}", priority=2)
            self._log_vram(prefix="[VRAM] after_vlm_load")
        except Exception as e:
            self._chartist_vlm_error = str(e)
            self._chartist_vlm_model = None
            self._chartist_vlm_processor = None
            self._log(f"Chartist VLM load failed: {e}", priority=2)

    def _read_prompt_yaml(self, path: str) -> Dict[str, str]:
        p = Path(str(path or "").strip())
        if not p.is_absolute():
            try:
                project_root = Path(__file__).resolve().parents[2]
                p = (project_root / p).resolve()
            except Exception:
                p = Path(str(path or "").strip())
        if not p.exists():
            return {}
        try:
            import yaml

            obj = yaml.safe_load(p.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                return {}
            sp = str(obj.get("system_prompt") or "").strip()
            up = str(obj.get("user_prompt") or "").strip()
            if sp and up:
                return {"system_prompt": sp, "user_prompt": up}
            return {}
        except Exception:
            return {}

    def _render_chartist_image(self, ticker: str) -> Any:
        try:
            import pandas as pd
            import mplfinance as mpf
            from PIL import Image
        except Exception:
            return None

        hist = list(self.price_history.get(str(ticker or "").upper(), []) or [])
        if len(hist) < 5:
            return None

        rows = hist[-60:]
        try:
            df = pd.DataFrame(rows)
        except Exception:
            return None

        if "time" not in df.columns:
            return None

        try:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.dropna(subset=["time"])
            df = df.set_index("time")
        except Exception:
            return None

        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass
        df = df.dropna(subset=["open", "high", "low", "close"])
        if df.empty:
            return None

        try:
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()
        except Exception:
            pass

        mav = None
        try:
            n = int(len(df))
            if n >= 200:
                mav = (20, 50, 200)
            elif n >= 50:
                mav = (20, 50)
            elif n >= 20:
                mav = (20,)
        except Exception:
            mav = None

        buf = io.BytesIO()
        try:
            plot_kwargs: Dict[str, Any] = {
                "type": "candle",
                "volume": True,
                "style": "yahoo",
                "savefig": dict(fname=buf, dpi=110, bbox_inches="tight"),
            }
            if mav is not None:
                plot_kwargs["mav"] = mav

            mpf.plot(df, **plot_kwargs)
        except Exception as e:
            try:
                now2 = time.time()
                last2 = float(getattr(self, "_chartist_render_fail_last_ts", 0.0) or 0.0)
                if (now2 - last2) >= 30.0:
                    setattr(self, "_chartist_render_fail_last_ts", now2)
                    self._log(f"Chartist (VLM): render_fail (ticker={str(ticker).upper()} err={e})", priority=2)
            except Exception:
                pass
            return None

        try:
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            return img
        except Exception:
            return None

    def generate_reply(self, user_msg: str, system_prompt: str = "") -> str:
        """Generate chat reply using the Secretary adapter (MoE hot-swap), falling back to base model if needed."""
        # Use the generic thread-safe method, preferring secretary adapter
        return self.generic_inference(
            user_msg=user_msg, 
            system_prompt=system_prompt, 
            adapter="secretary",
            temperature=0.7,
            max_new_tokens=int(getattr(self, "llm_max_new_tokens", 256) or 256)
        )

    def on_bar(self, market_data: dict) -> Optional[dict]:
        """
        Process incoming market data through Multi-Agent pipeline
        market_data: { "ticker": "NVDA", "close": 120.5, "volume": ..., "time": ... }
        """
        try:
            md = market_data if isinstance(market_data, dict) else {}
        except Exception:
            md = {}

        ticker_raw = md.get("ticker")
        if not ticker_raw:
            ticker_raw = md.get("symbol")
        if not ticker_raw:
            ticker_raw = md.get("code")
        ticker = str(ticker_raw or "").upper().strip()

        price_raw = md.get("close")
        if price_raw is None:
            price_raw = md.get("price")
        if price_raw is None:
            price_raw = md.get("last")
        if price_raw is None:
            price_raw = md.get("c")

        try:
            price = float(price_raw or 0.0)
        except Exception:
            price = 0.0

        try:
            volume = float(md.get("volume", 0.0) or 0.0)
        except Exception:
            volume = 0.0

        if (not ticker) or (price <= 0):
            try:
                keys = []
                try:
                    keys = sorted(list(md.keys()))
                except Exception:
                    keys = []
                self._log(
                    f"[MarketData] invalid payload: ticker={ticker!r} price={price!r} keys={keys}",
                    priority=2,
                )
            except Exception:
                pass
            return None
        
        # Update price history (normalized schema)
        try:
            o_raw = md.get("open")
            if o_raw is None:
                o_raw = md.get("o")
            h_raw = md.get("high")
            if h_raw is None:
                h_raw = md.get("h")
            l_raw = md.get("low")
            if l_raw is None:
                l_raw = md.get("l")
            try:
                o = float(o_raw) if o_raw is not None else float(price)
            except Exception:
                o = float(price)
            try:
                h = float(h_raw) if h_raw is not None else float(price)
            except Exception:
                h = float(price)
            try:
                l = float(l_raw) if l_raw is not None else float(price)
            except Exception:
                l = float(price)

            self._update_price_history(
                ticker,
                {
                    "time": md.get("time", datetime.now()),
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": float(price),
                    "volume": float(volume),
                },
            )
        except Exception:
            self._update_price_history(ticker, {"close": float(price), "volume": float(volume), "time": md.get("time", datetime.now())})
        
        # Calculate technical features
        features = self._compute_features(ticker)

        # News signal (optionally aligned to market time for offline playback)
        asof_dt: Optional[datetime] = None
        try:
            t0 = market_data.get("time")
            if isinstance(t0, datetime):
                asof_dt = t0
            elif isinstance(t0, (int, float)):
                asof_dt = datetime.fromtimestamp(float(t0))
            elif isinstance(t0, str) and t0.strip():
                try:
                    asof_dt = datetime.fromisoformat(t0.strip())
                except Exception:
                    asof_dt = None
        except Exception:
            asof_dt = None

        try:
            news_sig = self._get_news_signal(ticker, asof_time=asof_dt)
            if isinstance(news_sig, dict) and news_sig:
                features["news"] = dict(news_sig)
                sig = features.get("signal") if isinstance(features.get("signal"), dict) else {}
                sig["news_score"] = float(news_sig.get("news_score") or 0.0)
                sig["news_count"] = float(news_sig.get("news_count") or 0.0)
                features["signal"] = sig
        except Exception:
            pass

        backlog_hint = 0
        try:
            backlog_hint = int(getattr(self, "_md_backlog_hint", 0) or 0)
        except Exception:
            backlog_hint = 0
        
        # --- 1. Planner: Market Regime Assessment ---
        self._log(f"Planner scanning {ticker}...", priority=2)
        regime = self._planner_assess(ticker, features)
        try:
            self._log(f"Planner: {ticker} regime={regime}", priority=2)
        except Exception:
            pass
        
        if (not self.all_agents_mode) and regime == "cash_preservation":
            self._log(f"Planner: {ticker} - cash preservation mode, skip.", priority=2)
            return None
        
        # --- 2. Gatekeeper: RL-based Trade Filter ---
        approved_by_gatekeeper = self._gatekeeper_approve(ticker, features)
        try:
            self._log(f"Gatekeeper: {ticker} - {'approved' if approved_by_gatekeeper else 'rejected'}", priority=2)
        except Exception:
            pass
        if (not self.all_agents_mode) and (not approved_by_gatekeeper):
            self._log(f"Gatekeeper: {ticker} - rejected (low Q-value).", priority=2)
            return None

        if self.all_agents_mode:
            self._log(
                f"Gatekeeper: {ticker} - {'approved' if approved_by_gatekeeper else 'rejected'} (all_agents_mode continues)",
                priority=2,
            )
        
        # --- 3. MoE Router: Select Expert ---
        expert, router_meta = self._moe_route(ticker, features)
        self._log(f"MoE Router: {ticker} -> [{expert}] (vol={router_meta.get('vol', 0):.1f}%)", priority=2)

        # --- 4. Expert Inference ---
        degraded_all_agents = False
        try:
            th = int(getattr(self, "_perf_backlog_degrade_threshold", 40) or 40)
            degraded_all_agents = bool(self.all_agents_mode) and backlog_hint >= th
        except Exception:
            degraded_all_agents = False

        if self.all_agents_mode and (not degraded_all_agents):
            dec_scalper = self._expert_infer(ticker, features, "scalper")
            dec_analyst = self._expert_infer(ticker, features, "analyst")
            act_s = str(dec_scalper.get("decision", "HOLD") or "HOLD").upper()
            act_a = str(dec_analyst.get("decision", "HOLD") or "HOLD").upper()
            ana_s = str(dec_scalper.get("analysis", "") or "")
            ana_a = str(dec_analyst.get("analysis", "") or "")
            self._log(f"[scalper] {ticker}: {act_s} - {ana_s[:4000]}", priority=1)
            self._log(f"[analyst] {ticker}: {act_a} - {ana_a[:4000]}", priority=1)

            action = "HOLD"
            analysis = ""
            pol = str(getattr(self, "committee_policy", "conservative") or "conservative").strip().lower()
            if pol == "aggressive":
                if act_s in ("BUY", "SELL"):
                    action = act_s
                    analysis = ana_s
                elif act_a in ("BUY", "SELL"):
                    action = act_a
                    analysis = ana_a
                else:
                    action = "HOLD"
                    analysis = "committee: no_action"
            else:
                if act_s == act_a and act_s in ("BUY", "SELL"):
                    action = act_s
                    analysis = f"committee agree: {ana_s[:4000]}"
                else:
                    action = "HOLD"
                    analysis = "committee: disagree_or_hold"

            expert = "committee"
        else:
            if self.all_agents_mode and degraded_all_agents:
                try:
                    self._log(f"[Perf] backlog={backlog_hint} -> degrade all_agents_mode to single expert", priority=2)
                except Exception:
                    pass
            decision = self._expert_infer(ticker, features, expert)
            action = decision.get("decision", "HOLD").upper()
            analysis = decision.get("analysis", "")

        trace_id: Optional[str] = None
        trace_ids: list[str] = []
        
        if action == "HOLD":
            self._log(f"[{expert}] {ticker}: HOLD - {str(analysis or '')[:4000]}", priority=1)

            n0: Optional[int] = None
            try:
                n0 = int(self._hold_diag.get(ticker, 0) or 0) + 1
                self._hold_diag[ticker] = n0
                if n0 % 2 == 0:
                    try:
                        self._log(f"System 2 Debate: idle (action=HOLD)", priority=2)
                    except Exception:
                        pass
                    try:
                        macro_gear, macro_label = self._macro_governor_assess()
                        self._log(f"[MACRO] Macro Governor: regime={macro_label} (gear={macro_gear})", priority=2)
                    except Exception as e:
                        self._log(f"[MACRO] Macro Governor: error: {e}", priority=2)
            except Exception:
                pass

            forced: Optional[str] = None
            if bool(getattr(self, "sim_aggressive_entry", False)) and n0 and (n0 % 2 == 0):
                try:
                    br = getattr(self.engine, "broker", None)
                    pos = None
                    if br is not None and isinstance(getattr(br, "positions", None), dict):
                        pos = br.positions.get(str(ticker).upper())
                    forced = "SELL" if pos is not None else "BUY"
                    action = forced
                    analysis = f"sim_aggressive_entry: forced {forced} after HOLD"
                    self._log(f"[Sim] {ticker}: override HOLD -> {forced} (sim_aggressive_entry)", priority=2)
                except Exception:
                    forced = None

            if forced is None:
                return None

        try:
            print(f"\n[{expert}] proposal: {action} {ticker} :: {str(analysis or '')[:800]}")
        except Exception:
            pass

        if self.all_agents_mode:
            try:
                rid_s = evolution_recorder.record(
                    agent_id="scalper",
                    context=json.dumps(
                        {
                            "ticker": ticker,
                            "price": price,
                            "regime": regime,
                            "features": features,
                            "router": router_meta,
                            "proposed_action": str(act_s),
                        },
                        ensure_ascii=False,
                    ),
                    action=str(ana_s or ""),
                    outcome=0.0,
                    feedback="pending_pnl",
                )
                if rid_s:
                    trace_ids.append(str(rid_s))
            except Exception:
                pass

            try:
                rid_a = evolution_recorder.record(
                    agent_id="analyst",
                    context=json.dumps(
                        {
                            "ticker": ticker,
                            "price": price,
                            "regime": regime,
                            "features": features,
                            "router": router_meta,
                            "proposed_action": str(act_a),
                        },
                        ensure_ascii=False,
                    ),
                    action=str(ana_a or ""),
                    outcome=0.0,
                    feedback="pending_pnl",
                )
                if rid_a:
                    trace_ids.append(str(rid_a))
            except Exception:
                pass

            trace_id = trace_ids[0] if trace_ids else None
        else:
            try:
                trace_id = evolution_recorder.record(
                    agent_id=str(expert),
                    context=json.dumps(
                        {
                            "ticker": ticker,
                            "price": price,
                            "regime": regime,
                            "features": features,
                            "router": router_meta,
                            "proposed_action": action,
                        },
                        ensure_ascii=False,
                    ),
                    action=str(analysis or ""),
                    outcome=0.0,
                    feedback="pending_pnl",
                )
                if trace_id:
                    trace_ids = [str(trace_id)]
            except Exception:
                trace_id = None
                trace_ids = []
        
        if expert == "scalper":
            self._log(f"Scalper: {ticker} {action} - {str(analysis or '')[:4000]}", priority=2)
        else:
            self._log(f"Analyst (DPO): {ticker} {action} - {str(analysis or '')[:4000]}", priority=2)
        confidence = 0.75

        chart_score: Optional[int] = None

        news_score: Optional[float] = None
        news_sentiment: Optional[str] = None
        news_summary: Optional[str] = None
        try:
            ns = features.get("news") if isinstance(features.get("news"), dict) else {}
            news_score = float(ns.get("news_score")) if ("news_score" in ns) else None
            news_sentiment = str(ns.get("news_sentiment") or "").strip() or None
            news_summary = str(ns.get("news_summary") or "").strip() or None
        except Exception:
            news_score = None
            news_sentiment = None
            news_summary = None
        
        # --- 5. System 2 Debate (if enabled) ---
        if self.system2_enabled:
            if (not self.system2_buy_only) or (action in {"BUY", "SELL"}):
                skip_vlm = False
                try:
                    th = int(getattr(self, "_perf_backlog_skip_system2_vlm_threshold", 40) or 40)
                    if backlog_hint >= th:
                        self._log(f"[Perf] backlog={backlog_hint} -> skip VLM (System2 continues)", priority=2)
                        skip_vlm = True
                except Exception:
                    pass
                self._log("System 2 Debate: initiated...", priority=2)
                
                # Chartist Overlay
                if skip_vlm:
                    chart_score = 0
                else:
                    chart_score = int(self._chartist_overlay(ticker, action))
                    chart_view = "supports" if chart_score > 0 else ("opposes" if chart_score < 0 else "neutral")
                    self._log(f"Chartist (VLM): {ticker} pattern {chart_view} (score={int(chart_score)})", priority=2)
                
                # Macro Governor
                macro_gear, macro_label = self._macro_governor_assess()
                self._log(f"[MACRO] Macro Governor: regime={macro_label} (gear={macro_gear})", priority=2)
                
                # System2 Critic -> Judge (LLM via hot-swap); may override or block
                approved, final_action, reason = self._system2_debate(
                    ticker=ticker,
                    proposed_action=action,
                    proposed_analysis=analysis,
                    features=features,
                    chart_score=int(chart_score),
                    macro_gear=float(macro_gear),
                    macro_label=str(macro_label),
                )

                try:
                    final_action = str(final_action or "").strip().upper()
                except Exception:
                    final_action = str(action)

                if final_action in {"BUY", "SELL"} and final_action != action:
                    self._log(f"System 2 (Judge): override {action} -> {final_action} ({reason})", priority=2)
                    action = final_action

                if (not bool(approved)) or (str(final_action) == "HOLD"):
                    if bool(getattr(self, "system2_lenient", False)):
                        self._log(f"System 2 (Judge): LENIENT_BYPASS - {reason}", priority=2)
                    else:
                        self._log(f"System 2 (Judge): BLOCKED - {reason}", priority=2)
                        return None

                try:
                    evolution_recorder.record(
                        agent_id="system2",
                        context=json.dumps(
                            {
                                "ticker": ticker,
                                "proposed_action": action,
                                "expert": expert,
                                "chart_score": chart_score,
                                "macro_gear": macro_gear,
                                "features": features,
                            },
                            ensure_ascii=False,
                        ),
                        action="APPROVED" if approved else "REJECTED",
                        outcome=0.0,
                        feedback=str(reason or ""),
                    )
                except Exception:
                    pass
                
                self._log(f"System 2 (Judge): APPROVED (conf={confidence:.2f})", priority=2)
        
        
        # --- 6. Generate Signal ---
        signal = {
            "ticker": ticker,
            "action": action,
            "price": price,
            "shares": self._calculate_position_size(ticker, price, confidence),
            "expert": expert,
            "confidence": confidence,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
            "trace_id": trace_id,
            "trace_ids": trace_ids,
            "chart_score": chart_score,
            "news_score": news_score,
            "news_sentiment": news_sentiment,
            "news_summary": news_summary,
        }
        return signal

    def _get_news_signal(self, ticker: str, *, asof_time: Optional[datetime] = None) -> Dict[str, Any]:
        cfg = self._news_cfg if isinstance(self._news_cfg, dict) else {}
        if not bool(cfg.get("enabled", False)):
            return {}

        tk = str(ticker or "").upper().strip()
        if not tk:
            return {}

        # time_mode:
        # - wall: use wall clock for refresh and cache per ticker
        # - market: use bar time for refresh and cache per ticker+date (for offline playback)
        time_mode = "wall"
        try:
            time_mode = str(cfg.get("time_mode") or "wall").strip().lower() or "wall"
        except Exception:
            time_mode = "wall"

        cache_key = tk
        asof_date = ""
        if time_mode == "market" and isinstance(asof_time, datetime):
            try:
                asof_date = asof_time.strftime("%Y-%m-%d")
                cache_key = f"{tk}|{asof_date}"
            except Exception:
                cache_key = tk

        now = time.time()
        refresh_sec = 300.0
        try:
            refresh_sec = float(cfg.get("refresh_sec") or 300.0)
        except Exception:
            refresh_sec = 300.0
        refresh_sec = max(30.0, min(refresh_sec, 3600.0))

        last_ts = float(self._news_last_fetch_ts.get(cache_key, 0.0) or 0.0)
        if (now - last_ts) >= refresh_sec:
            try:
                fut = self._news_inflight.get(cache_key)
                if fut is None:
                    self._news_inflight[cache_key] = self._news_executor.submit(self._refresh_news, tk, asof_time, cache_key)
            except Exception:
                pass

        # Harvest finished futures (non-blocking)
        try:
            fut0 = self._news_inflight.get(cache_key)
            if fut0 is not None and bool(getattr(fut0, "done", lambda: False)()):
                try:
                    fut0.result(timeout=0.0)
                except Exception:
                    pass
                try:
                    self._news_inflight.pop(cache_key, None)
                except Exception:
                    pass
        except Exception:
            pass

        cached = self._news_cache.get(cache_key)
        if not isinstance(cached, dict):
            return {}
        out = cached.get("signal")
        return dict(out) if isinstance(out, dict) else {}

    def _refresh_news(self, ticker: str, asof_time: Optional[datetime] = None, cache_key: Optional[str] = None) -> None:
        tk = str(ticker or "").upper().strip()
        if not tk:
            return

        ck = str(cache_key or "").strip() or tk

        cfg = self._news_cfg if isinstance(self._news_cfg, dict) else {}
        limit = 12
        try:
            limit = int(cfg.get("limit") or 12)
        except Exception:
            limit = 12
        limit = max(3, min(limit, 40))

        items: List[Dict[str, Any]] = []
        time_mode = "wall"
        try:
            time_mode = str(cfg.get("time_mode") or "wall").strip().lower() or "wall"
        except Exception:
            time_mode = "wall"

        historical_enabled = False
        try:
            historical_enabled = bool(cfg.get("historical_enabled", False))
        except Exception:
            historical_enabled = False

        historical_source = "gdelt"
        try:
            historical_source = str(cfg.get("historical_source") or "gdelt").strip().lower() or "gdelt"
        except Exception:
            historical_source = "gdelt"

        if time_mode == "market" and historical_enabled and isinstance(asof_time, datetime) and historical_source == "gdelt":
            try:
                items = self._fetch_news_gdelt(tk, asof_time=asof_time, limit=limit)
            except Exception as e:
                try:
                    self._log(f"[News] gdelt fetch failed {tk}: {e}", priority=2)
                except Exception:
                    pass
                items = []
        else:
            rss_yahoo = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={tk}&region=US&lang=en-US"
            q = urllib.parse.quote_plus(f"{tk} stock")
            rss_google = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
            try:
                from src.data.fetcher import DataFetcher
            except Exception:
                return

            try:
                df = DataFetcher(source=str(cfg.get("source") or "yfinance").strip() or "yfinance")
                srcs: List[str] = []
                try:
                    if bool(cfg.get("yahoo_enabled", True)):
                        srcs.append(rss_yahoo)
                except Exception:
                    srcs.append(rss_yahoo)
                try:
                    if bool(cfg.get("google_enabled", False)):
                        srcs.append(rss_google)
                except Exception:
                    pass
                if not srcs:
                    srcs = [rss_yahoo]
                items = df.fetch_news(keywords=None, sources=srcs, limit=limit) or []
            except Exception as e:
                try:
                    self._log(f"[News] fetch failed {tk}: {e}", priority=2)
                except Exception:
                    pass
                items = []

        # If fetch returns nothing, keep previous cached signal to avoid flapping.
        try:
            if not items and ck in self._news_cache and isinstance(self._news_cache.get(ck), dict):
                self._news_last_fetch_ts[ck] = time.time()
                return
        except Exception:
            pass

        # Optionally run LLM-based parsing. Prefer 'news' adapter, fall back to 'analyst'.
        llm_enabled = False
        llm_adapter: Optional[str] = None
        try:
            llm_enabled = bool(cfg.get("llm_enabled", False))
        except Exception:
            llm_enabled = False
        try:
            loaded = getattr(self, "_adapters_loaded", set()) or set()
            if "news" in loaded:
                llm_adapter = "news"
            elif "analyst" in loaded:
                llm_adapter = "analyst"
            else:
                llm_adapter = None
        except Exception:
            llm_adapter = None

        if llm_enabled and (llm_adapter is not None):
            try:
                titles = []
                for it in list(items)[:limit]:
                    if not isinstance(it, dict):
                        continue
                    t0 = str(it.get("title") or "").strip()
                    if t0:
                        titles.append(t0)
                blob = "\n".join([f"- {t}" for t in titles[:12]])
                sp1 = "你是金融新闻分析员。根据标题列表，输出JSON：{\"news_sentiment\": \"positive|neutral|negative\", \"news_score\": -1..1, \"confidence\": 0..1, \"summary\": \"...\"}。只输出JSON。"
                raw1 = self._fast_generic_inference(user_msg=blob, system_prompt=sp1, adapter=llm_adapter, temperature=0.0, max_new_tokens=180)
                obj = repair_and_parse_json(extract_json_text(str(raw1 or ""))) if str(raw1 or "").strip() else None
                if isinstance(obj, dict) and ("news_score" in obj or "news_sentiment" in obj):
                    try:
                        avg_score = float(obj.get("news_score") or 0.0)
                    except Exception:
                        avg_score = 0.0
                    sentiment = str(obj.get("news_sentiment") or "neutral").strip().lower() or "neutral"
                    summary = str(obj.get("summary") or "").strip()
                    if len(summary) > 240:
                        summary = summary[:237] + "..."
                    src_label = "yahoo+google" if (bool(cfg.get("google_enabled", False))) else "yahoo_rss"
                    try:
                        if time_mode == "market" and historical_enabled and isinstance(asof_time, datetime):
                            src_label = "gdelt"
                    except Exception:
                        pass

                    sig = {
                        "news_count": int(len(items) or 0),
                        "news_score": float(avg_score),
                        "news_sentiment": sentiment,
                        "news_confidence": float(obj.get("confidence") or 0.0),
                        "news_summary": summary,
                        "news_source": src_label,
                        "asof": datetime.now().isoformat(),
                    }
                    try:
                        if time_mode == "market" and isinstance(asof_time, datetime):
                            sig["asof"] = asof_time.isoformat()
                    except Exception:
                        pass
                    self._news_cache[ck] = {"items": items[:limit], "signal": sig}
                    self._news_last_fetch_ts[ck] = time.time()
                    try:
                        self._log(f"[News] {tk} sentiment={sentiment} score={avg_score:.2f} n={len(items) or 0} :: {summary}", priority=2)
                    except Exception:
                        pass
                    return
            except Exception:
                pass

    def _fast_generic_inference(
        self,
        user_msg: str,
        system_prompt: str = "",
        adapter: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> str:
        if (not self.models_loaded) or (self.model is None) or (self.tokenizer is None):
            return ""

        target_adapter = str(adapter or "").strip()
        if target_adapter and target_adapter not in self._adapters_loaded:
            if "secretary" in self._adapters_loaded:
                target_adapter = "secretary"
            else:
                target_adapter = ""

        acquired = False
        try:
            acquired = bool(self._inference_lock.acquire(blocking=False))
        except Exception:
            acquired = False
        if not acquired:
            return ""

        model0 = None
        try:
            self._inference_lock_owner = "fast_generic_inference"
            self._inference_lock_hold_since_ts = time.time()

            import torch
            from contextlib import nullcontext

            adapter_ctx = nullcontext()
            model0 = getattr(self, "model", None)
            tok0 = getattr(self, "tokenizer", None)
            if model0 is None or tok0 is None:
                return ""

            if target_adapter:
                model0.set_adapter(target_adapter)
            else:
                if hasattr(model0, "disable_adapter"):
                    adapter_ctx = model0.disable_adapter()

            with adapter_ctx:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_msg})

                text = tok0.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                tk_kwargs: Dict[str, Any] = {"return_tensors": "pt"}
                if int(getattr(self, "llm_max_context", 0) or 0) > 0:
                    tk_kwargs.update({"truncation": True, "max_length": int(self.llm_max_context)})
                inputs = tok0([text], **tk_kwargs).to(model0.device)

                mn = int(max_new_tokens or 0)
                if mn <= 0:
                    mn = 64

                mt = float(getattr(self, "_gen_max_time_sec", 12.0) or 12.0)
                try:
                    mt = min(mt, 1.8)
                except Exception:
                    pass

                with torch.no_grad():
                    try:
                        gen_ids = model0.generate(
                            **inputs,
                            max_new_tokens=int(mn),
                            temperature=float(temperature),
                            do_sample=(float(temperature) > 0),
                            top_p=0.9,
                            max_time=float(mt),
                            pad_token_id=tok0.eos_token_id,
                        )
                    except TypeError as e:
                        if "max_time" in str(e):
                            gen_ids = model0.generate(
                                **inputs,
                                max_new_tokens=int(mn),
                                temperature=float(temperature),
                                do_sample=(float(temperature) > 0),
                                top_p=0.9,
                                pad_token_id=tok0.eos_token_id,
                            )
                        else:
                            raise

                try:
                    out = tok0.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                except Exception:
                    out = ""

            default = "scalper" if "scalper" in self._adapters_loaded else (list(self._adapters_loaded)[0] if self._adapters_loaded else None)
            if default:
                try:
                    model0.set_adapter(default)
                except Exception:
                    pass
            return str(out or "").strip()
        except Exception:
            try:
                default = "scalper" if "scalper" in self._adapters_loaded else (list(self._adapters_loaded)[0] if self._adapters_loaded else None)
                if default and model0 is not None:
                    model0.set_adapter(default)
            except Exception:
                pass
            return ""
        finally:
            try:
                self._inference_lock_owner = ""
                self._inference_lock_hold_since_ts = 0.0
            except Exception:
                pass
            try:
                self._inference_lock.release()
            except Exception:
                pass

        parsed = []
        try:
            from src.llm.news_parser import RuleBasedNewsParser

            parser = RuleBasedNewsParser()
            for it in items:
                if not isinstance(it, dict):
                    continue
                title = str(it.get("title") or "").strip()
                content = str(it.get("content") or "").strip()
                if not title and not content:
                    continue
                pn = parser.parse(title=title, content=content)
                parsed.append(pn)
        except Exception:
            parsed = []

        score = 0.0
        conf_sum = 0.0
        n = 0
        titles = []
        for i, pn in enumerate(parsed[:limit]):
            try:
                s = str(getattr(pn, "sentiment", "neutral") or "neutral").strip().lower()
                conf = float(getattr(pn, "confidence", 0.5) or 0.5)
                if s == "positive":
                    score += 1.0 * conf
                elif s == "negative":
                    score += -1.0 * conf
                conf_sum += conf
                n += 1
                t0 = str(getattr(pn, "title", "") or "").strip()
                if t0:
                    titles.append(t0)
            except Exception:
                continue

        avg_score = (score / conf_sum) if (conf_sum > 1e-9) else 0.0
        sentiment = "neutral"
        if avg_score >= 0.2:
            sentiment = "positive"
        elif avg_score <= -0.2:
            sentiment = "negative"

        summary = " | ".join(titles[:3]).strip()
        if len(summary) > 240:
            summary = summary[:237] + "..."

        sig = {
            "news_count": int(len(items) or 0),
            "news_score": float(avg_score),
            "news_sentiment": sentiment,
            "news_confidence": float(conf_sum / n) if n > 0 else 0.0,
            "news_summary": summary,
            "news_source": "yahoo+google" if (bool(cfg.get("google_enabled", False))) else "yahoo_rss",
            "asof": datetime.now().isoformat(),
        }

        try:
            if time_mode == "market" and isinstance(asof_time, datetime):
                sig["asof"] = asof_time.isoformat()
        except Exception:
            pass

        self._news_cache[ck] = {"items": items[:limit], "signal": sig}
        self._news_last_fetch_ts[ck] = time.time()
        try:
            self._log(f"[News] {tk} sentiment={sentiment} score={avg_score:.2f} n={len(items) or 0} :: {summary}", priority=2)
        except Exception:
            pass

    def _fetch_news_gdelt(self, ticker: str, *, asof_time: datetime, limit: int) -> List[Dict[str, Any]]:
        tk = str(ticker or "").upper().strip()
        if not tk:
            return []

        try:
            maxrec = int(limit) if limit is not None else 12
        except Exception:
            maxrec = 12
        maxrec = max(3, min(maxrec, 40))

        # GDELT expects UTC-ish yyyymmddHHMMSS. We align to the day boundary of asof_time.
        try:
            day0 = asof_time.strftime("%Y%m%d")
        except Exception:
            day0 = datetime.now().strftime("%Y%m%d")
        startdt = f"{day0}000000"
        enddt = f"{day0}235959"

        q = urllib.parse.quote_plus(f"{tk} stock")
        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc?query="
            + q
            + f"&mode=ArtList&format=json&startdatetime={startdt}&enddatetime={enddt}&maxrecords={maxrec}"
        )

        req0 = urllib.request.Request(url, headers={"User-Agent": "StockAppNews/0.1"})
        with urllib.request.urlopen(req0, timeout=12) as resp:
            raw = resp.read(2_000_000)
        try:
            js = json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception:
            js = {}

        arts = []
        try:
            arts = js.get("articles") if isinstance(js, dict) else []
        except Exception:
            arts = []

        out: List[Dict[str, Any]] = []
        if not isinstance(arts, list):
            return out
        for a in arts[:maxrec]:
            if not isinstance(a, dict):
                continue
            title = str(a.get("title") or "").strip()
            link = str(a.get("url") or "").strip()
            seen = str(a.get("seendate") or "").strip()
            if not title:
                continue
            out.append({"title": title, "content": title, "link": link, "published": seen, "source": "gdelt"})
        return out

    def inject_news_signal(self, ticker: str, signal: Dict[str, Any]) -> None:
        """Inject a precomputed news signal into cache (used by offline replay recordings)."""
        tk = str(ticker or "").upper().strip()
        if not tk or not isinstance(signal, dict):
            return

        cfg = self._news_cfg if isinstance(self._news_cfg, dict) else {}
        time_mode = "wall"
        try:
            time_mode = str(cfg.get("time_mode") or "wall").strip().lower() or "wall"
        except Exception:
            time_mode = "wall"

        ck = tk
        if time_mode == "market":
            try:
                asof_s = str(signal.get("asof") or "").strip()
                asof_dt = datetime.fromisoformat(asof_s) if asof_s else None
                if isinstance(asof_dt, datetime):
                    ck = f"{tk}|{asof_dt.strftime('%Y-%m-%d')}"
            except Exception:
                ck = tk

        try:
            self._news_cache[ck] = {"items": [], "signal": dict(signal)}
            self._news_last_fetch_ts[ck] = time.time()
        except Exception:
            return

        try:
            sent = str(signal.get("news_sentiment") or "").strip().lower()
            sc = float(signal.get("news_score") or 0.0)
            self._log(f"[News] inject {ck} sentiment={sent} score={sc:.2f}", priority=2)
        except Exception:
            pass

    def _flatten_gate_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        tech = features.get("technical") if isinstance(features.get("technical"), dict) else {}
        sig = features.get("signal") if isinstance(features.get("signal"), dict) else {}
        for k, v in tech.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        for k, v in sig.items():
            try:
                out["signal_" + str(k)] = float(v)
            except Exception:
                continue
        try:
            out["volatility_ann_pct"] = float(features.get("volatility_ann_pct") or 0.0)
        except Exception:
            out["volatility_ann_pct"] = 0.0
        return out

    def _update_price_history(self, ticker: str, data: dict) -> None:
        """Maintain rolling price history for technical analysis"""
        if ticker not in self.price_history:
            self.price_history[ticker] = []

        t = data.get("time", datetime.now())
        try:
            if isinstance(t, datetime):
                t = t.isoformat()
        except Exception:
            t = data.get("time", datetime.now())

        c_raw = data.get("close")
        if c_raw is None:
            c_raw = data.get("price")
        if c_raw is None:
            c_raw = data.get("last")
        if c_raw is None:
            c_raw = data.get("c")
        try:
            c = float(c_raw or 0.0)
        except Exception:
            c = 0.0
        if c <= 0:
            return

        try:
            o = float(data.get("open", c) or c)
        except Exception:
            o = c
        try:
            h = float(data.get("high", c) or c)
        except Exception:
            h = c
        try:
            l = float(data.get("low", c) or c)
        except Exception:
            l = c
        try:
            v = float(data.get("volume", 0.0) or 0.0)
        except Exception:
            v = 0.0

        self.price_history[ticker].append({
            "time": t,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
        })
        
        # Keep last 60 bars
        if len(self.price_history[ticker]) > 60:
            self.price_history[ticker] = self.price_history[ticker][-60:]

    def _compute_features(self, ticker: str) -> Dict[str, Any]:
        """Compute technical features from price history"""
        history = self.price_history.get(ticker, [])
        if len(history) < 1:
            return {"technical": {}, "signal": {}, "volatility_ann_pct": 0.0}
        
        closes = [float(bar.get("close") or 0.0) for bar in history]
        volumes = [float(bar.get("volume") or 0.0) for bar in history]
        closes = [c for c in closes if c > 0]
        if not closes:
            return {"technical": {}, "signal": {}, "volatility_ann_pct": 0.0}
        
        # Basic features
        current = closes[-1]
        ma5 = sum(closes[-5:]) / 5 if len(closes) >= 5 else current
        ma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else current
        
        # Returns
        ret_5d = (current / closes[-5] - 1) * 100 if len(closes) >= 5 else 0.0
        ret_20d = (current / closes[-20] - 1) * 100 if len(closes) >= 20 else 0.0
        
        # Volatility (annualized). We treat incoming bars as intraday bars.
        # Use a per-bar realized-vol estimate with a bars/year scaling so vol isn't stuck at a constant early on.
        try:
            returns = [(closes[i] / closes[i - 1] - 1) for i in range(1, len(closes))]
        except Exception:
            returns = []
        if len(returns) >= 2:
            w = min(20, len(returns))
            try:
                mean_r = sum(returns[-w:]) / float(w)
                var = sum((r - mean_r) ** 2 for r in returns[-w:]) / float(max(1, w - 1))
                std = math.sqrt(max(0.0, var))
            except Exception:
                std = 0.0
            # Approx bars/year: 252 trading days * 390 minutes/day
            bars_per_year = 252.0 * 390.0
            vol_ann = std * math.sqrt(bars_per_year) * 100.0
            if not (vol_ann > 0.0):
                vol_ann = 20.0
        else:
            vol_ann = 20.0
        
        # Volume ratio
        avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else (sum(volumes) / max(len(volumes), 1))
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1
        
        return {
            "technical": {
                "close": current,
                "price_vs_ma5": (current / ma5 - 1) * 100,
                "price_vs_ma20": (current / ma20 - 1) * 100,
                "return_5d": ret_5d,
                "return_21d": ret_20d,
                "volatility_20d": vol_ann,
                "vol_ratio": vol_ratio,
            },
            "signal": {
                "composite": 1 if current > ma20 else -1,
            },
            "volatility_ann_pct": vol_ann,
        }

    def _planner_assess(self, ticker: str, features: Dict) -> str:
        """Planner: Assess market regime"""
        try:
            if self._planner_model is not None and str(getattr(self, "planner_policy", "") or "").strip().lower() == "sft":
                flat = self._flatten_gate_features(features)
                dec = self._planner_model.decide(features=flat)
                st = str(getattr(dec, "strategy", "") or "").strip().lower()
                if st == "aggressive_long":
                    return "aggressive"
                if st == "defensive":
                    return "defensive"
                return "cash_preservation"
        except Exception:
            pass

        tech = features.get("technical", {})
        vol = tech.get("volatility_20d", 20)
        ret_5d = tech.get("return_5d", 0)
        
        # Simple rule-based planner (can be replaced with SFT model)
        if vol > 120 or ret_5d < -10:
            return "cash_preservation"
        elif vol > 80 or ret_5d < -5:
            return "defensive"
        else:
            return "aggressive"

    def _gatekeeper_approve(self, ticker: str, features: Dict) -> bool:
        """Gatekeeper: RL-based trade approval"""
        try:
            if self._gatekeeper_model is not None:
                flat = self._flatten_gate_features(features)
                d = self._gatekeeper_model.decide(feats=flat)
                allow = bool(getattr(d, "allow", False))
                q = float(getattr(d, "q_allow", 0.0) or 0.0)
                thr = float(getattr(d, "threshold", 0.0) or 0.0)
                self._log(f"Gatekeeper RL: {ticker} q_allow={q:.3f} thr={thr:.3f} -> {'ALLOW' if allow else 'DENY'}", priority=1)
                return bool(allow)

            if bool(getattr(self, "_force_real_gatekeeper", False)) and str(getattr(self, "gatekeeper_model_path", "") or "").strip():
                self._log(f"Gatekeeper RL model not loaded: {self.gatekeeper_model_path}", priority=2)
                return False
        except Exception:
            if bool(getattr(self, "_force_real_gatekeeper", False)) and str(getattr(self, "gatekeeper_model_path", "") or "").strip():
                self._log(f"Gatekeeper RL error: model={self.gatekeeper_model_path}", priority=2)
                return False

        tech = features.get("technical", {})
        vol = tech.get("volatility_20d", 20)

        if vol > self.vol_trigger_ann_pct:
            return False

        import random
        return random.random() > 0.3

    def _moe_route(self, ticker: str, features: Dict) -> Tuple[str, Dict]:
        """MoE Router: Select expert based on market features"""
        vol = features.get("volatility_ann_pct", 20)

        news_score = 0.0
        try:
            sig = features.get("signal") if isinstance(features.get("signal"), dict) else {}
            news_score = float(sig.get("news_score") or 0.0)
        except Exception:
            news_score = 0.0
        
        # Route to Analyst for high volatility or meaningful news signal
        use_analyst = vol >= self.moe_vol_threshold
        try:
            if bool(getattr(self, "moe_any_news", True)) and abs(float(news_score)) >= float(getattr(self, "moe_news_threshold", 0.8) or 0.8):
                use_analyst = True
        except Exception:
            pass
        
        expert = "analyst" if use_analyst else "scalper"
        meta = {"vol": vol, "expert": expert, "news_score": news_score}
        
        return expert, meta

    def _expert_infer(self, ticker: str, features: Dict, expert: str) -> Dict:
        """Expert inference (model or heuristic)"""
        try:
            if float(getattr(self, "_heuristic_only_until_ts", 0.0) or 0.0) > time.time():
                return self._heuristic_infer(ticker, features, expert)
        except Exception:
            pass
        if self.models_loaded and self.model is not None:
            try:
                if hasattr(self, "_adapters_loaded") and isinstance(self._adapters_loaded, set) and expert not in self._adapters_loaded:
                    return self._heuristic_infer(ticker, features, expert)
            except Exception:
                pass
            return self._model_infer(ticker, features, expert)
        else:
            return self._heuristic_infer(ticker, features, expert)

    def _heuristic_infer(self, ticker: str, features: Dict, expert: str) -> Dict:
        """Heuristic-based inference when models not loaded"""
        import random
        tech = features.get("technical", {})
        
        ret_5d = tech.get("return_5d", 0)
        price_vs_ma = tech.get("price_vs_ma20", 0)
        
        # Simple momentum logic
        if ret_5d > 3 and price_vs_ma > 2:
            decision = "BUY"
            analysis = f"Momentum: +{ret_5d:.1f}% 5d, above MA20"
        elif ret_5d < -3 and price_vs_ma < -2:
            decision = "SELL"
            analysis = f"Weakness: {ret_5d:.1f}% 5d, below MA20"
        else:
            decision = "HOLD"
            analysis = "No clear signal"
        
        # Add some randomness for demo
        if random.random() < 0.3:
            decision = random.choice(["BUY", "SELL", "HOLD"])
            analysis = f"[{expert}] technical analysis"
        
        return {"decision": decision, "analysis": analysis}

    def _model_infer(self, ticker: str, features: Dict, expert: str) -> Dict:
        """Real model inference using MoE"""
        # Set adapter based on expert
        adapter_name = "analyst" if expert == "analyst" else "scalper"

        t0_all = time.perf_counter()
        try:
            dev = str(getattr(self.model, "device", "")) if self.model is not None else ""
            self._log(f"[{expert}] [InferStart] {ticker} adapter={adapter_name} device={dev}", priority=2)
        except Exception:
            pass
        
        # Build prompt (simplified)
        tech = features.get("technical", {})
        prompt = f"""Ticker: {ticker}
Close: {tech.get('close', 0):.2f}
Return 5d: {tech.get('return_5d', 0):.2f}%
Volatility: {tech.get('volatility_20d', 0):.1f}%

Decide BUY/SELL/HOLD for next 5 days."""
        
        messages = [
            {"role": "system", "content": "Output JSON: {\"decision\": \"BUY|SELL|HOLD\", \"analysis\": \"brief reason\"}"},
            {"role": "user", "content": prompt}
        ]

        # If a previous inference task finished, clear inflight.
        try:
            if self._infer_future is not None and bool(getattr(self._infer_future, "done", lambda: False)()):
                self._infer_future = None
                self._infer_inflight_since_ts = 0.0
        except Exception:
            pass

        # If a worker inference is still running (possibly hung), do not block engine thread.
        try:
            if self._infer_future is not None:
                self._log(f"[{expert}] Inference busy: use heuristic for {ticker} (worker inflight)", priority=2)
                return self._heuristic_infer(ticker, features, expert)
        except Exception:
            pass

        def _run_generate_raw() -> str:
            acquired2 = False
            try:
                lt = float(getattr(self, "_inference_lock_timeout_sec", 1.0) or 1.0)
                lt = max(0.1, min(lt, 5.0))
                acquired2 = bool(self._inference_lock.acquire(timeout=float(lt)))
            except Exception:
                acquired2 = False
            if not acquired2:
                try:
                    owner = str(getattr(self, "_inference_lock_owner", "") or "")
                    held = 0.0
                    try:
                        ts0 = float(getattr(self, "_inference_lock_hold_since_ts", 0.0) or 0.0)
                        if ts0 > 0:
                            held = float(time.time() - ts0)
                    except Exception:
                        held = 0.0
                    self._log(f"[{expert}] [InferLockBusy] {ticker} owner={owner} held={held:.2f}s", priority=2)
                except Exception:
                    pass
                return ""
            try:
                try:
                    self._inference_lock_owner = f"model_infer:{expert}:{str(ticker or '').upper()}"
                    self._inference_lock_hold_since_ts = time.time()
                except Exception:
                    pass

                model0 = getattr(self, "model", None)
                tok0 = getattr(self, "tokenizer", None)
                if model0 is None or tok0 is None:
                    try:
                        self._log(f"[{expert}] [InferDecodeFail] {ticker} model_or_tokenizer_none", priority=2)
                    except Exception:
                        pass
                    return ""

                import torch
                from contextlib import nullcontext

                adapter_ctx = nullcontext()
                if adapter_name in self._adapters_loaded:
                    model0.set_adapter(adapter_name)
                else:
                    if "scalper" in self._adapters_loaded:
                        model0.set_adapter("scalper")
                    elif hasattr(model0, "disable_adapter"):
                        adapter_ctx = model0.disable_adapter()

                with adapter_ctx:
                    t0_tok = time.perf_counter()
                    text0 = tok0.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    tk_kwargs2: Dict[str, Any] = {"return_tensors": "pt"}
                    if int(getattr(self, "llm_max_context", 0) or 0) > 0:
                        tk_kwargs2.update({"truncation": True, "max_length": int(self.llm_max_context)})
                    inputs0 = tok0([text0], **tk_kwargs2).to(model0.device)

                    try:
                        dt_tok = time.perf_counter() - t0_tok
                        plen = int(getattr(getattr(inputs0, "input_ids", None), "shape", [0, 0])[1])
                        self._log(f"[{expert}] [InferTok] {ticker} prompt_tokens={plen} dt={dt_tok:.2f}s", priority=2)
                    except Exception:
                        pass

                    mn0 = int(getattr(self, "llm_max_new_tokens", 256) or 256)
                    cap_tokens = int(getattr(self, "_max_new_tokens_cap", 512) or 512)
                    cap_tokens = max(64, cap_tokens)
                    mn0 = max(32, min(mn0, cap_tokens))

                    try:
                        budget2 = float(getattr(self, "_tick_infer_budget_sec", 1.2) or 1.2)
                        cap_budget2 = float(getattr(self, "_tick_infer_budget_cap_sec", 6.0) or 6.0)
                        cap_budget2 = max(0.5, cap_budget2)
                        budget2 = max(0.2, min(budget2, cap_budget2))
                        if budget2 <= 1.5:
                            mn0 = min(mn0, 96)
                        elif budget2 <= 2.5:
                            mn0 = min(mn0, 160)
                    except Exception:
                        budget2 = float(getattr(self, "_tick_infer_budget_sec", 1.2) or 1.2)

                    kw = {
                        "max_new_tokens": mn0,
                        "temperature": 0.1,
                        "do_sample": False,
                        "pad_token_id": tok0.eos_token_id,
                    }
                    # max_time may not be supported on some transformers versions.
                    mt = float(getattr(self, "_gen_max_time_sec", 12.0) or 12.0)
                    try:
                        if str(expert or "").strip().lower() == "scalper":
                            mt = float(getattr(self, "_gen_max_time_sec_scalper", mt) or mt)
                        elif str(expert or "").strip().lower() == "analyst":
                            mt = float(getattr(self, "_gen_max_time_sec_analyst", mt) or mt)
                    except Exception:
                        pass
                    try:
                        mt_cap = float(getattr(self, "_gen_max_time_cap_sec", 12.0) or 12.0)
                        mt_cap = max(0.5, mt_cap)
                        mt = min(mt, mt_cap)
                    except Exception:
                        pass
                    kw["max_time"] = mt

                    with torch.no_grad():
                        try:
                            t0_gen = time.perf_counter()
                            gen_ids0 = model0.generate(**inputs0, **kw)
                            try:
                                dt_gen = time.perf_counter() - t0_gen
                                self._log(f"[{expert}] [InferGen] {ticker} max_new_tokens={mn0} dt={dt_gen:.2f}s", priority=2)
                            except Exception:
                                pass
                        except TypeError as e:
                            if "max_time" in str(e):
                                try:
                                    kw.pop("max_time", None)
                                    try:
                                        # Older transformers: max_time unsupported -> keep generation bounded by tokens.
                                        if float(budget2) <= 1.5:
                                            kw["max_new_tokens"] = min(int(kw.get("max_new_tokens") or 96), 96)
                                        elif float(budget2) <= 2.5:
                                            kw["max_new_tokens"] = min(int(kw.get("max_new_tokens") or 160), 160)
                                    except Exception:
                                        pass
                                    t0_gen = time.perf_counter()
                                    gen_ids0 = model0.generate(**inputs0, **kw)
                                    try:
                                        dt_gen = time.perf_counter() - t0_gen
                                        self._log(f"[{expert}] [InferGen] {ticker} max_new_tokens={mn0} dt={dt_gen:.2f}s (no max_time)", priority=2)
                                    except Exception:
                                        pass
                                except Exception:
                                    raise
                            else:
                                raise

                    try:
                        if gen_ids0 is None:
                            raise ValueError("gen_ids_none")
                        out_slice = gen_ids0[0][inputs0.input_ids.shape[1]:]
                        out0 = tok0.decode(out_slice, skip_special_tokens=True)
                        return str(out0 or "").strip()
                    except Exception as e:
                        try:
                            self._log(f"[{expert}] [InferDecodeFail] {ticker} err={e}", priority=2)
                        except Exception:
                            pass
                        return ""
            finally:
                try:
                    self._inference_lock_owner = ""
                    self._inference_lock_hold_since_ts = 0.0
                except Exception:
                    pass
                try:
                    if "scalper" in self._adapters_loaded and model0 is not None:
                        model0.set_adapter("scalper")
                except Exception:
                    pass
                try:
                    self._inference_lock.release()
                except Exception:
                    pass
        
        raw = ""
        try:
            self._infer_inflight_since_ts = time.time()

            # Ensure the outer future timeout is not shorter than the configured max_time,
            # otherwise we may fallback heuristic while the worker continues generating.
            mt_wait = None
            try:
                mt_wait = float(getattr(self, "_gen_max_time_sec", 12.0) or 12.0)
                if str(expert or "").strip().lower() == "scalper":
                    mt_wait = float(getattr(self, "_gen_max_time_sec_scalper", mt_wait) or mt_wait)
                elif str(expert or "").strip().lower() == "analyst":
                    mt_wait = float(getattr(self, "_gen_max_time_sec_analyst", mt_wait) or mt_wait)
                mt_cap2 = float(getattr(self, "_gen_max_time_cap_sec", 12.0) or 12.0)
                mt_cap2 = max(0.5, mt_cap2)
                mt_wait = min(float(mt_wait), float(mt_cap2))
            except Exception:
                mt_wait = None

            self._infer_future = self._infer_executor.submit(_run_generate_raw)
            budget = float(getattr(self, "_tick_infer_budget_sec", 1.2) or 1.2)
            cap_budget = float(getattr(self, "_tick_infer_budget_cap_sec", 6.0) or 6.0)
            cap_budget = max(0.5, cap_budget)
            budget = max(0.2, min(budget, cap_budget))

            try:
                if mt_wait is not None:
                    # small slack to allow decode/post-processing
                    budget = max(float(budget), float(mt_wait) + 0.2)
                    budget = min(float(budget), float(cap_budget))
            except Exception:
                pass

            raw = str(self._infer_future.result(timeout=budget) or "")
            self._infer_future = None
            self._infer_inflight_since_ts = 0.0
        except FuturesTimeout:
            try:
                self._log(f"[{expert}] [InferTimeout] {ticker} -> fallback heuristic (budget={budget:.2f}s)", priority=2)
            except Exception:
                pass
            try:
                self._heuristic_only_until_ts = time.time() + float(getattr(self, "_slow_infer_fuse_sec", 90.0) or 90.0)
            except Exception:
                pass
            # keep _infer_future as inflight; do not block
            return self._heuristic_infer(ticker, features, expert)
        except Exception as e:
            try:
                self._log(f"[{expert}] Model inference failed: {e}", priority=2)
            except Exception:
                pass
            self._infer_future = None
            self._infer_inflight_since_ts = 0.0
            return {}

        if not str(raw or "").strip():
            try:
                self._log(f"[{expert}] [InferEmpty] {ticker} -> fallback heuristic", priority=2)
            except Exception:
                pass
            return self._heuristic_infer(ticker, features, expert)

        try:
            dt_all = time.perf_counter() - t0_all
            if dt_all >= float(getattr(self, "_slow_infer_warn_sec", 25.0) or 25.0):
                self._log(f"[{expert}] [InferSlow] {ticker} dt={dt_all:.2f}s -> enable heuristic-only for {self._slow_infer_fuse_sec:.0f}s", priority=2)
                self._heuristic_only_until_ts = time.time() + float(getattr(self, "_slow_infer_fuse_sec", 90.0) or 90.0)
        except Exception:
            pass

        # Common: models wrap JSON in ```json ... ```
        # Common: models wrap JSON in ```json ... ```
        try:
            raw = raw.replace("```json", "").replace("```", "").strip()
        except Exception:
            raw = str(output or "").strip()

        def _normalize_decision(x: Any) -> str:
            d = str(x or "").strip().upper()
            if d in {"BUY", "SELL", "HOLD"}:
                return d
            return "HOLD"

        def _as_result(obj: Any) -> dict:
            if not isinstance(obj, dict):
                return {}
            decision = _normalize_decision(obj.get("decision"))
            analysis = str(obj.get("analysis") or "").strip()
            if not analysis:
                analysis = "(no analysis)"
            return {"decision": decision, "analysis": analysis}

        # Parse JSON (robust)
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                blob = raw[start : end + 1]
                try:
                    obj = json.loads(blob)
                    res = _as_result(obj)
                    if res:
                        return res
                except Exception:
                    try:
                        import ast
                        obj = ast.literal_eval(blob)
                        res = _as_result(obj)
                        if res:
                            return res
                    except Exception:
                        pass
        except Exception:
            pass

        # Fallback: infer decision from text
        tl = raw.lower()
        decision = "HOLD"
        if "buy" in tl or "买入" in tl or "做多" in tl:
            decision = "BUY"
        elif "sell" in tl or "卖出" in tl or "做空" in tl:
            decision = "SELL"
        elif "hold" in tl or "观望" in tl or "保持" in tl:
            decision = "HOLD"

        tail = raw.replace("\n", " ").strip()
        if len(tail) > 1600:
            tail = tail[:1597] + "..."
        return {"decision": decision, "analysis": f"Unparsed model output: {tail}"}

    def _chartist_overlay(self, ticker: str, proposed_action: str) -> int:
        """Chartist overlay: visual pattern analysis score"""
        cfg = self._chartist_vlm_cfg if isinstance(self._chartist_vlm_cfg, dict) else {}
        if bool(cfg.get("enabled", False)):
            try:
                self._maybe_load_chartist_vlm()
            except Exception:
                pass

            if self._chartist_vlm_model is not None and self._chartist_vlm_processor is not None:
                img = self._render_chartist_image(str(ticker or "").upper())
                if img is None:
                    try:
                        now2 = time.time()
                        last2 = float(getattr(self, "_chartist_noimg_last_ts", 0.0) or 0.0)
                        if (now2 - last2) >= 30.0:
                            setattr(self, "_chartist_noimg_last_ts", now2)
                            self._log(f"Chartist (VLM): no_image (ticker={str(ticker).upper()})", priority=2)
                    except Exception:
                        pass
                    return 0

                prompt_yaml = str(cfg.get("prompt_yaml") or "").strip()
                p = self._read_prompt_yaml(prompt_yaml) if prompt_yaml else {}
                sp = str(p.get("system_prompt") or "").strip()
                up_t = str(p.get("user_prompt") or "").strip()
                if not sp:
                    sp = "Analyze the candlestick chart image. Return only JSON {signal,confidence,reasoning}."
                if not up_t:
                    up_t = "Analyze this chart for ticker={ticker} asof={asof}. Return only the JSON object."
                try:
                    up = up_t.format(ticker=str(ticker).upper(), asof=str(datetime.now().date()))
                except Exception:
                    up = up_t

                try:
                    import torch

                    messages = [
                        {"role": "system", "content": sp},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": up},
                            ],
                        },
                    ]

                    proc = self._chartist_vlm_processor
                    model_obj = self._chartist_vlm_model

                    try:
                        from qwen_vl_utils import process_vision_info  # type: ignore
                    except Exception:
                        process_vision_info = None

                    prompt = ""
                    if hasattr(proc, "apply_chat_template"):
                        prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        if process_vision_info is not None:
                            image_inputs, video_inputs = process_vision_info(messages)
                            inputs = proc(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
                        else:
                            inputs = proc(text=[prompt], images=[img], padding=True, return_tensors="pt")
                    else:
                        inputs = proc(text=[str(up)], images=[img], padding=True, return_tensors="pt")

                    dev = getattr(model_obj, "device", None)
                    if dev is None:
                        try:
                            dev = next(model_obj.parameters()).device
                        except Exception:
                            dev = None
                    if dev is not None:
                        inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}
                    gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(cfg.get("max_new_tokens") or 256)}
                    temperature = float(cfg.get("temperature") or 0.2)
                    if temperature > 0:
                        gen_kwargs.update({"do_sample": True, "temperature": temperature})
                    else:
                        gen_kwargs.update({"do_sample": False})

                    with torch.inference_mode():
                        out_ids = model_obj.generate(**inputs, **gen_kwargs)
                    txts = proc.batch_decode(out_ids, skip_special_tokens=True)
                    out = str(txts[0] or "").strip() if txts else ""
                    if prompt and out.startswith(prompt):
                        out = out[len(prompt) :].strip()
                except Exception:
                    out = ""

                raw_json = extract_json_text(out)
                obj = None
                if raw_json is not None:
                    try:
                        obj = repair_and_parse_json(raw_json)
                    except Exception:
                        obj = None
                if not isinstance(obj, dict):
                    return 0

                sig = str(obj.get("signal") or "").strip().upper()
                conf = 0.0
                try:
                    conf = float(obj.get("confidence") or 0.0)
                except Exception:
                    conf = 0.0
                thr = float(cfg.get("confidence_threshold") or 0.7)
                try:
                    reason = str(obj.get("reasoning") or obj.get("analysis") or obj.get("reason") or "").replace("\n", " ").strip()
                    if len(reason) > 220:
                        reason = reason[:217] + "..."
                    self._log(
                        f"Chartist (VLM): {str(ticker).upper()} signal={sig or '?'} conf={conf:.2f} thr={thr:.2f} reason={reason}",
                        priority=2,
                    )
                except Exception:
                    pass
                if conf <= thr:
                    return 0

                act = str(proposed_action or "").upper().strip()
                if sig == "BULLISH":
                    if act == "BUY":
                        return 1
                    if act == "SELL":
                        return -1
                    return 0
                if sig == "BEARISH":
                    if act == "SELL":
                        return 1
                    if act == "BUY":
                        return -1
                    return 0
                return 0

            # Enabled but not ready (load failed or processor missing)
            try:
                now2 = time.time()
                last2 = float(getattr(self, "_chartist_notready_last_ts", 0.0) or 0.0)
                if (now2 - last2) >= 60.0:
                    setattr(self, "_chartist_notready_last_ts", now2)
                    err = str(getattr(self, "_chartist_vlm_error", "") or "").strip()
                    if err:
                        self._log(f"Chartist (VLM): not_ready (error={err})", priority=2)
                    else:
                        self._log("Chartist (VLM): not_ready", priority=2)
            except Exception:
                pass

        return 0

    def _system2_debate(
        self,
        *,
        ticker: str,
        proposed_action: str,
        proposed_analysis: str,
        features: Dict[str, Any],
        chart_score: int = 0,
        macro_gear: float = 0.0,
        macro_label: str = "",
    ) -> Tuple[bool, str, str]:
        action_up = str(proposed_action or "").strip().upper()
        if action_up not in {"BUY", "SELL", "HOLD"}:
            action_up = "HOLD"

        lines = []
        try:
            tech = features.get("technical") if isinstance(features.get("technical"), dict) else {}
            sig = features.get("signal") if isinstance(features.get("signal"), dict) else {}
            lines = [
                f"Ticker: {str(ticker).upper()}",
                f"Date: {str(datetime.now().date())}",
                f"Close: {tech.get('close', '')}",
                f"Price vs MA20: {tech.get('price_vs_ma20', '')}",
                f"Return 5d: {tech.get('return_5d', '')}",
                f"Volatility 20d: {tech.get('volatility_20d', '')}",
                f"Volume ratio: {tech.get('vol_ratio', '')}",
                f"Composite signal: {sig.get('composite', '')}",
                f"Chartist score: {int(chart_score)}",
                f"Macro regime: {str(macro_label)} (gear={float(macro_gear)})",
                "",
                f"Proposed decision: {action_up}",
                f"Proposed analysis: {str(proposed_analysis or '').strip()}",
            ]
        except Exception:
            lines = [f"Ticker: {str(ticker).upper()}", "", f"Proposed decision: {action_up}", f"Proposed analysis: {str(proposed_analysis or '').strip()}"]

        critic_sys = (
            "You are a strict trading decision critic.\n"
            "Response Format (STRICT JSON ONLY): {\"accept\": true|false, \"suggested_decision\": \"BUY\"|\"SELL\"|\"HOLD\"|\"CLEAR\", \"reasons\": [..3 strings..]}"
        )
        critic_user = "\n".join(lines)

        judge_sys = (
            "You are a strict trading decision judge.\n"
            "Response Format (STRICT JSON ONLY): {\"final_decision\": \"BUY\"|\"SELL\"|\"HOLD\"|\"CLEAR\", \"rationale\": \"...\"}"
        )

        adapter = "system2" if "system2" in self._adapters_loaded else "scalper"
        critic_raw = self.generic_inference(user_msg=critic_user, system_prompt=critic_sys, adapter=adapter, temperature=0.0, max_new_tokens=256)
        critic_json = None
        err = ""
        try:
            raw = extract_json_text(str(critic_raw or "").strip())
            if raw is None:
                err = "no json"
            else:
                critic_json = repair_and_parse_json(raw)
        except Exception as e:
            err = str(e)

        if not isinstance(critic_json, dict):
            return True, action_up, f"critic_parse_failed: {err}".strip()

        try:
            acc = bool(critic_json.get("accept", True))
            sug = str(critic_json.get("suggested_decision") or "").strip().upper()
            rs = critic_json.get("reasons")
            if not isinstance(rs, list):
                rs = []
            rs2 = [str(x or "").strip() for x in rs if str(x or "").strip()]
            msg = f"System 2 (Critic): accept={acc} suggested={sug} reasons={rs2[:3]}"
            self._log(msg[:1200], priority=2)
        except Exception:
            pass

        judge_user = "\n".join(
            [
                f"Ticker: {str(ticker).upper()}",
                f"Proposal JSON: {json.dumps({'decision': action_up, 'analysis': str(proposed_analysis or '').strip()}, ensure_ascii=False)}",
                f"Critic JSON: {json.dumps(critic_json, ensure_ascii=False)}",
            ]
        )
        judge_raw = self.generic_inference(user_msg=judge_user, system_prompt=judge_sys, adapter=adapter, temperature=0.0, max_new_tokens=192)
        judge_json = None
        jerr = ""
        try:
            raw = extract_json_text(str(judge_raw or "").strip())
            if raw is None:
                jerr = "no json"
            else:
                judge_json = repair_and_parse_json(raw)
        except Exception as e:
            jerr = str(e)

        if not isinstance(judge_json, dict):
            return True, action_up, f"judge_parse_failed: {jerr}".strip()

        final_dec = str(judge_json.get("final_decision") or "").strip().upper()
        rationale = str(judge_json.get("rationale") or "").strip()
        try:
            msg = f"System 2 (Judge): final={final_dec} rationale={rationale}"
            self._log(msg[:1600], priority=2)
        except Exception:
            pass
        if final_dec in {"CLEAR", "HOLD"}:
            return False, "HOLD", rationale or "system2_hold"
        if final_dec in {"BUY", "SELL"}:
            return True, final_dec, rationale
        return True, action_up, rationale

    def _macro_governor_assess(self) -> Tuple[float, str]:
        """Macro Governor: global risk assessment"""
        import random
        # Simulated risk score (0=risky, 1=safe)
        score = random.uniform(0.3, 0.8)
        
        if score >= 0.5:
            return 0.0, "NEUTRAL"
        elif score >= 0.3:
            return 0.5, "LOW"
        else:
            return 1.0, "DRIVE"

    def _system2_judge(
        self, action: str, chart_score: int, macro_gear: float, features: Dict
    ) -> Tuple[bool, float, str]:
        """System 2 Judge: final decision"""
        import random
        
        # Aggregate signals
        signals = []
        if chart_score > 0:
            signals.append(1)
        elif chart_score < 0:
            signals.append(-1)
        
        if macro_gear >= 0.5:
            signals.append(1)
        else:
            signals.append(-1)
        
        # Calculate confidence
        base_conf = 0.7
        conf = base_conf + 0.1 * sum(signals) / max(len(signals), 1)
        conf = max(0.5, min(0.95, conf))
        
        # Decision logic
        if action == "BUY":
            approved = chart_score >= 0 or random.random() > 0.4
            reason = "Chartist opposes" if chart_score < 0 else ""
        else:
            approved = random.random() > 0.3
            reason = "Risk budget exceeded" if not approved else ""
        
        return approved, conf, reason

    def _calculate_position_size(self, ticker: str, price: float, confidence: float) -> int:
        """Calculate position size based on confidence and risk"""
        base_shares = 100
        # Scale by confidence
        return int(base_shares * confidence)

    def _log(self, message: str, priority: int = 0) -> None:
        """Send log event to Mari"""
        try:
            q = getattr(self.engine, "events", None)
            qsz = int(q.qsize()) if (q is not None and hasattr(q, "qsize")) else 0
        except Exception:
            qsz = 0
        try:
            backlog_hint = int(getattr(self, "_md_backlog_hint", 0) or 0)
        except Exception:
            backlog_hint = 0

        try:
            th = int(getattr(self, "_perf_backlog_drop_logs_threshold", 40) or 40)
            if (priority <= 1) and ((qsz >= 12) or (backlog_hint >= th)):
                return
        except Exception:
            pass

        try:
            if (priority <= 0) and (qsz >= 6):
                return
        except Exception:
            pass

        self.engine.push_event(Event(EventType.LOG, datetime.now(), message, priority))

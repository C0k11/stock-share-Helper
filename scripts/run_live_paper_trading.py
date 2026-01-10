#!/usr/bin/env python
"""
Phase 3.4: Live Paper Trading Engine with Mari's Commentary
实时模拟盘 + Multi-Agent 思考过程语音解说

Usage:
    python scripts/run_live_paper_trading.py
"""
from __future__ import annotations

import json
import importlib
import argparse
import importlib
import json
import os
import random
from collections import deque
import socket
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.trading.engine import TradingEngine
from src.trading.strategy import MultiAgentStrategy
from src.trading.broker import PaperBroker
from src.trading.event import Event, EventType
from src.trading.data_feed import create_data_feed, DataFeed
from src.rl.online_learning import get_online_learning_manager

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


def _load_secretary_config() -> Dict[str, Any]:
    """Load secretary config for TTS settings"""
    cfg_path = project_root / "configs" / "secretary.yaml"
    if not cfg_path.exists():
        return {}
    try:
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _port_available(host: str, port: int) -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        s.bind((str(host), int(port)))
        return True
    except Exception:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


class MariVoice:
    """Mari TTS with GPT-SoVITS model loading + Shared LLM Inference (A2)"""

    def __init__(self, strategy=None):
        self.strategy = strategy
        self.tts_queue: list[str] = []
        self.llm_queue: list[tuple[str, bool]] = []
        self.lock = threading.Lock()
        self.max_queue_size = 3
        
        self.cfg = _load_secretary_config()
        voice_cfg = self.cfg.get("voice", {})
        try:
            self.enabled = bool((voice_cfg or {}).get("enabled", False))
        except Exception:
            self.enabled = False

        self.running = bool(self.enabled)
        self.gpt_sovits_cfg = voice_cfg.get("gpt_sovits", {})
        self.api_base = self.gpt_sovits_cfg.get("api_base", "http://127.0.0.1:9880")
        
        # Mari model paths
        self.gpt_path = self.gpt_sovits_cfg.get("gpt_path", "")
        self.sovits_path = self.gpt_sovits_cfg.get("sovits_path", "")
        
        # Reference audio
        presets = self.gpt_sovits_cfg.get("presets", {})
        gentle = presets.get("gentle", {})
        self.refer_wav = gentle.get("refer_wav_path", "")
        self.prompt_text = gentle.get("prompt_text", "先生…")
        
        # LLM config for generating Mari's speech (fallback)
        self.llm_cfg = self.cfg.get("llm", {})
        self.llm_base = self.llm_cfg.get("api_base", "http://localhost:11434/v1")
        self.llm_model = self.llm_cfg.get("model", "qwen2.5:7b-instruct")
        self.system_prompt = self.cfg.get("secretary", {}).get("system_prompt", "")
        
        # Chat history for display
        self.chat_log: List[Dict] = []
        
        if self.enabled and HAS_PYGAME:
            pygame.mixer.init()
        
        if self.enabled:
            # Load Mari's voice model (async) to avoid blocking API startup.
            try:
                t = threading.Thread(target=self._load_mari_model, daemon=True)
                t.start()
            except Exception:
                pass
            
            # Start workers
            self.llm_thread = threading.Thread(target=self._llm_worker, daemon=True)
            self.llm_thread.start()
            
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()

    def _load_mari_model(self) -> None:
        """Load Mari's trained GPT-SoVITS weights"""
        if not bool(getattr(self, "enabled", False)):
            return
        if not HAS_REQUESTS:
            return
        
        try:
            # Load GPT model
            if self.gpt_path and os.path.exists(self.gpt_path):
                resp = requests.get(
                    f"{self.api_base}/set_gpt_weights",
                    params={"weights_path": self.gpt_path},
                    timeout=60
                )
                if resp.status_code == 200:
                    print(f"[Mari] GPT model loaded: {Path(self.gpt_path).name}")
                else:
                    print(f"[Mari] GPT load failed: {resp.status_code}")
            
            # Load SoVITS model
            if self.sovits_path and os.path.exists(self.sovits_path):
                resp = requests.get(
                    f"{self.api_base}/set_sovits_weights",
                    params={"weights_path": self.sovits_path},
                    timeout=60
                )
                if resp.status_code == 200:
                    print(f"[Mari] SoVITS model loaded: {Path(self.sovits_path).name}")
                else:
                    print(f"[Mari] SoVITS load failed: {resp.status_code}")
        except Exception as e:
            print(f"[Mari] Model load error: {e}")

    def generate_commentary(self, event_context: str) -> str:
        """Use Shared Strategy LLM (preferred) or legacy API to generate commentary"""
        # 1. Try Shared Strategy LLM (A2 Architecture)
        # Note: generic_inference handles lazy loading, so we just check if strategy exists
        if self.strategy:
            try:
                # Use 'secretary' adapter for personality
                prompt = self.system_prompt + "\n\n请用一句话简洁转述以下事件，保持角色。"
                resp = self.strategy.generic_inference(
                    user_msg=event_context,
                    system_prompt=prompt,
                    adapter="secretary",
                    temperature=0.7,
                    max_new_tokens=80
                )
                if resp:
                    return resp.strip()
            except Exception as e:
                print(f"[Mari] Shared LLM error: {e}")

        # 2. Fallback to API (if configured and valid)
        if not HAS_REQUESTS:
            return event_context
        
        try:
            resp = requests.post(
                f"{self.llm_base}/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt + "\n\n请用一句话简洁转述以下事件，保持角色。"},
                        {"role": "user", "content": event_context}
                    ],
                    "max_tokens": 60,
                    "temperature": 0.7,
                },
                headers={"Authorization": f"Bearer {self.llm_cfg.get('api_key', 'ollama')}"},
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[LLM Error] {e}")
        return event_context

    def speak(self, text: str, use_llm: bool = True) -> None:
        """Enqueue text for processing (LLM -> TTS)"""
        if not bool(getattr(self, "enabled", False)):
            return
        with self.lock:
            # Drop old if full to keep up with live events
            if len(self.llm_queue) >= self.max_queue_size:
                self.llm_queue.pop(0)
            self.llm_queue.append((text, use_llm))

    def _llm_worker(self) -> None:
        """Background worker to process LLM generation queue"""
        while self.running:
            item = None
            with self.lock:
                if self.llm_queue:
                    item = self.llm_queue.pop(0)
            
            if item:
                text, use_llm = item
                final_text = text
                if use_llm:
                    final_text = self.generate_commentary(text)
                
                # Log to chat
                self.chat_log.append({"time": datetime.now().isoformat(), "speaker": "Mari", "text": final_text})
                print(f"\n[Chat] Mari: {final_text}")
                
                with self.lock:
                    if len(self.tts_queue) < self.max_queue_size:
                        self.tts_queue.append(final_text)
            else:
                time.sleep(0.1)

    def _tts_worker(self) -> None:
        """Background worker to process TTS queue"""
        while self.running:
            text = None
            with self.lock:
                if self.tts_queue:
                    text = self.tts_queue.pop(0)

            if text:
                self._speak_sync(text)
            else:
                time.sleep(0.1)

    def _speak_sync(self, text: str) -> None:
        """Synchronously speak text using GPT-SoVITS"""
        print(f"[TTS] Mari: {text}")
        
        if not HAS_REQUESTS or not HAS_PYGAME:
            return

        try:
            # Try GPT-SoVITS first
            resp = requests.post(
                f"{self.api_base}/tts",
                json={
                    "text": text,
                    "text_lang": "zh",
                    "ref_audio_path": self.refer_wav,
                    "prompt_text": self.prompt_text,
                    "prompt_lang": "ja",
                },
                timeout=30,
            )
            
            if resp.status_code == 200:
                # Save and play audio
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(resp.content)
                    tmp_path = f.name

                try:
                    pygame.mixer.music.load(tmp_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
            else:
                print(f"  [TTS Error: HTTP {resp.status_code}]")
                
        except Exception as e:
            print(f"  [TTS Error: {e}]")

    def stop(self) -> None:
        self.running = False
        if bool(getattr(self, "enabled", False)) and HAS_PYGAME:
            pygame.mixer.quit()


class LivePaperTradingRunner:
    """Main runner for live paper trading with Mari commentary"""

    def __init__(
        self,
        initial_cash: float = 500000.0,
        data_source: str = "auto",
        load_models: bool = False,
    ):
        self.engine = TradingEngine()
        self.broker = PaperBroker(self.engine, cash=initial_cash)

        base_model = None
        moe_scalper = None
        moe_analyst = None
        moe_secretary = None
        moe_system2 = None
        moe_news = None
        chartist_vlm_cfg = None
        news_cfg = None
        perf_cfg = None
        llm_max_context = None
        llm_max_new_tokens = None
        llm_max_tokens_scalper = None
        llm_max_tokens_analyst = None
        llm_repetition_penalty = None
        infer_cfg = None
        all_agents_mode = None
        committee_policy = None
        load_4bit = None
        planner_policy = None
        planner_sft_model = None
        gatekeeper_model = None
        gatekeeper_threshold = None
        system2_lenient = None
        sim_aggressive_entry = None
        allow_short = None
        broker_cfg = None
        risk_cfg = None
        data_feed_interval_sec = None
        md_queue_limit = None
        md_pending_limit = None
        md_symbols_per_tick = None
        tickers_cfg = None
        effective_scalper = None
        effective_analyst = None
        try:
            cfg_path = project_root / "configs" / "secretary.yaml"
            if cfg_path.exists():
                import yaml
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                trading_cfg = cfg.get("trading") if isinstance(cfg, dict) and isinstance(cfg.get("trading"), dict) else {}
                try:
                    ic = trading_cfg.get("initial_cash")
                    if ic is not None:
                        initial_cash = float(ic)
                except Exception:
                    pass
                base_model = str(trading_cfg.get("base_model") or "").strip() or None
                moe_scalper = str(trading_cfg.get("moe_scalper") or "").strip() or None
                moe_analyst = str(trading_cfg.get("moe_analyst") or "").strip() or None
                moe_secretary = str(trading_cfg.get("moe_secretary") or "").strip() or None
                moe_system2 = str(trading_cfg.get("moe_system2") or "").strip() or None
                moe_news = str(trading_cfg.get("moe_news") or "").strip() or None
                chartist_vlm_cfg = trading_cfg.get("chartist_vlm")
                news_cfg = trading_cfg.get("news")
                perf_cfg = trading_cfg.get("perf")
                llm_max_context = trading_cfg.get("llm_max_context")
                llm_max_new_tokens = trading_cfg.get("llm_max_new_tokens")
                infer_cfg = trading_cfg.get("infer")
                all_agents_mode = trading_cfg.get("all_agents_mode")
                committee_policy = str(trading_cfg.get("committee_policy") or "").strip() or None
                try:
                    lm = trading_cfg.get("load_models")
                    if lm is not None:
                        load_models = bool(lm)
                except Exception:
                    pass
                load_4bit = trading_cfg.get("load_4bit")
                planner_policy = str(trading_cfg.get("planner_policy") or "").strip() or None
                planner_sft_model = str(trading_cfg.get("planner_sft_model") or "").strip() or None
                gatekeeper_model = str(trading_cfg.get("gatekeeper_model") or "").strip() or None
                gatekeeper_threshold = trading_cfg.get("gatekeeper_threshold")
                system2_lenient = trading_cfg.get("system2_lenient")
                sim_aggressive_entry = trading_cfg.get("sim_aggressive_entry")
                allow_short = trading_cfg.get("allow_short")
                broker_cfg = trading_cfg.get("broker")
                risk_cfg = trading_cfg.get("risk")
                data_feed_interval_sec = trading_cfg.get("data_feed_interval_sec")
                md_queue_limit = trading_cfg.get("md_queue_limit")
                md_pending_limit = trading_cfg.get("md_pending_limit")
                md_symbols_per_tick = trading_cfg.get("md_symbols_per_tick")
                tickers_cfg = trading_cfg.get("tickers")
                try:
                    self._offline_playback_file = str(trading_cfg.get("offline_playback_file") or "").strip() or None
                except Exception:
                    self._offline_playback_file = None
                try:
                    self._record_enabled = bool(trading_cfg.get("record_enabled", True))
                except Exception:
                    self._record_enabled = True

                llm_cfg = cfg.get("llm") if isinstance(cfg, dict) and isinstance(cfg.get("llm"), dict) else {}
                lazy_load = llm_cfg.get("lazy_load")
                if isinstance(lazy_load, bool) and (not lazy_load):
                    # User explicitly disabled lazy loading -> force load_models=True
                    print("[Config] lazy_load=False -> forcing eager model loading")
                    load_models = True
                try:
                    v = llm_cfg.get("max_tokens_scalper")
                    if v is not None:
                        llm_max_tokens_scalper = int(v)
                except Exception:
                    pass
                try:
                    v = llm_cfg.get("max_tokens_analyst")
                    if v is not None:
                        llm_max_tokens_analyst = int(v)
                except Exception:
                    pass
                try:
                    v = llm_cfg.get("repetition_penalty")
                    if v is not None:
                        llm_repetition_penalty = float(v)
                except Exception:
                    pass
        except Exception:
            pass

        # Override trading adapters from active pointer file (if present)
        try:
            import json
            p = project_root / "data" / "finetune" / "evolution" / "active_trading_models.json"
            if p.exists():
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    s = str(obj.get("active_moe_scalper") or "").strip()
                    a = str(obj.get("active_moe_analyst") or "").strip()
                    if s:
                        moe_scalper = s
                    if a:
                        moe_analyst = a
        except Exception:
            pass

        # Apply potential initial_cash override from config to the already-created broker.
        try:
            self.broker.cash = float(initial_cash)
            self.broker.initial_cash = float(initial_cash)
        except Exception:
            pass

        # Broker leverage/margin settings (optional)
        try:
            if isinstance(broker_cfg, dict) and broker_cfg:
                if broker_cfg.get("max_leverage") is not None:
                    setattr(self.broker, "max_leverage", float(broker_cfg.get("max_leverage")))
                if broker_cfg.get("initial_margin") is not None:
                    setattr(self.broker, "initial_margin", float(broker_cfg.get("initial_margin")))
                if broker_cfg.get("maintenance_margin") is not None:
                    setattr(self.broker, "maintenance_margin", float(broker_cfg.get("maintenance_margin")))
                if broker_cfg.get("margin_interest_apr") is not None:
                    setattr(self.broker, "margin_interest_apr", float(broker_cfg.get("margin_interest_apr")))
                if broker_cfg.get("short_borrow_fee_apr") is not None:
                    setattr(self.broker, "short_borrow_fee_apr", float(broker_cfg.get("short_borrow_fee_apr")))
                if broker_cfg.get("settlement_interval_sec") is not None:
                    setattr(self.broker, "settlement_interval_sec", float(broker_cfg.get("settlement_interval_sec")))
                if broker_cfg.get("liquidation_enabled") is not None:
                    setattr(self.broker, "liquidation_enabled", bool(broker_cfg.get("liquidation_enabled")))
                if broker_cfg.get("liquidation_commission") is not None:
                    setattr(self.broker, "liquidation_commission", float(broker_cfg.get("liquidation_commission")))
        except Exception:
            pass

        effective_scalper = moe_scalper
        effective_analyst = moe_analyst

        st_kwargs = {"load_models": load_models}
        if isinstance(base_model, str) and base_model:
            st_kwargs["base_model"] = base_model
        if isinstance(moe_scalper, str) and moe_scalper:
            st_kwargs["moe_scalper"] = moe_scalper
        if isinstance(moe_analyst, str) and moe_analyst:
            st_kwargs["moe_analyst"] = moe_analyst
        if isinstance(moe_secretary, str) and moe_secretary:
            st_kwargs["moe_secretary"] = moe_secretary
        if isinstance(moe_system2, str) and moe_system2:
            st_kwargs["moe_system2"] = moe_system2
        if isinstance(moe_news, str) and moe_news:
            st_kwargs["moe_news"] = moe_news
        if isinstance(chartist_vlm_cfg, dict) and chartist_vlm_cfg:
            st_kwargs["chartist_vlm_cfg"] = dict(chartist_vlm_cfg)
        if isinstance(news_cfg, dict) and news_cfg:
            st_kwargs["news_cfg"] = dict(news_cfg)
        if isinstance(perf_cfg, dict) and perf_cfg:
            st_kwargs["perf_cfg"] = dict(perf_cfg)
        try:
            if isinstance(tickers_cfg, list) and tickers_cfg:
                st_kwargs["tickers"] = [str(x or "").upper().strip() for x in list(tickers_cfg) if str(x or "").strip()]
        except Exception:
            pass
        try:
            if llm_max_context is not None:
                st_kwargs["llm_max_context"] = int(llm_max_context)
        except Exception:
            pass
        try:
            if llm_max_new_tokens is not None:
                st_kwargs["llm_max_new_tokens"] = int(llm_max_new_tokens)
        except Exception:
            pass
        if isinstance(load_4bit, bool):
            st_kwargs["load_4bit"] = load_4bit
        if isinstance(planner_policy, str) and planner_policy:
            st_kwargs["planner_policy"] = planner_policy
        if isinstance(planner_sft_model, str) and planner_sft_model:
            st_kwargs["planner_sft_model_path"] = planner_sft_model
        if isinstance(gatekeeper_model, str) and gatekeeper_model:
            st_kwargs["gatekeeper_model_path"] = gatekeeper_model
        try:
            if gatekeeper_threshold is not None:
                st_kwargs["gatekeeper_threshold"] = float(gatekeeper_threshold)
        except Exception:
            pass
        try:
            if isinstance(system2_lenient, bool):
                st_kwargs["system2_lenient"] = bool(system2_lenient)
        except Exception:
            pass
        try:
            if isinstance(sim_aggressive_entry, bool):
                st_kwargs["sim_aggressive_entry"] = bool(sim_aggressive_entry)
        except Exception:
            pass
        try:
            if isinstance(allow_short, bool):
                st_kwargs["allow_short"] = bool(allow_short)
        except Exception:
            pass
        try:
            if isinstance(risk_cfg, dict) and risk_cfg:
                st_kwargs["risk_cfg"] = dict(risk_cfg)
        except Exception:
            pass

        self.strategy = MultiAgentStrategy(self.engine, **st_kwargs)
        try:
            if isinstance(infer_cfg, dict):
                if infer_cfg.get("tick_infer_budget_sec") is not None:
                    setattr(self.strategy, "_tick_infer_budget_sec", float(infer_cfg.get("tick_infer_budget_sec")))
                if infer_cfg.get("gen_max_time_sec") is not None:
                    setattr(self.strategy, "_gen_max_time_sec", float(infer_cfg.get("gen_max_time_sec")))
                if infer_cfg.get("scalper_gen_max_time_sec") is not None:
                    setattr(self.strategy, "_gen_max_time_sec_scalper", float(infer_cfg.get("scalper_gen_max_time_sec")))
                if infer_cfg.get("analyst_gen_max_time_sec") is not None:
                    setattr(self.strategy, "_gen_max_time_sec_analyst", float(infer_cfg.get("analyst_gen_max_time_sec")))
                if infer_cfg.get("tick_infer_budget_cap_sec") is not None:
                    setattr(self.strategy, "_tick_infer_budget_cap_sec", float(infer_cfg.get("tick_infer_budget_cap_sec")))
                if infer_cfg.get("gen_max_time_cap_sec") is not None:
                    setattr(self.strategy, "_gen_max_time_cap_sec", float(infer_cfg.get("gen_max_time_cap_sec")))
                if infer_cfg.get("max_new_tokens_cap") is not None:
                    setattr(self.strategy, "_max_new_tokens_cap", int(infer_cfg.get("max_new_tokens_cap")))
                if infer_cfg.get("inference_lock_timeout_sec") is not None:
                    setattr(self.strategy, "_inference_lock_timeout_sec", float(infer_cfg.get("inference_lock_timeout_sec")))
        except Exception:
            pass

        try:
            if llm_max_tokens_scalper is not None:
                setattr(self.strategy, "llm_max_new_tokens_scalper", int(llm_max_tokens_scalper))
        except Exception:
            pass
        try:
            if llm_max_tokens_analyst is not None:
                setattr(self.strategy, "llm_max_new_tokens_analyst", int(llm_max_tokens_analyst))
        except Exception:
            pass
        try:
            if llm_repetition_penalty is not None:
                setattr(self.strategy, "llm_repetition_penalty", float(llm_repetition_penalty))
        except Exception:
            pass
        try:
            if isinstance(all_agents_mode, bool):
                setattr(self.strategy, "all_agents_mode", bool(all_agents_mode))
        except Exception:
            pass
        try:
            if isinstance(committee_policy, str) and committee_policy:
                setattr(self.strategy, "committee_policy", str(committee_policy))
        except Exception:
            pass
        self.mari = MariVoice(strategy=self.strategy)

        self.load_models = bool(load_models)
        
        self.engine.broker = self.broker
        self.engine.strategy = self.strategy
        
        # Data feed
        self.data_feed: Optional[DataFeed] = None
        self.data_source = data_source
        
        # Paper trading data for system upgrade
        self.trade_log: List[Dict] = []
        self.pnl_history: List[Dict] = []
        self.price_history: Dict[str, List[Dict]] = {}  # For UI charts
        self.agent_logs: List[Dict] = []  # For dashboard terminal
        self.initial_cash = float(initial_cash)
        self.last_nav = float(initial_cash)
        self.currency = "CAD"
        
        # Significant event thresholds
        self.volatility_threshold = 0.02  # 2% move triggers alert
        self.profit_threshold = 1000  # $1000 gain triggers celebration
        self.loss_threshold = -500  # $500 loss triggers concern
        
        # Control terminal verbosity
        self.verbose_terminal = False  # Set to True for debug
        
        # Online RL learning manager
        self.rl_manager = get_online_learning_manager()
        self._pending_trades: Dict[str, Dict] = {}  # track open positions for RL
        
        # Trading mode (online = real-time, offline = backtest playback)
        self.trading_mode = "online"
        self._offline_thread: Optional[threading.Thread] = None
        self._offline_running = False

        self._record_enabled: bool = True
        self._record_path: Optional[Path] = None
        self._record_fp: Any = None
        self._record_last_news_keys: set[str] = set()
        self._offline_playback_file: Optional[str] = None

        self._started = False

        self._md_counter = 0

        self._md_pending: deque = deque()
        try:
            self._md_queue_limit = int(md_queue_limit) if md_queue_limit is not None else 3
        except Exception:
            self._md_queue_limit = 3
        try:
            self._md_pending_limit = int(md_pending_limit) if md_pending_limit is not None else 80
        except Exception:
            self._md_pending_limit = 80

        try:
            self._md_symbols_per_tick = int(md_symbols_per_tick) if md_symbols_per_tick is not None else 0
        except Exception:
            self._md_symbols_per_tick = 0

        try:
            self._data_feed_interval_sec = float(data_feed_interval_sec) if data_feed_interval_sec is not None else 5.0
        except Exception:
            self._data_feed_interval_sec = 5.0

        self._backfill_last_attempt_ts: float = 0.0
        self._backfill_cooldown_sec: float = 180.0

        try:
            t_str = datetime.now().strftime("%H:%M:%S")
            self.agent_logs.append(
                {
                    "time": t_str,
                    "type": "agent",
                    "priority": 2,
                    "message": f"[Models] scalper={effective_scalper} | analyst={effective_analyst}",
                }
            )
        except Exception:
            pass
        
        # Hook into event handling
        self._original_handle = self.engine._handle_event
        self.engine._handle_event = self._wrapped_handle_event

        # Session recorder (jsonl)
        try:
            if bool(getattr(self, "_record_enabled", True)):
                rec_dir = project_root / "data" / "paper_trading" / "recordings"
                rec_dir.mkdir(parents=True, exist_ok=True)
                ts0 = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._record_path = rec_dir / f"session_{ts0}.jsonl"
                self._record_fp = self._record_path.open("a", encoding="utf-8")
                t_str = datetime.now().strftime("%H:%M:%S")
                self.agent_logs.append({
                    "time": t_str,
                    "type": "system",
                    "priority": 2,
                    "message": f"[Record] started: {self._record_path.name}",
                })
        except Exception:
            self._record_fp = None
            self._record_path = None

    def _flush_md_pending(self) -> None:
        try:
            q = getattr(self.engine, "events", None)
            qsz = int(q.qsize()) if (q is not None and hasattr(q, "qsize")) else 0
        except Exception:
            qsz = 0

        pushed = 0
        while qsz < int(getattr(self, "_md_queue_limit", 3) or 3) and self._md_pending:
            try:
                bar = self._md_pending.popleft()
            except Exception:
                break
            try:
                event = Event(
                    EventType.MARKET_DATA,
                    datetime.now(),
                    bar,
                    priority=0,
                )
                self.engine.push_event(event)
                pushed += 1
            except Exception:
                break

            try:
                qsz = int(q.qsize()) if (q is not None and hasattr(q, "qsize")) else qsz + 1
            except Exception:
                qsz += 1

        if pushed <= 0:
            return

    def _wrapped_handle_event(self, event: Event) -> None:
        """Wrapped event handler - logs to agent_logs for dashboard, minimal terminal output"""
        self._original_handle(event)
        t_str = event.timestamp.strftime("%H:%M:%S")

        # Keep the engine busy: as soon as one MARKET_DATA tick is processed, enqueue the next
        # pending tick (bounded by _md_queue_limit). This reduces md_pending overflow.
        try:
            if event.type == EventType.MARKET_DATA:
                self._flush_md_pending()
        except Exception:
            pass

        if event.type == EventType.LOG:
            msg = str(event.payload)
            priority = event.priority

            # Record news logs (for later analysis) without blowing up disk.
            try:
                if self._record_fp is not None:
                    m0 = str(msg or "")
                    ml = m0.lower()
                    if "[news]" in ml or "[newssignal]" in ml:
                        self._record_fp.write(json.dumps({
                            "type": "log",
                            "ts": event.timestamp.isoformat(),
                            "message": m0,
                            "priority": int(priority),
                        }, ensure_ascii=False) + "\n")
                        self._record_fp.flush()
            except Exception:
                pass
            
            # Store in agent_logs for dashboard terminal
            self.agent_logs.append({
                "time": t_str,
                "type": "agent",
                "priority": priority,
                "message": msg,
            })
            # Keep logs bounded
            if len(self.agent_logs) > 500:
                self.agent_logs = self.agent_logs[-300:]
            
            # Only print high priority to terminal (reduces spam)
            if self.verbose_terminal or priority >= 2:
                print(f"[{t_str}] [Agent] {msg}")

        elif event.type == EventType.SIGNAL or event.type == EventType.ORDER:
            p = event.payload if isinstance(event.payload, dict) else {}
            ticker = str(p.get("ticker") or p.get("symbol") or "?").upper().strip()
            action = str(p.get("action") or "?").upper().strip()
            try:
                price = float(p.get("price") or 0.0)
            except Exception:
                price = 0.0
            try:
                shares = float(p.get("shares") or 0.0)
            except Exception:
                shares = 0.0
            expert = str(p.get("expert") or "").strip()
            et = "SIGNAL" if event.type == EventType.SIGNAL else "ORDER"

            try:
                msg = f"[Execution] {et} {action} {ticker} x{shares:g} @ ${price:.2f}" + (f" (expert={expert})" if expert else "")
                self.agent_logs.append({
                    "time": t_str,
                    "type": "fill",
                    "priority": 2,
                    "message": msg,
                })
                if len(self.agent_logs) > 500:
                    self.agent_logs = self.agent_logs[-300:]
            except Exception:
                pass

            try:
                if self._record_fp is not None and isinstance(p, dict):
                    self._record_fp.write(json.dumps({
                        "type": et.lower(),
                        "ts": event.timestamp.isoformat(),
                        "payload": p,
                    }, ensure_ascii=False) + "\n")
                    self._record_fp.flush()
            except Exception:
                pass

        elif event.type == EventType.FILL:
            fill = event.payload
            ticker = fill.get("ticker", "?")
            action = fill.get("action", "?")
            price = fill.get("price", 0)
            shares = fill.get("shares", 0)
            
            # Record trade for data collection
            trade_id = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            trade_record = {
                "time": datetime.now().isoformat(),
                "trade_id": trade_id,
                "ticker": ticker,
                "action": action,
                "price": price,
                "shares": shares,
            }
            self.trade_log.append(trade_record)

            try:
                if self._record_fp is not None and isinstance(fill, dict):
                    self._record_fp.write(json.dumps({
                        "type": "fill",
                        "ts": event.timestamp.isoformat(),
                        "payload": dict(fill),
                    }, ensure_ascii=False) + "\n")
                    self._record_fp.flush()
            except Exception:
                pass

            try:
                self.agent_logs.append({
                    "time": t_str,
                    "type": "fill",
                    "priority": 2,
                    "message": f"[Execution] FILL {action} {ticker} x{shares} @ ${float(price):.2f}",
                })
                if len(self.agent_logs) > 500:
                    self.agent_logs = self.agent_logs[-300:]
            except Exception:
                pass
            
            # Online RL: Track trades for learning
            if action == "BUY":
                try:
                    md = {
                        "expert": fill.get("expert", "unknown"),
                        "analysis": fill.get("analysis", ""),
                        "trace_id": fill.get("trace_id"),
                        "trace_ids": fill.get("trace_ids"),
                        "chart_score": fill.get("chart_score"),
                        "news_score": fill.get("news_score"),
                        "news_sentiment": fill.get("news_sentiment"),
                        "news_summary": fill.get("news_summary"),
                    }
                except Exception:
                    md = {}
                self._pending_trades[ticker] = {
                    "trade_id": trade_id,
                    "entry_price": price,
                    "shares": shares,
                    "entry_time": datetime.now(),
                    "state": self._get_current_state(ticker),
                    "metadata": md,
                }
                # Log decision for DPO preference pairs
                self.rl_manager.preference_logger.log_decision(
                    trade_id=trade_id,
                    context={
                        "ticker": ticker,
                        "price": price,
                        "trace_id": fill.get("trace_id"),
                        "trace_ids": fill.get("trace_ids"),
                        "chart_score": fill.get("chart_score"),
                    },
                    decision=action,
                    reasoning=fill.get("analysis", ""),
                    expert=fill.get("expert", "unknown"),
                )
            elif action == "SELL" and ticker in self._pending_trades:
                pending = self._pending_trades.pop(ticker)
                pnl = (price - pending["entry_price"]) * pending["shares"]
                hold_bars = 1  # simplified
                
                # Record experience for RL
                self.rl_manager.on_trade_complete(
                    trade_id=pending["trade_id"],
                    state=pending["state"],
                    action="BUY",
                    pnl=pnl,
                    drawdown_pct=0,
                    hold_bars=hold_bars,
                    exit_reason="signal",
                    metadata=pending.get("metadata") if isinstance(pending, dict) else None,
                )
            
            # Log to agent_logs
            self.agent_logs.append({
                "time": t_str,
                "type": "fill",
                "priority": 3,
                "message": f"[FILL] {action} {ticker} x{shares} @ ${price:.2f}",
            })
            
            # Always print fills to terminal (important)
            print(f"[{t_str}] [FILL] {action} {ticker} x{shares} @ ${price:.2f}")

        elif event.type == EventType.ERROR:
            self.agent_logs.append({
                "time": t_str,
                "type": "error",
                "priority": 3,
                "message": str(event.payload),
            })
            print(f"[{t_str}] [ERROR] {event.payload}")
    
    def _check_significant_events(self, nav: float) -> None:
        """Check for significant events - log only, Mari speaks when user asks"""
        pnl = nav - self.last_nav
        pnl_pct = pnl / self.last_nav if self.last_nav > 0 else 0
        
        # Record PnL snapshot
        self.pnl_history.append({
            "time": datetime.now().isoformat(),
            "nav": nav,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        })
        
        t_str = datetime.now().strftime("%H:%M:%S")
        
        # Log significant events (no auto-speak, Mari only talks when asked)
        if pnl >= self.profit_threshold:
            self.agent_logs.append({
                "time": t_str, "type": "pnl", "priority": 2,
                "message": f"[PnL] 盈利 ${pnl:.0f} ({pnl_pct*100:.2f}%)",
            })
            self.last_nav = nav
        elif pnl <= self.loss_threshold:
            self.agent_logs.append({
                "time": t_str, "type": "pnl", "priority": 2,
                "message": f"[PnL] 亏损 ${abs(pnl):.0f} ({pnl_pct*100:.2f}%)",
            })
            self.last_nav = nav
        elif abs(pnl_pct) >= self.volatility_threshold:
            direction = "上涨" if pnl > 0 else "下跌"
            self.agent_logs.append({
                "time": t_str, "type": "volatility", "priority": 2,
                "message": f"[波动] 市场{direction} {abs(pnl_pct)*100:.1f}%",
            })

    def _get_current_state(self, ticker: str) -> Dict[str, Any]:
        """Get current market state for RL"""
        state = {"ticker": ticker}
        if ticker in self.price_history and self.price_history[ticker]:
            recent = self.price_history[ticker][-20:]
            prices = [p.get("close", 0) for p in recent]
            if prices:
                state["current_price"] = prices[-1]
                state["price_mean_20"] = sum(prices) / len(prices)
                state["price_std_20"] = (sum((p - state["price_mean_20"])**2 for p in prices) / len(prices)) ** 0.5
        state["cash"] = self.broker.cash
        state["positions"] = len(getattr(self.broker, "positions", {}))
        return state

    def _on_market_data(self, data: Dict) -> None:
        """Handle incoming market data from data feed"""
        ticker = data.get("ticker", "")
        if not ticker:
            ticker = data.get("symbol", "")
        ticker = str(ticker or "").upper().strip()
        price = data.get("close", 0)
        try:
            price = float(price)
        except Exception:
            price = 0.0

        def _f(x: Any, fallback: float) -> float:
            try:
                v = float(x)
                if v != v or v in (float("inf"), float("-inf")):
                    return float(fallback)
                return v
            except Exception:
                return float(fallback)
        
        # Store for UI charts
        if ticker not in self.price_history:
            self.price_history[ticker] = []
        ts = data.get("time", datetime.now())
        time_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

        src = ""
        try:
            src = str(data.get("source") or "").strip().lower()
        except Exception:
            src = ""
        try:
            last_src = ""
            if self.price_history[ticker]:
                last_src = str(self.price_history[ticker][-1].get("source") or "").strip().lower()
            if src and last_src and src != last_src:
                if src == "yfinance":
                    self.price_history[ticker] = []
                if src == "simulated" and last_src == "yfinance":
                    # Do not pollute real history with simulated fallback ticks
                    return
        except Exception:
            pass

        o = _f(data.get("open", price), price)
        h = _f(data.get("high", price), price)
        l = _f(data.get("low", price), price)
        c = _f(data.get("close", price), price)
        if c <= 0 and price > 0:
            c = float(price)
        if o <= 0:
            o = c
        if h <= 0:
            h = max(o, c)
        if l <= 0:
            l = min(o, c)
        hi = max(o, h, l, c)
        lo = min(o, h, l, c)

        bar = {
            "ticker": str(ticker or "").upper(),
            "time": time_str,
            "open": o,
            "high": hi,
            "low": lo,
            "close": c,
            "volume": _f(data.get("volume", 0), 0.0),
            "source": data.get("source", ""),
        }
        if self.price_history[ticker] and self.price_history[ticker][-1].get("time") == time_str:
            self.price_history[ticker][-1] = bar
        else:
            self.price_history[ticker].append(bar)
        if len(self.price_history[ticker]) > 800:
            self.price_history[ticker] = self.price_history[ticker][-800:]
        
        print(f">> {ticker} @ ${price:.2f}")

        # Record the market bar (raw input for future replay)
        try:
            if self._record_fp is not None:
                self._record_fp.write(json.dumps({
                    "type": "market_data",
                    "ts": time_str,
                    "bar": bar,
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

        # Record the news signal actually used by strategy cache (once per ticker/date)
        try:
            if self._record_fp is not None and getattr(self, "strategy", None) is not None:
                asof_dt = None
                try:
                    asof_dt = datetime.fromisoformat(str(time_str))
                except Exception:
                    asof_dt = None
                ns = {}
                try:
                    ns = self.strategy._get_news_signal(ticker, asof_time=asof_dt)  # non-blocking cache read
                except Exception:
                    ns = {}
                if isinstance(ns, dict) and ns:
                    k0 = ticker
                    try:
                        if isinstance(ns.get("asof"), str) and ns.get("asof"):
                            ad = datetime.fromisoformat(str(ns.get("asof")))
                            k0 = f"{ticker}|{ad.strftime('%Y-%m-%d')}"
                    except Exception:
                        k0 = ticker
                    if k0 not in self._record_last_news_keys:
                        self._record_last_news_keys.add(k0)
                        self._record_fp.write(json.dumps({
                            "type": "news_signal",
                            "ts": str(ns.get("asof") or time_str),
                            "ticker": ticker,
                            "signal": ns,
                        }, ensure_ascii=False) + "\n")
        except Exception:
            pass

        try:
            if self._record_fp is not None:
                self._record_fp.flush()
        except Exception:
            pass

        # Small-window feeding: buffer ticks and only enqueue when engine queue is small.
        try:
            # Deduplicate pending bars per ticker: keep only the most recent bar for each ticker.
            replaced = False
            try:
                for i in range(len(self._md_pending) - 1, -1, -1):
                    try:
                        b0 = self._md_pending[i]
                    except Exception:
                        b0 = None
                    try:
                        if isinstance(b0, dict) and str(b0.get("ticker") or "").upper().strip() == ticker:
                            self._md_pending[i] = bar
                            replaced = True
                            break
                    except Exception:
                        continue
            except Exception:
                replaced = False

            if not replaced:
                self._md_pending.append(bar)
            try:
                if getattr(self, "strategy", None) is not None:
                    setattr(self.strategy, "_md_backlog_hint", int(len(self._md_pending)))
            except Exception:
                pass
            if len(self._md_pending) > int(getattr(self, "_md_pending_limit", 80) or 80):
                # Drop oldest to keep recent market data.
                drop_n = len(self._md_pending) - int(getattr(self, "_md_pending_limit", 80) or 80)
                for _ in range(max(1, int(drop_n))):
                    try:
                        self._md_pending.popleft()
                    except Exception:
                        break

                try:
                    self._md_counter = int(getattr(self, "_md_counter", 0) or 0) + 1
                    if self._md_counter % 10 == 0:
                        t_str = datetime.now().strftime("%H:%M:%S")
                        self.agent_logs.append({
                            "time": t_str,
                            "type": "agent",
                            "priority": 2,
                            "message": f"[Engine] md_pending overflow: dropped old ticks (pending={len(self._md_pending)})",
                        })
                        if len(self.agent_logs) > 500:
                            self.agent_logs = self.agent_logs[-300:]
                except Exception:
                    pass
        except Exception:
            return

        try:
            self._flush_md_pending()
        except Exception:
            return

        try:
            self._md_counter = int(getattr(self, "_md_counter", 0) or 0) + 1
            if self._md_counter % 30 == 0:
                t_str = datetime.now().strftime("%H:%M:%S")
                qsz = None
                eng_ok = None
                try:
                    eng_ok = bool(getattr(self.engine, "is_running", False))
                except Exception:
                    eng_ok = None
                try:
                    q = getattr(self.engine, "events", None)
                    if q is not None and hasattr(q, "qsize"):
                        qsz = int(q.qsize())
                except Exception:
                    qsz = None
                self.agent_logs.append({
                    "time": t_str,
                    "type": "agent",
                    "priority": 2,
                    "message": f"[MarketData] feed tick {str(ticker or '').upper()} close={c:.4f} source={bar.get('source','')} engine={eng_ok} q={qsz}",
                })
                if len(self.agent_logs) > 500:
                    self.agent_logs = self.agent_logs[-300:]
        except Exception:
            pass

    def reset_price_history(self, ticker: str | None = None) -> None:
        try:
            if ticker is None:
                self.price_history = {}
                return
            tk = str(ticker or "").upper()
            if tk:
                self.price_history[tk] = []
        except Exception:
            return

    def backfill_intraday(self, *, max_bars: int = 600) -> bool:
        try:
            import yfinance as yf
        except Exception:
            return False

        ok = False
        tickers = [str(x or "").upper().strip() for x in list(getattr(self.strategy, "tickers", []) or [])]
        tickers = [x for x in tickers if x]
        if not tickers:
            return False

        dl_err = ""
        def _try_download(*, period: str, interval: str):
            nonlocal dl_err
            try:
                dl_err = ""
                return yf.download(
                    tickers=" ".join(tickers),
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=False,
                    prepost=True,
                    threads=False,
                    progress=False,
                )
            except Exception as e:
                dl_err = str(e)
                return None

        last_err: Dict[str, str] = {}

        picked = ""
        picked_hist = None
        for period, interval in (("1d", "1m"), ("5d", "1m"), ("5d", "5m"), ("1mo", "15m"), ("1mo", "1h")):
            h0 = _try_download(period=period, interval=interval)
            if h0 is None or getattr(h0, "empty", True):
                continue
            picked = f"{period}/{interval}"
            picked_hist = h0
            break

        def _build_bars_from_hist(*, hist: Any, tk: str, only_last_day: bool) -> list[dict]:
            out: list[dict] = []
            try:
                if hist is None or getattr(hist, "empty", True):
                    return out
            except Exception:
                return out

            df = None
            try:
                cols = getattr(hist, "columns", None)
                if cols is not None:
                    # Single-ticker DataFrame
                    try:
                        if ("Open" in cols and "Close" in cols):
                            df = hist
                        else:
                            df = None
                    except Exception:
                        df = None

                    # Multi-ticker download (group_by='ticker') often returns a DataFrame with MultiIndex columns.
                    if df is None:
                        try:
                            if int(getattr(cols, "nlevels", 1) or 1) > 1:
                                df = hist[tk]
                        except Exception:
                            df = None

                if df is None:
                    try:
                        df = hist.get(tk)
                    except Exception:
                        df = None
            except Exception:
                df = None

            if df is None or getattr(df, "empty", True):
                return out

            idx = list(getattr(df, "index", []) or [])
            if not idx:
                return out

            last_day = None
            if only_last_day:
                last_ts = idx[-1]
                try:
                    if hasattr(last_ts, "to_pydatetime"):
                        last_ts = last_ts.to_pydatetime()
                except Exception:
                    pass
                last_day = last_ts.date() if hasattr(last_ts, "date") else None

            close_s = df.get("Close")
            open_s = df.get("Open")
            high_s = df.get("High")
            low_s = df.get("Low")
            vol_s = df.get("Volume")
            n = len(idx)

            def _f(x: Any, fallback: float) -> float:
                try:
                    v = float(x)
                    if v != v or v in (float("inf"), float("-inf")):
                        return float(fallback)
                    return v
                except Exception:
                    return float(fallback)

            for i in range(n):
                ts = idx[i]
                try:
                    if hasattr(ts, "to_pydatetime"):
                        ts = ts.to_pydatetime()
                except Exception:
                    pass
                if only_last_day and last_day is not None:
                    try:
                        if hasattr(ts, "date") and ts.date() != last_day:
                            continue
                    except Exception:
                        pass
                time_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                c = _f(close_s.iloc[i], 0.0) if close_s is not None else 0.0
                if c <= 0:
                    continue
                o = _f(open_s.iloc[i], c) if open_s is not None else c
                h = _f(high_s.iloc[i], max(o, c)) if high_s is not None else max(o, c)
                l = _f(low_s.iloc[i], min(o, c)) if low_s is not None else min(o, c)
                hi = max(o, h, l, c)
                lo = min(o, h, l, c)
                v = _f(vol_s.iloc[i], 0.0) if vol_s is not None else 0.0
                out.append({
                    "time": time_str,
                    "open": o,
                    "high": hi,
                    "low": lo,
                    "close": c,
                    "volume": v,
                    "source": "yfinance",
                })
            return out

        for tk in tickers:
            try:
                hist = None
                if picked_hist is not None:
                    hist = picked_hist
                else:
                    try:
                        hist = yf.Ticker(tk).history(period="5d", interval="5m", auto_adjust=False, prepost=True)
                        picked = "5d/5m"
                    except Exception as e:
                        last_err[tk] = str(e)
                        hist = None

                if hist is None or getattr(hist, "empty", True):
                    continue

                try:
                    bars = _build_bars_from_hist(hist=hist, tk=tk, only_last_day=True)
                    if len(bars) < 50:
                        bars = _build_bars_from_hist(hist=hist, tk=tk, only_last_day=False)
                except Exception:
                    continue

                if not bars:
                    continue

                if max_bars > 0 and len(bars) > int(max_bars):
                    bars = bars[-int(max_bars):]

                self.price_history[tk] = bars
                if len(bars) >= 5:
                    ok = True

                try:
                    t_str = datetime.now().strftime("%H:%M:%S")
                    self.agent_logs.append(
                        {
                            "time": t_str,
                            "type": "agent",
                            "priority": 2,
                            "message": f"[Backfill] {tk} source=yfinance picked={picked or '?'} bars={len(bars)}",
                        }
                    )
                    if len(self.agent_logs) > 500:
                        self.agent_logs = self.agent_logs[-300:]
                except Exception:
                    pass
            except Exception:
                try:
                    last_err[tk] = "unknown error"
                except Exception:
                    pass
                continue

        if not ok:
            try:
                t_str = datetime.now().strftime("%H:%M:%S")
                err_sample = None
                try:
                    if last_err:
                        k0 = sorted(list(last_err.keys()))[0]
                        err_sample = f"{k0}: {last_err.get(k0)}"
                except Exception:
                    err_sample = None
                if (not err_sample) and dl_err:
                    err_sample = f"download: {dl_err}"
                self.agent_logs.append(
                    {
                        "time": t_str,
                        "type": "agent",
                        "priority": 2,
                        "message": f"[Backfill] yfinance intraday FAIL" + (f" | err={err_sample}" if err_sample else ""),
                    }
                )
                if len(self.agent_logs) > 500:
                    self.agent_logs = self.agent_logs[-300:]
            except Exception:
                pass

        return ok

    def start(self) -> None:
        """Start the live paper trading engine"""
        if self._started:
            return
        self._started = True
        self.engine.start()

        try:
            self._restore_price_history()
        except Exception:
            pass

        try:
            if bool(getattr(self, "load_models", False)):
                stg = getattr(self, "strategy", None)
                if stg is not None:
                    if not bool(getattr(stg, "models_loaded", False)):
                        fn = getattr(stg, "load_models", None)
                        if callable(fn):
                            try:
                                t_str = datetime.now().strftime("%H:%M:%S")
                                self.agent_logs.append({
                                    "time": t_str,
                                    "type": "agent",
                                    "priority": 2,
                                    "message": "[LoadModels] start",
                                })
                                if len(self.agent_logs) > 500:
                                    self.agent_logs = self.agent_logs[-300:]
                            except Exception:
                                pass
                            fn()
                            try:
                                t_str = datetime.now().strftime("%H:%M:%S")
                                self.agent_logs.append({
                                    "time": t_str,
                                    "type": "agent",
                                    "priority": 2,
                                    "message": "[LoadModels] done",
                                })
                                if len(self.agent_logs) > 500:
                                    self.agent_logs = self.agent_logs[-300:]
                            except Exception:
                                pass
                    fn2 = getattr(stg, "_warmup_kv_cache", None)
                    if callable(fn2):
                        try:
                            t_str = datetime.now().strftime("%H:%M:%S")
                            self.agent_logs.append({
                                "time": t_str,
                                "type": "agent",
                                "priority": 2,
                                "message": "[Warmup] start",
                            })
                            if len(self.agent_logs) > 500:
                                self.agent_logs = self.agent_logs[-300:]
                        except Exception:
                            pass

                        fn2()

                        try:
                            t_str = datetime.now().strftime("%H:%M:%S")
                            self.agent_logs.append({
                                "time": t_str,
                                "type": "agent",
                                "priority": 2,
                                "message": "[Warmup] done",
                            })
                            if len(self.agent_logs) > 500:
                                self.agent_logs = self.agent_logs[-300:]
                        except Exception:
                            pass
        except Exception:
            pass
        
        # Initialize data feed
        self.data_feed = create_data_feed(
            self.strategy.tickers,
            source=self.data_source,
            interval_sec=float(getattr(self, "_data_feed_interval_sec", 5.0) or 5.0),
            symbols_per_tick=int(getattr(self, "_md_symbols_per_tick", 0) or 0),
        )
        self.data_feed.subscribe(self._on_market_data)
        self.data_feed.start()

        try:
            t_str = datetime.now().strftime("%H:%M:%S")
            src = str(getattr(self.data_feed, "source", "") or "")
            itv = float(getattr(self.data_feed, "interval_sec", 0.0) or 0.0)
            spt = int(getattr(self.data_feed, "_symbols_per_tick", 0) or 0)
            self.agent_logs.append({
                "time": t_str,
                "type": "agent",
                "priority": 2,
                "message": f"[DataFeed] source={src} interval_sec={itv:.1f} symbols_per_tick={spt}",
            })
            if len(self.agent_logs) > 500:
                self.agent_logs = self.agent_logs[-300:]
        except Exception:
            pass

        try:
            ds = str(getattr(self, "data_source", "auto") or "auto").lower()
        except Exception:
            ds = "auto"

        try:
            actual = str(getattr(self.data_feed, "source", "") or "").strip().lower()
            if actual and actual != ds:
                self.data_source = actual
                ds = actual
        except Exception:
            pass

        def _seed_strategy_history() -> None:
            try:
                ph = getattr(self, "price_history", {})
                st = getattr(self, "strategy", None)
                if st is None or not isinstance(ph, dict):
                    return
                for tk, bars in ph.items():
                    if not isinstance(bars, list) or not bars:
                        continue
                    picked_bars = list(bars)
                    if ds in {"yfinance", "auto"}:
                        try:
                            yf_bars = [b for b in picked_bars if isinstance(b, dict) and str(b.get("source") or "").lower() == "yfinance"]
                            if yf_bars:
                                picked_bars = yf_bars
                        except Exception:
                            pass
                    try:
                        st.price_history[str(tk).upper()] = list(picked_bars)[-60:]
                    except Exception:
                        continue
            except Exception:
                return

        try:
            _seed_strategy_history()
        except Exception:
            pass

        def _maybe_backfill() -> None:
            try:
                import time

                need = False
                try:
                    ph = getattr(self, "price_history", {})
                    if not isinstance(ph, dict) or not ph:
                        need = True
                    else:
                        for tk in list(getattr(self.strategy, "tickers", []) or []):
                            tku = str(tk or "").upper()
                            if not tku:
                                continue
                            bars = ph.get(tku)
                            if not isinstance(bars, list):
                                need = True
                                break
                            if ds in {"yfinance", "auto"}:
                                try:
                                    yf_n = sum(1 for b in bars if isinstance(b, dict) and str(b.get("source") or "").lower() == "yfinance")
                                except Exception:
                                    yf_n = 0
                                if yf_n < 120:
                                    need = True
                                    break
                            else:
                                if len(bars) < 120:
                                    need = True
                                    break
                except Exception:
                    need = True

                if ds not in {"yfinance", "auto"}:
                    return
                if not need:
                    return

                try:
                    now = float(time.time())
                    last = float(getattr(self, "_backfill_last_attempt_ts", 0.0) or 0.0)
                    cooldown = float(getattr(self, "_backfill_cooldown_sec", 180.0) or 180.0)
                    if last > 0 and (now - last) < cooldown:
                        return
                    self._backfill_last_attempt_ts = now
                except Exception:
                    pass

                ok = self.backfill_intraday(max_bars=800)
                try:
                    # backfill_intraday already logs FAIL with an error sample.
                    # Only emit a startup OK marker to avoid duplicate FAIL spam.
                    if ok:
                        t_str = datetime.now().strftime("%H:%M:%S")
                        self.agent_logs.append(
                            {
                                "time": t_str,
                                "type": "agent",
                                "priority": 2,
                                "message": "[Backfill] startup yfinance intraday OK",
                            }
                        )
                        if len(self.agent_logs) > 500:
                            self.agent_logs = self.agent_logs[-300:]
                except Exception:
                    pass

                try:
                    _seed_strategy_history()
                except Exception:
                    pass
            except Exception:
                return

        try:
            threading.Thread(target=_maybe_backfill, daemon=True).start()
        except Exception:
            pass
        
        print("=" * 60)
        print("Phase 3.4: Live Paper Trading Engine")
        print("=" * 60)
        print(f"Initial Cash: ${self.broker.cash:,.2f}")
        print(f"Tickers: {', '.join(self.strategy.tickers)}")
        print(f"Data Source: {self.data_source}")
        print("=" * 60)
        
        # Mari startup announcement disabled
        # self.mari.speak("实盘模拟系统已启动，Mari 开始监视 Agent 活动。")

    def stop(self) -> None:
        """Stop the engine and save trading data"""
        if not self._started:
            return
        self.stop_offline_playback()
        if self.data_feed:
            self.data_feed.stop()
        self.engine.stop()
        self.mari.stop()
        
        # Save paper trading data for system upgrade
        self._save_trading_data()

        try:
            if self._record_fp is not None:
                self._record_fp.flush()
                self._record_fp.close()
        except Exception:
            pass
        self._record_fp = None
        print("System Shutdown.")
        self._started = False
    
    def start_offline_playback(self) -> None:
        """Start offline backtest playback mode"""
        if self._offline_running:
            return
        
        self._offline_running = True
        self.trading_mode = "offline"
        
        # Reset for fresh backtest
        self.broker.cash = self.initial_cash
        self.broker.positions = {}
        self.trade_log.clear()
        self.pnl_history.clear()
        self.price_history.clear()
        self.agent_logs.clear()
        
        self.agent_logs.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": "system",
            "priority": 2,
            "message": "[System] Offline mode started - replaying historical data",
        })
        
        def _playback_thread():
            # Prefer replay from a recorded session jsonl.
            data_file = None
            try:
                if isinstance(getattr(self, "_offline_playback_file", None), str) and self._offline_playback_file:
                    p0 = Path(str(self._offline_playback_file))
                    if not p0.is_absolute():
                        p0 = (project_root / p0).resolve()
                    if p0.exists():
                        data_file = p0
            except Exception:
                data_file = None

            if data_file is None:
                # Load historical data from results
                data_file = project_root / "data" / "historical" / "sample_ohlc.csv"
            if not data_file.exists():
                # Try to find any CSV in data folder
                for f in (project_root / "data").rglob("*.csv"):
                    if "ohlc" in f.name.lower() or "price" in f.name.lower():
                        data_file = f
                        break
            
            if not data_file.exists():
                self.agent_logs.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "error",
                    "priority": 3,
                    "message": "[System] No historical data found for offline mode",
                })
                self._offline_running = False
                return
            
            try:
                # jsonl recording replay
                if str(data_file).lower().endswith(".jsonl"):
                    lines = data_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                    for ln in lines:
                        if not self._offline_running:
                            break
                        if not ln.strip():
                            continue
                        try:
                            obj = json.loads(ln)
                        except Exception:
                            continue
                        tp = str(obj.get("type") or "").strip().lower()
                        if tp == "news_signal":
                            try:
                                tk0 = str(obj.get("ticker") or "").upper().strip()
                                sig0 = obj.get("signal") if isinstance(obj.get("signal"), dict) else None
                                if tk0 and isinstance(sig0, dict) and getattr(self, "strategy", None) is not None:
                                    fn = getattr(self.strategy, "inject_news_signal", None)
                                    if callable(fn):
                                        fn(tk0, sig0)
                            except Exception:
                                pass
                            continue
                        if tp != "market_data":
                            continue
                        bar0 = obj.get("bar") if isinstance(obj.get("bar"), dict) else {}
                        if not isinstance(bar0, dict):
                            continue
                        self._on_market_data(dict(bar0))
                        time.sleep(0.2)
                else:
                    import pandas as pd
                    df = pd.read_csv(data_file)
                    
                    for _, row in df.iterrows():
                        if not self._offline_running:
                            break
                        
                        ticker = row.get("ticker", row.get("symbol", "NVDA"))
                        t_row = None
                        try:
                            for k in ("time", "datetime", "date", "timestamp"):
                                if k in row and str(row.get(k) or "").strip():
                                    t_row = row.get(k)
                                    break
                        except Exception:
                            t_row = None
                        try:
                            if t_row is not None:
                                t_parsed = pd.to_datetime(t_row, errors="coerce")
                                if pd.notna(t_parsed):
                                    t_row = t_parsed.to_pydatetime()
                                else:
                                    t_row = None
                        except Exception:
                            t_row = None

                        data = {
                            "ticker": str(ticker).upper(),
                            "open": float(row.get("open", row.get("Open", 0))),
                            "high": float(row.get("high", row.get("High", 0))),
                            "low": float(row.get("low", row.get("Low", 0))),
                            "close": float(row.get("close", row.get("Close", 0))),
                            "volume": float(row.get("volume", row.get("Volume", 0))),
                            "time": t_row or datetime.now(),
                        }
                        
                        self._on_market_data(data)
                        time.sleep(0.5)  # Simulate real-time pace
                
                self.agent_logs.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "system",
                    "priority": 2,
                    "message": "[System] Offline playback completed",
                })
            except Exception as e:
                self.agent_logs.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "error",
                    "priority": 3,
                    "message": f"[System] Offline playback error: {e}",
                })
            finally:
                self._offline_running = False
        
        self._offline_thread = threading.Thread(target=_playback_thread, daemon=True)
        self._offline_thread.start()
    
    def stop_offline_playback(self) -> None:
        """Stop offline backtest playback"""
        self._offline_running = False
        self.trading_mode = "online"
        if self._offline_thread and self._offline_thread.is_alive():
            self._offline_thread.join(timeout=1.0)
    
    def _save_trading_data(self) -> None:
        """Save trading data for future system improvements"""
        data_dir = project_root / "data" / "paper_trading"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trade log
        if self.trade_log:
            trade_file = data_dir / f"trades_{timestamp}.json"
            trade_file.write_text(json.dumps(self.trade_log, indent=2, ensure_ascii=False))
            print(f"Saved {len(self.trade_log)} trades to {trade_file.name}")
        
        # Save PnL history
        if self.pnl_history:
            pnl_file = data_dir / f"pnl_{timestamp}.json"
            pnl_file.write_text(json.dumps(self.pnl_history, indent=2, ensure_ascii=False))
            print(f"Saved {len(self.pnl_history)} PnL snapshots to {pnl_file.name}")
        
        # Save Mari chat log
        if self.mari.chat_log:
            chat_file = data_dir / f"chat_{timestamp}.json"
            chat_file.write_text(json.dumps(self.mari.chat_log, indent=2, ensure_ascii=False))
            print(f"Saved {len(self.mari.chat_log)} chat messages to {chat_file.name}")
        
        # Save price history for charts
        if self.price_history:
            price_file = data_dir / f"prices_{timestamp}.json"
            price_file.write_text(json.dumps(self.price_history, indent=2, ensure_ascii=False))
            print(f"Saved price history for {len(self.price_history)} tickers to {price_file.name}")

    def _restore_price_history(self) -> None:
        try:
            data_dir = project_root / "data" / "paper_trading"
            if not data_dir.exists():
                return
            cand = sorted(list(data_dir.glob("prices_*.json")), key=lambda p: p.stat().st_mtime, reverse=True)
            if not cand:
                return
            raw = cand[0].read_text(encoding="utf-8")
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                return
            out: Dict[str, List[Dict]] = {}
            for tk, bars in obj.items():
                tku = str(tk or "").upper().strip()
                if not tku or not isinstance(bars, list):
                    continue
                cleaned: List[Dict] = []
                for b in bars[-800:]:
                    if not isinstance(b, dict):
                        continue
                    try:
                        c = float(b.get("close") or 0.0)
                    except Exception:
                        c = 0.0
                    if c <= 0:
                        continue
                    t = b.get("time")
                    if not t:
                        continue
                    cleaned.append(dict(b))

                try:
                    ds = str(getattr(self, "data_source", "auto") or "auto").strip().lower()
                except Exception:
                    ds = "auto"

                if ds in {"yfinance", "auto"}:
                    try:
                        yf_only = [x for x in cleaned if str(x.get("source") or "").strip().lower() == "yfinance"]
                        if yf_only:
                            cleaned = yf_only
                    except Exception:
                        pass

                if cleaned:
                    out[tku] = cleaned
            if out:
                self.price_history.update(out)
                try:
                    t_str = datetime.now().strftime("%H:%M:%S")
                    self.agent_logs.append(
                        {
                            "time": t_str,
                            "type": "agent",
                            "priority": 2,
                            "message": f"[Chart] restored price_history from {cand[0].name} (tickers={len(out)})",
                        }
                    )
                    if len(self.agent_logs) > 500:
                        self.agent_logs = self.agent_logs[-300:]
                except Exception:
                    pass
        except Exception:
            return

    def get_chart_data(self, ticker: str) -> List[Dict]:
        """Get price history for UI chart rendering"""
        return self.price_history.get(ticker.upper(), [])

    def get_trade_markers(self) -> List[Dict]:
        """Get trade markers for chart overlay (buy/sell points)"""
        return self.trade_log


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3.4: Live Paper Trading Engine")
    parser.add_argument("--cash", type=float, default=500000.0, help="Initial cash")
    parser.add_argument("--data-source", default="auto", choices=["auto", "yfinance", "simulated"])
    parser.add_argument("--load-models", action="store_true", help="Load real MoE models (requires GPU)")
    parser.add_argument("--with-api", action="store_true", help="Start FastAPI server for live dashboard")
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Phase 3.4: Live Paper Trading Engine")
    print("=" * 60)
    print(f"Data Source: {args.data_source}")
    print(f"Load Models: {args.load_models}")
    runner: Optional[LivePaperTradingRunner] = None
    api_mod = None

    if args.with_api:
        try:
            if not _port_available(str(args.api_host), int(args.api_port)):
                print(
                    f"[API] Port already in use: http://{args.api_host}:{args.api_port} (skip starting API). "
                    "Stop the process that is using the port, or run with --api-port <other>."
                )
                args.with_api = False
        except Exception:
            pass

    if not args.with_api:
        print("Loading Mari's voice model...")
        runner = LivePaperTradingRunner(
            initial_cash=args.cash,
            data_source=args.data_source,
            load_models=args.load_models,
        )
        runner.start()
        try:
            print("\n[Press Ctrl+C to stop]\n")
            tick_count = 0
            while True:
                time.sleep(1)  # Just wait, data feed handles ticks
                tick_count += 1
                if tick_count % 20 == 0:
                    nav = runner.broker.cash
                    for ticker, pos in getattr(runner.broker, 'positions', {}).items():
                        if ticker in runner.price_history and runner.price_history[ticker]:
                            last_price = runner.price_history[ticker][-1].get("close", 0)
                            nav += float(getattr(pos, "shares", 0.0)) * float(last_price or 0.0)
                    runner._check_significant_events(nav)
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            runner.stop()
        return

    # ===== with-api path: run uvicorn in main thread (reliable port binding), init runner in background =====

    try:
        api_mod = importlib.import_module("src.api.main")
        uvicorn = importlib.import_module("uvicorn")
    except Exception as e:
        print(f"[API] Failed to import API server: {e}")
        return

    def _boot_runner() -> None:
        nonlocal runner
        try:
            print("Loading Mari's voice model...")
            r = LivePaperTradingRunner(
                initial_cash=args.cash,
                data_source=args.data_source,
                load_models=args.load_models,
            )
            runner = r
            try:
                api_mod.set_live_runner(r)
            except Exception as e:
                print(f"[API] Warning: failed to attach live runner: {e}")
            r.start()
        except Exception as e:
            print(f"[Runner] boot failed: {e}")

    threading.Thread(target=_boot_runner, daemon=True).start()

    print(f"Live API starting: http://{args.api_host}:{args.api_port}/api/v1/live/status")
    try:
        uvicorn.run(api_mod.app, host=str(args.api_host), port=int(args.api_port), log_level="warning")
    finally:
        try:
            if runner is not None:
                runner.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()

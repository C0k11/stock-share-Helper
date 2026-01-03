#!/usr/bin/env python
"""
Phase 3.4: Live Paper Trading Engine with Mari's Commentary
实时模拟盘 + Multi-Agent 思考过程语音解说

Usage:
    python scripts/run_live_paper_trading.py
"""
from __future__ import annotations

import json
import os
import random
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


class MariVoice:
    """Mari TTS with GPT-SoVITS model loading"""

    def __init__(self):
        self.queue: list[str] = []
        self.lock = threading.Lock()
        self.running = True
        self.max_queue_size = 2
        
        self.cfg = _load_secretary_config()
        voice_cfg = self.cfg.get("voice", {})
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
        
        # LLM config for generating Mari's speech
        self.llm_cfg = self.cfg.get("llm", {})
        self.llm_base = self.llm_cfg.get("api_base", "http://localhost:11434/v1")
        self.llm_model = self.llm_cfg.get("model", "qwen2.5:7b-instruct")
        self.system_prompt = self.cfg.get("secretary", {}).get("system_prompt", "")
        
        # Chat history for display
        self.chat_log: List[Dict] = []
        
        if HAS_PYGAME:
            pygame.mixer.init()
        
        # Load Mari's voice model
        self._load_mari_model()
        
        # Start TTS worker thread
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _load_mari_model(self) -> None:
        """Load Mari's trained GPT-SoVITS weights"""
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
        """Use LLM to generate Mari's commentary in character"""
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
        """Add text to TTS queue, optionally generating through LLM first"""
        if use_llm:
            text = self.generate_commentary(text)
        
        # Log to chat
        self.chat_log.append({"time": datetime.now().isoformat(), "speaker": "Mari", "text": text})
        print(f"\n[Chat] Mari: {text}")
        
        with self.lock:
            while len(self.queue) >= self.max_queue_size:
                self.queue.pop(0)
            if len(text) > 60:
                text = text[:57] + "..."
            self.queue.append(text)

    def _worker(self) -> None:
        """Background worker to process TTS queue"""
        while self.running:
            text = None
            with self.lock:
                if self.queue:
                    text = self.queue.pop(0)

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
        if HAS_PYGAME:
            pygame.mixer.quit()


class LivePaperTradingRunner:
    """Main runner for live paper trading with Mari commentary"""

    def __init__(self, initial_cash: float = 500000.0):
        self.engine = TradingEngine()
        self.broker = PaperBroker(self.engine, cash=initial_cash)
        self.strategy = MultiAgentStrategy(self.engine)
        self.mari = MariVoice()
        
        self.engine.broker = self.broker
        self.engine.strategy = self.strategy
        
        # Paper trading data for system upgrade
        self.trade_log: List[Dict] = []
        self.pnl_history: List[Dict] = []
        self.initial_cash = initial_cash
        self.last_nav = initial_cash
        
        # Significant event thresholds
        self.volatility_threshold = 0.02  # 2% move triggers alert
        self.profit_threshold = 1000  # $1000 gain triggers celebration
        self.loss_threshold = -500  # $500 loss triggers concern
        
        # Hook into event handling
        self._original_handle = self.engine._handle_event
        self.engine._handle_event = self._wrapped_handle_event

    def _wrapped_handle_event(self, event: Event) -> None:
        """Wrapped event handler with Mari commentary"""
        self._original_handle(event)

        if event.type == EventType.LOG:
            msg = str(event.payload)
            priority = event.priority
            t_str = event.timestamp.strftime("%H:%M:%S")
            print(f"[{t_str}] [Agent] {msg}")
            # Only speak on significant decisions (priority >= 2)
            # Mari doesn't need to narrate every little thing

        elif event.type == EventType.FILL:
            fill = event.payload
            ticker = fill.get("ticker", "?")
            action = fill.get("action", "?")
            price = fill.get("price", 0)
            shares = fill.get("shares", 0)
            
            # Record trade for data collection
            trade_record = {
                "time": datetime.now().isoformat(),
                "ticker": ticker,
                "action": action,
                "price": price,
                "shares": shares,
            }
            self.trade_log.append(trade_record)
            print(f"[FILL] {action} {ticker} x{shares} @ ${price:.2f}")
            
            # Mari only announces trades (significant events)
            self.mari.speak(f"{action} {ticker}，成交价 {price:.2f}")

        elif event.type == EventType.ERROR:
            print(f"[ERROR] {event.payload}")
    
    def _check_significant_events(self, nav: float) -> None:
        """Check for significant events that Mari should announce"""
        pnl = nav - self.last_nav
        pnl_pct = pnl / self.last_nav if self.last_nav > 0 else 0
        
        # Record PnL snapshot
        self.pnl_history.append({
            "time": datetime.now().isoformat(),
            "nav": nav,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        })
        
        # Significant profit
        if pnl >= self.profit_threshold:
            self.mari.speak(f"赚了 ${pnl:.0f}，这是 Sensei 平日积累的福报。")
            self.last_nav = nav
        # Significant loss
        elif pnl <= self.loss_threshold:
            self.mari.speak(f"亏损 ${abs(pnl):.0f}，这也是一种试炼呢。")
            self.last_nav = nav
        # High volatility
        elif abs(pnl_pct) >= self.volatility_threshold:
            direction = "上涨" if pnl > 0 else "下跌"
            self.mari.speak(f"市场波动剧烈，{direction} {abs(pnl_pct)*100:.1f}%。")

    def start(self) -> None:
        """Start the live paper trading engine"""
        self.engine.start()
        
        print("=" * 60)
        print("Phase 3.4: Live Paper Trading Engine")
        print("=" * 60)
        print(f"Initial Cash: ${self.broker.cash:,.2f}")
        print(f"Tickers: {', '.join(self.strategy.tickers)}")
        print("=" * 60)
        
        # Mari announces startup (through LLM)
        self.mari.speak("实盘模拟系统已启动，Mari 开始监视 Agent 活动。")

    def stop(self) -> None:
        """Stop the engine and save trading data"""
        self.engine.stop()
        self.mari.stop()
        
        # Save paper trading data for system upgrade
        self._save_trading_data()
        print("System Shutdown.")
    
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

    def simulate_market_tick(self, ticker: Optional[str] = None) -> None:
        """Simulate a market tick (for demo purposes)"""
        if ticker is None:
            ticker = random.choice(self.strategy.tickers)
        
        price = round(random.uniform(100, 900), 2)
        
        print(f"\n>> Market Tick: {ticker} @ ${price:.2f}")
        
        event = Event(
            EventType.MARKET_DATA,
            datetime.now(),
            {"ticker": ticker, "close": price},
            priority=0,
        )
        self.engine.push_event(event)


def main():
    print("Initializing Paper Trading Engine (Phase 3.4)...")
    print("Loading Mari's voice model...")
    
    runner = LivePaperTradingRunner(initial_cash=500000.0)
    runner.start()

    try:
        print("\n[Press Ctrl+C to stop]\n")
        tick_count = 0
        while True:
            time.sleep(4)
            runner.simulate_market_tick()
            tick_count += 1
            
            # Periodically check for significant PnL events
            if tick_count % 5 == 0:
                # In real implementation, calculate NAV from positions
                # For now, simulate NAV changes
                simulated_nav = runner.initial_cash + random.uniform(-2000, 3000)
                runner._check_significant_events(simulated_nav)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        runner.stop()


if __name__ == "__main__":
    main()

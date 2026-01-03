#!/usr/bin/env python
"""
Phase 3.3: Live Paper Trading Engine with Mari's Commentary
实时模拟盘 + Multi-Agent 思考过程语音解说

Usage:
    python scripts/run_live_paper_trading.py
"""
from __future__ import annotations

import asyncio
import os
import random
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.trading.engine import TradingEngine
from src.trading.strategy import MultiAgentStrategy
from src.trading.broker import PaperBroker
from src.trading.event import Event, EventType

# TTS imports
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


class TTSQueue:
    """Async TTS queue to prevent voice lag"""

    def __init__(self):
        self.queue: list[str] = []
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        
        self.cfg = _load_secretary_config()
        voice_cfg = self.cfg.get("voice", {})
        self.gpt_sovits_cfg = voice_cfg.get("gpt_sovits", {})
        self.api_base = self.gpt_sovits_cfg.get("api_base", "http://127.0.0.1:9880")
        
        # Presets for reference audio
        presets = self.gpt_sovits_cfg.get("presets", {})
        gentle = presets.get("gentle", {})
        self.refer_wav = gentle.get("refer_wav_path", "")
        self.prompt_text = gentle.get("prompt_text", "先生…")

        if HAS_PYGAME:
            pygame.mixer.init()

    def speak(self, text: str) -> None:
        """Add text to TTS queue"""
        with self.lock:
            # Skip if queue is too long (voice lag prevention)
            if len(self.queue) < 5:
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
        print(f"🗣️ Mari: {text}")
        
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
        self.tts = TTSQueue()

        self.engine.broker = self.broker
        self.strategy = self.strategy
        self.engine.strategy = self.strategy

        # Hook into event handling
        self._original_handle = self.engine._handle_event
        self.engine._handle_event = self._wrapped_handle_event

    def _wrapped_handle_event(self, event: Event) -> None:
        """Wrapped event handler with Mari commentary"""
        # Call original handler
        self._original_handle(event)

        # Process for Mari's commentary
        if event.type == EventType.LOG:
            msg = str(event.payload)
            priority = event.priority
            t_str = event.timestamp.strftime("%H:%M:%S")
            
            print(f"[{t_str}] 🤖 {msg}")

            if priority >= 2:  # High priority -> Always speak
                self.tts.speak(f"Sensei，{msg}")
            elif priority == 1:  # Medium -> Occasionally speak
                if random.random() > 0.6:
                    self.tts.speak(msg)

        elif event.type == EventType.FILL:
            fill = event.payload
            ticker = fill.get("ticker", "?")
            action = fill.get("action", "?")
            price = fill.get("price", 0)
            
            msg = f"成交报告：{action} {ticker}，价格 {price:.2f}"
            print(f"[FILL] 💰 {msg}")
            self.tts.speak(f"好消息！{msg}")

        elif event.type == EventType.ERROR:
            err = event.payload
            print(f"[ERROR] ❌ {err}")

    def start(self) -> None:
        """Start the live paper trading engine"""
        self.engine.start()
        
        print("=" * 60)
        print("🚀 Phase 3.3: Live Paper Trading Engine")
        print("=" * 60)
        print(f"💰 Initial Cash: ${self.broker.cash:,.2f}")
        print(f"📊 Tickers: {', '.join(self.strategy.tickers)}")
        print("=" * 60)
        
        self.tts.speak("Sensei，实盘模拟系统已启动。Mari 正在监视所有 Agent 的活动呢。")

    def stop(self) -> None:
        """Stop the engine"""
        self.engine.stop()
        self.tts.stop()
        print("🛑 System Shutdown.")

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
    print("🚀 Initializing Paper Trading Engine (Phase 3.3)...")
    
    runner = LivePaperTradingRunner(initial_cash=500000.0)
    runner.start()

    try:
        print("\n[Press Ctrl+C to stop]\n")
        while True:
            # Simulate market data every 4 seconds
            time.sleep(4)
            runner.simulate_market_tick()

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        runner.stop()


if __name__ == "__main__":
    main()

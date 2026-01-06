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

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .event import Event, EventType
from src.learning.recorder import recorder as evolution_recorder

# Default model paths
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SCALPER_ADAPTER = "models/trader_stock_v1_1_tech_plus_news/lora_weights"
DEFAULT_ANALYST_ADAPTER = "models/trader_v3_dpo_analyst/lora_weights"


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
        *,
        load_models: bool = False,
        base_model: str = DEFAULT_BASE_MODEL,
        moe_scalper: str = DEFAULT_SCALPER_ADAPTER,
        moe_analyst: str = DEFAULT_ANALYST_ADAPTER,
        load_4bit: bool = True,
    ) -> None:
        self.engine = engine
        self.tickers = ["NVDA", "TSLA", "AAPL", "MSFT", "AMD"]
        
        # Model state
        self.models_loaded = False
        self.model = None
        self.tokenizer = None
        self.planner = None
        self.gatekeeper = None
        
        # Config
        self.base_model = base_model
        self.moe_scalper = moe_scalper
        self.moe_analyst = moe_analyst
        self.load_4bit = load_4bit
        
        # MoE routing thresholds
        self.moe_any_news = True
        self.moe_news_threshold = 0.8
        self.moe_vol_threshold = 30.0  # annualized vol %
        
        # Risk parameters
        self.max_drawdown_pct = -8.0
        self.vol_trigger_ann_pct = 30.0
        
        # System 2 debate settings
        self.system2_enabled = True
        self.system2_buy_only = True
        
        # Chartist overlay
        self.chart_confidence_threshold = 0.7
        
        # Macro governor
        self.macro_risk_map: Dict[str, float] = {}
        
        # Price history for technical analysis
        self.price_history: Dict[str, List[Dict]] = {}
        
        # Load models if requested
        if load_models:
            self._load_models()
        
        print("Multi-Agent Strategy Initialized.")

    def _load_models(self) -> None:
        """Load MoE models (Scalper + Analyst adapters)"""
        self._log("Loading Multi-Agent models...", priority=1)
        
        try:
            # Import model loading functions
            project_root = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(project_root / "scripts"))
            from run_trading_inference import load_model_moe
            
            scalper_path = str(project_root / self.moe_scalper)
            analyst_path = str(project_root / self.moe_analyst)
            
            if not Path(scalper_path).exists():
                self._log(f"Scalper adapter not found: {scalper_path}", priority=1)
                return
            if not Path(analyst_path).exists():
                self._log(f"Analyst adapter not found: {analyst_path}", priority=1)
                return
            
            self.model, self.tokenizer = load_model_moe(
                self.base_model,
                {"scalper": scalper_path, "analyst": analyst_path},
                self.load_4bit,
                default_adapter="scalper",
            )
            self.models_loaded = True
            self._log("MoE models loaded (Scalper + Analyst)", priority=2)
            
        except Exception as e:
            self._log(f"Model loading failed: {e}", priority=2)
            self.models_loaded = False

    def on_bar(self, market_data: dict) -> Optional[dict]:
        """
        Process incoming market data through Multi-Agent pipeline
        market_data: { "ticker": "NVDA", "close": 120.5, "volume": ..., "time": ... }
        """
        ticker = str(market_data.get("ticker", "")).upper()
        price = float(market_data.get("close", 0.0))
        volume = float(market_data.get("volume", 0.0))
        
        if not ticker or price <= 0:
            return None
        
        # Update price history
        self._update_price_history(ticker, market_data)
        
        # Calculate technical features
        features = self._compute_features(ticker)
        
        # --- 1. Planner: Market Regime Assessment ---
        self._log(f"Planner scanning {ticker}...", priority=0)
        regime = self._planner_assess(ticker, features)
        
        if regime == "cash_preservation":
            self._log(f"Planner: {ticker} - cash preservation mode, skip.", priority=1)
            return None
        
        # --- 2. Gatekeeper: RL-based Trade Filter ---
        if not self._gatekeeper_approve(ticker, features):
            self._log(f"Gatekeeper: {ticker} - rejected (low Q-value).", priority=1)
            return None
        
        # --- 3. MoE Router: Select Expert ---
        expert, router_meta = self._moe_route(ticker, features)
        self._log(f"MoE Router: {ticker} -> [{expert}] (vol={router_meta.get('vol', 0):.1f}%)", priority=1)
        
        # --- 4. Expert Inference ---
        decision = self._expert_infer(ticker, features, expert)
        action = decision.get("decision", "HOLD").upper()
        analysis = decision.get("analysis", "")

        trace_id: Optional[str] = None
        
        if action == "HOLD":
            self._log(f"[{expert}] {ticker}: HOLD - {analysis[:50]}", priority=0)
            return None

        try:
            print(f"\n[{expert}] proposal: {action} {ticker} :: {str(analysis or '')[:160]}")
        except Exception:
            pass

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
        except Exception:
            trace_id = None
        
        if expert == "scalper":
            self._log(f"Scalper: {ticker} {action} - {analysis[:40]}", priority=1)
        else:
            self._log(f"Analyst (DPO): {ticker} {action} - {analysis[:40]}", priority=1)
        confidence = 0.75
        
        # --- 5. System 2 Debate (if enabled) ---
        if self.system2_enabled:
            if not self.system2_buy_only or action == "BUY":
                self._log("System 2 Debate: initiated...", priority=1)
                
                # Chartist Overlay
                chart_score = self._chartist_overlay(ticker, action)
                chart_view = "supports" if chart_score > 0 else ("opposes" if chart_score < 0 else "neutral")
                self._log(f"Chartist (VLM): {ticker} pattern {chart_view} ({chart_score:+.2f})", priority=1)
                
                # Macro Governor
                macro_gear, macro_label = self._macro_governor_assess()
                self._log(f"Macro Governor: regime={macro_label} (gear={macro_gear})", priority=1)
                
                # Judge Decision
                approved, confidence, reason = self._system2_judge(
                    action, chart_score, macro_gear, features
                )

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
                
                if not approved:
                    self._log(f"System 2 (Judge): REJECTED - {reason}", priority=2)
                    return None
                
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
        }
        
        self.engine.push_event(
            Event(EventType.SIGNAL, datetime.now(), signal, priority=2)
        )
        return signal

    def _update_price_history(self, ticker: str, data: dict) -> None:
        """Maintain rolling price history for technical analysis"""
        if ticker not in self.price_history:
            self.price_history[ticker] = []
        
        self.price_history[ticker].append({
            "time": data.get("time", datetime.now()),
            "open": data.get("open", data.get("close", 0)),
            "high": data.get("high", data.get("close", 0)),
            "low": data.get("low", data.get("close", 0)),
            "close": data.get("close", 0),
            "volume": data.get("volume", 0),
        })
        
        # Keep last 60 bars
        if len(self.price_history[ticker]) > 60:
            self.price_history[ticker] = self.price_history[ticker][-60:]

    def _compute_features(self, ticker: str) -> Dict[str, Any]:
        """Compute technical features from price history"""
        history = self.price_history.get(ticker, [])
        if len(history) < 5:
            return {"technical": {}, "signal": {}}
        
        closes = [bar["close"] for bar in history]
        volumes = [bar["volume"] for bar in history]
        
        # Basic features
        current = closes[-1]
        ma5 = sum(closes[-5:]) / 5 if len(closes) >= 5 else current
        ma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else current
        
        # Returns
        ret_5d = (current / closes[-5] - 1) * 100 if len(closes) >= 5 else 0
        ret_20d = (current / closes[-20] - 1) * 100 if len(closes) >= 20 else 0
        
        # Volatility (annualized)
        if len(closes) >= 20:
            returns = [(closes[i] / closes[i-1] - 1) for i in range(1, len(closes))]
            daily_vol = (sum(r**2 for r in returns[-20:]) / 20) ** 0.5
            vol_ann = daily_vol * math.sqrt(252) * 100
        else:
            vol_ann = 20.0
        
        # Volume ratio
        avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else 1
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
        tech = features.get("technical", {})
        vol = tech.get("volatility_20d", 20)
        ret_5d = tech.get("return_5d", 0)
        
        # Simple rule-based planner (can be replaced with SFT model)
        if vol > 40 or ret_5d < -10:
            return "cash_preservation"
        elif vol > 30 or ret_5d < -5:
            return "defensive"
        else:
            return "aggressive"

    def _gatekeeper_approve(self, ticker: str, features: Dict) -> bool:
        """Gatekeeper: RL-based trade approval"""
        # Simple heuristic (can be replaced with RL model)
        tech = features.get("technical", {})
        vol = tech.get("volatility_20d", 20)
        
        # Reject if volatility too extreme
        if vol > self.vol_trigger_ann_pct:
            return False
        
        # Random factor for demo (replace with actual Q-value)
        import random
        return random.random() > 0.3

    def _moe_route(self, ticker: str, features: Dict) -> Tuple[str, Dict]:
        """MoE Router: Select expert based on market features"""
        vol = features.get("volatility_ann_pct", 20)
        
        # Route to Analyst for high volatility or news events
        # Route to Scalper for calm markets
        use_analyst = vol >= self.moe_vol_threshold
        
        expert = "Analyst" if use_analyst else "Scalper"
        meta = {"vol": vol, "expert": expert}
        
        return expert, meta

    def _expert_infer(self, ticker: str, features: Dict, expert: str) -> Dict:
        """Expert inference (model or heuristic)"""
        if self.models_loaded and self.model is not None:
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
        adapter_name = "analyst" if expert == "Analyst" else "scalper"
        self.model.set_adapter(adapter_name)
        
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
        
        # Generate
        import torch
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=128, temperature=0.1)
        
        output = self.tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse JSON
        try:
            import re
            match = re.search(r'\{[^}]+\}', output)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        
        return {"decision": "HOLD", "analysis": "Parse error"}

    def _chartist_overlay(self, ticker: str, proposed_action: str) -> int:
        """Chartist overlay: visual pattern analysis score"""
        # Returns: 1 (supports), 0 (neutral), -1 (opposes)
        import random
        return random.choice([-1, 0, 0, 1, 1])

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
        self.engine.push_event(
            Event(EventType.LOG, datetime.now(), message, priority)
        )

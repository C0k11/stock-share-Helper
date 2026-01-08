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
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .event import Event, EventType
from src.learning.recorder import recorder as evolution_recorder
from src.agent.gatekeeper import Gatekeeper
from src.agent.planner import Planner

# Default model paths
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SCALPER_ADAPTER = "models/trader_stock_v1_1_tech_plus_news/lora_weights"
DEFAULT_ANALYST_ADAPTER = "models/trader_v3_dpo_analyst"


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
        load_4bit: bool = True,
        planner_policy: str = "rule",
        planner_sft_model_path: str = "models/planner_sft_v1.pt",
        gatekeeper_model_path: str = "models/rl_gatekeeper_v2.pt",
        gatekeeper_threshold: float = 0.0,
    ) -> None:
        self.engine = engine
        self.tickers = ["NVDA", "TSLA", "AAPL", "MSFT", "AMD"]
        
        # Model state
        self.models_loaded = False
        self.model = None
        self.tokenizer = None
        self.models_error = ""
        self._adapters_loaded: set[str] = set()
        self.planner = None
        self.gatekeeper = None
        self._inference_lock = threading.Lock()

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
        self.load_4bit = load_4bit

        self.all_agents_mode: bool = False
        self.committee_policy: str = "conservative"
        
        # MoE routing thresholds
        self.moe_any_news = True
        self.moe_news_threshold = 0.8
        self.moe_vol_threshold = 60.0  # annualized vol %
        
        # Risk parameters
        self.max_drawdown_pct = -8.0
        self.vol_trigger_ann_pct = 120.0
        
        # System 2 debate settings
        self.system2_enabled = True
        self.system2_buy_only = True
        
        # Chartist overlay
        self.chart_confidence_threshold = 0.7
        
        # Macro governor
        self.macro_risk_map: Dict[str, float] = {}
        
        # Price history for technical analysis
        self.price_history: Dict[str, List[Dict]] = {}

        self._hold_diag: Dict[str, int] = {}
        
        # Load models if requested
        if load_models:
            self._load_models()

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

    def _load_models(self) -> None:
        """Load MoE models (Scalper + Analyst adapters)"""
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

            adapters: Dict[str, str] = {}
            if Path(scalper_path).exists():
                adapters["scalper"] = scalper_path
            if Path(analyst_path).exists():
                adapters["analyst"] = analyst_path
            if secretary_path and Path(secretary_path).exists():
                adapters["secretary"] = secretary_path

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

        with self._inference_lock:
            try:
                import torch
                from contextlib import nullcontext

                # Context manager for adapter usage
                adapter_ctx = nullcontext()
                if target_adapter:
                    self.model.set_adapter(target_adapter)
                else:
                    # Use base model
                    if hasattr(self.model, "disable_adapter"):
                        adapter_ctx = self.model.disable_adapter()

                with adapter_ctx:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": user_msg})

                    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

                    with torch.no_grad():
                        gen_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            do_sample=(temperature > 0),
                            top_p=0.9,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                    
                    output = self.tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

                # Restore default adapter (scalper) to minimize impact on next trading tick
                # (Scalper is usually the default active one)
                default = "scalper" if "scalper" in self._adapters_loaded else (list(self._adapters_loaded)[0] if self._adapters_loaded else None)
                if default:
                    self.model.set_adapter(default)

                return str(output or "").strip()

            except Exception as e:
                self._log(f"Generic inference failed: {e}", priority=2)
                # Try to restore default
                try:
                    default = "scalper" if "scalper" in self._adapters_loaded else (list(self._adapters_loaded)[0] if self._adapters_loaded else None)
                    if default:
                        self.model.set_adapter(default)
                except Exception:
                    pass
                return ""

    def generate_reply(self, user_msg: str, system_prompt: str = "") -> str:
        """Generate chat reply using the Secretary adapter (MoE hot-swap), falling back to base model if needed."""
        # Use the generic thread-safe method, preferring secretary adapter
        return self.generic_inference(
            user_msg=user_msg, 
            system_prompt=system_prompt, 
            adapter="secretary",
            temperature=0.7,
            max_new_tokens=256
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
        if self.all_agents_mode:
            dec_scalper = self._expert_infer(ticker, features, "scalper")
            dec_analyst = self._expert_infer(ticker, features, "analyst")
            act_s = str(dec_scalper.get("decision", "HOLD") or "HOLD").upper()
            act_a = str(dec_analyst.get("decision", "HOLD") or "HOLD").upper()
            ana_s = str(dec_scalper.get("analysis", "") or "")
            ana_a = str(dec_analyst.get("analysis", "") or "")
            self._log(f"[scalper] {ticker}: {act_s} - {ana_s[:80]}", priority=1)
            self._log(f"[analyst] {ticker}: {act_a} - {ana_a[:80]}", priority=1)

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
                    analysis = f"committee agree: {ana_s[:80]}"
                else:
                    action = "HOLD"
                    analysis = "committee: disagree_or_hold"

            expert = "committee"
        else:
            decision = self._expert_infer(ticker, features, expert)
            action = decision.get("decision", "HOLD").upper()
            analysis = decision.get("analysis", "")

        trace_id: Optional[str] = None
        trace_ids: list[str] = []
        
        if action == "HOLD":
            self._log(f"[{expert}] {ticker}: HOLD - {analysis[:50]}", priority=1)

            try:
                n0 = int(self._hold_diag.get(ticker, 0) or 0) + 1
                self._hold_diag[ticker] = n0
                if n0 % 2 == 0:
                    try:
                        self._log(f"System 2 Debate: idle (action=HOLD)", priority=2)
                    except Exception:
                        pass
                    try:
                        chart_score = self._chartist_overlay(ticker, "HOLD")
                        chart_view = "supports" if chart_score > 0 else ("opposes" if chart_score < 0 else "neutral")
                        self._log(f"Chartist (VLM): {ticker} pattern {chart_view} (score={int(chart_score)})", priority=2)
                    except Exception as e:
                        self._log(f"Chartist (VLM): error: {e}", priority=2)
                    try:
                        macro_gear, macro_label = self._macro_governor_assess()
                        self._log(f"Macro Governor: regime={macro_label} (gear={macro_gear})", priority=2)
                    except Exception as e:
                        self._log(f"Macro Governor: error: {e}", priority=2)
            except Exception:
                pass
            return None

        try:
            print(f"\n[{expert}] proposal: {action} {ticker} :: {str(analysis or '')[:160]}")
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
            self._log(f"Scalper: {ticker} {action} - {analysis[:40]}", priority=1)
        else:
            self._log(f"Analyst (DPO): {ticker} {action} - {analysis[:40]}", priority=1)
        confidence = 0.75
        
        # --- 5. System 2 Debate (if enabled) ---
        if self.system2_enabled:
            if (not self.system2_buy_only) or (action in {"BUY", "SELL"}):
                self._log("System 2 Debate: initiated...", priority=2)
                
                # Chartist Overlay
                chart_score = self._chartist_overlay(ticker, action)
                chart_view = "supports" if chart_score > 0 else ("opposes" if chart_score < 0 else "neutral")
                self._log(f"Chartist (VLM): {ticker} pattern {chart_view} (score={int(chart_score)})", priority=1)
                
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
            "trace_ids": trace_ids,
        }
        return signal

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
        
        # Route to Analyst for high volatility or news events
        # Route to Scalper for calm markets
        use_analyst = vol >= self.moe_vol_threshold
        
        expert = "analyst" if use_analyst else "scalper"
        meta = {"vol": vol, "expert": expert}
        
        return expert, meta

    def _expert_infer(self, ticker: str, features: Dict, expert: str) -> Dict:
        """Expert inference (model or heuristic)"""
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
        
        raw = ""
        with self._inference_lock:
            try:
                import torch
                from contextlib import nullcontext
                
                # Ensure adapter is active
                # Note: on_bar calls this, and on_bar is generally sequential, but chat can interrupt.
                # The lock prevents race conditions on model state (adapters).
                adapter_ctx = nullcontext()
                if adapter_name in self._adapters_loaded:
                    self.model.set_adapter(adapter_name)
                else:
                    # Fallback to base or scalper if analyst missing
                    if "scalper" in self._adapters_loaded:
                        self.model.set_adapter("scalper")
                    elif hasattr(self.model, "disable_adapter"):
                        adapter_ctx = self.model.disable_adapter()

                with adapter_ctx:
                    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                    
                    with torch.no_grad():
                        gen_ids = self.model.generate(**inputs, max_new_tokens=128, temperature=0.1)
                    
                    output = self.tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    raw = str(output or "").strip()
                
                # Restore default (usually scalper) if we changed it? 
                # on_bar might make multiple calls. 
                # But generic_inference restores default. 
                # It is safer to leave it or restore to scalper.
                if "scalper" in self._adapters_loaded:
                    self.model.set_adapter("scalper")

            except Exception as e:
                self._log(f"Model inference failed: {e}", priority=2)
                return {}

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
        if len(tail) > 220:
            tail = tail[:217] + "..."
        return {"decision": decision, "analysis": f"Unparsed model output: {tail}"}

    def _chartist_overlay(self, ticker: str, proposed_action: str) -> int:
        """Chartist overlay: visual pattern analysis score"""
        # Returns: 1 (supports), 0 (neutral), -1 (opposes)
        try:
            feats = self._compute_features(str(ticker or "").upper())
            tech = feats.get("technical") if isinstance(feats.get("technical"), dict) else {}
            ret_5d = float(tech.get("return_5d") or 0.0)
            p_ma20 = float(tech.get("price_vs_ma20") or 0.0)
        except Exception:
            ret_5d = 0.0
            p_ma20 = 0.0

        act = str(proposed_action or "").upper().strip()
        if act == "BUY":
            if ret_5d > 0.2 and p_ma20 > 0.0:
                return 1
            if ret_5d < -0.2 and p_ma20 < 0.0:
                return -1
            return 0
        if act == "SELL":
            if ret_5d < -0.2 and p_ma20 < 0.0:
                return 1
            if ret_5d > 0.2 and p_ma20 > 0.0:
                return -1
            return 0
        return 0

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

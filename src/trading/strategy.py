# src/trading/strategy.py
"""
Multi-Agent Strategy for Live Paper Trading (Phase 3.3)
封装 run_trading_inference.py 的逻辑，通过事件总线推送 Agent 思考过程
"""
from __future__ import annotations

import random
import time
from datetime import datetime
from typing import Any, Optional

from .event import Event, EventType


class MultiAgentStrategy:
    """
    Multi-Agent 策略类
    - 接收 MARKET_DATA 事件
    - 模拟 Planner -> MoE Router -> Expert -> System2 Debate 流程
    - 通过 LOG 事件推送思考过程给 Mari
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.tickers = ["NVDA", "TSLA", "AAPL", "MSFT", "AMD"]

        # 真实模型加载区 (Phase 3.4 填入)
        # self.planner = load_planner(...)
        # self.moe = load_moe(...)
        # self.system2 = load_system2(...)
        # self.gatekeeper = load_gatekeeper(...)

        print("🧠 Multi-Agent Strategy Initialized.")

    def on_bar(self, market_data: dict) -> Optional[dict]:
        """
        当新的 K 线到达时触发
        market_data: { "ticker": "NVDA", "close": 120.5, "time": ... }
        """
        ticker = str(market_data.get("ticker", "")).upper()
        price = float(market_data.get("close", 0.0))

        if not ticker or price <= 0:
            return None

        # --- 1. Planner & Gatekeeper ---
        self._log(f"Planner 正在扫描 {ticker}...", priority=0)
        time.sleep(0.3)  # 模拟思考延迟

        # 模拟 Gatekeeper 决策
        if random.random() > 0.7:
            self._log(f"Gatekeeper: {ticker} 波动率过低，跳过。", priority=1)
            return None

        # --- 2. MoE Router ---
        experts = ["Scalper", "Analyst", "Momentum", "MeanReversion"]
        expert = random.choice(experts)
        self._log(f"MoE Router: 市场特征匹配，路由至 [{expert}] 专家。", priority=1)
        time.sleep(0.3)

        # --- 3. Expert Inference (SFT/LoRA) ---
        action = random.choice(["BUY", "SELL", "HOLD"])

        if action == "HOLD":
            self._log(f"[{expert}] 专家建议 HOLD，观望。", priority=0)
            return None

        reason = self._generate_reason(action, expert)
        self._log(f"[{expert}] 专家建议 {action}。理由：{reason}", priority=1)

        # --- 4. System 2 Debate (Overlays) ---
        self._log("System 2: 启动辩论程序...", priority=1)
        time.sleep(0.5)

        # 模拟 Chartist Overlay
        chartist_view = random.choice(["支持", "中立", "反对"])
        self._log(f"Chartist Overlay: {chartist_view}当前提案。", priority=0)

        # 模拟 Macro Governor
        macro_risk = random.choice(["低", "中", "高"])
        self._log(f"Macro Governor: 宏观风险评估为 {macro_risk}。", priority=0)

        # 模拟 Judge 裁决
        debate_result = "PASS" if random.random() > 0.35 else "REJECT"

        if debate_result == "PASS":
            confidence = round(random.uniform(0.7, 0.95), 2)
            self._log(
                f"System 2 (Judge): 驳回空方观点，批准交易。置信度 {confidence}。",
                priority=2,
            )

            # --- 5. 发出信号 ---
            signal = {
                "ticker": ticker,
                "action": action,
                "price": price,
                "shares": 100,
                "expert": expert,
                "confidence": confidence,
            }
            self.engine.push_event(
                Event(EventType.SIGNAL, datetime.now(), signal, priority=2)
            )
            return signal
        else:
            reject_reason = random.choice([
                "宏观风险过高 (Macro Governor)",
                "技术面不支持 (Chartist)",
                "风险预算不足 (Risk Budget)",
            ])
            self._log(
                f"System 2 (Judge): 否决提案。原因：{reject_reason}",
                priority=2,
            )
            return None

    def _generate_reason(self, action: str, expert: str) -> str:
        """生成模拟的交易理由"""
        reasons = {
            "BUY": [
                "MACD 金叉且突破布林带上轨",
                "RSI 超卖反弹信号",
                "均线多头排列确认",
                "放量突破关键阻力位",
            ],
            "SELL": [
                "MACD 死叉且跌破布林带下轨",
                "RSI 超买回落信号",
                "均线空头排列确认",
                "放量跌破关键支撑位",
            ],
        }
        return random.choice(reasons.get(action, ["技术指标综合判断"]))

    def _log(self, message: str, priority: int = 0) -> None:
        """发送日志事件给 Mari"""
        self.engine.push_event(
            Event(EventType.LOG, datetime.now(), message, priority)
        )

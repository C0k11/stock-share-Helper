"""
决策解释模块 - 生成用户可理解的投资建议解释
"""

from typing import Dict, List, Optional
from loguru import logger


EXPLANATION_PROMPT = """你是一个专业的投资顾问助手。根据以下信息，生成一段简洁易懂的投资建议解释。

当前市场状态：
- 风险状态：{regime}
- VIX指数：{vix}
- SPY趋势：{spy_trend}

标的 {symbol} 的信号：
- 趋势信号：{trend_signal}
- 动量信号：{momentum_signal}
- 波动率：{volatility}

建议动作：{action}
目标仓位：{target_position}%

相关新闻摘要：
{news_summary}

请用2-3句话解释为什么给出这个建议，要点：
1. 用简单易懂的语言
2. 说明主要原因（技术面/风险状态/新闻事件）
3. 提示潜在风险"""


class DecisionExplainer:
    """决策解释生成器"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        use_local: bool = True
    ):
        self.model_name = model_name
        self.use_local = use_local
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """加载模型"""
        if self.model is not None:
            return
        
        logger.info(f"Loading explainer model: {self.model_name}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            logger.info("Explainer model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_explanation(
        self,
        symbol: str,
        action: str,
        target_position: float,
        regime: str,
        signals: Dict,
        news_summary: str = "",
        vix: Optional[float] = None
    ) -> str:
        """
        生成决策解释
        
        Args:
            symbol: 标的代码
            action: 建议动作
            target_position: 目标仓位
            regime: 风险状态
            signals: 信号字典
            news_summary: 新闻摘要
            vix: VIX指数
        
        Returns:
            解释文本
        """
        if self.model is None:
            self.load_model()
        
        # 构建提示
        prompt = EXPLANATION_PROMPT.format(
            symbol=symbol,
            action=action,
            target_position=round(target_position * 100, 1),
            regime=regime,
            vix=vix if vix else "N/A",
            spy_trend=signals.get("spy_trend", "N/A"),
            trend_signal=signals.get("trend", "N/A"),
            momentum_signal=signals.get("momentum", "N/A"),
            volatility=signals.get("volatility", "N/A"),
            news_summary=news_summary if news_summary else "无重要新闻"
        )
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7
            )
            
            explanation = self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            )
            
            return explanation.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return self._fallback_explanation(symbol, action, target_position, regime, signals)
    
    def _fallback_explanation(
        self,
        symbol: str,
        action: str,
        target_position: float,
        regime: str,
        signals: Dict
    ) -> str:
        """模板式回退解释"""
        regime_text = {
            "risk_on": "市场风险偏好较高",
            "risk_off": "市场处于避险模式",
            "transition": "市场处于过渡期"
        }.get(regime, "市场状态不明")
        
        trend_text = "趋势向上" if signals.get("trend", 0) > 0 else "趋势向下"
        
        if action in ["买入", "加仓", "buy", "add"]:
            return f"{symbol}：{regime_text}，{trend_text}，建议将仓位调整至{target_position*100:.0f}%。注意控制风险，设置好止损。"
        elif action in ["减仓", "清仓", "reduce", "clear"]:
            return f"{symbol}：{regime_text}，{trend_text}，建议降低仓位至{target_position*100:.0f}%以控制风险。"
        else:
            return f"{symbol}：当前{regime_text}，建议维持观望，等待更明确的信号。"


class TemplateExplainer:
    """模板式解释器（不需要LLM）"""
    
    TEMPLATES = {
        "risk_on_long": "{symbol}：市场处于Risk-On状态，趋势向上，动量为正。建议仓位{position}%。",
        "risk_off_reduce": "{symbol}：市场转为Risk-Off，VIX上升，建议降低仓位至{position}%以控制风险。",
        "transition_cautious": "{symbol}：市场处于过渡期，信号不明确，建议保持谨慎，仓位{position}%。",
        "stop_loss": "{symbol}：触发止损条件，建议清仓以控制损失。",
        "take_profit": "{symbol}：达到止盈目标，建议减仓锁定部分收益。"
    }
    
    def generate(
        self,
        symbol: str,
        action: str,
        target_position: float,
        regime: str,
        signals: Dict,
        triggered_rules: List[str] = None
    ) -> str:
        """生成模板式解释"""
        position = round(target_position * 100, 1)
        
        # 检查特殊规则
        if triggered_rules:
            if "stop_loss" in triggered_rules:
                return self.TEMPLATES["stop_loss"].format(symbol=symbol)
            if "take_profit" in triggered_rules:
                return self.TEMPLATES["take_profit"].format(symbol=symbol)
        
        # 根据regime选择模板
        if regime == "risk_on" and signals.get("trend", 0) > 0:
            template = self.TEMPLATES["risk_on_long"]
        elif regime == "risk_off":
            template = self.TEMPLATES["risk_off_reduce"]
        else:
            template = self.TEMPLATES["transition_cautious"]
        
        return template.format(symbol=symbol, position=position)

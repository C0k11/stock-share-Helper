"""
新闻解析模块 - 使用LLM将新闻结构化
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class ParsedNews:
    """结构化新闻"""
    title: str
    event_type: str
    sentiment: str
    impact_timeframe: str
    confidence: float
    impact_equity: int
    impact_bond: int
    impact_gold: int
    summary: str
    raw_content: str = ""


# 系统提示词
SYSTEM_PROMPT = """你是一个专业的金融新闻分析师。你的任务是分析金融新闻，提取结构化信息。

分析要点：
1. 事件类型：monetary_policy(货币政策)、economic_data(经济数据)、geopolitical(地缘政治)、earnings(财报)、regulatory(监管)、systemic_risk(系统风险)、other
2. 情绪方向：positive(利好)、negative(利空)、neutral(中性)
3. 影响时效：immediate(即时1-3天)、short_term(短期1-2周)、long_term(长期)
4. 对各类资产的影响：-1(利空)、0(无影响)、1(利多)

请严格以JSON格式输出，不要添加任何其他文字。"""


USER_PROMPT_TEMPLATE = """分析以下新闻：

标题：{title}
内容：{content}

请输出JSON：
{{
    "event_type": "事件类型",
    "sentiment": "情绪方向",
    "impact_timeframe": "影响时效",
    "confidence": 0.0-1.0,
    "impact_equity": -1/0/1,
    "impact_bond": -1/0/1,
    "impact_gold": -1/0/1,
    "summary": "一句话摘要"
}}"""


class NewsParser:
    """新闻解析器"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        use_local: bool = True
    ):
        """
        Args:
            model_name: 模型名称
            device: 设备
            use_local: 是否使用本地模型
        """
        self.model_name = model_name
        self.device = device
        self.use_local = use_local
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """加载模型"""
        if self.model is not None:
            return
        
        logger.info(f"Loading model: {self.model_name}")
        
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
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def parse(self, title: str, content: str) -> ParsedNews:
        """
        解析单条新闻
        
        Args:
            title: 新闻标题
            content: 新闻内容
        
        Returns:
            ParsedNews对象
        """
        if self.model is None:
            self.load_model()
        
        # 构建消息
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(title=title, content=content)}
        ]
        
        # 生成
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )
            
            response = self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            )
            
            # 解析JSON
            result = self._parse_json(response)
            
            return ParsedNews(
                title=title,
                event_type=result.get("event_type", "other"),
                sentiment=result.get("sentiment", "neutral"),
                impact_timeframe=result.get("impact_timeframe", "short_term"),
                confidence=result.get("confidence", 0.5),
                impact_equity=result.get("impact_equity", 0),
                impact_bond=result.get("impact_bond", 0),
                impact_gold=result.get("impact_gold", 0),
                summary=result.get("summary", ""),
                raw_content=content
            )
            
        except Exception as e:
            logger.error(f"Failed to parse news: {e}")
            return ParsedNews(
                title=title,
                event_type="other",
                sentiment="neutral",
                impact_timeframe="short_term",
                confidence=0.0,
                impact_equity=0,
                impact_bond=0,
                impact_gold=0,
                summary="解析失败",
                raw_content=content
            )
    
    def parse_batch(self, news_list: List[Dict]) -> List[ParsedNews]:
        """批量解析新闻"""
        results = []
        for news in news_list:
            result = self.parse(news.get("title", ""), news.get("content", ""))
            results.append(result)
        return results
    
    def _parse_json(self, text: str) -> Dict:
        """从文本中提取JSON"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except:
            pass
        
        # 尝试提取JSON块
        import re
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        return {}


# 不加载模型的轻量级解析（基于规则）
class RuleBasedNewsParser:
    """基于规则的新闻解析器（不需要LLM）"""
    
    KEYWORDS = {
        "monetary_policy": ["加息", "降息", "利率", "fed", "央行", "qe", "量化宽松", "鸽派", "鹰派"],
        "economic_data": ["cpi", "gdp", "就业", "失业", "通胀", "pmi", "非农"],
        "geopolitical": ["战争", "冲突", "制裁", "关税", "贸易战"],
        "systemic_risk": ["银行危机", "违约", "暴雷", "流动性"],
        "earnings": ["财报", "业绩", "营收", "利润", "eps"]
    }
    
    SENTIMENT_KEYWORDS = {
        "positive": ["上涨", "利好", "超预期", "增长", "突破", "新高"],
        "negative": ["下跌", "利空", "不及预期", "下滑", "暴跌", "新低", "危机"]
    }
    
    def parse(self, title: str, content: str) -> ParsedNews:
        """基于关键词的简单解析"""
        text = (title + " " + content).lower()
        
        # 检测事件类型
        event_type = "other"
        for etype, keywords in self.KEYWORDS.items():
            if any(kw in text for kw in keywords):
                event_type = etype
                break
        
        # 检测情绪
        sentiment = "neutral"
        pos_count = sum(1 for kw in self.SENTIMENT_KEYWORDS["positive"] if kw in text)
        neg_count = sum(1 for kw in self.SENTIMENT_KEYWORDS["negative"] if kw in text)
        
        if pos_count > neg_count:
            sentiment = "positive"
        elif neg_count > pos_count:
            sentiment = "negative"
        
        # 简单的影响判断
        impact_equity = 1 if sentiment == "positive" else (-1 if sentiment == "negative" else 0)
        impact_bond = -impact_equity if event_type == "monetary_policy" else 0
        impact_gold = 1 if event_type in ["geopolitical", "systemic_risk"] and sentiment == "negative" else 0
        
        return ParsedNews(
            title=title,
            event_type=event_type,
            sentiment=sentiment,
            impact_timeframe="short_term",
            confidence=0.5,
            impact_equity=impact_equity,
            impact_bond=impact_bond,
            impact_gold=impact_gold,
            summary=title[:100],
            raw_content=content
        )

"""
新闻因子模块 - 新闻事件分类、情绪分析、影响因子
"""

import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class EventType(Enum):
    """事件类型"""
    MONETARY_POLICY = "monetary_policy"     # 货币政策（加息/降息/QE）
    ECONOMIC_DATA = "economic_data"         # 经济数据（CPI/GDP/就业）
    GEOPOLITICAL = "geopolitical"           # 地缘政治
    EARNINGS = "earnings"                   # 财报季
    REGULATORY = "regulatory"               # 监管政策
    SYSTEMIC_RISK = "systemic_risk"         # 系统性风险（银行危机等）
    OTHER = "other"


class Sentiment(Enum):
    """情绪方向"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ImpactTimeframe(Enum):
    """影响时效"""
    IMMEDIATE = "immediate"     # 即时冲击（1-3天）
    SHORT_TERM = "short_term"   # 短期（1-2周）
    LONG_TERM = "long_term"     # 长期结构性


@dataclass
class NewsEvent:
    """结构化新闻事件"""
    title: str
    content: str
    source: str
    published_at: str
    url: Optional[str] = None
    
    # 结构化字段（由LLM或规则填充）
    event_type: Optional[EventType] = None
    sentiment: Optional[Sentiment] = None
    impact_timeframe: Optional[ImpactTimeframe] = None
    confidence: float = 0.0
    
    # 对各资产的影响方向
    impact_equity: Optional[int] = None   # -1/0/1
    impact_bond: Optional[int] = None
    impact_gold: Optional[int] = None
    
    # 摘要
    summary: Optional[str] = None


class NewsFeatures:
    """新闻因子计算"""
    
    # 事件类型对资产的历史统计影响（简化版，实际应从数据学习）
    EVENT_IMPACT_MAP = {
        EventType.MONETARY_POLICY: {
            "hawkish": {"equity": -1, "bond": -1, "gold": 0},
            "dovish": {"equity": 1, "bond": 1, "gold": 1}
        },
        EventType.GEOPOLITICAL: {
            "risk_up": {"equity": -1, "bond": 1, "gold": 1}
        },
        EventType.SYSTEMIC_RISK: {
            "crisis": {"equity": -1, "bond": 1, "gold": 1}
        }
    }
    
    def __init__(self):
        self.events: List[NewsEvent] = []
    
    def add_event(self, event: NewsEvent):
        """添加新闻事件"""
        self.events.append(event)
    
    def compute_daily_factors(
        self,
        date: str
    ) -> Dict:
        """
        计算指定日期的新闻因子
        
        Returns:
            {
                "news_count": int,
                "sentiment_score": float,  # -1 to 1
                "risk_event_count": int,
                "equity_impact": float,
                "bond_impact": float,
                "gold_impact": float
            }
        """
        # 筛选当天事件
        daily_events = [e for e in self.events if e.published_at.startswith(date)]
        
        if not daily_events:
            return {
                "news_count": 0,
                "sentiment_score": 0.0,
                "risk_event_count": 0,
                "equity_impact": 0.0,
                "bond_impact": 0.0,
                "gold_impact": 0.0
            }
        
        # 统计
        sentiment_scores = []
        risk_count = 0
        equity_impacts = []
        bond_impacts = []
        gold_impacts = []
        
        for event in daily_events:
            # 情绪分数
            if event.sentiment == Sentiment.POSITIVE:
                sentiment_scores.append(1)
            elif event.sentiment == Sentiment.NEGATIVE:
                sentiment_scores.append(-1)
            else:
                sentiment_scores.append(0)
            
            # 风险事件计数
            if event.event_type in [EventType.GEOPOLITICAL, EventType.SYSTEMIC_RISK]:
                risk_count += 1
            
            # 影响方向
            if event.impact_equity is not None:
                equity_impacts.append(event.impact_equity * event.confidence)
            if event.impact_bond is not None:
                bond_impacts.append(event.impact_bond * event.confidence)
            if event.impact_gold is not None:
                gold_impacts.append(event.impact_gold * event.confidence)
        
        return {
            "news_count": len(daily_events),
            "sentiment_score": sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
            "risk_event_count": risk_count,
            "equity_impact": sum(equity_impacts) / len(equity_impacts) if equity_impacts else 0,
            "bond_impact": sum(bond_impacts) / len(bond_impacts) if bond_impacts else 0,
            "gold_impact": sum(gold_impacts) / len(gold_impacts) if gold_impacts else 0
        }
    
    def get_risk_alerts(self) -> List[Dict]:
        """获取风险预警事件"""
        alerts = []
        for event in self.events[-50:]:  # 最近50条
            if event.event_type in [EventType.GEOPOLITICAL, EventType.SYSTEMIC_RISK]:
                if event.sentiment == Sentiment.NEGATIVE:
                    alerts.append({
                        "title": event.title,
                        "type": event.event_type.value if event.event_type else "unknown",
                        "published_at": event.published_at,
                        "summary": event.summary
                    })
        return alerts


# LLM结构化新闻的Prompt模板
NEWS_PARSE_PROMPT = """
分析以下金融新闻，提取结构化信息：

标题：{title}
内容：{content}

请以JSON格式输出：
{{
    "event_type": "monetary_policy|economic_data|geopolitical|earnings|regulatory|systemic_risk|other",
    "sentiment": "positive|negative|neutral",
    "impact_timeframe": "immediate|short_term|long_term",
    "confidence": 0.0-1.0,
    "impact_equity": -1|0|1,
    "impact_bond": -1|0|1,
    "impact_gold": -1|0|1,
    "summary": "一句话摘要"
}}
"""

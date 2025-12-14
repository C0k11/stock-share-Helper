"""
微调数据集构建
"""

import json
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger


class FineTuneDataset:
    """微调数据集管理"""
    
    def __init__(self, data_path: str = "data/finetune"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.samples: List[Dict] = []
    
    def add_news_sample(
        self,
        title: str,
        content: str,
        event_type: str,
        sentiment: str,
        impact_equity: int,
        impact_bond: int,
        impact_gold: int,
        summary: str,
        meta: Optional[Dict] = None
    ):
        """添加新闻解析样本"""
        sample = {
            "type": "news_parsing",
            "input": f"标题：{title}\n内容：{content}",
            "output": json.dumps({
                "event_type": event_type,
                "sentiment": sentiment,
                "impact_equity": impact_equity,
                "impact_bond": impact_bond,
                "impact_gold": impact_gold,
                "summary": summary
            }, ensure_ascii=False)
        }
        if meta is not None:
            sample["meta"] = meta
        self.samples.append(sample)
    
    def add_explanation_sample(
        self,
        context: str,
        explanation: str,
        meta: Optional[Dict] = None
    ):
        """添加决策解释样本"""
        sample = {
            "type": "explanation",
            "input": context,
            "output": explanation
        }
        if meta is not None:
            sample["meta"] = meta
        self.samples.append(sample)
    
    def to_conversation_format(self) -> List[Dict]:
        """转换为对话格式（适用于大多数LLM微调）"""
        conversations = []
        
        for sample in self.samples:
            if sample["type"] == "news_parsing":
                system = "你是一个专业的金融新闻分析师。分析新闻并输出结构化JSON。"
                user = sample["input"] + "\n\n请输出JSON格式的分析结果。"
            else:
                system = "你是一个专业的投资顾问助手。"
                user = sample["input"]
            
            item = {
                "conversations": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": sample["output"]}
                ]
            }
            if "meta" in sample:
                item["meta"] = sample["meta"]
            conversations.append(item)
        
        return conversations
    
    def save(self, filename: str = "train.json"):
        """保存数据集"""
        output_path = self.data_path / filename
        data = self.to_conversation_format()
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(data)} samples to {output_path}")
    
    def load(self, filename: str = "train.json"):
        """加载数据集"""
        input_path = self.data_path / filename
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} samples from {input_path}")
        return data
    
    def get_statistics(self) -> Dict:
        """获取数据集统计"""
        news_count = sum(1 for s in self.samples if s["type"] == "news_parsing")
        explanation_count = sum(1 for s in self.samples if s["type"] == "explanation")
        
        return {
            "total_samples": len(self.samples),
            "news_parsing": news_count,
            "explanation": explanation_count
        }


def create_sample_dataset():
    """创建示例数据集（用于测试）"""
    dataset = FineTuneDataset()
    
    # 添加一些示例
    dataset.add_news_sample(
        title="美联储宣布加息25个基点",
        content="美联储在周三的会议后宣布将联邦基金利率上调25个基点，符合市场预期。鲍威尔表示将继续关注通胀数据。",
        event_type="monetary_policy",
        sentiment="negative",
        impact_equity=-1,
        impact_bond=-1,
        impact_gold=0,
        summary="美联储加息25bp，符合预期，鲍威尔态度偏鹰。"
    )
    
    dataset.add_news_sample(
        title="中东地缘冲突升级",
        content="中东地区紧张局势再度升级，原油价格大涨3%，避险情绪上升。",
        event_type="geopolitical",
        sentiment="negative",
        impact_equity=-1,
        impact_bond=1,
        impact_gold=1,
        summary="中东冲突升级，避险情绪上升，利好黄金和债券。"
    )
    
    dataset.add_explanation_sample(
        context="市场状态：Risk-Off，VIX：25，SPY跌破200日均线。SPY信号：趋势-1，动量-1。建议：减仓至30%。",
        explanation="当前市场处于避险模式，VIX升至25显示恐慌情绪上升，SPY跌破重要均线支撑。建议将股票仓位降至30%，增持债券和黄金以对冲风险。密切关注后续走势，若企稳可逐步加仓。"
    )
    
    return dataset

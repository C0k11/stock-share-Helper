"""LLM模块 - 新闻解析、决策解释、微调、本地聊天"""

from .news_parser import NewsParser
from .explainer import DecisionExplainer
from .local_chat import LocalChatModel, chat as local_chat, simple_chat

__all__ = ["NewsParser", "DecisionExplainer", "LocalChatModel", "local_chat", "simple_chat"]

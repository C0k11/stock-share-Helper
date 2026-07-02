"""QuantAI - 多 Agent LLM 量化交易系统（refactor/v2 干净重构）。

包结构（自底向上）：
    config   -> 唯一的类型化配置系统（本次重构第 1 个模块）
    data     -> 数据层：行情抓取 / parquet 存储 / US 交易日历
    features -> 特征工程（严格因果）
    ...      -> signals / llm / agents / execution / risk / backtest / live / evolution / api / ui
"""

__version__ = "0.2.0"

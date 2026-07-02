"""quantai.portfolio — 真实持仓录入 + 组合分析。

- `loader`：`portfolio.local.yaml`/CSV → 类型化 `Portfolio`（真实文件永不入库）。
- `analyzer`：持仓 × 价格历史 → 盈亏/暴露/风险/指标快照（价格注入、可离线测试）。
"""

from quantai.portfolio.analyzer import (
    PortfolioAnalyzer,
    PortfolioSnapshot,
    PositionSnapshot,
    format_snapshot_text,
)
from quantai.portfolio.loader import Portfolio, Position, load_portfolio

__all__ = [
    "Portfolio",
    "Position",
    "load_portfolio",
    "PortfolioAnalyzer",
    "PortfolioSnapshot",
    "PositionSnapshot",
    "format_snapshot_text",
]

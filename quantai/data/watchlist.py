"""自选股 watchlist：`watchlist.local.yaml`（本地、不入库）的加载/保存/增删。

与持仓（portfolio.local.yaml）的分工：watchlist 是**关注**不是**持有**——
只跟踪行情/指标/新闻，不参与盈亏与风险统计。仓库 --load 会把 watchlist 一并
抓价入库（fact_prices/fact_signals/fact_news 都会覆盖到）。

格式：
```yaml
symbols: [SPCX, NVDA, BTC-USD]   # 加密货币用 yfinance 后缀口径（BTC-USD）
```
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _normalize(symbols) -> list[str]:
    """大写、去空、按首次出现去重。

    标量字符串包成单元素列表——`symbols: NVDA`（YAML 自然单标的写法）解析出来
    是 str，直接迭代会逐字符炸成 ['N','V','D','A']（N/V/D/A 都是真实 ticker，
    会静默抓错公司入库，审查实锤）。
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    seen: dict[str, None] = {}
    for s in symbols or []:
        sym = str(s).strip().upper()
        if sym:
            seen.setdefault(sym, None)
    return list(seen)


def load_watchlist(path: str | Path = "watchlist.local.yaml") -> list[str]:
    """读 watchlist。文件不存在 → 空列表（watchlist 是可选功能，不报错）。"""
    p = Path(path)
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if isinstance(raw, list):  # 容忍裸列表写法
        return _normalize(raw)
    if not isinstance(raw, dict):
        raise ValueError(f"watchlist 文件顶层应为 mapping 或 list，收到 {type(raw).__name__}")
    return _normalize(raw.get("symbols", []))


def save_watchlist(symbols, path: str | Path = "watchlist.local.yaml") -> list[str]:
    """写回 watchlist（归一化后）。返回最终列表。"""
    norm = _normalize(symbols)
    Path(path).write_text(
        "# QuantAI 自选股（本地文件，不入库）。加密货币用 yfinance 口径（如 BTC-USD）。\n"
        + yaml.safe_dump({"symbols": norm}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return norm


def add_symbol(symbol: str, path: str | Path = "watchlist.local.yaml") -> list[str]:
    """追加一个标的（幂等）。"""
    return save_watchlist(load_watchlist(path) + [symbol], path)


def remove_symbol(symbol: str, path: str | Path = "watchlist.local.yaml") -> list[str]:
    """移除一个标的（不存在则不变）。"""
    sym = str(symbol).strip().upper()
    return save_watchlist([s for s in load_watchlist(path) if s != sym], path)

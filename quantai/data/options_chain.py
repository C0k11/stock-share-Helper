"""期权链抓取（yfinance 免费源）。

诚实边界：
- Yahoo 期权报价/IV 延迟约 15 分钟，零售参考级——对冲**成本估算**够用；
- 新上市标的（如 SPCX）可能根本没有期权——返回 None，调用方诚实降级；
- `ticker_factory` 注入（测试用假 Ticker，零网络）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from loguru import logger

_CHAIN_COLS = ("strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility")


def _records(df) -> list[dict]:
    cols = [c for c in _CHAIN_COLS if c in df.columns]
    return df[cols].to_dict("records")


class OptionChainFetcher:
    """抓单标的的近月期权链（对冲口径：默认选 20-60 天后到期的第一个）。"""

    def __init__(self, ticker_factory: Optional[Callable] = None) -> None:
        if ticker_factory is None:
            import yfinance as yf

            ticker_factory = yf.Ticker
        self._ticker = ticker_factory

    def fetch(self, symbol: str, min_days: int = 20, max_days: int = 60) -> Optional[dict]:
        """返回 {symbol, expiry, days_to_expiry, calls, puts} 或 None（无期权/抓取失败）。

        到期日选择：[min_days, max_days] 窗口内最早的；窗口内没有则退而求
        ≥7 天的最近一个（太近的末日期权时间价值畸形，不做对冲参考）。
        """
        try:
            t = self._ticker(symbol)
            expiries = list(getattr(t, "options", ()) or ())
            if not expiries:
                return None
            today = datetime.now().date()
            dated = []
            for e in expiries:
                try:
                    d = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
                except ValueError:
                    continue
                if d >= 7:
                    dated.append((d, e))
            if not dated:
                return None
            in_window = [x for x in dated if min_days <= x[0] <= max_days]
            days, expiry = (min(in_window) if in_window else min(dated))
            oc = t.option_chain(expiry)
            return {
                "symbol": str(symbol).upper(),
                "expiry": expiry,
                "days_to_expiry": int(days),
                "calls": _records(oc.calls),
                "puts": _records(oc.puts),
            }
        except Exception as exc:  # noqa: BLE001 - 单标的失败不炸调用方
            logger.warning(f"option chain fetch failed for {symbol}: {exc}")
            return None

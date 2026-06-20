"""实盘/模拟数据源 —— 忠实迁移自 `src/trading/data_feed.py`。

- `DataFeed`：基类，后台线程按 `interval_sec` 轮询 -> `_fetch_and_publish` -> `_publish` 回调订阅者。
- `YFinanceDataFeed`：用 yfinance 抓 1m bar（免费、~15min 延迟）；支持每 tick 只抓一批（轮转）省配额。
- `SimulatedDataFeed`：随机游走**模拟**数据源（测试/演示用）。**明确标注** `source="simulated"`，
  不伪装真实行情。新增可选 `seed` 让模拟可复现（默认仍随机，行为不变）。
- `create_data_feed`：工厂，auto 时优先 yfinance，失败/无库回退 simulated 并打印 WARNING。

与旧版差异：`SimulatedDataFeed` 用独立的 `random.Random(seed)` 实例（可注入种子）替代全局
`random` 模块——既不改默认行为（seed=None 仍随机），又让单测可确定性断言。
"""
from __future__ import annotations

import math
import random
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

try:
    import yfinance as yf

    HAS_YFINANCE = True
except Exception:  # pragma: no cover - 取决于环境是否装 yfinance
    HAS_YFINANCE = False


class DataFeed:
    """市场数据源基类：后台线程轮询，回调推送给订阅者。"""

    def __init__(self, tickers: List[str], interval_sec: float = 5.0) -> None:
        self.tickers = [t.upper() for t in tickers]
        self.interval_sec = interval_sec
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callbacks: List[Callable[[Dict], None]] = []
        self._last_prices: Dict[str, float] = {}
        self.source: str = "unknown"

    def subscribe(self, callback: Callable[[Dict], None]) -> None:
        self._callbacks.append(callback)

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run_loop(self) -> None:
        while self.running and not self._stop_event.is_set():
            try:
                self._fetch_and_publish()
            except Exception as exc:
                print(f"[DataFeed Error] {exc}")
            if self._stop_event.wait(timeout=float(self.interval_sec)):
                break

    def _fetch_and_publish(self) -> None:
        raise NotImplementedError

    def _publish(self, data: Dict) -> None:
        if (not self.running) or self._stop_event.is_set():
            return
        for cb in self._callbacks:
            try:
                cb(data)
            except Exception as exc:
                print(f"[DataFeed Callback Error] {exc}")

    # 共享：每 tick 选一批 ticker（轮转），用于省 API 配额。
    def _select_batch(self, tickers: List[str]) -> List[str]:
        n = int(getattr(self, "_symbols_per_tick", 0) or 0)
        if n > 0 and n < len(tickers) and len(tickers) <= max(12, n * 3):
            n = 0
        if n <= 0 or n >= len(tickers):
            return tickers
        start = int(getattr(self, "_rr_index", 0) or 0) % len(tickers)
        batch = [tickers[(start + i) % len(tickers)] for i in range(n)]
        self._rr_index = (start + n) % len(tickers)
        return batch


class YFinanceDataFeed(DataFeed):
    """yfinance 实时数据源（免费、约 15 分钟延迟）。"""

    def __init__(
        self, tickers: List[str], interval_sec: float = 10.0, symbols_per_tick: int = 0
    ) -> None:
        super().__init__(tickers, interval_sec)
        if not HAS_YFINANCE:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        self._symbols_per_tick = max(0, int(symbols_per_tick or 0))
        self._rr_index = 0
        self._tk_cache: Dict[str, Any] = {}

    def _fetch_and_publish(self) -> None:
        tickers = list(self.tickers)
        if not tickers:
            return
        for ticker in self._select_batch(tickers):
            try:
                tk = self._tk_cache.get(ticker)
                if tk is None:
                    tk = yf.Ticker(ticker)
                    self._tk_cache[ticker] = tk
                hist = tk.history(period="1d", interval="1m", prepost=True)
                if hist.empty:
                    continue
                latest = hist.iloc[-1]
                bar_time = hist.index[-1]
                if hasattr(bar_time, "to_pydatetime"):
                    bar_time = bar_time.to_pydatetime()
                price = float(latest["Close"])
                self._last_prices[ticker] = price
                self._publish(
                    {
                        "ticker": ticker,
                        "time": bar_time,
                        "open": float(latest["Open"]),
                        "high": float(latest["High"]),
                        "low": float(latest["Low"]),
                        "close": price,
                        "volume": int(latest["Volume"]),
                        "source": "yfinance",
                    }
                )
            except Exception as exc:
                print(f"[YFinance] {ticker} fetch error: {exc}")


class SimulatedDataFeed(DataFeed):
    """随机游走模拟数据源（测试/演示）。明确 `source='simulated'`，非真实行情。"""

    def __init__(
        self,
        tickers: List[str],
        interval_sec: float = 4.0,
        base_prices: Optional[Dict[str, float]] = None,
        symbols_per_tick: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(tickers, interval_sec)
        self._rng = random.Random(seed)
        self._base_prices = dict(base_prices or {})
        for ticker in self.tickers:
            if ticker not in self._base_prices:
                self._base_prices[ticker] = self._rng.uniform(100, 500)
        self._current_prices = dict(self._base_prices)
        self._symbols_per_tick = max(0, int(symbols_per_tick or 0))
        self._rr_index = 0

    def _fetch_and_publish(self) -> None:
        tickers = list(self.tickers)
        if not tickers:
            return
        rng = self._rng
        for ticker in self._select_batch(tickers):
            current = self._current_prices[ticker]
            itv = max(0.5, min(float(self.interval_sec or 4.0), 3600.0))
            trading_day_sec = 6.5 * 3600.0
            sigma = 0.005 * math.sqrt(itv / trading_day_sec)
            new_price = current * (1 + rng.gauss(0, sigma))
            new_price = max(10.0, min(2000.0, new_price))
            self._current_prices[ticker] = new_price
            self._publish(
                {
                    "ticker": ticker,
                    "time": datetime.now(),
                    "open": round(current, 2),
                    "high": round(new_price * (1 + abs(rng.gauss(0, 0.002))), 2),
                    "low": round(new_price * (1 - abs(rng.gauss(0, 0.002))), 2),
                    "close": round(new_price, 2),
                    "volume": rng.randint(10000, 1000000),
                    "source": "simulated",
                }
            )


def create_data_feed(
    tickers: List[str],
    source: str = "auto",
    interval_sec: float = 5.0,
    symbols_per_tick: int = 0,
    base_prices: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> DataFeed:
    """工厂：source='yfinance'|'simulated'|'auto'。auto 优先 yfinance，失败/无库回退 simulated。"""
    if source == "yfinance" or (source == "auto" and HAS_YFINANCE):
        try:
            interval_sec = max(float(interval_sec), 15.0)
        except Exception:
            interval_sec = 15.0
        try:
            feed: DataFeed = YFinanceDataFeed(
                tickers, interval_sec, symbols_per_tick=symbols_per_tick
            )
            feed.source = "yfinance"
            print("[DataFeed] Using REAL market data (yfinance)")
            return feed
        except Exception as exc:
            if source == "yfinance":
                raise
            print(f"[DataFeed] yfinance failed: {exc}, falling back to simulated")

    print("[DataFeed] WARNING: Using SIMULATED data (not real market!)")
    feed = SimulatedDataFeed(
        tickers, interval_sec, base_prices=base_prices, symbols_per_tick=symbols_per_tick, seed=seed
    )
    feed.source = "simulated"
    return feed

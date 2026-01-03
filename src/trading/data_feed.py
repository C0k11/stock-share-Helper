# src/trading/data_feed.py
"""
Phase 3.4: Real-time Market Data Feed

Supports:
- yfinance (free, 15-min delayed)
- Simulated data (for testing)
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


class DataFeed:
    """Base class for market data feeds"""
    
    def __init__(self, tickers: List[str], interval_sec: float = 5.0):
        self.tickers = [t.upper() for t in tickers]
        self.interval_sec = interval_sec
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[Dict], None]] = []
        self._last_prices: Dict[str, float] = {}
    
    def subscribe(self, callback: Callable[[Dict], None]) -> None:
        """Subscribe to market data updates"""
        self._callbacks.append(callback)
    
    def start(self) -> None:
        """Start the data feed"""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"DataFeed started: {', '.join(self.tickers)}")
    
    def stop(self) -> None:
        """Stop the data feed"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        print("DataFeed stopped.")
    
    def _run_loop(self) -> None:
        """Main data fetch loop"""
        while self.running:
            try:
                self._fetch_and_publish()
            except Exception as e:
                print(f"[DataFeed Error] {e}")
            time.sleep(self.interval_sec)
    
    def _fetch_and_publish(self) -> None:
        """Override in subclass"""
        raise NotImplementedError
    
    def _publish(self, data: Dict) -> None:
        """Publish data to all subscribers"""
        for cb in self._callbacks:
            try:
                cb(data)
            except Exception as e:
                print(f"[DataFeed Callback Error] {e}")


class YFinanceDataFeed(DataFeed):
    """Real-time data feed using yfinance (15-min delayed)"""
    
    def __init__(self, tickers: List[str], interval_sec: float = 10.0):
        super().__init__(tickers, interval_sec)
        if not HAS_YFINANCE:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
    
    def _fetch_and_publish(self) -> None:
        """Fetch latest quotes from yfinance"""
        for ticker in self.tickers:
            try:
                tk = yf.Ticker(ticker)
                # Get latest 1-day data with 1-minute interval
                hist = tk.history(period="1d", interval="1m")
                
                if hist.empty:
                    continue
                
                latest = hist.iloc[-1]
                price = float(latest["Close"])
                
                # Skip if price unchanged
                if ticker in self._last_prices and abs(price - self._last_prices[ticker]) < 0.01:
                    continue
                
                self._last_prices[ticker] = price
                
                data = {
                    "ticker": ticker,
                    "time": datetime.now(),
                    "open": float(latest["Open"]),
                    "high": float(latest["High"]),
                    "low": float(latest["Low"]),
                    "close": price,
                    "volume": int(latest["Volume"]),
                    "source": "yfinance",
                }
                self._publish(data)
                
            except Exception as e:
                print(f"[YFinance] {ticker} fetch error: {e}")


class SimulatedDataFeed(DataFeed):
    """Simulated data feed for testing (random walk)"""
    
    def __init__(
        self,
        tickers: List[str],
        interval_sec: float = 4.0,
        base_prices: Optional[Dict[str, float]] = None,
    ):
        super().__init__(tickers, interval_sec)
        import random
        
        # Initialize base prices
        self._base_prices = base_prices or {}
        for ticker in self.tickers:
            if ticker not in self._base_prices:
                self._base_prices[ticker] = random.uniform(100, 500)
        
        self._current_prices = dict(self._base_prices)
    
    def _fetch_and_publish(self) -> None:
        """Generate simulated price data"""
        import random
        
        for ticker in self.tickers:
            # Random walk with drift
            current = self._current_prices[ticker]
            change_pct = random.gauss(0, 0.005)  # 0.5% daily vol
            new_price = current * (1 + change_pct)
            
            # Clamp to reasonable range
            new_price = max(10, min(2000, new_price))
            self._current_prices[ticker] = new_price
            
            # Generate OHLC
            high = new_price * (1 + abs(random.gauss(0, 0.002)))
            low = new_price * (1 - abs(random.gauss(0, 0.002)))
            open_price = current
            
            data = {
                "ticker": ticker,
                "time": datetime.now(),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(new_price, 2),
                "volume": random.randint(10000, 1000000),
                "source": "simulated",
            }
            self._publish(data)


def create_data_feed(
    tickers: List[str],
    source: str = "auto",
    interval_sec: float = 5.0,
) -> DataFeed:
    """Factory function to create appropriate data feed"""
    
    if source == "yfinance" or (source == "auto" and HAS_YFINANCE):
        try:
            feed = YFinanceDataFeed(tickers, interval_sec)
            print(f"[DataFeed] Using REAL market data (yfinance)")
            return feed
        except Exception as e:
            print(f"[DataFeed] yfinance failed: {e}, falling back to simulated")
    
    # Fallback to simulated
    print("[DataFeed] WARNING: Using SIMULATED data (not real market!)")
    return SimulatedDataFeed(tickers, interval_sec)

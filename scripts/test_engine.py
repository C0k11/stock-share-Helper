import time
from datetime import datetime

from src.trading.broker import PaperBroker
from src.trading.engine import TradingEngine
from src.trading.event import Event, EventType


class DummyStrategy:
    def on_bar(self, bar: dict):
        ticker = str(bar.get("ticker") or "").upper().strip()
        price = float(bar.get("close") or 0.0)
        if not ticker or price <= 0:
            return None
        return {
            "ticker": ticker,
            "action": "BUY",
            "price": price,
            "shares": 10,
        }


class DummyPortfolio:
    def __init__(self):
        self.fills: list[dict] = []

    def on_fill(self, fill: dict) -> None:
        self.fills.append(dict(fill))
        print(f"portfolio >> FILL: {fill}")


def main() -> None:
    engine = TradingEngine()
    engine.strategy = DummyStrategy()
    engine.portfolio = DummyPortfolio()
    engine.broker = PaperBroker(engine)

    engine.start()

    bar = {"ticker": "NVDA", "close": 500.0}
    engine.push_event(Event(type=EventType.MARKET_DATA, timestamp=datetime.now(), payload=bar))

    time.sleep(1.0)
    engine.stop()


if __name__ == "__main__":
    main()

"""Goal-based data agent: fetch real market data and compute required indicators."""
from dataclasses import dataclass
from core.data import load_history
from core.indicators import enrich

@dataclass
class DataAgent:
    source: str = 'yahoo'

    def __post_init__(self):
        if self.source != 'yahoo':
            raise ValueError("Only real market data ('yahoo') is supported; offline synthetic data is disabled.")

    def collect(self, ticker: str = 'AAPL', period: str = '2y'):
        prices = load_history(ticker=ticker, period=period, source=self.source)
        return enrich(prices)

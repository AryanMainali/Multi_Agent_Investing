"""Model-based reflex: tracks previous and current market state."""
from dataclasses import dataclass
import pandas as pd

@dataclass
class MarketAgent:
    volume_threshold: float = 0.75

    def scan(self, df: pd.DataFrame) -> dict:
        last = df.iloc[-1]
        previous = df.iloc[-2]
        volume_ratio = float(last['Volume'] / last['AVG_VOLUME_20']) if last['AVG_VOLUME_20'] > 0 else 0.0
        return {
            'trend': 'UP' if last['SMA_FAST'] > last['SMA_SLOW'] else 'DOWN',
            'previous_trend': 'UP' if previous['SMA_FAST'] > previous['SMA_SLOW'] else 'DOWN',
            'cross_up_today': bool(last['CROSS_UP']),
            'cross_down_today': bool(last['CROSS_DOWN']),
            'volume_ratio': round(volume_ratio, 3),
            'passes_volume_filter': volume_ratio >= self.volume_threshold,
        }

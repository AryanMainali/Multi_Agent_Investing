"""Stage 1 [Market Data]: Model-based reflex agent tracking market trend, crossovers, volume, and volatility."""
from dataclasses import dataclass
import pandas as pd


@dataclass
class MarketAgent:
    volume_threshold: float = 0.75

    def scan(self, df: pd.DataFrame) -> dict:
        last = df.iloc[-1]
        previous = df.iloc[-2]
        volume_ratio = float(last['Volume'] / last['AVG_VOLUME_20']) if last['AVG_VOLUME_20'] > 0 else 0.0
        vol_20 = float(last['VOLATILITY_20']) if pd.notna(last['VOLATILITY_20']) else 0.0
        mom_20 = float(last['MOMENTUM_20']) if pd.notna(last['MOMENTUM_20']) else 0.0
        return {
            'trend': 'UP' if last['SMA_FAST'] > last['SMA_SLOW'] else 'DOWN',
            'previous_trend': 'UP' if previous['SMA_FAST'] > previous['SMA_SLOW'] else 'DOWN',
            'cross_up_today': bool(last['CROSS_UP']),
            'cross_down_today': bool(last['CROSS_DOWN']),
            'volume_ratio': round(volume_ratio, 3),
            'passes_volume_filter': volume_ratio >= self.volume_threshold,
            'price': round(float(last['Close']), 2),
            'sma_20': round(float(last['SMA_FAST']), 2),
            'sma_50': round(float(last['SMA_SLOW']), 2),
            'volatility_20_pct': round(vol_20 * 100, 2),
            'momentum_20_pct': round(mom_20 * 100, 2),
            'bars_count': int(len(df)),
            'as_of': str(df.index[-1].date()),
        }

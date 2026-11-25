"""Utility-based scoring prototype; no fitted forecasting or LLM claims."""
from dataclasses import dataclass
import pandas as pd

@dataclass
class AnalystAgent:
    def evaluate(self, df: pd.DataFrame, market: dict) -> dict:
        last = df.iloc[-1]
        score = 0
        score += 2 if last['SMA_FAST'] > last['SMA_SLOW'] else -2
        score += 1 if last['Close'] > last['SMA_FAST'] else -1
        score += 1 if last['MOMENTUM_20'] > 0 else -1
        score += 1 if market['passes_volume_filter'] else 0
        if market['cross_up_today']:
            signal = 'ENTRY CANDIDATE'
        elif market['cross_down_today']:
            signal = 'EXIT CANDIDATE'
        else:
            signal = 'WATCH / NO NEW CROSSOVER'
        return {
            'signal': signal,
            'heuristic_score': int(score),
            'price': round(float(last['Close']), 2),
            'sma_20': round(float(last['SMA_FAST']), 2),
            'sma_50': round(float(last['SMA_SLOW']), 2),
            'momentum_20_pct': round(100 * float(last['MOMENTUM_20']), 2),
            'note': 'Rule-based screening score, not an expected-return forecast.',
        }

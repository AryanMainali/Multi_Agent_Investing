"""Catalog of past-only daily-bar trading strategies.

Every strategy computes ENTRY_SIGNAL and EXIT_SIGNAL strictly from information
available at or before bar T's close. Orders execute no earlier than bar T+1's open.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from core.indicators import enrich


@dataclass(frozen=True)
class StrategySpec:
    key: str
    name: str
    category: str
    description: str
    entry_reason: str
    exit_reason: str
    uses_risk_policy: bool = True


STRATEGIES: dict[str, StrategySpec] = {
    'sma_crossover': StrategySpec(
        key='sma_crossover',
        name='SMA 20/50 Crossover',
        category='Trend Following',
        description='Buy when 20-day SMA crosses above 50-day SMA; exit when 20-day SMA crosses below 50-day SMA.',
        entry_reason='SMA 20/50 cross up',
        exit_reason='SMA 20/50 cross down',
    ),
    'ema_crossover': StrategySpec(
        key='ema_crossover',
        name='EMA 12/26 Crossover',
        category='Momentum',
        description='Buy when faster 12-day EMA crosses above 26-day EMA; exit when 12-day EMA crosses below 26-day EMA.',
        entry_reason='EMA 12/26 cross up',
        exit_reason='EMA 12/26 cross down',
    ),
    'macd_crossover': StrategySpec(
        key='macd_crossover',
        name='MACD (12/26/9) Signal',
        category='Momentum',
        description='Buy when MACD line crosses above its 9-day EMA signal line; exit when MACD crosses below signal.',
        entry_reason='MACD cross above signal',
        exit_reason='MACD cross below signal',
    ),
    'rsi_mean_reversion': StrategySpec(
        key='rsi_mean_reversion',
        name='RSI (14) Mean Reversion',
        category='Mean Reversion',
        description='Buy when 14-day Wilder RSI dips below 30 (oversold); exit when RSI rises above 70 (overbought).',
        entry_reason='RSI(14) oversold (<30)',
        exit_reason='RSI(14) overbought (>70)',
    ),
    'bollinger_reversion': StrategySpec(
        key='bollinger_reversion',
        name='Bollinger Bands (20, 2σ)',
        category='Mean Reversion',
        description='Buy when close pierces below the lower 20-day 2σ band; exit when price reverts to the 20-day SMA.',
        entry_reason='Below lower Bollinger Band',
        exit_reason='Reverted to 20d SMA',
    ),
    'donchian_breakout': StrategySpec(
        key='donchian_breakout',
        name='Donchian 20/10 Breakout',
        category='Breakout',
        description='Buy when close breaks above prior 20-day high; exit when close breaks below prior 10-day low.',
        entry_reason='20-day high breakout',
        exit_reason='10-day low breakdown',
    ),
    'buy_and_hold': StrategySpec(
        key='buy_and_hold',
        name='Buy & Hold (Benchmark)',
        category='Benchmark',
        description='Allocate starting cash into whole shares on the first available open (with fees/slippage) and hold.',
        entry_reason='Initial benchmark buy',
        exit_reason='End of period',
        uses_risk_policy=False,
    ),
}


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    loss_zero = (avg_gain > 0.0) & (avg_loss == 0.0)
    rsi = rsi.mask(loss_zero, 100.0).mask(both_zero, 50.0)
    return rsi


def generate_signals(
    prices: pd.DataFrame,
    strategy: str = 'sma_crossover',
    fast: int = 20,
    slow: int = 50,
) -> tuple[pd.DataFrame, StrategySpec]:
    """Attach past-only boolean ENTRY_SIGNAL and EXIT_SIGNAL columns for the requested strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGIES)}")
    spec = STRATEGIES[strategy]

    if strategy == 'sma_crossover':
        x = enrich(prices, fast=fast, slow=slow)
        x['ENTRY_SIGNAL'] = x['CROSS_UP'].astype(bool)
        x['EXIT_SIGNAL'] = x['CROSS_DOWN'].astype(bool)
        return x, spec

    x = prices.copy()
    close = x['Close']

    if strategy == 'ema_crossover':
        ema_fast = close.ewm(span=12, min_periods=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, min_periods=26, adjust=False).mean()
        above = ema_fast > ema_slow
        prior = above.shift(1, fill_value=False)
        valid = ema_slow.notna() & ema_slow.shift(1).notna()
        x['ENTRY_SIGNAL'] = (above & ~prior & valid).astype(bool)
        x['EXIT_SIGNAL'] = (~above & prior & valid).astype(bool)

    elif strategy == 'macd_crossover':
        ema_12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()
        above = macd > signal
        prior = above.shift(1, fill_value=False)
        valid = signal.notna() & signal.shift(1).notna()
        x['ENTRY_SIGNAL'] = (above & ~prior & valid).astype(bool)
        x['EXIT_SIGNAL'] = (~above & prior & valid).astype(bool)

    elif strategy == 'rsi_mean_reversion':
        rsi = _compute_rsi(close, period=14)
        valid = rsi.notna() & rsi.shift(1).notna()
        x['ENTRY_SIGNAL'] = ((rsi < 30.0) & (rsi.shift(1) >= 30.0) & valid).astype(bool)
        x['EXIT_SIGNAL'] = ((rsi > 70.0) & rsi.notna()).astype(bool)

    elif strategy == 'bollinger_reversion':
        bb_mid = close.rolling(20, min_periods=20).mean()
        bb_std = close.rolling(20, min_periods=20).std(ddof=0)
        bb_lower = bb_mid - 2.0 * bb_std
        valid = bb_lower.notna() & bb_lower.shift(1).notna()
        x['ENTRY_SIGNAL'] = ((close < bb_lower) & (close.shift(1) >= bb_lower.shift(1)) & valid).astype(bool)
        x['EXIT_SIGNAL'] = ((close >= bb_mid) & bb_mid.notna()).astype(bool)

    elif strategy == 'donchian_breakout':
        prior_high_20 = x['High'].shift(1).rolling(20, min_periods=20).max()
        prior_low_10 = x['Low'].shift(1).rolling(10, min_periods=10).min()
        x['ENTRY_SIGNAL'] = ((close > prior_high_20) & prior_high_20.notna()).astype(bool)
        x['EXIT_SIGNAL'] = ((close < prior_low_10) & prior_low_10.notna()).astype(bool)

    elif strategy == 'buy_and_hold':
        entry = pd.Series(False, index=x.index)
        if len(entry) > 0:
            entry.iloc[0] = True
        x['ENTRY_SIGNAL'] = entry
        x['EXIT_SIGNAL'] = pd.Series(False, index=x.index)

    return x, spec

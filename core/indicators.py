"""Past-only technical indicators. Daily signal occurs after bar close."""
import pandas as pd


def enrich(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    if fast < 2 or slow <= fast:
        raise ValueError('Require 2 <= fast < slow moving-average periods')
    x = df.copy()
    x['SMA_FAST'] = x['Close'].rolling(fast, min_periods=fast).mean()
    x['SMA_SLOW'] = x['Close'].rolling(slow, min_periods=slow).mean()
    x['RET_1D'] = x['Close'].pct_change()
    x['VOLATILITY_20'] = x['RET_1D'].rolling(20).std() * (252 ** 0.5)
    x['AVG_VOLUME_20'] = x['Volume'].rolling(20).mean()
    x['MOMENTUM_20'] = x['Close'].pct_change(20)
    above = x['SMA_FAST'] > x['SMA_SLOW']
    prior = above.shift(1, fill_value=False)
    # Exclude the first observation with defined slow SMA from generating a fake cross.
    valid = x['SMA_SLOW'].notna() & x['SMA_SLOW'].shift(1).notna()
    x['CROSS_UP'] = (above & ~prior & valid).astype(bool)
    x['CROSS_DOWN'] = (~above & prior & valid).astype(bool)
    return x

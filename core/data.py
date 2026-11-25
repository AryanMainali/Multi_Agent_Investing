"""Historical OHLCV retrieval using real market data only (Yahoo Finance)."""
import re
import pandas as pd

REQUIRED = ['Open', 'High', 'Low', 'Close', 'Volume']


def validate_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate daily OHLCV bars. Never forward-fill missing quotes."""
    if frame.empty:
        raise ValueError('The data source returned no price history.')
    df = frame.copy()
    if not set(REQUIRED).issubset(df.columns):
        raise ValueError(f'Price data must contain: {REQUIRED}')
    df = df[REQUIRED].copy()
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep='last')].sort_index()
    for col in REQUIRED:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=REQUIRED)
    if len(df) < 60:
        raise ValueError('At least 60 daily OHLCV rows are required for 20/50 SMA.')
    if ((df[['Open', 'High', 'Low', 'Close']] <= 0).any().any() or
        (df['Volume'] < 0).any() or
        (df['High'] < df[['Open', 'Close', 'Low']].max(axis=1)).any() or
        (df['Low'] > df[['Open', 'Close', 'High']].min(axis=1)).any()):
        raise ValueError('Invalid OHLCV prices, ranges, or volume.')
    return df


def load_history(ticker: str = 'AAPL', period: str = '2y', source: str = 'yahoo') -> pd.DataFrame:
    """Fetch real historical daily OHLCV data from Yahoo Finance. Synthetic/offline data is disabled."""
    if source != 'yahoo':
        raise ValueError("Only real market data ('yahoo') is supported; offline synthetic data is disabled.")
    symbol = ticker.strip().upper()
    if not re.fullmatch(r'[A-Z0-9.\-^]{1,15}', symbol):
        raise ValueError('Ticker must be a valid symbol, e.g. AAPL or MSFT.')
    import yfinance as yf
    df = yf.Ticker(symbol).history(period=period, interval='1d', auto_adjust=True)
    return validate_bars(df)

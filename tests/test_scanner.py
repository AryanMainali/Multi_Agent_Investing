import numpy as np
import pandas as pd
import yfinance as yf
from agents.risk_agent import get_risk_policy
from core.scanner import extract_symbol_bars, scan_market_and_suggest, scan_universe_technicals
from core.universe import get_universe_tickers, normalize_yahoo_symbol


def _build_multi_ticker_batch(specs: dict[str, tuple[float, float]], n: int = 160) -> pd.DataFrame:
    """Build a MultiIndex (symbol, field) OHLCV DataFrame matching yf.download(..., group_by='ticker')."""
    dates = pd.bdate_range('2026-03-02', periods=n)
    t = np.arange(n, dtype=float)
    frames = {}
    for sym, (base_price, slope) in specs.items():
        close = base_price + slope * t + 1.2 * np.sin(t / 4.0)
        open_ = close - 0.25
        high = close + 0.9
        low = open_ - 0.8
        vol = np.full(n, 35_000_000, dtype=int)
        frames[sym] = pd.DataFrame(
            {'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': vol},
            index=dates,
        )
    return pd.concat(frames, axis=1)


def test_universe_helpers():
    assert normalize_yahoo_symbol('brk.b') == 'BRK-B'
    sp100 = get_universe_tickers('sp100')
    sp500 = get_universe_tickers('sp500', refresh_web=False)
    assert len(sp100) >= 95
    assert len(sp500) >= 400
    assert 'AAPL' in sp100
    assert 'BRK-B' in sp100


def test_scanner_filters_unaffordable_stocks_and_shortlists_top_k(monkeypatch, tmp_path):
    # BKNG (~$4000) exceeds $2000 max allocation on $10,000 Medium risk (20%);
    # NVDA, AAPL, MSFT, AMZN, META, GOOGL are affordable; INTC is in a downtrend.
    ticker_specs = {
        'BKNG': (3900.0, 2.0),
        'NVDA': (110.0, 0.55),
        'AAPL': (180.0, 0.45),
        'MSFT': (320.0, 0.50),
        'AMZN': (160.0, 0.40),
        'META': (350.0, 0.48),
        'GOOGL': (145.0, 0.35),
        'INTC': (45.0, -0.15),
    }
    batch_df = _build_multi_ticker_batch(ticker_specs)

    monkeypatch.setattr(yf, 'download', lambda *args, **kwargs: batch_df)

    class DummyTicker:
        def __init__(self, symbol):
            self.symbol = symbol
            self.info = {'shortName': f'{symbol} Corp', 'sector': 'Technology', 'trailingPE': 24.0}

        def history(self, period='2y', interval='1d', auto_adjust=True):
            return extract_symbol_bars(batch_df, self.symbol, list(ticker_specs.keys()))

        def get_news(self, count=6):
            return [{'title': f'{self.symbol} reports strong demand and profit growth', 'publisher': 'Yahoo Finance'}]

    monkeypatch.setattr(yf, 'Ticker', DummyTicker)

    stage1 = scan_universe_technicals(
        tickers=list(ticker_specs.keys()),
        bankroll=10000.0,
        policy=get_risk_policy('medium'),
    )
    assert stage1['universe_size'] == 8
    assert stage1['valid_data_count'] == 8
    # BKNG (~$4000) is filtered out as unaffordable for 1 whole share under $2000 cap
    assert 'BKNG' not in stage1['leaderboard_df']['Ticker'].values
    assert stage1['affordable_count'] == 7

    db = tmp_path / 'scan.db'
    scan_res = scan_market_and_suggest(
        scope='sp100',
        bankroll=10000.0,
        policy=get_risk_policy('medium'),
        top_k=5,
        tickers_override=list(ticker_specs.keys()),
        db_path=db,
    )
    assert len(scan_res['candidates']) == 5
    assert scan_res['passed_count'] >= 4
    assert 'BKNG' not in scan_res['shortlist_summary_df']['Ticker'].values
    assert scan_res['candidates'][0]['workflow']['outcome'] == 'SUGGESTION_READY'


def test_scanner_diversifies_shortlist_across_sectors(monkeypatch, tmp_path):
    # 4 Tech stocks + JPM (Financials) + JNJ (Healthcare) + XOM (Energy) under Low risk (max 1 stock/sector)
    ticker_specs = {
        'NVDA': (105.0, 0.55),
        'AAPL': (150.0, 0.50),
        'AMD': (120.0, 0.48),
        'ORCL': (130.0, 0.46),
        'JPM': (180.0, 0.40),
        'JNJ': (155.0, 0.35),
        'XOM': (110.0, 0.32),
    }
    batch_df = _build_multi_ticker_batch(ticker_specs)
    monkeypatch.setattr(yf, 'download', lambda *args, **kwargs: batch_df)

    class DummyTicker:
        def __init__(self, symbol):
            self.symbol = symbol
            self.info = {'shortName': f'{symbol} Corp', 'trailingPE': 20.0}

        def history(self, period='2y', interval='1d', auto_adjust=True):
            return extract_symbol_bars(batch_df, self.symbol, list(ticker_specs.keys()))

        def get_news(self, count=6):
            return [{'title': f'{self.symbol} strong profit growth', 'publisher': 'Yahoo Finance'}]

    monkeypatch.setattr(yf, 'Ticker', DummyTicker)

    scan_res = scan_market_and_suggest(
        scope='sp100',
        bankroll=25000.0,
        policy=get_risk_policy('low'),  # max 1 position per sector!
        top_k=4,
        tickers_override=list(ticker_specs.keys()),
        db_path=tmp_path / 'div.db',
    )
    sectors = scan_res['shortlist_summary_df']['Sector'].tolist()
    # Under max 1 per sector with 4 distinct sectors available, all 4 shortlisted candidates come from distinct sectors!
    assert len(sectors) == 4
    assert len(set(sectors)) == 4


from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from streamlit.testing.v1 import AppTest
import core.scanner as scanner_mod
import core.storage as storage_mod
from core.strategies import STRATEGIES

APP_PATH = Path(__file__).resolve().parents[1] / 'app.py'


def _make_batch(specs: dict[str, tuple[float, float]], n: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range('2026-02-02', periods=n, tz='America/New_York')
    t = np.arange(n, dtype=float)
    frames = {}
    for sym, (base, slope) in specs.items():
        close = base + slope * t + 1.5 * np.sin(t / 5.0)
        frames[sym] = pd.DataFrame(
            {
                'Open': close - 0.3,
                'High': close + 1.1,
                'Low': close - 0.9,
                'Close': close,
                'Volume': np.full(n, 40_000_000, dtype=int),
            },
            index=dates,
        )
    return pd.concat(frames, axis=1)


def test_onboarding_has_no_ticker_input_and_scans_sp_market(monkeypatch, tmp_path):
    specs = {
        'AAPL': (160.0, 0.45),
        'MSFT': (280.0, 0.50),
        'NVDA': (110.0, 0.55),
        'AMZN': (150.0, 0.40),
        'GOOGL': (140.0, 0.35),
        'META': (310.0, 0.42),
    }
    batch_df = _make_batch(specs)
    test_db = tmp_path / 'app_scan.db'
    monkeypatch.setattr(storage_mod, 'DEFAULT_DB', test_db)
    monkeypatch.setattr(scanner_mod, 'get_universe_tickers', lambda scope='sp100': list(specs.keys()))
    monkeypatch.setattr(yf, 'download', lambda *args, **kwargs: batch_df)

    class DummyTicker:
        def __init__(self, symbol):
            self.symbol = symbol
            self.info = {'shortName': f'{symbol} Inc.', 'sector': 'Technology', 'trailingPE': 23.5}

        def history(self, period='2y', interval='1d', auto_adjust=True):
            return batch_df[self.symbol].copy()

        def get_news(self, count=6):
            return [{'title': f'{self.symbol} beats expectations on strong growth', 'publisher': 'Yahoo Finance'}]

    monkeypatch.setattr(yf, 'Ticker', DummyTicker)

    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception

    # Onboarding asks ONLY for paper trading amount, scan scope, and risk tolerance — NO ticker input!
    assert len(at.text_input) == 0
    assert any('Paper trading amount' in ni.label for ni in at.number_input)
    assert any('How much risk are you willing to take?' in r.label for r in at.radio)

    at.number_input[0].set_value(25000.0)
    at.radio[0].set_value('High (Aggressive)')
    at.run()
    assert not at.exception

    start_btn = [b for b in at.button if b.label == 'Scan Market & Start Paper Trading'][0]
    start_btn.click().run()
    assert not at.exception

    # Verify automatic S&P market scan ran, shortlisted Top 5, and rendered the #1 candidate's report & backtests
    assert at.session_state['onboarding_complete'] is True
    assert at.session_state['bankroll'] == 25000.0
    assert any('Market Scan Overview — S&P 100 Discovery Funnel' in s.value for s in at.subheader)
    assert any('Risk Gate PASSED' in s.value for s in at.success)
    assert len(at.multiselect) == 1
    assert at.multiselect[0].value == [spec.name for spec in STRATEGIES.values()]

    # Verify human manual paper approval opens a tracked paper position AND logs decision
    approve_btn = [b for b in at.button if b.label == 'Act Manually (Log Paper Approval)'][0]
    approve_btn.click().run()
    assert not at.exception
    assert len(storage_mod.list_decisions(test_db)) == 1
    assert len(storage_mod.list_paper_positions('OPEN', test_db)) == 1

    # Verify manual close of that open paper position from the Live Paper Portfolio tab
    close_btn = [b for b in at.button if b.label == 'Close Selected Paper Position'][0]
    close_btn.click().run()
    assert not at.exception
    assert len(storage_mod.list_paper_positions('OPEN', test_db)) == 0
    assert len(storage_mod.list_paper_positions('CLOSED', test_db)) == 1

    # Switch sidebar to Manual Ticker Lookup override and verify single-ticker workflow works
    mode_radio = [r for r in at.sidebar.radio if r.label == 'Stock discovery mode'][0]
    mode_radio.set_value('Manual Ticker Lookup').run()
    assert not at.exception
    assert any('1. Six-agent suggestion workflow — AAPL' in s.value for s in at.subheader)


def test_app_manual_mode_dropped_branch_disables_approval(monkeypatch, tmp_path):
    dates = pd.bdate_range('2026-02-02', periods=180, tz='America/New_York')
    t = np.arange(180, dtype=float)
    close = 260.0 - 0.65 * t
    bear_bars = pd.DataFrame(
        {'Open': close + 0.3, 'High': close + 1.0, 'Low': close - 1.0, 'Close': close, 'Volume': 30_000_000},
        index=dates,
    )
    batch_df = pd.concat({'WEAK': bear_bars}, axis=1)
    test_db = tmp_path / 'app_drop.db'
    monkeypatch.setattr(storage_mod, 'DEFAULT_DB', test_db)
    monkeypatch.setattr(scanner_mod, 'get_universe_tickers', lambda scope='sp100': ['WEAK'])
    monkeypatch.setattr(yf, 'download', lambda *args, **kwargs: batch_df)

    class BearishTicker:
        def __init__(self, symbol):
            self.info = {'shortName': 'Bearish Co'}

        def history(self, period='2y', interval='1d', auto_adjust=True):
            return bear_bars

        def get_news(self, count=6):
            return [{'title': 'Weak guidance and earnings miss', 'publisher': 'News'}]

    monkeypatch.setattr(yf, 'Ticker', BearishTicker)

    at = AppTest.from_file(APP_PATH).run()
    start_btn = [b for b in at.button if b.label == 'Scan Market & Start Paper Trading'][0]
    start_btn.click().run()
    assert not at.exception
    assert any('Risk Gate DROPPED' in w.value for w in at.warning)

    approve_btn = [b for b in at.button if b.label == 'Act Manually (Log Paper Approval)'][0]
    assert approve_btn.disabled is True
    assert len(storage_mod.list_dropped_ideas(test_db)) == 1

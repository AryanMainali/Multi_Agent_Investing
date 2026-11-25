from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from streamlit.testing.v1 import AppTest
from core.strategies import STRATEGIES

APP_PATH = Path(__file__).resolve().parents[1] / 'app.py'


def _mock_yahoo_bars(n: int = 200) -> pd.DataFrame:
    dates = pd.bdate_range('2025-01-02', periods=n, tz='America/New_York')
    t = np.arange(n, dtype=float)
    close = 190.0 + 0.12 * t + 18.0 * np.sin(t / 9.0) + 5.0 * np.cos(t / 3.0)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 1.5
    low = np.minimum(open_, close) - 1.5
    volume = np.full(n, 40_000_000, dtype=int)
    return pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume}, index=dates)


def test_onboarding_and_multi_strategy_comparison(monkeypatch):
    bars = _mock_yahoo_bars()

    class DummyTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period='2y', interval='1d', auto_adjust=True):
            return bars

    monkeypatch.setattr(yf, 'Ticker', DummyTicker)

    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    assert any('Paper trading amount' in ni.label for ni in at.number_input)
    assert any('How much risk are you willing to take?' in r.label for r in at.radio)
    assert not any('Offline synthetic' in str(r.options) for r in at.radio)

    # Set custom paper trading amount ($25,000) and High risk, then start
    at.number_input[0].set_value(25000.0)
    at.radio[0].set_value('High (Aggressive)')
    at.run()
    assert not at.exception

    start_btn = [b for b in at.button if b.label == 'Start Paper Trading'][0]
    start_btn.click().run()
    assert not at.exception

    # Verify dashboard loaded with real Yahoo Finance data, $25,000 capital, High risk, and all strategies
    assert at.session_state['onboarding_complete'] is True
    assert at.session_state['bankroll'] == 25000.0
    assert at.session_state['risk_label'] == 'High (Aggressive)'
    assert any('1. Four agents — AAPL' in s.value for s in at.subheader)
    assert any('2. Multi-strategy historical backtest — AAPL' in s.value for s in at.subheader)
    assert any('$25,000.00' in m.value for m in at.sidebar.metric)

    # Verify multiselect defaults to all strategies and can inspect another strategy
    assert len(at.multiselect) == 1
    expected_names = [spec.name for spec in STRATEGIES.values()]
    assert at.multiselect[0].value == expected_names

    inspect_select = [sb for sb in at.selectbox if sb.label == 'Choose a strategy to inspect'][0]
    inspect_select.set_value('RSI (14) Mean Reversion').run()
    assert not at.exception

    # Verify user can return to change amount / risk
    change_btn = [b for b in at.sidebar.button if b.label == 'Change Paper Amount / Risk'][0]
    change_btn.click().run()
    assert not at.exception
    assert at.session_state['onboarding_complete'] is False

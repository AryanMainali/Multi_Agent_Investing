import numpy as np
import pandas as pd
import pytest
import yfinance as yf
from agents.data_agent import DataAgent
from agents.risk_agent import RiskPolicy, get_risk_policy, size_position
from core.data import load_history
from core.storage import list_decisions, save_decision
from core.workflow import analyze


def _make_ohlcv_bars(n: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range('2025-01-02', periods=n, tz='America/New_York')
    t = np.arange(n, dtype=float)
    close = 180.0 + 0.15 * t + 12.0 * np.sin(t / 9.0)
    open_ = np.roll(close, 1)
    open_[0] = close[0] - 0.5
    high = np.maximum(open_, close) + 1.5
    low = np.minimum(open_, close) - 1.5
    volume = 50_000_000 + (5_000_000 * np.cos(t / 5.0)).astype(int)
    return pd.DataFrame(
        {'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume},
        index=dates,
    )


def test_real_data_pipeline_and_risk_scaling(monkeypatch):
    bars = _make_ohlcv_bars()

    class DummyTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period='2y', interval='1d', auto_adjust=True):
            return bars

    monkeypatch.setattr(yf, 'Ticker', DummyTicker)

    low_policy = get_risk_policy('low')
    high_policy = get_risk_policy('high')

    data_low, res_low = analyze(ticker='aapl', bankroll=20000, policy=low_policy)
    _, res_high = analyze(ticker='AAPL', bankroll=20000, policy=high_policy)

    assert len(data_low) == len(bars)
    assert data_low.index.is_monotonic_increasing
    assert res_low['source'] == 'yahoo'
    assert res_low['ticker'] == 'AAPL'
    assert res_low['risk']['notional'] <= 20000 * low_policy.max_allocation + 0.01
    assert res_high['risk']['notional'] <= 20000 * high_policy.max_allocation + 0.01
    assert res_high['risk']['shares'] > res_low['risk']['shares']


def test_synthetic_and_unknown_sources_disabled():
    with pytest.raises(ValueError, match='offline synthetic data is disabled'):
        load_history(ticker='AAPL', source='sample')
    with pytest.raises(ValueError):
        load_history(ticker='AAPL', source='unknown')
    with pytest.raises(ValueError):
        DataAgent(source='sample')
    with pytest.raises(ValueError):
        analyze(ticker='AAPL', bankroll=10000, source='sample')


def test_risk_profiles_and_sizing_caps():
    result = size_position(100, 10000, 10000, RiskPolicy())
    assert result['shares'] == 20
    assert result['nominal_risk'] <= 100
    assert result['notional'] <= 2000
    assert get_risk_policy('conservative') == get_risk_policy('low')
    assert get_risk_policy('moderate') == get_risk_policy('medium')
    assert get_risk_policy('aggressive') == get_risk_policy('high')
    with pytest.raises(ValueError):
        get_risk_policy('extreme')


def test_no_cash_no_shares():
    assert size_position(100, 20, 10000)['shares'] == 0


def test_invalid_policy():
    with pytest.raises(ValueError):
        RiskPolicy(max_allocation=1.5)


def test_decision_audit(tmp_path):
    db = tmp_path / 'log.db'
    save_decision('AAPL', 'yahoo', 'REJECT', 195.5, 'Test', db)
    rows = list_decisions(db)
    assert len(rows) == 1
    assert rows[0][1] == 'AAPL'
    assert rows[0][2] == 'yahoo'
    assert rows[0][3] == 'REJECT'
    with pytest.raises(ValueError):
        save_decision('AAPL', 'yahoo', 'REAL_BUY', 195.5, db_path=db)

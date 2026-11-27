import numpy as np
import pandas as pd
import pytest
from core.portfolio import evaluate_paper_portfolio, get_paper_cash_summary
from core.storage import close_paper_position, list_paper_positions, open_paper_position


def _make_bars(last_close: float, last_low: float | None = None, n: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range('2026-06-01', periods=n)
    close = np.full(n, last_close, dtype=float)
    low = np.full(n, last_close - 1.0, dtype=float)
    if last_low is not None:
        low[-1] = last_low
    return pd.DataFrame(
        {'Open': close, 'High': close + 1.0, 'Low': low, 'Close': close, 'Volume': 20_000_000},
        index=dates,
    )


def test_open_mark_to_market_alerts_and_manual_close(tmp_path):
    db = tmp_path / 'portfolio.db'

    pos_id = open_paper_position(
        ticker='AAPL',
        strategy='SMA 20/50 Crossover',
        shares=10,
        entry_price=200.0,
        stop_price=190.0,
        target_price=220.0,
        notes='Initial paper entry',
        db_path=db,
    )
    assert pos_id == 1

    # Duplicate open position on the same ticker is blocked
    with pytest.raises(ValueError, match='already exists'):
        open_paper_position('AAPL', 'SMA 20/50 Crossover', 5, 201.0, 190.0, 222.0, db_path=db)

    cash_summary = get_paper_cash_summary(10000.0, db_path=db)
    assert cash_summary['open_cost_basis'] == 2000.0
    assert cash_summary['available_cash'] == 8000.0
    assert cash_summary['open_tickers'] == {'AAPL'}

    # Mark to market above 2R target ($225 >= $220) -> Unrealized PnL +$250, TARGET REACHED alert
    port_up = evaluate_paper_portfolio(
        10000.0,
        db_path=db,
        bars_by_ticker={'AAPL': _make_bars(225.0)},
    )
    assert port_up['unrealized_pnl'] == 250.0
    assert port_up['total_equity'] == 10250.0
    assert port_up['alerts'][0]['severity'] == 'INFO'
    assert 'TARGET REACHED' in port_up['open_positions'][0]['alert_status']
    # Never auto-closed!
    assert len(list_paper_positions('OPEN', db_path=db)) == 1

    # Mark to market below stop ($188 <= $190) -> HIGH severity STOP BREACHED alert, still never auto-closed
    port_stop = evaluate_paper_portfolio(
        10000.0,
        db_path=db,
        bars_by_ticker={'AAPL': _make_bars(188.0, last_low=187.5)},
    )
    assert port_stop['unrealized_pnl'] == -120.0
    assert port_stop['alerts'][0]['severity'] == 'HIGH'
    assert 'STOP BREACHED' in port_stop['open_positions'][0]['alert_status']
    assert len(list_paper_positions('OPEN', db_path=db)) == 1

    # Human manually closes at $215 -> Realized PnL +$150, cash becomes $10,150
    closed = close_paper_position(pos_id, exit_price=215.0, close_note='Took profit manually', db_path=db)
    assert closed['realized_pnl'] == 150.0

    after_close = evaluate_paper_portfolio(10000.0, db_path=db)
    assert after_close['available_cash'] == 10150.0
    assert after_close['realized_pnl'] == 150.0
    assert after_close['total_equity'] == 10150.0
    assert len(after_close['open_positions']) == 0
    assert len(after_close['closed_positions']) == 1

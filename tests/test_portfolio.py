import sqlite3
import numpy as np
import pandas as pd
import pytest
from agents.risk_agent import evaluate_risk_gate, get_risk_policy
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


def test_portfolio_heat_and_sector_diversification_gates():
    policy = get_risk_policy('medium')  # 4% max heat ($400 on $10k), max 2 per sector
    idea = {'action': 'SUGGEST_LONG', 'conviction_score': 3, 'entry_reference_price': 100.0, 'summary': 'ok'}
    market = {'volatility_20_pct': 22.0, 'volume_ratio': 1.1}
    research = {'sentiment': 'BULLISH', 'sentiment_score': 0.5, 'fundamentals': {'sector': 'Technology'}}

    # Case 1: $370 of $400 heat already used -> only $30 heat left -> at $5 stop risk/share ($100 * 5%), caps at 6 shares!
    partial_gate = evaluate_risk_gate(
        strategy_idea=idea,
        market=market,
        research=research,
        bankroll=10000.0,
        policy=policy,
        open_portfolio_risk=370.0,
        sector_counts={'Technology': 1},
    )
    assert partial_gate['passed'] is True
    assert partial_gate['shares'] == 6
    assert partial_gate['open_portfolio_risk'] + partial_gate['nominal_risk'] <= 400.0

    # Case 2: $398 of $400 heat already used (< $5 needed for 1 share) -> DROPPED by portfolio heat cap
    maxed_heat_gate = evaluate_risk_gate(
        strategy_idea=idea,
        market=market,
        research=research,
        bankroll=10000.0,
        policy=policy,
        open_portfolio_risk=398.0,
        sector_counts={'Technology': 1},
    )
    assert maxed_heat_gate['passed'] is False
    assert any('Portfolio heat cap reached' in r for r in maxed_heat_gate['drop_reasons'])

    # Case 3: Already holding 2 Technology positions -> 3rd Tech stock DROPPED, Healthcare stock PASSES
    sector_blocked_gate = evaluate_risk_gate(
        strategy_idea=idea,
        market=market,
        research=research,
        bankroll=10000.0,
        policy=policy,
        open_portfolio_risk=150.0,
        sector_counts={'Technology': 2},
    )
    assert sector_blocked_gate['passed'] is False
    assert any('Sector concentration cap reached' in r for r in sector_blocked_gate['drop_reasons'])

    healthcare_gate = evaluate_risk_gate(
        strategy_idea=idea,
        market=market,
        research={'sentiment': 'BULLISH', 'sentiment_score': 0.5, 'fundamentals': {'sector': 'Healthcare'}},
        bankroll=10000.0,
        policy=policy,
        open_portfolio_risk=150.0,
        sector_counts={'Technology': 2},
    )
    assert healthcare_gate['passed'] is True


def test_legacy_paper_positions_schema_auto_migrates_sector_column(tmp_path):
    db = tmp_path / 'legacy.db'
    with sqlite3.connect(db) as conn:
        conn.execute('''CREATE TABLE paper_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            opened_at_utc TEXT NOT NULL,
            closed_at_utc TEXT,
            status TEXT NOT NULL,
            ticker TEXT NOT NULL,
            strategy TEXT NOT NULL,
            shares INTEGER NOT NULL,
            entry_price REAL NOT NULL,
            stop_price REAL NOT NULL,
            target_price REAL NOT NULL,
            exit_price REAL,
            realized_pnl REAL,
            notes TEXT NOT NULL
        )''')
    pos_id = open_paper_position(
        ticker='JPM',
        strategy='SMA 20/50 Crossover',
        shares=5,
        entry_price=190.0,
        stop_price=180.5,
        target_price=209.0,
        sector='Financial Services',
        db_path=db,
    )
    assert pos_id == 1
    summary = get_paper_cash_summary(10000.0, db_path=db)
    assert summary['sector_counts'] == {'Financial Services': 1}


import sqlite3
import numpy as np
import pandas as pd
import pytest
from agents.risk_agent import evaluate_risk_gate, get_risk_policy
from core.exports import (
    build_portfolio_report_markdown,
    build_suggestion_report_markdown,
    dataframe_to_csv_bytes,
)
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


def test_spy_benchmark_alpha_and_markdown_csv_exports(tmp_path):
    db = tmp_path / 'spy_bench.db'
    pos_id = open_paper_position(
        ticker='AAPL',
        strategy='SMA 20/50 Crossover',
        shares=20,
        entry_price=150.0,
        stop_price=142.5,
        target_price=165.0,
        sector='Technology',
        db_path=db,
    )
    # Backdate opened_at_utc to 10 business days into our 80-day bar series so multi-day curve builds
    dates = pd.bdate_range('2026-06-01', periods=80)
    inception_iso = dates[50].isoformat()
    with sqlite3.connect(db) as conn:
        conn.execute('UPDATE paper_positions SET opened_at_utc = ? WHERE id = ?', (inception_iso, pos_id))

    # AAPL rises from $150 to $180 (+$600 on $10,000 = +6.0% portfolio return)
    aapl_close = np.linspace(140.0, 180.0, 80)
    aapl_close[50] = 150.0
    aapl_bars = pd.DataFrame(
        {'Open': aapl_close, 'High': aapl_close + 1, 'Low': aapl_close - 1, 'Close': aapl_close, 'Volume': 20_000_000},
        index=dates,
    )
    # SPY rises from $500 at index 50 to $515 at index 79 (+3.0% SPY return)
    spy_close = np.linspace(480.0, 515.0, 80)
    spy_close[50] = 500.0
    spy_bars = pd.DataFrame(
        {'Open': spy_close, 'High': spy_close + 1, 'Low': spy_close - 1, 'Close': spy_close, 'Volume': 50_000_000},
        index=dates,
    )

    port = evaluate_paper_portfolio(
        starting_bankroll=10000.0,
        db_path=db,
        bars_by_ticker={'AAPL': aapl_bars, 'SPY': spy_bars},
    )
    bench = port['spy_benchmark']
    assert bench['available'] is True
    assert port['total_return_pct'] == 6.0
    assert bench['spy_return_pct'] == 3.0
    assert bench['alpha_vs_spy_pct'] == 3.0
    assert len(bench['curve_df']) == 30
    assert list(bench['curve_df'].columns) == ['Paper Portfolio ($)', 'SPY Benchmark ($)']

    # Verify Markdown & CSV exporters
    port_md = build_portfolio_report_markdown(port, get_risk_policy('medium'))
    assert 'SPY Benchmark' in port_md
    assert '+3.00%' in port_md
    assert 'AAPL' in port_md

    csv_bytes = dataframe_to_csv_bytes(bench['curve_df'], include_index=True)
    assert b'Paper Portfolio ($)' in csv_bytes

    dummy_wf_result = {
        'ticker': 'AAPL',
        'as_of': '2026-09-18',
        'outcome': 'SUGGESTION_READY',
        'market': {'price': 180.0, 'trend': 'UP', 'sma_20': 172.0, 'sma_50': 162.0, 'momentum_20_pct': 5.2, 'volatility_20_pct': 18.5, 'volume_ratio': 1.1},
        'research': {'sentiment': 'BULLISH', 'sentiment_score': 0.6, 'ai_mode': 'deterministic', 'summary': 'Strong demand.'},
        'strategy': {'strategy_name': 'SMA 20/50 Crossover', 'action': 'SUGGEST_LONG', 'conviction_score': 4, 'bullish_strategies_count': 5, 'total_active_strategies': 6},
        'risk': {'sector': 'Technology', 'passed': True, 'drop_reasons': []},
        'report': {'markdown_report': '### Paper Trade Suggestion: AAPL\nAgents only suggest.'},
        'monitor': {'status': 'HEALTHY', 'warnings': []},
    }
    dossier_md = build_suggestion_report_markdown(dummy_wf_result, None, 10000.0, get_risk_policy('medium'))
    assert '# Multi-Agent Suggestion Dossier: AAPL (Technology)' in dossier_md
    assert 'Agents only suggest' in dossier_md



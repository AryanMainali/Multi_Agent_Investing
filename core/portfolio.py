"""Paper portfolio valuation, cash ledger, and advisory exit/stop alert monitor.

Strictly advisory: marks open paper positions to real Yahoo Finance daily bars and
surfaces alerts when a stop-loss, 2R target, or strategy exit signal is reached.
Never auto-closes positions or executes broker trades.
"""
from pathlib import Path
import pandas as pd
import yfinance as yf
from agents.strategy_agent import ACTIVE_STRATEGY_KEYS, _evaluate_strategy_stance
from core.data import validate_bars
from core.indicators import enrich
from core.scanner import extract_symbol_bars
from core.storage import list_paper_positions
from core.strategies import STRATEGIES

NAME_TO_STRATEGY_KEY = {spec.name: key for key, spec in STRATEGIES.items()}


def get_paper_cash_summary(starting_bankroll: float, db_path: Path | None = None) -> dict:
    """Compute realized P&L, open cost basis, available cash, and open ticker set without network calls."""
    if starting_bankroll <= 0:
        raise ValueError('Starting bankroll must be positive')
    all_positions = list_paper_positions(db_path=db_path)
    open_pos = [p for p in all_positions if p['status'] == 'OPEN']
    closed_pos = [p for p in all_positions if p['status'] == 'CLOSED']

    open_cost_basis = round(sum(p['shares'] * p['entry_price'] for p in open_pos), 2)
    realized_pnl = round(sum(float(p['realized_pnl'] or 0.0) for p in closed_pos), 2)
    available_cash = round(float(starting_bankroll) + realized_pnl - open_cost_basis, 2)

    return {
        'starting_bankroll': round(float(starting_bankroll), 2),
        'available_cash': available_cash,
        'open_cost_basis': open_cost_basis,
        'realized_pnl': realized_pnl,
        'open_tickers': {p['ticker'] for p in open_pos},
        'open_positions': open_pos,
        'closed_positions': closed_pos,
    }


def evaluate_paper_portfolio(
    starting_bankroll: float,
    db_path: Path | None = None,
    bars_by_ticker: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Mark open paper positions to real Yahoo Finance closes and generate manual review alerts."""
    summary = get_paper_cash_summary(starting_bankroll=starting_bankroll, db_path=db_path)
    open_pos = summary['open_positions']
    closed_pos = summary['closed_positions']
    known_bars = dict(bars_by_ticker or {})

    missing_syms = [p['ticker'] for p in open_pos if p['ticker'] not in known_bars]
    if missing_syms:
        try:
            raw_batch = yf.download(
                tickers=missing_syms,
                period='6mo',
                interval='1d',
                group_by='ticker',
                auto_adjust=True,
                threads=True,
                progress=False,
                multi_level_index=True,
            )
            for sym in missing_syms:
                sub = extract_symbol_bars(raw_batch, sym, missing_syms)
                if sub is not None:
                    try:
                        known_bars[sym] = validate_bars(sub)
                    except ValueError:
                        pass
        except Exception:
            pass

    enriched_open = []
    alerts = []
    open_market_value = 0.0
    unrealized_pnl_total = 0.0

    for pos in open_pos:
        sym = pos['ticker']
        shares = int(pos['shares'])
        entry = float(pos['entry_price'])
        stop = float(pos['stop_price'])
        target = float(pos['target_price'])
        bars = known_bars.get(sym)

        strategy_exit = False
        if bars is not None and not bars.empty:
            curr_price = round(float(bars['Close'].iloc[-1]), 2)
            last_low = float(bars['Low'].iloc[-1])
            strat_key = NAME_TO_STRATEGY_KEY.get(pos['strategy'], 'sma_crossover')
            if strat_key in ACTIVE_STRATEGY_KEYS and len(bars) >= 60:
                stance_info = _evaluate_strategy_stance(enrich(bars), strat_key)
                strategy_exit = bool(stance_info['exit_signal_today']) or stance_info['stance'].startswith('BEARISH')
        else:
            curr_price = round(entry, 2)
            last_low = curr_price

        mkt_val = round(shares * curr_price, 2)
        unrl_pnl = round((curr_price - entry) * shares, 2)
        unrl_pct = round((curr_price / entry - 1.0) * 100.0, 2)
        open_market_value += mkt_val
        unrealized_pnl_total += unrl_pnl

        if curr_price <= stop or last_low <= stop:
            alert_status = 'STOP BREACHED — Review manual exit'
            alerts.append({
                'severity': 'HIGH',
                'ticker': sym,
                'message': f"{sym} traded at/below protective stop (${stop:.2f}; last close ${curr_price:.2f}). Review manual exit.",
            })
        elif curr_price >= target:
            alert_status = 'TARGET REACHED — Review profit-taking'
            alerts.append({
                'severity': 'INFO',
                'ticker': sym,
                'message': f"{sym} reached its 2R reference target (${target:.2f}; last close ${curr_price:.2f}).",
            })
        elif strategy_exit:
            alert_status = 'STRATEGY BEARISH / EXIT — Review hold'
            alerts.append({
                'severity': 'WARN',
                'ticker': sym,
                'message': f"{sym} ({pos['strategy']}) turned bearish or flashed an exit signal at ${curr_price:.2f}.",
            })
        else:
            alert_status = 'HOLDING — Within bounds'

        enriched_open.append({
            **pos,
            'current_price': curr_price,
            'market_value': mkt_val,
            'unrealized_pnl': unrl_pnl,
            'unrealized_pnl_pct': unrl_pct,
            'alert_status': alert_status,
        })

    open_market_value = round(open_market_value, 2)
    unrealized_pnl_total = round(unrealized_pnl_total, 2)
    total_equity = round(summary['available_cash'] + open_market_value, 2)
    total_return_pct = round((total_equity / summary['starting_bankroll'] - 1.0) * 100.0, 2)

    return {
        'starting_bankroll': summary['starting_bankroll'],
        'available_cash': summary['available_cash'],
        'open_cost_basis': summary['open_cost_basis'],
        'open_market_value': open_market_value,
        'total_equity': total_equity,
        'unrealized_pnl': unrealized_pnl_total,
        'realized_pnl': summary['realized_pnl'],
        'total_return_pct': total_return_pct,
        'open_positions': enriched_open,
        'closed_positions': closed_pos,
        'alerts': alerts,
    }

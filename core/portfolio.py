"""Paper portfolio valuation, cash ledger, portfolio heat, sector exposure, SPY benchmark, and advisory alerts.

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
from core.universe import get_known_sector

NAME_TO_STRATEGY_KEY = {spec.name: key for key, spec in STRATEGIES.items()}
BENCHMARK_SYMBOL = 'SPY'


def _resolve_position_sector(pos: dict) -> str:
    sec = str(pos.get('sector') or '').strip()
    if sec and sec != 'Unknown':
        return sec
    return get_known_sector(pos['ticker'])


def _to_naive_date(iso_ts: str | None) -> pd.Timestamp | None:
    if not iso_ts:
        return None
    ts = pd.to_datetime(iso_ts, utc=True, errors='coerce')
    if pd.isna(ts):
        return None
    return ts.tz_localize(None).normalize()


def get_paper_cash_summary(starting_bankroll: float, db_path: Path | None = None) -> dict:
    """Compute cash, open stop-loss risk (portfolio heat), and sector counts without network calls."""
    if starting_bankroll <= 0:
        raise ValueError('Starting bankroll must be positive')
    all_positions = list_paper_positions(db_path=db_path)
    open_pos = [p for p in all_positions if p['status'] == 'OPEN']
    closed_pos = [p for p in all_positions if p['status'] == 'CLOSED']

    open_cost_basis = round(sum(p['shares'] * p['entry_price'] for p in open_pos), 2)
    realized_pnl = round(sum(float(p['realized_pnl'] or 0.0) for p in closed_pos), 2)
    available_cash = round(float(starting_bankroll) + realized_pnl - open_cost_basis, 2)

    open_portfolio_risk = round(
        sum(max(0.0, (float(p['entry_price']) - float(p['stop_price'])) * int(p['shares'])) for p in open_pos),
        2,
    )
    open_portfolio_heat_pct = round((open_portfolio_risk / float(starting_bankroll)) * 100.0, 2)

    sector_counts: dict[str, int] = {}
    for p in open_pos:
        sec = _resolve_position_sector(p)
        p['sector'] = sec
        if sec != 'Unknown':
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

    return {
        'starting_bankroll': round(float(starting_bankroll), 2),
        'available_cash': available_cash,
        'open_cost_basis': open_cost_basis,
        'open_portfolio_risk': open_portfolio_risk,
        'open_portfolio_heat_pct': open_portfolio_heat_pct,
        'sector_counts': sector_counts,
        'realized_pnl': realized_pnl,
        'open_tickers': {p['ticker'] for p in open_pos},
        'open_positions': open_pos,
        'closed_positions': closed_pos,
    }


def _compute_spy_benchmark(
    starting_bankroll: float,
    total_equity: float,
    total_return_pct: float,
    all_positions: list[dict],
    known_bars: dict[str, pd.DataFrame],
) -> dict:
    """Build daily Paper Portfolio ($) vs. SPY Benchmark ($) equity curve from earliest paper trade."""
    spy_bars = known_bars.get(BENCHMARK_SYMBOL)
    if not all_positions or spy_bars is None or spy_bars.empty:
        return {
            'available': False,
            'inception_date': None,
            'spy_ending_equity': round(float(starting_bankroll), 2),
            'spy_return_pct': 0.0,
            'alpha_vs_spy_pct': 0.0,
            'curve_df': pd.DataFrame(columns=['Paper Portfolio ($)', 'SPY Benchmark ($)']),
        }

    open_dates = [_to_naive_date(p.get('opened_at_utc')) for p in all_positions]
    valid_open_dates = [d for d in open_dates if d is not None]
    if not valid_open_dates:
        return {
            'available': False,
            'inception_date': None,
            'spy_ending_equity': round(float(starting_bankroll), 2),
            'spy_return_pct': 0.0,
            'alpha_vs_spy_pct': 0.0,
            'curve_df': pd.DataFrame(columns=['Paper Portfolio ($)', 'SPY Benchmark ($)']),
        }

    earliest_open = min(valid_open_dates)
    spy_dates = spy_bars.index
    on_or_after = spy_dates[spy_dates >= earliest_open]
    start_bar_date = on_or_after[0] if len(on_or_after) > 0 else spy_dates[-1]

    window_spy = spy_bars.loc[spy_bars.index >= start_bar_date, 'Close']
    spy_start_close = float(window_spy.iloc[0])
    spy_curve = (window_spy / spy_start_close) * float(starting_bankroll)

    # Reconstruct daily paper portfolio equity across the same trading dates
    parsed_positions = []
    for p in all_positions:
        o_dt = _to_naive_date(p.get('opened_at_utc')) or start_bar_date
        c_dt = _to_naive_date(p.get('closed_at_utc')) if p.get('status') == 'CLOSED' else None
        parsed_positions.append({
            'ticker': p['ticker'],
            'shares': int(p['shares']),
            'entry_price': float(p['entry_price']),
            'realized_pnl': float(p['realized_pnl'] or 0.0),
            'status': p['status'],
            'open_date': o_dt,
            'close_date': c_dt,
        })

    port_values = []
    for dt in window_spy.index:
        day_pnl = 0.0
        for item in parsed_positions:
            if dt < item['open_date']:
                continue
            if item['status'] == 'CLOSED' and item['close_date'] is not None and dt >= item['close_date']:
                day_pnl += item['realized_pnl']
            else:
                t_bars = known_bars.get(item['ticker'])
                if t_bars is not None and not t_bars.empty:
                    elig = t_bars.loc[t_bars.index <= dt, 'Close']
                    px = float(elig.iloc[-1]) if len(elig) > 0 else item['entry_price']
                else:
                    px = item['entry_price']
                day_pnl += (px - item['entry_price']) * item['shares']
        port_values.append(round(float(starting_bankroll) + day_pnl, 2))

    # Ensure final bar matches exact current mark-to-market total_equity (e.g. weekend/same-day open/close)
    if port_values:
        port_values[-1] = round(float(total_equity), 2)

    curve_df = pd.DataFrame(
        {
            'Paper Portfolio ($)': port_values,
            'SPY Benchmark ($)': spy_curve.round(2).values,
        },
        index=window_spy.index,
    )
    spy_end_equity = round(float(spy_curve.iloc[-1]), 2)
    spy_ret_pct = round((spy_end_equity / float(starting_bankroll) - 1.0) * 100.0, 2)
    alpha_pct = round(float(total_return_pct) - spy_ret_pct, 2)

    return {
        'available': True,
        'inception_date': str(start_bar_date.date()),
        'spy_ending_equity': spy_end_equity,
        'spy_return_pct': spy_ret_pct,
        'alpha_vs_spy_pct': alpha_pct,
        'curve_df': curve_df,
    }


def evaluate_paper_portfolio(
    starting_bankroll: float,
    db_path: Path | None = None,
    bars_by_ticker: dict[str, pd.DataFrame] | None = None,
    include_spy_benchmark: bool = True,
) -> dict:
    """Mark open paper positions to real Yahoo Finance closes, compare vs SPY, and generate alerts."""
    summary = get_paper_cash_summary(starting_bankroll=starting_bankroll, db_path=db_path)
    open_pos = summary['open_positions']
    closed_pos = summary['closed_positions']
    all_positions = open_pos + closed_pos
    known_bars = dict(bars_by_ticker or {})

    needed_syms = {p['ticker'] for p in all_positions}
    if include_spy_benchmark and all_positions:
        needed_syms.add(BENCHMARK_SYMBOL)

    missing_syms = sorted(s for s in needed_syms if s not in known_bars)
    if missing_syms:
        try:
            raw_batch = yf.download(
                tickers=missing_syms,
                period='1y',
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
    sector_agg: dict[str, dict] = {}

    for pos in open_pos:
        sym = pos['ticker']
        sec = _resolve_position_sector(pos)
        shares = int(pos['shares'])
        entry = float(pos['entry_price'])
        stop = float(pos['stop_price'])
        target = float(pos['target_price'])
        pos_risk = round(max(0.0, (entry - stop) * shares), 2)
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

        bucket = sector_agg.setdefault(
            sec,
            {'Sector': sec, 'Open Positions': 0, 'Market Value ($)': 0.0, 'Open Stop Risk ($)': 0.0},
        )
        bucket['Open Positions'] += 1
        bucket['Market Value ($)'] = round(bucket['Market Value ($)'] + mkt_val, 2)
        bucket['Open Stop Risk ($)'] = round(bucket['Open Stop Risk ($)'] + pos_risk, 2)

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
            'sector': sec,
            'open_stop_risk': pos_risk,
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
    sector_breakdown = sorted(sector_agg.values(), key=lambda x: x['Market Value ($)'], reverse=True)

    spy_benchmark = (
        _compute_spy_benchmark(
            starting_bankroll=summary['starting_bankroll'],
            total_equity=total_equity,
            total_return_pct=total_return_pct,
            all_positions=all_positions,
            known_bars=known_bars,
        )
        if include_spy_benchmark
        else {
            'available': False,
            'inception_date': None,
            'spy_ending_equity': summary['starting_bankroll'],
            'spy_return_pct': 0.0,
            'alpha_vs_spy_pct': 0.0,
            'curve_df': pd.DataFrame(columns=['Paper Portfolio ($)', 'SPY Benchmark ($)']),
        }
    )

    return {
        'starting_bankroll': summary['starting_bankroll'],
        'available_cash': summary['available_cash'],
        'open_cost_basis': summary['open_cost_basis'],
        'open_portfolio_risk': summary['open_portfolio_risk'],
        'open_portfolio_heat_pct': summary['open_portfolio_heat_pct'],
        'sector_counts': summary['sector_counts'],
        'sector_breakdown': sector_breakdown,
        'open_market_value': open_market_value,
        'total_equity': total_equity,
        'unrealized_pnl': unrealized_pnl_total,
        'realized_pnl': summary['realized_pnl'],
        'total_return_pct': total_return_pct,
        'open_positions': enriched_open,
        'closed_positions': closed_pos,
        'spy_benchmark': spy_benchmark,
        'alerts': alerts,
    }

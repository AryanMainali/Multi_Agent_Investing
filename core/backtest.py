"""Educational long-only daily-bar multi-strategy backtester.

Signals use the current CLOSE and execute no earlier than the NEXT day's OPEN.
An intraday stop exits at its trigger price or opening gap price, whichever is lower.
Fees and slippage are estimated; this is not a production-grade fill simulator.
"""
from dataclasses import dataclass
from math import sqrt
import pandas as pd
from agents.risk_agent import RiskPolicy, size_position
from core.strategies import STRATEGIES, generate_signals


@dataclass
class BacktestResult:
    metrics: dict
    equity: pd.DataFrame
    trades: pd.DataFrame


def run_backtest(
    prices: pd.DataFrame,
    starting_cash: float = 10000.0,
    fast: int = 20,
    slow: int = 50,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    policy: RiskPolicy | None = None,
    strategy: str = 'sma_crossover',
) -> BacktestResult:
    """Run a single-symbol, long-only strategy simulation with whole shares."""
    if starting_cash <= 0 or fee_bps < 0 or slippage_bps < 0:
        raise ValueError('Starting cash must be positive and trading costs nonnegative')
    if len(prices) <= slow + 2:
        raise ValueError('Need more history than the slow moving-average window')

    df, spec = generate_signals(prices, strategy=strategy, fast=fast, slow=slow)
    if spec.uses_risk_policy:
        active_policy = policy or RiskPolicy()
        use_stop = True
    else:
        active_policy = RiskPolicy(max_allocation=1.0, max_risk_fraction=1.0, stop_loss_fraction=0.99)
        use_stop = False

    cash = float(starting_cash)
    shares = 0
    entry_cash_spent = 0.0
    stop = None
    pending = None
    rows = []
    trades = []
    roundtrip_pnls = []
    fee = fee_bps / 10000
    slip = slippage_bps / 10000

    for timestamp, bar in df.iterrows():
        op = float(bar['Open'])
        low = float(bar['Low'])
        close = float(bar['Close'])
        # Act on yesterday's close signal at today's open.
        if pending == 'SELL' and shares > 0:
            fill = op * (1 - slip)
            received = shares * fill * (1 - fee)
            pnl = received - entry_cash_spent
            roundtrip_pnls.append(pnl)
            trades.append({
                'Date': timestamp,
                'Action': 'SELL',
                'Shares': shares,
                'Fill': round(fill, 4),
                'Reason': spec.exit_reason,
                'PnL': round(pnl, 2),
            })
            cash += received
            shares = 0
            stop = None
            entry_cash_spent = 0.0
        elif pending == 'BUY' and shares == 0:
            fill = op * (1 + slip)
            sized = size_position(fill, cash, cash, active_policy, cost_fraction=fee)
            qty = sized['shares']
            if qty:
                paid = qty * fill * (1 + fee)
                cash -= paid
                shares = qty
                entry_cash_spent = paid
                stop = fill * (1 - active_policy.stop_loss_fraction) if use_stop else None
                trades.append({
                    'Date': timestamp,
                    'Action': 'BUY',
                    'Shares': qty,
                    'Fill': round(fill, 4),
                    'Reason': spec.entry_reason,
                    'PnL': None,
                })
        pending = None

        # Conservative daily OHLC stop approximation; gap-through fills at open.
        if shares > 0 and stop is not None and low <= stop:
            fill = min(op, stop) * (1 - slip)
            received = shares * fill * (1 - fee)
            pnl = received - entry_cash_spent
            roundtrip_pnls.append(pnl)
            trades.append({
                'Date': timestamp,
                'Action': 'STOP',
                'Shares': shares,
                'Fill': round(fill, 4),
                'Reason': f'{active_policy.stop_loss_fraction * 100:g}% protective stop / gap',
                'PnL': round(pnl, 2),
            })
            cash += received
            shares = 0
            stop = None
            entry_cash_spent = 0.0

        rows.append({
            'Date': timestamp,
            'Equity': cash + shares * close,
            'Cash': cash,
            'Shares': shares,
            'Close': close,
        })

        # Only information available after today's close can generate tomorrow's order.
        if bool(bar['ENTRY_SIGNAL']) and shares == 0:
            pending = 'BUY'
        elif bool(bar['EXIT_SIGNAL']) and shares > 0:
            pending = 'SELL'

    equity = pd.DataFrame(rows).set_index('Date')
    equity['DailyReturn'] = equity['Equity'].pct_change().fillna(0.0)
    high_water = equity['Equity'].cummax()
    drawdown_pct = (equity['Equity'] / high_water - 1) * 100
    volatility = equity['DailyReturn'].std(ddof=0)
    sharpe = (sqrt(252) * equity['DailyReturn'].mean() / volatility) if volatility > 0 else 0.0
    closed_count = len(roundtrip_pnls)
    metrics = {
        'strategy_key': spec.key,
        'strategy_name': spec.name,
        'strategy_category': spec.category,
        'starting_cash': round(starting_cash, 2),
        'ending_equity': round(float(equity['Equity'].iloc[-1]), 2),
        'strategy_return_pct': round((float(equity['Equity'].iloc[-1]) / starting_cash - 1) * 100, 2),
        'price_buy_hold_pct': round((float(df['Close'].iloc[-1]) / float(df['Open'].iloc[0]) - 1) * 100, 2),
        'max_drawdown_pct': round(float(-drawdown_pct.min()), 2),
        'sharpe_zero_rf': round(float(sharpe), 3),
        'closed_trades': closed_count,
        'win_rate_pct': round(100 * sum(p > 0 for p in roundtrip_pnls) / closed_count, 1) if closed_count else None,
        'open_shares_at_end': shares,
        'fee_bps_per_side': fee_bps,
        'slippage_bps_per_side': slippage_bps,
    }
    return BacktestResult(
        metrics=metrics,
        equity=equity,
        trades=pd.DataFrame(trades, columns=['Date', 'Action', 'Shares', 'Fill', 'Reason', 'PnL']),
    )


def compare_strategies(
    prices: pd.DataFrame,
    starting_cash: float = 10000.0,
    policy: RiskPolicy | None = None,
    strategy_keys: list[str] | None = None,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, BacktestResult]]:
    """Backtest multiple strategies on the same historical bars and return summary, equity curves, and full results."""
    keys = strategy_keys if strategy_keys is not None else list(STRATEGIES.keys())
    if not keys:
        raise ValueError('At least one strategy must be provided')

    summary_rows = []
    equity_curves = {}
    results_by_key: dict[str, BacktestResult] = {}

    for key in keys:
        spec = STRATEGIES[key]
        res = run_backtest(
            prices,
            starting_cash=starting_cash,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            policy=policy,
            strategy=key,
        )
        results_by_key[key] = res
        equity_curves[spec.name] = res.equity['Equity']
        m = res.metrics
        summary_rows.append({
            'Strategy': spec.name,
            'Category': spec.category,
            'Ending Equity ($)': m['ending_equity'],
            'Return (%)': m['strategy_return_pct'],
            'Sharpe': m['sharpe_zero_rf'],
            'Max Drawdown (%)': m['max_drawdown_pct'],
            'Closed Trades': m['closed_trades'],
            'Win Rate (%)': m['win_rate_pct'],
            'Open Shares': m['open_shares_at_end'],
            'Rule Summary': spec.description,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by='Return (%)', ascending=False).reset_index(drop=True)
    equity_df = pd.DataFrame(equity_curves)
    return summary_df, equity_df, results_by_key

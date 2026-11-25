"""Run real-data CLI multi-strategy analysis and backtesting; does not place any trading orders."""
import argparse
import sys
from agents.risk_agent import get_risk_policy
from core.backtest import compare_strategies
from core.strategies import STRATEGIES
from core.workflow import analyze


def prompt_startup_settings(bankroll: float | None, risk: str | None) -> tuple[float, str]:
    interactive = sys.stdin.isatty()
    if bankroll is None:
        if interactive:
            raw_cash = input('Enter starting paper trading amount ($) [10000]: ').strip()
            bankroll = float(raw_cash) if raw_cash else 10000.0
        else:
            bankroll = 10000.0
    if risk is None:
        if interactive:
            raw_risk = input('How much risk are you willing to take? (low / medium / high) [medium]: ').strip().lower()
            risk = raw_risk if raw_risk else 'medium'
        else:
            risk = 'medium'
    return bankroll, risk


def main():
    strategy_choices = ['all'] + list(STRATEGIES.keys())
    parser = argparse.ArgumentParser(description='Multi-agent stock paper-trading research (real Yahoo Finance data only)')
    parser.add_argument('--ticker', default='AAPL', help='Stock ticker symbol on Yahoo Finance (default: AAPL)')
    parser.add_argument('--bankroll', type=float, default=None, help='Starting paper trading capital ($)')
    parser.add_argument('--risk', choices=['low', 'medium', 'high'], default=None, help='Risk tolerance level')
    parser.add_argument('--period', default='2y', help='Yahoo Finance history window (e.g. 6mo, 1y, 2y, 5y)')
    parser.add_argument('--strategy', choices=strategy_choices, default='all', help='Strategy key to backtest, or all')
    args = parser.parse_args()

    bankroll, risk_level = prompt_startup_settings(args.bankroll, args.risk)
    policy = get_risk_policy(risk_level)
    ticker = args.ticker.strip().upper()
    keys = list(STRATEGIES.keys()) if args.strategy == 'all' else [args.strategy]

    prices, result = analyze(ticker=ticker, bankroll=bankroll, policy=policy, period=args.period, source='yahoo')
    summary_df, _, results_by_key = compare_strategies(
        prices,
        starting_cash=bankroll,
        policy=policy,
        strategy_keys=keys,
    )

    print(f'\n=== PAPER PORTFOLIO ({risk_level.upper()} RISK, ${bankroll:,.2f}) ===')
    print(
        f'max_allocation: {policy.max_allocation * 100:g}% | '
        f'max_risk_per_trade: {policy.max_risk_fraction * 100:g}% | '
        f'stop_loss: {policy.stop_loss_fraction * 100:g}%'
    )
    print('\n=== FOUR AGENTS (REAL YAHOO FINANCE DATA) ===')
    for key in ('ticker', 'source', 'as_of', 'market', 'analyst', 'risk'):
        print(f'{key}: {result[key]}')

    print('\n=== STRATEGY BACKTEST COMPARISON (historical simulation, NO trades placed) ===')
    cols = ['Strategy', 'Category', 'Ending Equity ($)', 'Return (%)', 'Sharpe', 'Max Drawdown (%)', 'Closed Trades', 'Win Rate (%)']
    print(summary_df[cols].to_string(index=False))

    if len(keys) == 1:
        bt = results_by_key[keys[0]]
        print(f'\nLast trades ({STRATEGIES[keys[0]].name}):')
        print(bt.trades.tail(8).to_string(index=False) if not bt.trades.empty else '(none)')

    print('\nUse: streamlit run app.py  for the interactive multi-strategy dashboard.')


if __name__ == '__main__':
    main()

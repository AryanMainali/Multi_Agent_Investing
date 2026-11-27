"""Run real-data CLI S&P market scanner (or manual ticker lookup), 6-agent workflow, and backtests.

Core Design Rule: Agents only suggest. Humans decide and trade manually.
Does NOT connect to any broker or submit any trading orders.
"""
import argparse
import sys
from agents.risk_agent import get_risk_policy
from core.backtest import compare_strategies
from core.portfolio import evaluate_paper_portfolio
from core.scanner import scan_market_and_suggest
from core.strategies import STRATEGIES
from core.workflow import run_suggestion_workflow


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
    active_keys = [k for k, spec in STRATEGIES.items() if spec.uses_risk_policy]
    strategy_choices = ['all'] + list(STRATEGIES.keys())
    parser = argparse.ArgumentParser(description='S&P market scanner & 6-agent stock suggestion workflow (manual human decisions only)')
    parser.add_argument('--scan', choices=['sp100', 'sp500'], default='sp100', help='Market scan universe when --ticker is omitted (default: sp100)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top candidates to run through the 6-agent pipeline (default: 5)')
    parser.add_argument('--ticker', default=None, help='Optional manual stock ticker override (skips market scan if provided)')
    parser.add_argument('--bankroll', type=float, default=None, help='Starting paper trading capital ($)')
    parser.add_argument('--risk', choices=['low', 'medium', 'high'], default=None, help='Risk tolerance level')
    parser.add_argument('--period', default='2y', help='Yahoo Finance history window (e.g. 6mo, 1y, 2y, 5y)')
    parser.add_argument('--strategy', choices=strategy_choices, default='all', help='Strategy key for backtest comparison, or all')
    args = parser.parse_args()

    bankroll, risk_level = prompt_startup_settings(args.bankroll, args.risk)
    policy = get_risk_policy(risk_level)
    primary_strategy = args.strategy if args.strategy in active_keys else 'sma_crossover'
    keys = list(STRATEGIES.keys()) if args.strategy == 'all' else [args.strategy]

    portfolio = evaluate_paper_portfolio(starting_bankroll=bankroll)
    print(f'\n=== PAPER PORTFOLIO ({risk_level.upper()} RISK, Equity ${portfolio["total_equity"]:,.2f}, Cash ${portfolio["available_cash"]:,.2f}) ===')
    print(
        f'max_allocation: {policy.max_allocation * 100:g}% | '
        f'max_risk_per_trade: {policy.max_risk_fraction * 100:g}% | '
        f'stop_loss: {policy.stop_loss_fraction * 100:g}% | '
        f'open_positions: {len(portfolio["open_positions"])}'
    )
    for alert in portfolio['alerts']:
        print(f"[PORTFOLIO ALERT - {alert['severity']}] {alert['message']}")

    if args.ticker:
        ticker = args.ticker.strip().upper()
        prices, result = run_suggestion_workflow(
            ticker=ticker,
            bankroll=bankroll,
            policy=policy,
            strategy=primary_strategy,
            period=args.period,
            source='yahoo',
        )
    else:
        print(f"\nScanning {args.scan.upper()} universe for top {args.top_k} candidates...")
        scan_res = scan_market_and_suggest(
            scope=args.scan,
            bankroll=bankroll,
            policy=policy,
            primary_strategy=primary_strategy,
            top_k=args.top_k,
            history_period=args.period,
        )
        print(
            f"Scanned {scan_res['valid_data_count']}/{scan_res['universe_size']} stocks | "
            f"Affordable: {scan_res['affordable_count']} | "
            f"Passed Risk Gate: {scan_res['passed_count']}/{len(scan_res['candidates'])}"
        )
        if scan_res['shortlist_summary_df'].empty:
            print('No eligible candidates matched your capital and risk constraints.')
            return
        print('\n=== TOP SHORTLISTED S&P CANDIDATES ===')
        print(scan_res['shortlist_summary_df'].to_string(index=False))
        top_pick = scan_res['candidates'][0]
        prices, result = top_pick['data'], top_pick['workflow']

    summary_df, _, _ = compare_strategies(
        prices,
        starting_cash=bankroll,
        policy=policy,
        strategy_keys=keys,
    )

    print(f"\n=== TOP CANDIDATE 6-AGENT DETAIL ({result['ticker']}, as of {result['as_of']}) ===")
    print(f"[1. Market Data]   trend={result['market']['trend']} | close=${result['market']['price']:.2f} | vol_20d={result['market']['volatility_20_pct']:.1f}%")
    print(f"[2. Research (AI)] sentiment={result['research']['sentiment']} ({result['research']['sentiment_score']:+.2f}) | mode={result['research']['ai_mode']}")
    print(f"[3. Strategy]      {result['strategy']['strategy_name']} -> {result['strategy']['action']} (conviction {result['strategy']['conviction_score']:+d}/5)")
    print(f"[4. Risk Gate]     passed={result['risk']['passed']} ({result['risk']['decision']})")
    print(f"[6. Monitor Agent] status={result['monitor']['status']} | warnings={len(result['monitor']['warnings'])}")

    if result['outcome'] == 'SUGGESTION_READY' and result['report'] is not None:
        print('\n=== [5. REPORT (AI)] PLAIN-ENGLISH SUGGESTION ===')
        print(result['report']['markdown_report'])
    else:
        print('\n=== [DROP & LOG REASON] ===')
        for reason in result['risk']['drop_reasons']:
            print(f'- {reason}')

    print(f"\n=== MULTI-STRATEGY HISTORICAL BACKTEST COMPARISON ({result['ticker']}, NO trades placed) ===")
    cols = ['Strategy', 'Category', 'Ending Equity ($)', 'Return (%)', 'Sharpe', 'Max Drawdown (%)', 'Closed Trades', 'Win Rate (%)']
    print(summary_df[cols].to_string(index=False))
    print('\nUse: streamlit run app.py  for the interactive dashboard.')


if __name__ == '__main__':
    main()

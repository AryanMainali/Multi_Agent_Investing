"""Clean Markdown and CSV export builders for suggestion dossiers, backtests, and paper portfolios."""
import pandas as pd
from agents.risk_agent import RiskPolicy


def dataframe_to_csv_bytes(df: pd.DataFrame, include_index: bool = False) -> bytes:
    """Serialize a DataFrame to UTF-8 CSV bytes for Streamlit download buttons."""
    return df.to_csv(index=include_index).encode('utf-8')


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return '*(No rows)*'
    cols = [str(c) for c in df.columns]
    header = '| ' + ' | '.join(cols) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
    body_lines = []
    for _, row in df.iterrows():
        vals = [str(row[c]).replace('|', '/').replace('\n', ' ') for c in df.columns]
        body_lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join([header, sep] + body_lines)


def build_suggestion_report_markdown(
    result: dict,
    backtest_summary_df: pd.DataFrame | None,
    bankroll: float,
    policy: RiskPolicy,
) -> str:
    """Build a standalone Markdown dossier for a single 6-agent candidate run."""
    ticker = result['ticker']
    sector = result['risk'].get('sector', 'Unknown')
    as_of = result['as_of']
    mkt = result['market']
    res = result['research']
    strat = result['strategy']
    risk = result['risk']
    mon = result['monitor']

    lines = [
        f"# Multi-Agent Suggestion Dossier: {ticker} ({sector})",
        '',
        f"- **As of Daily Close:** {as_of} (${mkt['price']:.2f})",
        f"- **Outcome:** `{result['outcome']}` | **Monitor Status:** `{mon['status']}`",
        f"- **Paper Capital:** ${bankroll:,.2f} | **Max Allocation:** {policy.max_allocation * 100:g}% | "
        f"**Max Risk/Trade:** {policy.max_risk_fraction * 100:g}% | **Stop-Loss:** {policy.stop_loss_fraction * 100:g}% | "
        f"**Max Portfolio Heat:** {policy.max_portfolio_heat * 100:g}% | **Max/Sector:** {policy.max_positions_per_sector}",
        '',
        '---',
        '',
    ]

    if result['outcome'] == 'SUGGESTION_READY' and result.get('report'):
        lines.append(result['report']['markdown_report'])
    else:
        lines.append(f"## Risk Gate Rejected {ticker}")
        lines.append('')
        for r in risk.get('drop_reasons', []):
            lines.append(f"- {r}")

    lines.extend([
        '',
        '---',
        '',
        '## Agent Stage Breakdown',
        '',
        f"### 1. Market Data\n- Trend: `{mkt['trend']}` | 20d SMA: `${mkt['sma_20']:.2f}` | "
        f"50d SMA: `${mkt['sma_50']:.2f}` | 20d Momentum: `{mkt['momentum_20_pct']:+.2f}%` | "
        f"20d Volatility: `{mkt['volatility_20_pct']:.2f}%` | Volume Ratio: `{mkt['volume_ratio']}x`",
        '',
        f"### 2. Research (AI)\n- Sentiment: `{res['sentiment']}` (`{res['sentiment_score']:+.2f}`) | Mode: `{res['ai_mode']}`",
        f"- Summary: {res['summary']}",
        '',
        f"### 3. Strategy\n- Primary Strategy: **{strat['strategy_name']}** | Action: `{strat['action']}` | "
        f"Conviction: `{strat['conviction_score']:+d}/5` | Bullish Consensus: `{strat['bullish_strategies_count']}/{strat['total_active_strategies']}`",
        '',
        f"### 6. Monitor Agent\n- Status: `{mon['status']}` | Warnings: {len(mon['warnings'])}",
    ])
    for w in mon.get('warnings', []):
        lines.append(f"  - Warning: {w}")

    if backtest_summary_df is not None and not backtest_summary_df.empty:
        cols = [
            c for c in [
                'Strategy', 'Category', 'Ending Equity ($)', 'Return (%)',
                'Sharpe', 'Max Drawdown (%)', 'Closed Trades', 'Win Rate (%)',
            ]
            if c in backtest_summary_df.columns
        ]
        lines.extend([
            '',
            '---',
            '',
            f'## Historical Multi-Strategy Backtest Comparison ({ticker})',
            '',
            _df_to_markdown_table(backtest_summary_df[cols]),
        ])

    lines.extend([
        '',
        '---',
        '*Core Rule: Agents only suggest. Humans decide and trade manually. Educational paper-research report — not investment advice.*',
        '',
    ])
    return '\n'.join(lines)


def build_portfolio_report_markdown(portfolio: dict, policy: RiskPolicy) -> str:
    """Build a standalone Markdown report of the live paper portfolio and SPY benchmark comparison."""
    bench = portfolio.get('spy_benchmark', {})
    lines = [
        '# Live Paper Portfolio & SPY Benchmark Report',
        '',
        f"- **Starting Paper Capital:** ${portfolio['starting_bankroll']:,.2f}",
        f"- **Total Paper Equity:** ${portfolio['total_equity']:,.2f} ({portfolio['total_return_pct']:+.2f}%)",
        f"- **Available Paper Cash:** ${portfolio['available_cash']:,.2f}",
        f"- **Open Portfolio Heat:** ${portfolio['open_portfolio_risk']:,.2f} "
        f"({portfolio['open_portfolio_heat_pct']:.2f}% / {policy.max_portfolio_heat * 100:g}% cap)",
        f"- **Unrealized P&L:** ${portfolio['unrealized_pnl']:+,.2f} | **Realized P&L:** ${portfolio['realized_pnl']:+,.2f}",
    ]

    if bench.get('available'):
        lines.append(
            f"- **SPY Benchmark (since {bench['inception_date']}):** {bench['spy_return_pct']:+.2f}% | "
            f"**Excess Return (Alpha vs. SPY):** {bench['alpha_vs_spy_pct']:+.2f}%"
        )

    if portfolio.get('alerts'):
        lines.extend(['', '## Active Advisory Alerts (Manual Review Suggested)', ''])
        for a in portfolio['alerts']:
            lines.append(f"- **[{a['severity']}]** {a['message']}")

    if portfolio.get('sector_breakdown'):
        lines.extend([
            '',
            f'## Sector Concentration (Max {policy.max_positions_per_sector} / Sector)',
            '',
            _df_to_markdown_table(pd.DataFrame(portfolio['sector_breakdown'])),
        ])

    open_pos = portfolio.get('open_positions', [])
    lines.extend(['', '## Open Paper Positions', ''])
    if open_pos:
        open_df = pd.DataFrame([
            {
                'Ticker': p['ticker'],
                'Sector': p['sector'],
                'Shares': p['shares'],
                'Entry ($)': f"{p['entry_price']:.2f}",
                'Current ($)': f"{p['current_price']:.2f}",
                'Unrealized PnL ($)': f"{p['unrealized_pnl']:+.2f}",
                'Stop ($)': f"{p['stop_price']:.2f}",
                'Status': p['alert_status'],
            }
            for p in open_pos
        ])
        lines.append(_df_to_markdown_table(open_df))
    else:
        lines.append('*(No open paper positions)*')

    closed_pos = portfolio.get('closed_positions', [])
    lines.extend(['', '## Closed Paper Positions', ''])
    if closed_pos:
        closed_df = pd.DataFrame([
            {
                'Ticker': p['ticker'],
                'Sector': p['sector'],
                'Shares': p['shares'],
                'Entry ($)': f"{p['entry_price']:.2f}",
                'Exit ($)': f"{p['exit_price']:.2f}",
                'Realized PnL ($)': f"{p['realized_pnl']:+.2f}",
                'Closed UTC': p['closed_at_utc'],
            }
            for p in closed_pos
        ])
        lines.append(_df_to_markdown_table(closed_df))
    else:
        lines.append('*(No closed paper positions)*')

    lines.extend([
        '',
        '---',
        '*Agents only suggest. Humans decide and trade manually. No broker orders placed.*',
        '',
    ])
    return '\n'.join(lines)

"""Streamlit app: onboarding, autonomous S&P 100/500 market scanner, 6-agent suggestion workflow,
portfolio heat & sector diversification caps, multi-strategy backtesting, and live paper portfolio tracker.

Core Design Rule: Agents only suggest. Humans decide and trade manually.
"""
import pandas as pd
import streamlit as st
from agents.risk_agent import RiskPolicy, get_risk_policy
from core.backtest import compare_strategies
from core.exports import (
    build_portfolio_report_markdown,
    build_suggestion_report_markdown,
    dataframe_to_csv_bytes,
)
from core.llm import LLMConfig
from core.portfolio import evaluate_paper_portfolio, get_paper_cash_summary
from core.scanner import scan_market_and_suggest
from core.storage import (
    clear_paper_positions,
    close_paper_position,
    get_idea_scorecard,
    list_decisions,
    list_dropped_ideas,
    list_heartbeats,
    list_workflow_audits,
    open_paper_position,
    save_decision,
)
from core.strategies import STRATEGIES
from core.workflow import run_suggestion_workflow

st.set_page_config(page_title='Stock Multi-Agent Lab', page_icon='📈', layout='wide')

RISK_OPTIONS = [
    'Low (Conservative)',
    'Medium (Moderate)',
    'High (Aggressive)',
    'Custom',
]

RISK_DESCRIPTIONS = {
    'Low (Conservative)': '10% max alloc · 0.5% risk/trade · 4% stop · 2% max portfolio heat · max 1 stock/sector',
    'Medium (Moderate)': '20% max alloc · 1.0% risk/trade · 5% stop · 4% max portfolio heat · max 2 stocks/sector',
    'High (Aggressive)': '35% max alloc · 2.5% risk/trade · 7% stop · 8% max portfolio heat · max 3 stocks/sector',
    'Custom': 'Define your own position allocation, per-trade risk, stop-loss, portfolio heat, and sector cap',
}

SCAN_SCOPE_LABELS = [
    'S&P 100 (Fast ~5s)',
    'Full S&P 500 (~20–30s)',
]

NAME_TO_STRATEGY_KEY = {spec.name: key for key, spec in STRATEGIES.items()}
ALL_STRATEGY_NAMES = list(NAME_TO_STRATEGY_KEY.keys())
ACTIVE_STRATEGY_NAMES = [spec.name for spec in STRATEGIES.values() if spec.uses_risk_policy]
TOP_K_SHORTLIST = 5


def resolve_onboarding_policy(
    choice: str,
    alloc_pct: float,
    risk_pct: float,
    stop_pct: float,
    heat_pct: float,
    max_per_sector: int,
) -> RiskPolicy:
    if choice.startswith('Low'):
        return get_risk_policy('low')
    if choice.startswith('Medium'):
        return get_risk_policy('medium')
    if choice.startswith('High'):
        return get_risk_policy('high')
    return RiskPolicy(
        max_allocation=alloc_pct / 100.0,
        max_risk_fraction=risk_pct / 100.0,
        stop_loss_fraction=stop_pct / 100.0,
        max_portfolio_heat=heat_pct / 100.0,
        max_positions_per_sector=int(max_per_sector),
    )


def render_candidate_deep_dive(data: pd.DataFrame, result: dict, bankroll: float, policy: RiskPolicy) -> None:
    """Render the 6-agent suggestion report, multi-strategy backtests, and manual paper decision entry."""
    source = 'yahoo'
    st.subheader(f"1. Six-agent suggestion workflow — {result['ticker']} ({result['risk']['sector']})")
    st.caption(
        f"As of {result['as_of']} | [1. Market Data] + [2. Research AI] → [3. Strategy] → [4. Risk Gate] → "
        f"{'[5. Report AI]' if result['risk']['passed'] else '[Drop & Log Reason]'} ← [6. Monitor Agent]"
    )

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric('Last daily close', f"${result['market']['price']:.2f}")
    m_col2.metric('20-day / 50-day SMA', f"${result['market']['sma_20']:.2f} / ${result['market']['sma_50']:.2f}")
    m_col3.metric('Strategy conviction', f"{result['strategy']['conviction_score']:+d} / 5", result['strategy']['action'])
    m_col4.metric('Monitor Agent status', result['monitor']['status'], f"Risk Gate: {'YES' if result['risk']['passed'] else 'NO'}")

    chart = data[['Close', 'SMA_FAST', 'SMA_SLOW']].rename(columns={'SMA_FAST': 'SMA 20', 'SMA_SLOW': 'SMA 50'})
    st.line_chart(chart)

    if result['outcome'] == 'SUGGESTION_READY' and result['report'] is not None:
        st.success(f"Risk Gate PASSED — {result['report']['headline']}")
        with st.container(border=True):
            st.markdown(result['report']['markdown_report'])
            st.caption(
                f"Report generator mode: {result['report']['ai_mode']} | "
                f"Sector: {result['risk']['sector']} ({result['risk']['sector_open_count']}/{policy.max_positions_per_sector} open) | "
                f"Projected portfolio heat: {result['risk']['projected_portfolio_heat_pct']:.2f}% / {policy.max_portfolio_heat * 100:g}% | "
                f"Monitor: {result['monitor']['status']}"
            )
    else:
        st.warning(f"Risk Gate DROPPED {result['ticker']} — Idea rejected and logged to SQLite (no Report issued).")
        with st.container(border=True):
            st.markdown('**Why the Risk Gate dropped this idea:**')
            for reason_item in result['risk']['drop_reasons']:
                st.markdown(f"- {reason_item}")
            st.caption('Logged automatically to the `dropped_ideas` table in `outputs/decisions.db`.')

    tab_mkt, tab_res, tab_strat, tab_risk, tab_mon = st.tabs([
        '1. Market Data',
        '2. Research (AI)',
        '3. Strategy',
        '4. Risk Gate',
        '6. Monitor Agent',
    ])

    with tab_mkt:
        st.write(f"Fetched **{len(data)}** real daily OHLCV bars from Yahoo Finance through **{result['as_of']}**.")
        st.json(result['market'])

    with tab_res:
        res = result['research']
        st.markdown(
            f"**Sentiment:** `{res['sentiment']}` (score `{res['sentiment_score']:+.2f}`) · "
            f"**Sector:** `{result['risk']['sector']}` · **Mode:** `{res['ai_mode']}`"
        )
        st.write(res['summary'])
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown('**Catalysts**')
            for c_item in res['catalysts']:
                st.markdown(f"- {c_item}")
        with rc2:
            st.markdown('**Key Risks**')
            for r_item in res['risks']:
                st.markdown(f"- {r_item}")
        if res.get('event_flags'):
            st.markdown('**Structured News Event Flags (Severity & Category)**')
            st.dataframe(pd.DataFrame(res['event_flags']), width='stretch', hide_index=True)
        if res['headlines']:
            st.markdown('**Recent Market Headlines**')
            st.dataframe(pd.DataFrame(res['headlines']), width='stretch', hide_index=True)
        if res['fundamentals']:
            with st.expander('Company fundamentals snapshot'):
                st.json(res['fundamentals'])

    with tab_strat:
        strat = result['strategy']
        st.markdown(
            f"**Selected Strategy:** {strat['strategy_name']} "
            f"(`{strat['action']}`, conviction `{strat['conviction_score']:+d}/5`)"
        )
        ml_f = strat.get('ml_forecast')
        if isinstance(ml_f, dict) and ml_f.get('available'):
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(
                f"Walk-Forward {ml_f['horizon_days']}d ML P(Up)",
                f"{ml_f['p_up']:.1%}",
                ml_f['direction'].upper(),
            )
            mc2.metric(
                'Held-Out Backtest Accuracy',
                f"{ml_f['backtest_accuracy']:.1%}",
                f"Last {ml_f['test_window']} settled bars (with embargo)",
                delta_color='off',
            )
            mc3.metric('Training Sample Rows', str(ml_f['train_samples']))
        for bullet in strat['rationale']:
            st.markdown(f"- {bullet}")
        votes_df = pd.DataFrame(list(strat['strategy_votes'].values()))
        st.markdown('**Current stance across all active strategies**')
        st.dataframe(votes_df, width='stretch', hide_index=True)

    with tab_risk:
        st.markdown(f"**Gate Decision:** `{result['risk']['decision']}`")
        checks_df = pd.DataFrame(result['risk']['gate_checks'])
        st.dataframe(checks_df, width='stretch', hide_index=True)
        st.json({k: v for k, v in result['risk'].items() if k not in ('gate_checks', 'drop_reasons')})

    with tab_mon:
        mon = result['monitor']
        st.markdown(f"**Overall Monitor Status:** `{mon['status']}` (checked at `{mon['checked_at_utc']}`)")
        if mon['warnings']:
            for w in mon['warnings']:
                st.info(f"Monitor warning: {w}")
        if mon['critical_errors']:
            for err in mon['critical_errors']:
                st.error(f"Critical alert: {err}")
        checks_rows = [{'Stage': stage, **details} for stage, details in mon['agent_checks'].items()]
        st.dataframe(pd.DataFrame(checks_rows), width='stretch', hide_index=True)

        beats = list_heartbeats()
        if beats:
            st.markdown('**Agent Pipeline Heartbeats**')
            st.dataframe(pd.DataFrame(beats), width='stretch', hide_index=True)

        scorecard_df = get_idea_scorecard()
        st.markdown('**Retrospective Idea Scorecard (Approved vs. Risk-Rejected After Horizon Passes)**')
        if not scorecard_df.empty:
            st.dataframe(scorecard_df, width='stretch', hide_index=True)
        else:
            st.caption(
                'Proposals (both Approved and Risk-Rejected) are automatically scored once their forward '
                'horizon (default 5 trading days) passes so you can compare hit rates against a 50% coin flip.'
            )

    st.divider()
    st.subheader(f"2. Multi-strategy historical backtest — {result['ticker']}")
    st.caption(
        f"Next-day opening fills | Active strategies use your risk profile "
        f"(max {policy.max_allocation * 100:g}% allocation, {policy.max_risk_fraction * 100:g}% risk/trade, "
        f"{policy.stop_loss_fraction * 100:g}% protective stop) | Whole shares | 10 bps fee & 5 bps slippage per side"
    )

    with st.expander('View all available strategies & rules'):
        catalog_rows = [
            {
                'Strategy': spec.name,
                'Category': spec.category,
                'Risk-Managed': 'Yes' if spec.uses_risk_policy else 'Passive Benchmark (100% cash, no stop)',
                'Rules': spec.description,
            }
            for spec in STRATEGIES.values()
        ]
        st.dataframe(pd.DataFrame(catalog_rows), width='stretch', hide_index=True)

    selected_names = st.multiselect(
        'Select strategies to compare on historical data',
        options=ALL_STRATEGY_NAMES,
        default=ALL_STRATEGY_NAMES,
    )

    if not selected_names:
        st.warning('Select at least one strategy above to run historical backtests.')
        summary_df_for_export = None
    else:
        selected_keys = [NAME_TO_STRATEGY_KEY[name] for name in selected_names]
        summary_df, equity_df, results_by_key = compare_strategies(
            data,
            starting_cash=bankroll,
            policy=policy,
            strategy_keys=selected_keys,
        )
        summary_df_for_export = summary_df

        best_return_row = summary_df.iloc[0]
        best_sharpe_row = summary_df.loc[summary_df['Sharpe'].idxmax()]
        lowest_dd_row = summary_df.loc[summary_df['Max Drawdown (%)'].idxmin()]

        k1, k2, k3 = st.columns(3)
        k1.metric(
            'Top Return Strategy',
            str(best_return_row['Strategy']),
            f"{best_return_row['Return (%)']:+.2f}% (${best_return_row['Ending Equity ($)']:,.2f})",
        )
        k2.metric(
            'Best Sharpe Ratio',
            str(best_sharpe_row['Strategy']),
            f"Sharpe {best_sharpe_row['Sharpe']:.3f}",
        )
        k3.metric(
            'Lowest Max Drawdown',
            str(lowest_dd_row['Strategy']),
            f"-{lowest_dd_row['Max Drawdown (%)']:.2f}%",
            delta_color='off',
        )

        st.markdown('**Strategy performance comparison (ranked by return)**')
        st.dataframe(
            summary_df,
            width='stretch',
            hide_index=True,
            column_config={
                'Ending Equity ($)': st.column_config.NumberColumn(format='$%.2f'),
                'Return (%)': st.column_config.NumberColumn(format='%.2f%%'),
                'Sharpe': st.column_config.NumberColumn(format='%.3f'),
                'Max Drawdown (%)': st.column_config.NumberColumn(format='%.2f%%'),
                'Win Rate (%)': st.column_config.NumberColumn(format='%.1f%%'),
            },
        )

        exp_c1, exp_c2 = st.columns(2)
        dossier_md = build_suggestion_report_markdown(result, summary_df_for_export, bankroll, policy)
        exp_c1.download_button(
            f"Download {result['ticker']} Suggestion Dossier (.md)",
            data=dossier_md.encode('utf-8'),
            file_name=f"{result['ticker']}_suggestion_dossier_{result['as_of']}.md",
            mime='text/markdown',
            key=f"dl_dossier_{result['ticker']}",
            width='stretch',
        )
        exp_c2.download_button(
            f"Download {result['ticker']} Strategy Comparison (.csv)",
            data=dataframe_to_csv_bytes(summary_df),
            file_name=f"{result['ticker']}_strategy_comparison_{result['as_of']}.csv",
            mime='text/csv',
            key=f"dl_bt_summary_{result['ticker']}",
            width='stretch',
        )

        st.markdown('**Portfolio equity curves ($)**')
        st.line_chart(equity_df)

        st.markdown('**Inspect single strategy details & simulated trades**')
        inspect_name = st.selectbox('Choose a strategy to inspect', options=selected_names)
        inspect_key = NAME_TO_STRATEGY_KEY[inspect_name]
        inspect_spec = STRATEGIES[inspect_key]
        bt = results_by_key[inspect_key]
        m = bt.metrics

        st.caption(f"**{inspect_spec.name}** ({inspect_spec.category}): {inspect_spec.description}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Ending equity', f"${m['ending_equity']:,.2f}", f"{m['strategy_return_pct']:+.2f}%")
        c2.metric('Price buy & hold', f"{m['price_buy_hold_pct']:+.2f}%")
        c3.metric('Maximum drawdown', f"{m['max_drawdown_pct']:.2f}%")
        c4.metric('Closed trades', str(m['closed_trades']))
        win_rate_label = f"{m['win_rate_pct']}%" if m['win_rate_pct'] is not None else 'N/A'
        st.caption(
            f"Sharpe (0% assumed risk-free): {m['sharpe_zero_rf']} | "
            f"Win rate: {win_rate_label} | "
            f"Ending open shares: {m['open_shares_at_end']}"
        )
        with st.expander(f"View simulated trade events — {inspect_spec.name}"):
            if bt.trades.empty:
                st.info('No trades triggered for this strategy in this historical window.')
            else:
                st.dataframe(bt.trades, width='stretch', hide_index=True)
                st.download_button(
                    f"Download {inspect_spec.name} Trade Log (.csv)",
                    data=dataframe_to_csv_bytes(bt.trades),
                    file_name=f"{result['ticker']}_{inspect_key}_trades.csv",
                    mime='text/csv',
                    key=f"dl_trades_{result['ticker']}_{inspect_key}",
                )

    st.divider()
    st.subheader('3. Human decision & paper position entry')
    st.info(
        'Agents only suggest — humans decide and trade manually. '
        'Approving below opens a tracked position in your local SQLite Paper Portfolio and NEVER submits a broker order.'
    )

    gate_passed = bool(result['risk']['passed'])
    max_shares = max(1, int(result['risk']['shares']))
    in_col1, in_col2 = st.columns([1, 2])
    with in_col1:
        shares_to_record = st.number_input(
            'Paper shares to open (capped by Risk Gate)',
            min_value=1,
            max_value=max_shares,
            value=max_shares,
            step=1,
            disabled=not gate_passed,
        )
    with in_col2:
        reason = st.text_input('Notes / reason for your manual decision (optional)', max_chars=250)

    formatted_reason = (
        f"[{result['strategy']['strategy_name']}] {reason}".strip()
        if reason
        else f"[{result['strategy']['strategy_name']}]"
    )
    col1, col2 = st.columns(2)
    if col1.button(
        'Act Manually (Log Paper Approval)',
        type='primary',
        disabled=not gate_passed,
        help='Opens a paper position in your local SQLite tracker and logs your manual paper approval.',
    ):
        try:
            open_paper_position(
                ticker=result['ticker'],
                strategy=result['strategy']['strategy_name'],
                shares=int(shares_to_record),
                entry_price=float(result['market']['price']),
                stop_price=float(result['risk']['stop_price']),
                target_price=float(result['risk']['target_price']),
                notes=reason,
                sector=result['risk']['sector'],
            )
            save_decision(
                result['ticker'],
                source,
                'APPROVE_PAPER',
                result['market']['price'],
                f"{formatted_reason} (Opened {int(shares_to_record)} paper shares, sector={result['risk']['sector']})",
            )
            st.session_state.scan_cache.clear()
            st.session_state.single_cache.clear()
            st.session_state.flash_success = (
                f"Opened paper position: {int(shares_to_record)} share(s) of {result['ticker']} "
                f"({result['risk']['sector']}) @ ${result['market']['price']:.2f} "
                f"(Stop ${result['risk']['stop_price']:.2f})."
            )
            st.rerun()
        except ValueError as err:
            st.error(str(err))

    if col2.button('Skip / Reject Idea'):
        save_decision(result['ticker'], source, 'SKIP_IDEA', result['market']['price'], formatted_reason)
        st.success('Skip / rejection logged in SQLite.')

    log_tab_dec, log_tab_drop, log_tab_mon = st.tabs([
        'Human Decisions',
        'Dropped Ideas (Risk Gate)',
        'Monitor Agent Audits',
    ])

    with log_tab_dec:
        logged = list_decisions()
        if logged:
            st.dataframe(
                pd.DataFrame(logged, columns=['UTC time', 'Ticker', 'Source', 'Decision', 'Price', 'Reason']),
                width='stretch',
                hide_index=True,
            )
        else:
            st.caption('No manual decisions logged yet.')

    with log_tab_drop:
        dropped = list_dropped_ideas()
        if dropped:
            st.dataframe(
                pd.DataFrame(
                    dropped,
                    columns=['UTC time', 'Ticker', 'Strategy', 'Price', 'Conviction', 'Drop Reasons', 'Monitor Status'],
                ),
                width='stretch',
                hide_index=True,
            )
        else:
            st.caption('No Risk-Gate dropped ideas logged yet.')

    with log_tab_mon:
        audits = list_workflow_audits()
        if audits:
            st.dataframe(
                pd.DataFrame(
                    audits,
                    columns=['UTC time', 'Ticker', 'Strategy', 'Outcome', 'Monitor Status', 'Warnings', 'Headline'],
                ),
                width='stretch',
                hide_index=True,
            )
        else:
            st.caption('No workflow audits logged yet.')


def render_paper_portfolio_tab(portfolio: dict, policy: RiskPolicy) -> None:
    """Render live paper portfolio valuation, portfolio heat, sector concentration, alerts, and close form."""
    st.subheader('Live Paper Portfolio & P/L Tracker')
    st.caption(
        'Marks your manually opened paper positions against real Yahoo Finance closes. '
        'Advisory alerts notify you when a stop, target, or strategy exit is reached — positions are NEVER auto-closed.'
    )

    max_heat_dollars = portfolio['starting_bankroll'] * policy.max_portfolio_heat
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric(
        'Total Paper Equity',
        f"${portfolio['total_equity']:,.2f}",
        f"{portfolio['total_return_pct']:+.2f}% vs ${portfolio['starting_bankroll']:,.2f}",
    )
    p2.metric(
        'Available Paper Cash',
        f"${portfolio['available_cash']:,.2f}",
        f"Cost basis: ${portfolio['open_cost_basis']:,.2f}",
        delta_color='off',
    )
    p3.metric(
        'Portfolio Heat (Stop Risk)',
        f"${portfolio['open_portfolio_risk']:,.2f}",
        f"{portfolio['open_portfolio_heat_pct']:.2f}% / {policy.max_portfolio_heat * 100:g}% (${max_heat_dollars:,.2f})",
        delta_color='off',
    )
    p4.metric('Unrealized P&L (Open)', f"${portfolio['unrealized_pnl']:+,.2f}")
    p5.metric('Realized P&L (Closed)', f"${portfolio['realized_pnl']:+,.2f}")

    # --- SPY Benchmark Comparison & Export Row ---
    bench = portfolio.get('spy_benchmark', {})
    st.markdown('### Performance vs. SPY Benchmark')
    if bench.get('available'):
        b1, b2, b3 = st.columns(3)
        b1.metric(
            'Your Paper Portfolio Return',
            f"{portfolio['total_return_pct']:+.2f}%",
            f"${portfolio['total_equity']:,.2f} current equity",
        )
        b2.metric(
            f"SPY Return (Since {bench['inception_date']})",
            f"{bench['spy_return_pct']:+.2f}%",
            f"${bench['spy_ending_equity']:,.2f} equivalent",
        )
        b3.metric(
            'Excess Return (Alpha vs. SPY)',
            f"{bench['alpha_vs_spy_pct']:+.2f}%",
            'Outperforming SPY' if bench['alpha_vs_spy_pct'] >= 0 else 'Trailing SPY',
        )
        if len(bench['curve_df']) >= 2:
            st.line_chart(bench['curve_df'])
        else:
            st.caption(
                f"Inception began on the latest trading session ({bench['inception_date']}). "
                'Multi-day equity curves vs. SPY will plot automatically as new daily bars close.'
            )
    else:
        st.caption('Open your first paper position to start tracking your live paper equity curve against SPY.')

    port_report_md = build_portfolio_report_markdown(portfolio, policy)
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    dl_col1.download_button(
        'Download Portfolio & SPY Report (.md)',
        data=port_report_md.encode('utf-8'),
        file_name='paper_portfolio_report.md',
        mime='text/markdown',
        key='dl_port_md',
        width='stretch',
    )
    if portfolio['open_positions']:
        dl_col2.download_button(
            'Download Open Positions (.csv)',
            data=dataframe_to_csv_bytes(pd.DataFrame(portfolio['open_positions'])),
            file_name='open_paper_positions.csv',
            mime='text/csv',
            key='dl_open_pos_csv',
            width='stretch',
        )
    if portfolio['closed_positions']:
        dl_col3.download_button(
            'Download Closed Trades (.csv)',
            data=dataframe_to_csv_bytes(pd.DataFrame(portfolio['closed_positions'])),
            file_name='closed_paper_positions.csv',
            mime='text/csv',
            key='dl_closed_pos_csv',
            width='stretch',
        )

    if portfolio['alerts']:
        st.markdown('**Advisory Position Alerts (Manual Review Suggested)**')
        for alert in portfolio['alerts']:
            if alert['severity'] == 'HIGH':
                st.error(alert['message'])
            elif alert['severity'] == 'WARN':
                st.warning(alert['message'])
            else:
                st.info(alert['message'])

    if portfolio['sector_breakdown']:
        st.markdown(f"### Sector Concentration (Max {policy.max_positions_per_sector} Open Position(s) / Sector)")
        sec_rows = [
            {
                'Sector': row['Sector'],
                'Slots Used': f"{row['Open Positions']} / {policy.max_positions_per_sector}",
                'Market Value ($)': row['Market Value ($)'],
                'Open Stop Risk ($)': row['Open Stop Risk ($)'],
            }
            for row in portfolio['sector_breakdown']
        ]
        st.dataframe(
            pd.DataFrame(sec_rows),
            width='stretch',
            hide_index=True,
            column_config={
                'Market Value ($)': st.column_config.NumberColumn(format='$%.2f'),
                'Open Stop Risk ($)': st.column_config.NumberColumn(format='$%.2f'),
            },
        )

    st.markdown('### Open Paper Positions')
    open_pos = portfolio['open_positions']
    if not open_pos:
        st.info('No open paper positions yet. Approve a Risk-Gate-passing suggestion in the Discovery tab to open one.')
    else:
        open_df = pd.DataFrame([
            {
                'ID': p['id'],
                'Ticker': p['ticker'],
                'Sector': p['sector'],
                'Strategy': p['strategy'],
                'Shares': p['shares'],
                'Entry ($)': p['entry_price'],
                'Last Close ($)': p['current_price'],
                'Market Value ($)': p['market_value'],
                'Unrealized P&L ($)': p['unrealized_pnl'],
                'Return (%)': p['unrealized_pnl_pct'],
                'Stop ($)': p['stop_price'],
                'Stop Risk ($)': p['open_stop_risk'],
                'Target ($)': p['target_price'],
                'Advisory Status': p['alert_status'],
            }
            for p in open_pos
        ])
        st.dataframe(
            open_df,
            width='stretch',
            hide_index=True,
            column_config={
                'Entry ($)': st.column_config.NumberColumn(format='$%.2f'),
                'Last Close ($)': st.column_config.NumberColumn(format='$%.2f'),
                'Market Value ($)': st.column_config.NumberColumn(format='$%.2f'),
                'Unrealized P&L ($)': st.column_config.NumberColumn(format='$%.2f'),
                'Return (%)': st.column_config.NumberColumn(format='%.2f%%'),
                'Stop ($)': st.column_config.NumberColumn(format='$%.2f'),
                'Stop Risk ($)': st.column_config.NumberColumn(format='$%.2f'),
                'Target ($)': st.column_config.NumberColumn(format='$%.2f'),
            },
        )

        st.markdown('#### Manually Close an Open Paper Position')
        pos_labels = [
            f"#{p['id']} {p['ticker']} ({p['sector']}) — {p['shares']} sh "
            f"(Entry ${p['entry_price']:.2f} → Current ${p['current_price']:.2f} | {p['alert_status']})"
            for p in open_pos
        ]
        chosen_pos_label = st.selectbox('Select open paper position to close', options=pos_labels)
        chosen_pos = open_pos[pos_labels.index(chosen_pos_label)]

        cl1, cl2 = st.columns([1, 2])
        with cl1:
            manual_exit_price = st.number_input(
                'Manual exit fill price ($)',
                min_value=0.01,
                value=float(chosen_pos['current_price']),
                step=0.10,
                format='%.2f',
            )
        with cl2:
            close_reason = st.text_input('Exit note (optional)', max_chars=200)

        if st.button('Close Selected Paper Position', type='primary'):
            closed_summary = close_paper_position(
                position_id=chosen_pos['id'],
                exit_price=float(manual_exit_price),
                close_note=close_reason,
            )
            st.session_state.scan_cache.clear()
            st.session_state.single_cache.clear()
            st.session_state.flash_success = (
                f"Closed {closed_summary['shares']} share(s) of {closed_summary['ticker']} "
                f"@ ${closed_summary['exit_price']:.2f} | Realized P&L: ${closed_summary['realized_pnl']:+,.2f}."
            )
            st.rerun()

    st.divider()
    st.markdown('### Closed Paper Positions History')
    closed_pos = portfolio['closed_positions']
    if not closed_pos:
        st.caption('No closed paper positions recorded yet.')
    else:
        closed_df = pd.DataFrame([
            {
                'ID': p['id'],
                'Ticker': p['ticker'],
                'Sector': p['sector'],
                'Strategy': p['strategy'],
                'Shares': p['shares'],
                'Entry ($)': p['entry_price'],
                'Exit ($)': p['exit_price'],
                'Realized P&L ($)': p['realized_pnl'],
                'Opened UTC': p['opened_at_utc'],
                'Closed UTC': p['closed_at_utc'],
                'Notes': p['notes'],
            }
            for p in closed_pos
        ])
        st.dataframe(
            closed_df,
            width='stretch',
            hide_index=True,
            column_config={
                'Entry ($)': st.column_config.NumberColumn(format='$%.2f'),
                'Exit ($)': st.column_config.NumberColumn(format='$%.2f'),
                'Realized P&L ($)': st.column_config.NumberColumn(format='$%.2f'),
            },
        )


# --- Session Initialization & Onboarding ---
if 'onboarding_complete' not in st.session_state:
    st.session_state.onboarding_complete = False
if 'scan_cache' not in st.session_state:
    st.session_state.scan_cache = {}
if 'single_cache' not in st.session_state:
    st.session_state.single_cache = {}

if not st.session_state.onboarding_complete:
    st.title('Multi-Agent Stock Analysis Lab')
    st.caption('Agents only suggest — humans decide and trade manually | Real Yahoo Finance data only | NO brokerage orders')
    st.divider()

    st.subheader('Welcome! Set up your paper trading profile')
    st.write(
        'Enter your starting paper trading capital and risk tolerance. '
        'Our agents will automatically scan S&P stocks to find diversified candidates that fit your exact budget, portfolio heat, and sector rules.'
    )

    default_bankroll = float(st.session_state.get('bankroll', 10000.0))
    default_choice = st.session_state.get('risk_label', 'Medium (Moderate)')
    default_index = RISK_OPTIONS.index(default_choice) if default_choice in RISK_OPTIONS else 1
    default_scope_idx = 0 if st.session_state.get('scan_scope', 'sp100') == 'sp100' else 1

    col_amount, col_scope = st.columns([2, 1])
    with col_amount:
        input_bankroll = st.number_input(
            'Paper trading amount ($)',
            min_value=100.0,
            max_value=10_000_000.0,
            value=default_bankroll,
            step=500.0,
            format='%.2f',
        )
    with col_scope:
        input_scope_label = st.selectbox(
            'Market scan universe',
            options=SCAN_SCOPE_LABELS,
            index=default_scope_idx,
            help='Scan S&P 100 liquid leaders quickly by default, or scan all ~500 S&P 500 stocks.',
        )

    risk_choice = st.radio(
        'How much risk are you willing to take?',
        options=RISK_OPTIONS,
        index=default_index,
        captions=[RISK_DESCRIPTIONS[opt] for opt in RISK_OPTIONS],
    )

    prev_policy: RiskPolicy = st.session_state.get('risk_policy', RiskPolicy())
    if risk_choice == 'Custom':
        c_alloc, c_risk, c_stop, c_heat, c_sec = st.columns(5)
        with c_alloc:
            custom_alloc = st.slider(
                'Max position allocation (%)',
                min_value=5.0,
                max_value=100.0,
                value=round(prev_policy.max_allocation * 100, 1),
                step=1.0,
            )
        with c_risk:
            custom_risk = st.slider(
                'Max risk per trade (%)',
                min_value=0.1,
                max_value=10.0,
                value=round(prev_policy.max_risk_fraction * 100, 1),
                step=0.1,
            )
        with c_stop:
            custom_stop = st.slider(
                'Protective stop-loss (%)',
                min_value=1.0,
                max_value=25.0,
                value=round(prev_policy.stop_loss_fraction * 100, 1),
                step=0.5,
            )
        with c_heat:
            custom_heat = st.slider(
                'Max portfolio heat (%)',
                min_value=0.5,
                max_value=25.0,
                value=round(prev_policy.max_portfolio_heat * 100, 1),
                step=0.5,
            )
        with c_sec:
            custom_sector_cap = st.slider(
                'Max stocks / sector',
                min_value=1,
                max_value=10,
                value=int(prev_policy.max_positions_per_sector),
                step=1,
            )
    else:
        custom_alloc, custom_risk, custom_stop, custom_heat, custom_sector_cap = 20.0, 1.0, 5.0, 4.0, 2

    preview_policy = resolve_onboarding_policy(
        risk_choice,
        custom_alloc,
        custom_risk,
        custom_stop,
        custom_heat,
        custom_sector_cap,
    )

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric(
        'Max Position Size',
        f"${input_bankroll * preview_policy.max_allocation:,.2f}",
        f"{preview_policy.max_allocation * 100:g}% of portfolio",
    )
    p2.metric(
        'Max Risk / Trade',
        f"${input_bankroll * preview_policy.max_risk_fraction:,.2f}",
        f"{preview_policy.max_risk_fraction * 100:g}% of portfolio",
    )
    p3.metric(
        'Max Portfolio Heat',
        f"${input_bankroll * preview_policy.max_portfolio_heat:,.2f}",
        f"{preview_policy.max_portfolio_heat * 100:g}% combined risk",
    )
    p4.metric(
        'Sector Limit',
        f"{preview_policy.max_positions_per_sector} stock(s)",
        'Max per sector',
    )
    p5.metric(
        'Protective Stop',
        f"{preview_policy.stop_loss_fraction * 100:g}%",
        'Below entry fill',
    )

    if st.button('Scan Market & Start Paper Trading', type='primary'):
        st.session_state.bankroll = float(input_bankroll)
        st.session_state.risk_label = risk_choice
        st.session_state.risk_policy = preview_policy
        st.session_state.scan_scope = 'sp100' if input_scope_label.startswith('S&P 100') else 'sp500'
        st.session_state.onboarding_complete = True
        st.rerun()

    st.stop()

# --- Main Dashboard (after onboarding) ---
st.title('Multi-Agent Stock Analysis Lab')
st.caption('Core Rule: Agents only suggest. Humans decide and trade manually. | Real Yahoo Finance data | NO automated order execution')

if st.session_state.get('flash_success'):
    st.success(st.session_state.pop('flash_success'))

bankroll: float = st.session_state.bankroll
policy: RiskPolicy = st.session_state.risk_policy
risk_label: str = st.session_state.risk_label
cash_ledger = get_paper_cash_summary(starting_bankroll=bankroll)

with st.sidebar:
    st.header('Discovery & market scan')
    discovery_mode = st.radio(
        'Stock discovery mode',
        ['Automatic S&P Market Scan', 'Manual Ticker Lookup'],
        index=0,
    )

    if discovery_mode == 'Automatic S&P Market Scan':
        curr_scope_idx = 0 if st.session_state.get('scan_scope', 'sp100') == 'sp100' else 1
        scope_choice = st.radio('S&P scan scope', SCAN_SCOPE_LABELS, index=curr_scope_idx)
        active_scope = 'sp100' if scope_choice.startswith('S&P 100') else 'sp500'
        st.session_state.scan_scope = active_scope
        primary_strategy_name = st.selectbox('Primary workflow strategy', ACTIVE_STRATEGY_NAMES, index=0)
        primary_strategy_key = NAME_TO_STRATEGY_KEY[primary_strategy_name]
        history_period = st.selectbox('Backtest history window', ['6mo', '1y', '2y', '5y'], index=2)
        if st.button('Refresh Market Scan', width='stretch'):
            st.session_state.scan_cache.clear()
            st.rerun()
    else:
        manual_symbol = st.text_input(
            'Stock ticker (Yahoo Finance)',
            value=st.session_state.get('manual_ticker', 'AAPL'),
        ).upper().strip()
        if manual_symbol:
            st.session_state.manual_ticker = manual_symbol
        primary_strategy_name = st.selectbox('Primary workflow strategy', ACTIVE_STRATEGY_NAMES, index=0)
        primary_strategy_key = NAME_TO_STRATEGY_KEY[primary_strategy_name]
        history_period = st.selectbox('Historical window', ['6mo', '1y', '2y', '5y'], index=2)

    st.caption('Uses real historical daily OHLCV bars & news from Yahoo Finance. Offline synthetic data is disabled.')

    st.divider()
    st.header('Paper portfolio & risk')
    st.metric('Starting paper capital', f"${bankroll:,.2f}")
    st.metric(
        'Available paper cash',
        f"${cash_ledger['available_cash']:,.2f}",
        f"{len(cash_ledger['open_positions'])} open position(s)",
        delta_color='off',
    )
    st.metric(
        'Open portfolio heat',
        f"{cash_ledger['open_portfolio_heat_pct']:.2f}% / {policy.max_portfolio_heat * 100:g}%",
        f"${cash_ledger['open_portfolio_risk']:,.2f} at stop risk",
        delta_color='off',
    )
    st.write(f"**Risk tolerance:** {risk_label}")
    st.caption(
        f"Max alloc: {policy.max_allocation * 100:g}% | "
        f"Risk/trade: {policy.max_risk_fraction * 100:g}% | "
        f"Stop: {policy.stop_loss_fraction * 100:g}% | "
        f"Max {policy.max_positions_per_sector}/sector"
    )
    if st.button('Change Paper Amount / Risk', width='stretch'):
        st.session_state.onboarding_complete = False
        st.rerun()
    if cash_ledger['open_positions'] or cash_ledger['closed_positions']:
        if st.button('Reset Paper Positions', width='stretch'):
            clear_paper_positions()
            st.session_state.scan_cache.clear()
            st.session_state.single_cache.clear()
            st.rerun()

    with st.expander('AI Settings (Research & Report)'):
        provider_label = st.selectbox(
            'AI mode',
            ['Auto (env key or built-in)', 'OpenAI', 'Anthropic', 'Google Gemini', 'Built-in Synthesis Only'],
            index=0,
        )
        provider_map = {
            'Auto (env key or built-in)': 'auto',
            'OpenAI': 'openai',
            'Anthropic': 'anthropic',
            'Google Gemini': 'gemini',
            'Built-in Synthesis Only': 'none',
        }
        chosen_provider = provider_map[provider_label]
        user_api_key = ''
        if chosen_provider in ('openai', 'anthropic', 'gemini'):
            user_api_key = st.text_input('API key (session only)', type='password')
        llm_cfg = LLMConfig(provider=chosen_provider, api_key=user_api_key or None)


open_count = len(cash_ledger['open_positions'])
main_tab_discover, main_tab_portfolio = st.tabs([
    'Market Discovery & 6-Agent Workflow',
    f'Live Paper Portfolio ({open_count} Open)',
])

known_bars_map: dict[str, pd.DataFrame] = {}

with main_tab_discover:
    if discovery_mode == 'Automatic S&P Market Scan':
        scan_key = (
            st.session_state.scan_scope,
            bankroll,
            cash_ledger['available_cash'],
            cash_ledger['open_portfolio_risk'],
            tuple(sorted(cash_ledger['sector_counts'].items())),
            tuple(sorted(cash_ledger['open_tickers'])),
            policy,
            primary_strategy_key,
            history_period,
            llm_cfg,
        )
        if scan_key not in st.session_state.scan_cache:
            prog_bar = st.progress(0.0, text='Initializing S&P market scan...')

            def _update_prog(done: int, total: int, msg: str) -> None:
                frac = min(1.0, max(0.0, done / max(total, 1)))
                prog_bar.progress(frac, text=msg)

            try:
                scan_res = scan_market_and_suggest(
                    scope=st.session_state.scan_scope,
                    bankroll=bankroll,
                    policy=policy,
                    primary_strategy=primary_strategy_key,
                    top_k=TOP_K_SHORTLIST,
                    history_period=history_period,
                    llm_config=llm_cfg,
                    available_cash=cash_ledger['available_cash'],
                    open_tickers=cash_ledger['open_tickers'],
                    open_portfolio_risk=cash_ledger['open_portfolio_risk'],
                    sector_counts=cash_ledger['sector_counts'],
                    progress_callback=_update_prog,
                )
                st.session_state.scan_cache[scan_key] = scan_res
            except Exception as exc:
                prog_bar.empty()
                st.error(f'Market scan failed: {exc}')
                st.info('Check your internet connection and Yahoo Finance availability, or try refreshing.')
                st.stop()
            prog_bar.empty()

        scan_res = st.session_state.scan_cache[scan_key]
        for c_item in scan_res['candidates']:
            known_bars_map[c_item['ticker']] = c_item['data']

        scope_title = 'S&P 100' if scan_res['scope'] == 'sp100' else 'S&P 500'

        st.subheader(f'Market Scan Overview — {scope_title} Discovery Funnel')
        f1, f2, f3, f4 = st.columns(4)
        f1.metric('Universe Scanned', f"{scan_res['valid_data_count']} / {scan_res['universe_size']} stocks")
        f2.metric(
            'Affordable Under Cash & Heat',
            f"{scan_res['affordable_count']} stocks",
            f"{scan_res.get('sector_blocked_count', 0)} at sector cap",
            delta_color='off',
        )
        f3.metric('Diversified Shortlist', f"{len(scan_res['candidates'])} candidates")
        f4.metric(
            'Passed Risk Gate',
            f"{scan_res['passed_count']} ready",
            f"{scan_res['dropped_count']} dropped",
            delta_color='normal',
        )

        if not scan_res['candidates']:
            st.warning(
                f"No new S&P stocks met whole-share affordability, portfolio heat "
                f"({cash_ledger['open_portfolio_heat_pct']:.2f}% / {policy.max_portfolio_heat * 100:g}%), "
                f"and sector caps under ${cash_ledger['available_cash']:,.2f} available cash. "
                f"Review your open paper positions or adjust capital/risk settings."
            )
        else:
            st.markdown(f"**Top {len(scan_res['candidates'])} sector-diversified S&P candidates evaluated by the 6-agent pipeline**")
            st.dataframe(
                scan_res['shortlist_summary_df'],
                width='stretch',
                hide_index=True,
                column_config={
                    'Close ($)': st.column_config.NumberColumn(format='$%.2f'),
                    'Notional ($)': st.column_config.NumberColumn(format='$%.2f'),
                    'Stop ($)': st.column_config.NumberColumn(format='$%.2f'),
                },
            )

            with st.expander(f"View wider {scope_title} technical scan leaderboard (Top 25 eligible stocks)"):
                st.dataframe(scan_res['leaderboard_df'].head(25), width='stretch', hide_index=True)
                st.download_button(
                    f"Download {scope_title} Scan Leaderboard (.csv)",
                    data=dataframe_to_csv_bytes(scan_res['leaderboard_df']),
                    file_name=f"{scan_res['scope']}_scan_leaderboard.csv",
                    mime='text/csv',
                    key='dl_scan_leaderboard_csv',
                )

            candidate_options = [
                f"{c['ticker']} ({c['workflow']['risk']['sector']}) — "
                f"{'PASSED' if c['workflow']['risk']['passed'] else 'DROPPED'} "
                f"(${c['workflow']['market']['price']:.2f} | Conviction {c['workflow']['strategy']['conviction_score']:+d}/5)"
                for c in scan_res['candidates']
            ]
            selected_candidate_label = st.selectbox(
                'Select a surfaced candidate to inspect its Plain-English Report, Backtests & Decision',
                options=candidate_options,
                index=0,
            )
            chosen_idx = candidate_options.index(selected_candidate_label)
            chosen_candidate = scan_res['candidates'][chosen_idx]

            st.divider()
            render_candidate_deep_dive(
                data=chosen_candidate['data'],
                result=chosen_candidate['workflow'],
                bankroll=bankroll,
                policy=policy,
            )
    else:
        ticker = st.session_state.get('manual_ticker', 'AAPL')
        already_held = ticker.upper() in cash_ledger['open_tickers']
        single_key = (
            ticker,
            bankroll,
            cash_ledger['available_cash'],
            cash_ledger['open_portfolio_risk'],
            tuple(sorted(cash_ledger['sector_counts'].items())),
            already_held,
            policy,
            primary_strategy_key,
            history_period,
            llm_cfg,
        )
        if single_key not in st.session_state.single_cache:
            try:
                with st.spinner(f'Running 6-agent suggestion workflow for {ticker}...'):
                    st.session_state.single_cache[single_key] = run_suggestion_workflow(
                        ticker=ticker,
                        bankroll=bankroll,
                        policy=policy,
                        strategy=primary_strategy_key,
                        period=history_period,
                        source='yahoo',
                        llm_config=llm_cfg,
                        available_cash=cash_ledger['available_cash'],
                        already_held=already_held,
                        open_portfolio_risk=cash_ledger['open_portfolio_risk'],
                        sector_counts=cash_ledger['sector_counts'],
                    )
            except Exception as exc:
                st.error(f"Unable to load real market history for '{ticker}': {exc}")
                st.info('Verify your internet connection and check that the ticker symbol exists on Yahoo Finance (e.g. AAPL, MSFT, NVDA, SPY).')
                st.stop()

        data, result = st.session_state.single_cache[single_key]
        known_bars_map[ticker] = data
        render_candidate_deep_dive(data=data, result=result, bankroll=bankroll, policy=policy)

with main_tab_portfolio:
    portfolio_state = evaluate_paper_portfolio(
        starting_bankroll=bankroll,
        bars_by_ticker=known_bars_map,
    )
    render_paper_portfolio_tab(portfolio_state, policy)

st.caption('Limitations: educational paper-research tool, daily OHLC stop approximation, no automated trade execution. Not investment advice.')

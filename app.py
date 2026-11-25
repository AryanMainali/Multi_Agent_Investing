"""Streamlit app: onboarding for paper capital & risk tolerance, real-data multi-agent analysis, multi-strategy backtesting, and audit log."""
import pandas as pd
import streamlit as st
from agents.risk_agent import RiskPolicy, get_risk_policy
from core.backtest import compare_strategies
from core.storage import list_decisions, save_decision
from core.strategies import STRATEGIES
from core.workflow import analyze

st.set_page_config(page_title='Stock Multi-Agent Lab', page_icon='📈', layout='wide')

RISK_OPTIONS = [
    'Low (Conservative)',
    'Medium (Moderate)',
    'High (Aggressive)',
    'Custom',
]

RISK_DESCRIPTIONS = {
    'Low (Conservative)': 'Up to 10% per position · 0.5% max portfolio risk per trade · 4% protective stop-loss',
    'Medium (Moderate)': 'Up to 20% per position · 1.0% max portfolio risk per trade · 5% protective stop-loss',
    'High (Aggressive)': 'Up to 35% per position · 2.5% max portfolio risk per trade · 7% protective stop-loss',
    'Custom': 'Define your own position allocation, per-trade risk budget, and stop-loss percentage',
}

NAME_TO_STRATEGY_KEY = {spec.name: key for key, spec in STRATEGIES.items()}
ALL_STRATEGY_NAMES = list(NAME_TO_STRATEGY_KEY.keys())


def resolve_onboarding_policy(choice: str, alloc_pct: float, risk_pct: float, stop_pct: float) -> RiskPolicy:
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
    )


if 'onboarding_complete' not in st.session_state:
    st.session_state.onboarding_complete = False

if not st.session_state.onboarding_complete:
    st.title('Multi-Agent Stock Analysis Lab')
    st.caption('Paper-trading research prototype | Real Yahoo Finance market data only | NO brokerage orders')
    st.divider()

    st.subheader('Welcome! Set up your paper trading profile')
    st.write('Before analyzing real market data, enter your starting paper trading capital and how much risk you are willing to take.')

    default_bankroll = float(st.session_state.get('bankroll', 10000.0))
    default_choice = st.session_state.get('risk_label', 'Medium (Moderate)')
    default_index = RISK_OPTIONS.index(default_choice) if default_choice in RISK_OPTIONS else 1

    col_amount, col_ticker = st.columns([2, 1])
    with col_amount:
        input_bankroll = st.number_input(
            'Paper trading amount ($)',
            min_value=100.0,
            max_value=10_000_000.0,
            value=default_bankroll,
            step=500.0,
            format='%.2f',
        )
    with col_ticker:
        input_ticker = st.text_input(
            'Initial stock ticker',
            value=st.session_state.get('ticker', 'AAPL'),
            help='Real stock or ETF symbol on Yahoo Finance (e.g. AAPL, MSFT, NVDA, SPY).',
        ).upper().strip()

    risk_choice = st.radio(
        'How much risk are you willing to take?',
        options=RISK_OPTIONS,
        index=default_index,
        captions=[RISK_DESCRIPTIONS[opt] for opt in RISK_OPTIONS],
    )

    prev_policy: RiskPolicy = st.session_state.get('risk_policy', RiskPolicy())
    if risk_choice == 'Custom':
        c_alloc, c_risk, c_stop = st.columns(3)
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
                'Max portfolio risk per trade (%)',
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
    else:
        custom_alloc, custom_risk, custom_stop = 20.0, 1.0, 5.0

    preview_policy = resolve_onboarding_policy(risk_choice, custom_alloc, custom_risk, custom_stop)

    p1, p2, p3 = st.columns(3)
    p1.metric(
        'Max Position Allocation',
        f"${input_bankroll * preview_policy.max_allocation:,.2f}",
        f"{preview_policy.max_allocation * 100:g}% of portfolio",
    )
    p2.metric(
        'Max Risk Per Trade',
        f"${input_bankroll * preview_policy.max_risk_fraction:,.2f}",
        f"{preview_policy.max_risk_fraction * 100:g}% of portfolio",
    )
    p3.metric(
        'Protective Stop-Loss',
        f"{preview_policy.stop_loss_fraction * 100:g}%",
        'Below entry fill price',
    )

    if st.button('Start Paper Trading', type='primary'):
        if not input_ticker:
            st.error('Please enter a valid stock ticker symbol (e.g. AAPL).')
        else:
            st.session_state.bankroll = float(input_bankroll)
            st.session_state.risk_label = risk_choice
            st.session_state.risk_policy = preview_policy
            st.session_state.ticker = input_ticker
            st.session_state.onboarding_complete = True
            st.rerun()

    st.stop()

# --- Main Dashboard (after onboarding) ---
st.title('Multi-Agent Stock Analysis Lab')
st.caption('Educational prototype | Paper decisions only | Real Yahoo Finance market data | NO brokerage connection or order placement')

bankroll: float = st.session_state.bankroll
policy: RiskPolicy = st.session_state.risk_policy
risk_label: str = st.session_state.risk_label
source = 'yahoo'

with st.sidebar:
    st.header('Market data')
    symbol = st.text_input('Stock ticker (Yahoo Finance)', value=st.session_state.ticker).upper().strip()
    if symbol:
        st.session_state.ticker = symbol
    ticker = st.session_state.ticker
    period = st.selectbox('Historical window', ['6mo', '1y', '2y', '5y'], index=2)
    st.caption('Uses real historical daily OHLCV bars from Yahoo Finance. Offline synthetic data is disabled.')

    st.divider()
    st.header('Paper portfolio & risk')
    st.metric('Starting paper capital', f"${bankroll:,.2f}")
    st.write(f"**Risk tolerance:** {risk_label}")
    st.caption(
        f"Max allocation: {policy.max_allocation * 100:g}% | "
        f"Max risk/trade: {policy.max_risk_fraction * 100:g}% | "
        f"Stop-loss: {policy.stop_loss_fraction * 100:g}%"
    )
    if st.button('Change Paper Amount / Risk', width='stretch'):
        st.session_state.onboarding_complete = False
        st.rerun()


@st.cache_data(ttl=1800, show_spinner=False)
def get_result(ticker_symbol: str, starting_cash: float, active_policy: RiskPolicy, history_period: str):
    return analyze(
        ticker=ticker_symbol,
        bankroll=starting_cash,
        policy=active_policy,
        period=history_period,
        source='yahoo',
    )


try:
    with st.spinner(f'Fetching real market history for {ticker} from Yahoo Finance...'):
        data, result = get_result(ticker, bankroll, policy, period)
except Exception as exc:
    st.error(f"Unable to load real market history for '{ticker}': {exc}")
    st.info('Verify your internet connection and check that the ticker symbol exists on Yahoo Finance (e.g. AAPL, MSFT, NVDA, SPY).')
    st.stop()

st.subheader(f"1. Four agents — {result['ticker']}")
st.caption(f"As of {result['as_of']} | source: YAHOO FINANCE (real historical daily bars, not streaming quotes)")
a, b, c = st.columns(3)
a.metric('Last daily close', f"${result['analyst']['price']:.2f}")
b.metric('20-day moving average', f"${result['analyst']['sma_20']:.2f}")
c.metric('50-day moving average', f"${result['analyst']['sma_50']:.2f}")
chart = data[['Close', 'SMA_FAST', 'SMA_SLOW']].rename(columns={'SMA_FAST': 'SMA 20', 'SMA_SLOW': 'SMA 50'})
st.line_chart(chart)
left, right = st.columns(2)
with left:
    st.markdown('**Market Agent**')
    st.json(result['market'])
    st.markdown('**Data Agent**')
    st.write(f"Fetched {len(data)} real trading-day bars from Yahoo Finance; generated volume, momentum, and volatility indicators.")
with right:
    st.markdown('**Analyst Agent**')
    st.json(result['analyst'])
    st.markdown('**Risk Manager**')
    st.json(result['risk'])

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
else:
    selected_keys = [NAME_TO_STRATEGY_KEY[name] for name in selected_names]
    summary_df, equity_df, results_by_key = compare_strategies(
        data,
        starting_cash=bankroll,
        policy=policy,
        strategy_keys=selected_keys,
    )

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

st.divider()
st.subheader('3. Human-in-the-loop paper decision')
st.info('This logs an educational approval or rejection ONLY. It does NOT execute a trade or alter the backtest.')
chosen_strategy = st.selectbox(
    'Strategy context for decision log',
    options=['Multi-Agent Screening'] + ALL_STRATEGY_NAMES,
)
reason = st.text_input('Reason for decision (optional)', max_chars=250)
formatted_reason = f"[{chosen_strategy}] {reason}".strip() if reason else f"[{chosen_strategy}]"
col1, col2 = st.columns(2)
if col1.button('Approve for paper research'):
    save_decision(result['ticker'], source, 'APPROVE_PAPER', result['analyst']['price'], formatted_reason)
    st.success('Paper-only approval saved in SQLite.')
if col2.button('Reject suggestion'):
    save_decision(result['ticker'], source, 'REJECT', result['analyst']['price'], formatted_reason)
    st.success('Rejection saved in SQLite.')
logged = list_decisions()
if logged:
    st.dataframe(
        pd.DataFrame(logged, columns=['UTC time', 'Ticker', 'Source', 'Decision', 'Price', 'Reason']),
        width='stretch',
    )

st.caption('Limitations: single ticker per run, rule-based strategies, daily OHLC stop approximation, no ML or LLM, no live orders. Do not treat the output as investment advice.')

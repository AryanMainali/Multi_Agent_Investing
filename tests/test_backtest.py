import numpy as np
import pandas as pd
import pytest
from agents.risk_agent import get_risk_policy
from core.backtest import compare_strategies, run_backtest
from core.data import validate_bars
from core.indicators import enrich
from core.strategies import STRATEGIES, generate_signals


def _make_crossover_prices(n: int = 240) -> pd.DataFrame:
    dates = pd.bdate_range('2025-01-02', periods=n)
    t = np.arange(n, dtype=float)
    close = 120.0 + 0.08 * t + 22.0 * np.sin(t / 9.0) + 6.0 * np.cos(t / 3.0)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 1.5
    low = np.minimum(open_, close) - 1.5
    volume = np.full(n, 25_000_000, dtype=int)
    return validate_bars(
        pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume}, index=dates)
    )


def test_backtest_metrics_and_equity():
    prices = _make_crossover_prices()
    result = run_backtest(prices, starting_cash=15000.0, policy=get_risk_policy('medium'))
    assert len(result.equity) == len(prices)
    assert np.isfinite(result.equity['Equity']).all()
    assert (result.equity['Equity'] >= 0).all()
    assert result.metrics['starting_cash'] == 15000.0
    assert result.metrics['max_drawdown_pct'] >= 0
    assert result.metrics['ending_equity'] > 0
    assert result.metrics['closed_trades'] == result.trades['Action'].isin(['SELL', 'STOP']).sum()


@pytest.mark.parametrize('strategy_key', list(STRATEGIES.keys()))
def test_every_strategy_executes_without_lookahead(strategy_key):
    prices = _make_crossover_prices()
    signaled, _ = generate_signals(prices, strategy=strategy_key)
    result = run_backtest(prices, starting_cash=10000.0, strategy=strategy_key)
    assert len(result.equity) == len(prices)
    assert np.isfinite(result.equity['Equity']).all()
    assert (result.equity['Equity'] > 0).all()

    # Every BUY fill must occur on bar i > 0 strictly after bar i-1 ENTRY_SIGNAL was True
    buy_dates = result.trades.loc[result.trades['Action'] == 'BUY', 'Date']
    for date in buy_dates:
        i = prices.index.get_loc(date)
        assert i > 0
        assert bool(signaled.iloc[i - 1]['ENTRY_SIGNAL'])


def test_compare_strategies_ranks_all_strategies():
    prices = _make_crossover_prices()
    summary_df, equity_df, results = compare_strategies(
        prices,
        starting_cash=20000.0,
        policy=get_risk_policy('medium'),
    )
    assert len(summary_df) == len(STRATEGIES)
    assert list(equity_df.columns) == [spec.name for spec in STRATEGIES.values()]
    assert len(equity_df) == len(prices)
    assert len(results) == len(STRATEGIES)
    # Summary table is sorted descending by Return (%)
    returns = summary_df['Return (%)'].tolist()
    assert returns == sorted(returns, reverse=True)


def test_cross_signal_executes_at_next_bar_open():
    prices = _make_crossover_prices()
    enriched = enrich(prices)
    result = run_backtest(prices)
    buy_dates = result.trades.loc[result.trades['Action'] == 'BUY', 'Date']
    assert len(buy_dates) > 0
    for date in buy_dates:
        i = prices.index.get_loc(date)
        assert i > 0
        assert bool(enriched.iloc[i - 1]['CROSS_UP'])


def test_risk_policy_affects_backtest_position_size():
    prices = _make_crossover_prices()
    low_bt = run_backtest(prices, starting_cash=20000.0, policy=get_risk_policy('low'))
    high_bt = run_backtest(prices, starting_cash=20000.0, policy=get_risk_policy('high'))
    first_low_buy = low_bt.trades.loc[low_bt.trades['Action'] == 'BUY', 'Shares'].iloc[0]
    first_high_buy = high_bt.trades.loc[high_bt.trades['Action'] == 'BUY', 'Shares'].iloc[0]
    assert first_high_buy > first_low_buy


def test_costs_do_not_improve_portfolio():
    prices = _make_crossover_prices()
    low = run_backtest(prices, fee_bps=0, slippage_bps=0)
    high = run_backtest(prices, fee_bps=20, slippage_bps=10)
    assert high.metrics['ending_equity'] <= low.metrics['ending_equity'] + 1e-7


def test_reject_bad_args():
    prices = _make_crossover_prices()
    with pytest.raises(ValueError):
        run_backtest(prices, starting_cash=-1)
    with pytest.raises(ValueError, match='Unknown strategy'):
        run_backtest(prices, strategy='nonexistent_strategy')
    with pytest.raises(ValueError):
        compare_strategies(prices, strategy_keys=[])

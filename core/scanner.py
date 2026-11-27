"""Two-stage S&P market scanner for autonomous stock discovery.

Stage 1A (Wide Technical & Capital Scan):
  Downloads real Yahoo Finance daily OHLCV bars in batches, filters out stocks that
  cannot be sized to >= 1 whole share within the user's paper capital & RiskPolicy
  (or breach volatility/liquidity limits), and ranks all eligible stocks by technical
  and multi-strategy strength.

Stage 1B (Deep 6-Agent Pipeline on Top K):
  Runs the full 6-agent suggestion workflow ([1. Market Data] + [2. Research AI] ->
  [3. Strategy] -> [4. Risk Gate] -> [5. Report AI] / [Drop & Log] <- [6. Monitor Agent])
  on the Top K shortlisted candidates.
"""
from collections.abc import Callable
from pathlib import Path
import pandas as pd
import yfinance as yf
from agents.analyst_agent import AnalystAgent
from agents.market_agent import MarketAgent
from agents.risk_agent import RiskPolicy, size_position
from agents.strategy_agent import ACTIVE_STRATEGY_KEYS, _evaluate_strategy_stance
from core.data import REQUIRED, validate_bars
from core.indicators import enrich
from core.llm import LLMConfig
from core.strategies import STRATEGIES
from core.universe import get_universe_tickers
from core.workflow import run_suggestion_workflow


def extract_symbol_bars(batch_df: pd.DataFrame | None, symbol: str, batch_symbols: list[str]) -> pd.DataFrame | None:
    """Extract a single ticker's OHLCV DataFrame from a yfinance.download result."""
    if batch_df is None or batch_df.empty:
        return None
    if isinstance(batch_df.columns, pd.MultiIndex):
        lvl0 = batch_df.columns.get_level_values(0)
        lvl1 = batch_df.columns.get_level_values(1)
        if symbol in lvl0:
            sub = batch_df[symbol].copy()
        elif symbol in lvl1:
            sub = batch_df.xs(symbol, axis=1, level=1).copy()
        else:
            return None
    else:
        if len(batch_symbols) == 1 and batch_symbols[0] == symbol:
            sub = batch_df.copy()
        else:
            return None

    if not set(REQUIRED).issubset(sub.columns):
        return None
    sub = sub.dropna(how='all')
    return sub if not sub.empty else None


def score_candidate_technicals(
    df: pd.DataFrame,
    primary_strategy: str = 'sma_crossover',
) -> dict:
    """Compute fast technical & multi-strategy alignment score on past-only bars."""
    key = primary_strategy if primary_strategy in ACTIVE_STRATEGY_KEYS else 'sma_crossover'
    enriched = enrich(df)
    market = MarketAgent().scan(enriched)
    analyst = AnalystAgent().evaluate(enriched, market)

    votes = {k: _evaluate_strategy_stance(enriched, k) for k in ACTIVE_STRATEGY_KEYS}
    primary_vote = votes[key]
    bullish_count = sum(1 for v in votes.values() if v['bullish'])

    score = 0.0
    if primary_vote['entry_signal_today']:
        score += 3.0
    elif primary_vote['exit_signal_today']:
        score -= 3.0
    elif primary_vote['bullish']:
        score += 1.5
    else:
        score -= 1.5

    score += 0.5 * bullish_count
    score += 0.4 * float(analyst['heuristic_score'])
    mom_pct = float(market['momentum_20_pct'])
    score += max(-1.5, min(1.5, mom_pct / 10.0))
    if market['passes_volume_filter']:
        score += 0.3

    return {
        'scan_score': round(score, 2),
        'market': market,
        'analyst': analyst,
        'primary_stance': primary_vote['stance'],
        'entry_signal_today': primary_vote['entry_signal_today'],
        'exit_signal_today': primary_vote['exit_signal_today'],
        'bullish_strategies': bullish_count,
        'total_strategies': len(ACTIVE_STRATEGY_KEYS),
    }


def scan_universe_technicals(
    tickers: list[str],
    bankroll: float,
    policy: RiskPolicy,
    primary_strategy: str = 'sma_crossover',
    scan_period: str = '6mo',
    chunk_size: int = 100,
    max_annual_volatility_pct: float = 85.0,
    min_volume_ratio: float = 0.30,
    available_cash: float | None = None,
    open_tickers: set[str] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Batch-download OHLCV across tickers, filter by capital/risk eligibility, and rank."""
    if bankroll <= 0:
        raise ValueError('Bankroll must be positive')
    if not tickers:
        raise ValueError('Ticker universe must not be empty')

    cash_budget = max(0.0, float(available_cash)) if available_cash is not None else float(bankroll)
    held_set = {t.upper() for t in (open_tickers or set())}
    valid_data_count = 0
    affordable_count = 0
    eligible_rows = []
    total = len(tickers)

    for start_idx in range(0, total, chunk_size):
        chunk = tickers[start_idx : start_idx + chunk_size]
        done_count = min(start_idx + len(chunk), total)
        if progress_callback:
            progress_callback(start_idx, total, f'Scanning market bars ({start_idx + 1}–{done_count} of {total})...')

        raw_batch = yf.download(
            tickers=chunk,
            period=scan_period,
            interval='1d',
            group_by='ticker',
            auto_adjust=True,
            threads=True,
            progress=False,
            multi_level_index=True,
        )

        for sym in chunk:
            sub = extract_symbol_bars(raw_batch, sym, chunk)
            if sub is None:
                continue
            try:
                bars = validate_bars(sub)
            except ValueError:
                continue
            valid_data_count += 1

            tech = score_candidate_technicals(bars, primary_strategy=primary_strategy)
            mkt = tech['market']
            price = float(mkt['price'])
            sized = size_position(price=price, cash=cash_budget, equity=bankroll, policy=policy)
            if sized['shares'] >= 1:
                affordable_count += 1
            else:
                continue

            if sym.upper() in held_set:
                continue
            if mkt['volatility_20_pct'] > max_annual_volatility_pct:
                continue
            if mkt['volume_ratio'] < min_volume_ratio:
                continue
            if tech['exit_signal_today']:
                continue

            eligible_rows.append({
                'Ticker': sym,
                'Scan Score': tech['scan_score'],
                'Close ($)': price,
                'Max Shares': sized['shares'],
                'Max Notional ($)': sized['notional'],
                'Trend': mkt['trend'],
                'Primary Stance': tech['primary_stance'],
                'Strategy Consensus': f"{tech['bullish_strategies']}/{tech['total_strategies']}",
                '20d Momentum (%)': mkt['momentum_20_pct'],
                '20d Volatility (%)': mkt['volatility_20_pct'],
                'Volume Ratio': mkt['volume_ratio'],
            })

    if progress_callback:
        progress_callback(total, total, f'Completed market scan of {total} stocks.')

    if not eligible_rows:
        leaderboard_df = pd.DataFrame(columns=[
            'Ticker', 'Scan Score', 'Close ($)', 'Max Shares', 'Max Notional ($)',
            'Trend', 'Primary Stance', 'Strategy Consensus', '20d Momentum (%)',
            '20d Volatility (%)', 'Volume Ratio',
        ])
    else:
        leaderboard_df = (
            pd.DataFrame(eligible_rows)
            .sort_values(by=['Scan Score', '20d Momentum (%)'], ascending=[False, False])
            .reset_index(drop=True)
        )

    return {
        'universe_size': total,
        'valid_data_count': valid_data_count,
        'affordable_count': affordable_count,
        'eligible_count': len(leaderboard_df),
        'leaderboard_df': leaderboard_df,
    }


def scan_market_and_suggest(
    scope: str = 'sp100',
    bankroll: float = 10000.0,
    policy: RiskPolicy | None = None,
    primary_strategy: str = 'sma_crossover',
    top_k: int = 5,
    history_period: str = '2y',
    llm_config: LLMConfig | None = None,
    tickers_override: list[str] | None = None,
    db_path: Path | None = None,
    log_to_db: bool = True,
    available_cash: float | None = None,
    open_tickers: set[str] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """End-to-end two-stage scanner: batch screen universe, then run 6-agent pipeline on Top K."""
    if top_k < 1:
        raise ValueError('top_k must be >= 1')
    active_policy = policy or RiskPolicy()
    tickers = tickers_override if tickers_override is not None else get_universe_tickers(scope=scope)
    held_set = {t.upper() for t in (open_tickers or set())}

    stage1 = scan_universe_technicals(
        tickers=tickers,
        bankroll=bankroll,
        policy=active_policy,
        primary_strategy=primary_strategy,
        scan_period='6mo',
        available_cash=available_cash,
        open_tickers=held_set,
        progress_callback=progress_callback,
    )

    leaderboard_df = stage1['leaderboard_df']
    if leaderboard_df.empty:
        return {
            'scope': scope,
            'primary_strategy': STRATEGIES.get(primary_strategy, STRATEGIES['sma_crossover']).name,
            **stage1,
            'shortlist_summary_df': pd.DataFrame(),
            'candidates': [],
            'passed_count': 0,
            'dropped_count': 0,
        }

    top_rows = leaderboard_df.head(top_k).to_dict(orient='records')
    candidates = []
    for idx, row in enumerate(top_rows, start=1):
        sym = row['Ticker']
        if progress_callback:
            progress_callback(idx, len(top_rows), f'Running 6-agent deep evaluation on {sym} ({idx}/{len(top_rows)})...')
        try:
            data, wf_res = run_suggestion_workflow(
                ticker=sym,
                bankroll=bankroll,
                policy=active_policy,
                strategy=primary_strategy,
                period=history_period,
                source='yahoo',
                llm_config=llm_config,
                db_path=db_path,
                log_to_db=log_to_db,
                available_cash=available_cash,
                already_held=(sym.upper() in held_set),
            )
            candidates.append({
                'ticker': sym,
                'scan_score': row['Scan Score'],
                'data': data,
                'workflow': wf_res,
            })
        except Exception:
            continue

    # Rank Risk-Gate PASSED candidates first by conviction & scan_score, then DROPPED candidates
    candidates.sort(
        key=lambda item: (
            1 if item['workflow']['risk']['passed'] else 0,
            item['workflow']['strategy']['conviction_score'],
            item['scan_score'],
        ),
        reverse=True,
    )

    summary_records = []
    passed_count = 0
    dropped_count = 0
    for c in candidates:
        wf = c['workflow']
        passed = bool(wf['risk']['passed'])
        if passed:
            passed_count += 1
        else:
            dropped_count += 1
        summary_records.append({
            'Ticker': c['ticker'],
            'Risk Gate': 'PASSED' if passed else 'DROPPED',
            'Close ($)': wf['market']['price'],
            'Max Shares': wf['risk']['shares'],
            'Notional ($)': wf['risk']['notional'],
            'Stop ($)': wf['risk']['stop_price'],
            'Conviction': f"{wf['strategy']['conviction_score']:+d}/5",
            'Research Sentiment': wf['research']['sentiment'],
            'Monitor Status': wf['monitor']['status'],
            'Scan Score': c['scan_score'],
        })

    return {
        'scope': scope,
        'primary_strategy': STRATEGIES.get(primary_strategy, STRATEGIES['sma_crossover']).name,
        **stage1,
        'shortlist_summary_df': pd.DataFrame(summary_records),
        'candidates': candidates,
        'passed_count': passed_count,
        'dropped_count': dropped_count,
    }

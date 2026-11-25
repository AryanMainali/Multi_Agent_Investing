# System architecture and evaluation

User onboarding (paper trading capital & risk tolerance) -> Ticker selection (Yahoo Finance real data) -> **Data Agent** (historical bars & indicators) -> **Market Agent** (state and volume) -> **Analyst Agent** (rule-based decision score) -> **Risk Manager** (hard constraints scaled to user risk profile) -> **Human checkpoint** (approve/reject paper suggestion) -> **SQLite audit**.

The **Backtesting Engine** replays the analyst's well-defined 20/50 SMA crossover rule against *past only* bars. The dashboard shows its simulated results next to human-reviewed suggestions. The approval UI intentionally does not trigger or rewrite the backtest.

## Architecture mapping

| Component | Agent architecture | Starter implementation | Later expansion |
|---|---|---|---|
| Market | Model-based reflex | Current/previous trend, crossover state, volume filter | Multi-ticker watchlist and persistent market state |
| Data | Goal-based | Source selection + data validation + indicator computation | SEC financials, extra providers, data freshness planning |
| Analyst | Utility-based with learning *planned* | Transparent heuristic score and SMA signals | Train/test probabilistic ranker; measure calibration |
| Risk | Simple reflex/rule-based | Whole-share sizing, max allocation, risk budget, stop level | Portfolio caps, drawdown and daily stop rules |

The starter does **not** pretend to implement a trained agent. It provides separate architectural modules that are easy to explain and extend.

## Four-week plan

- Week 1: data model, offline reference dataset, risk constraints, interface contracts.
- Week 2: agents, market signals, deterministic tests, multi-ticker exploration if time.
- Week 3: Streamlit UI, SQLite audit, historical backtester and metrics.
- Week 4: out-of-sample evaluation, integrated testing, report and demo.

## Academic integrity of backtests

Never feed a feature computed from future prices into historical decisions. Use the next day's opening price after a signal is known; keep training and evaluation periods separate if you add model fitting. A favorable return on a synthetic demo is **not evidence** of an investable edge. Compare risk-adjusted results, drawdown, trade frequency, and data handling. Do not add automatic trading to this class project.

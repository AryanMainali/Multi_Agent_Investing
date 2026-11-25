# Multi-Agent Stock Analysis & Backtesting — Starter

An educational, **paper-only** Python project adapted from a four-agent investing workflow. Uses **real historical market data** from Yahoo Finance via `yfinance` (offline synthetic data is disabled) and asks the user for their **paper trading capital** and **risk tolerance** on startup. **No brokers, credentials, order endpoints, or real trades.**

## Quickstart (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py               # interactive dashboard (prompts for paper capital & risk on start)
python run_demo.py                 # CLI runner (prompts for paper capital & risk, fetches Yahoo Finance)
pytest -q                          # unit & UI tests
```

On Windows PowerShell, activate using `.venv\Scripts\Activate.ps1`; on Windows cmd, use `.venv\Scripts\activate.bat`.

You can also pass startup parameters directly on the CLI:

```bash
python run_demo.py --ticker AAPL --bankroll 10000 --risk medium
```

The app exclusively uses **Yahoo Finance** real historical bars (requires an internet connection). If the feed is unavailable or a symbol is invalid, the app displays an error; it **never substitutes synthetic or fake prices**.

## Structure

- `agents/market_agent.py`: model-based reflex; tracks trend state, crossovers, volume relative to average.
- `agents/data_agent.py`: goal-based data gathering from Yahoo Finance and indicator preparation.
- `agents/analyst_agent.py`: transparent, **rule-based** utility score; ML/learning is a future phase, not implemented.
- `agents/risk_agent.py`: deterministic caps on allocation, per-trade nominal loss, and stop-loss based on user risk tolerance (`Low`, `Medium`, `High`, or `Custom`).
- `core/workflow.py`: sequential orchestration; optional LangGraph extension later.
- `core/strategies.py`: past-only signal generators for 7 strategies (`SMA 20/50 Crossover`, `EMA 12/26 Crossover`, `MACD (12/26/9) Signal`, `RSI (14) Mean Reversion`, `Bollinger Bands (20, 2σ)`, `Donchian 20/10 Breakout`, and `Buy & Hold`).
- `core/backtest.py`: daily-bar multi-strategy backtester and side-by-side comparison engine with next-open fills, stop approximation, fees, and slippage.
- `core/storage.py`: local SQLite log of explicit human **paper** decisions, not broker orders.
- `tests/`: unit and Streamlit onboarding tests.
- `docs/ARCHITECTURE.md`: limitations and development roadmap.

## Backtest assumptions (important)

- Indicators are calculated from information through the **end of each day**.
- A crossover at day T close is executed at **day T+1 open**, not day T close.
- Applies a 10-basis-point fee per side and 5-basis-point modeled slippage per side by default (illustrative, not a broker quote).
- Uses whole shares, up to 20% of portfolio equity allocated to a trade, and a nominal 1% equity risk budget based on a 5% protective stop.
- Stop filled at trigger or opening price if gapped below, with modeled sell slippage; daily bars cannot reproduce intraday execution perfectly.
- Holdings still open at the last date are marked to market, not falsely counted as closed winning trades.
- Buy-and-hold comparison is a simple *price change* reference that excludes costs and may be affected by adjusted price conventions, so it is not fully apples-to-apples.
- No survivorship-bias controls, no walk-forward optimization, no ML forecasting, no multi-asset portfolio engine, and no guarantee of profit.

## Proposed three-member split

1. Market/Data Agent, data source, indicator calculations.
2. Analyst/Risk Agents, backtest evaluation and tests.
3. Workflow, dashboard, local preferences/decisions log and demo.

## Next upgrades

1. Add a defined universe of 5-10 symbols and repeated historical snapshots.
2. Add separate train/test or walk-forward evaluations without look-ahead leakage.
3. Document data limitations, costs, missing bars, splits, and benchmark assumptions.
4. Only if required for the course, introduce LangGraph human-interrupt and persisted state **after** MVP tests pass.

## Data-source notice

`yfinance` is an unofficial Yahoo Finance research interface and may fail or be rate-limited; consult the project's notice and applicable Yahoo terms: https://github.com/ranaroussi/yfinance . Offline synthetic data has been removed so all analyses run exclusively on real historical market bars.

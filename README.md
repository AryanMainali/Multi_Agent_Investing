# Multi-Agent Stock Suggestion & Backtesting Lab

An educational, **suggestion-only** multi-agent stock discovery, walk-forward ML, and backtesting system built exclusively on real market data. **Agents only suggest — humans decide and trade manually.** There are **no** brokerage order-execution integrations anywhere in this codebase (`tests/test_no_order_code.py` enforces this at test time).

> *Not financial advice. This is a research and learning project.*

## Architecture & Six-Agent Pipeline

```text
[S&P 100 / S&P 500 / Watchlist]
             │
             ▼
[1. Market Data Batch Scanner] ──┐  (filters affordable whole-share fits & ranks diversified Top 5)
                                 ├──► [3. Strategy + Walk-Forward ML] ──► [4. Risk Gate]
[2. Research (AI) + Event Flags] ┘                                               │
                                                                         ┌───────┴───────┐
                                                                       (No)            (Yes)
                                                                         ▼               ▼
                                                               [Drop & Log Reason]  [5. Report (AI)] ──► Human Decision
                                                                         │               │
                                                                         └───────┬───────┘
                                                                                 ▼
                                                                        [6. Monitor Agent]
                                                       (audits stages, heartbeats & retrospective scorecard)
```

## Quickstart (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # optional: add LLM, read-only Alpaca news, or Telegram alert keys
streamlit run app.py               # interactive dashboard (asks only for paper capital & risk, then scans S&P)
python run_demo.py                 # CLI runner: scans S&P 100 for Top 5 diversified candidates
pytest -q                          # unit, scanner, workflow, ML, scorecard, no-order guardrail, and UI tests
```

## CLI Modes (`run_demo.py`)

| Command | What it does |
|---|---|
| `python run_demo.py` | Scan S&P 100 for Top 5 sector-diversified fits, run 6-agent pipeline & 8-strategy backtest |
| `python run_demo.py --scan sp500` | Run full S&P 500 universe discovery scan |
| `python run_demo.py --mode watchlist` | Evaluate the team watchlist configured in `config/settings.yaml` |
| `python run_demo.py --ticker AAPL` | Run the 6-agent suggestion workflow & backtester on a single symbol |
| `python run_demo.py --mode monitor` | Check agent heartbeats, score past approved & rejected ideas, and write daily summary |
| `python run_demo.py --mode schedule` | Keep running: pipeline weekdays at 16:45 ET, Monitor health & scorecard hourly |

## Key Capabilities

- **Team Settings (`config/settings.yaml` & `.env.example`)**: Shared team-edited configuration for watchlist symbols, ML forecast horizon, walk-forward accuracy windows, risk thresholds, high-severity news blocks, and Monitor staleness alerts.
- **Autonomous S&P 100 / S&P 500 Scanner (`core/universe.py`, `core/scanner.py`)**: Two-stage batch OHLCV & capital-aware market scanner enforcing whole-share affordability, remaining portfolio heat, and sector diversification.
- **Structured News Event Flagging (`agents/research_agent.py`, `core/llm.py`)**: Reads Yahoo Finance (plus optional read-only Alpaca News) headlines and flags events (`earnings`, `guidance`, `legal`, `regulatory`, `merger`, `analyst_rating`, `product`, `macro`) with `low`/`medium`/`high` severity via Claude tool-schema / OpenAI / Gemini or built-in keyword rules.
- **Multi-Strategy + Walk-Forward ML (`agents/strategy_agent.py`, `core/ml_model.py`, `core/strategies.py`)**: Combines 8 strategies (`SMA 20/50`, `EMA 12/26`, `MACD`, `RSI 14`, `Bollinger Bands`, `Donchian 20/10`, `ML Walk-Forward Logistic (5d)`, and `Buy & Hold`) with embargoed out-of-sample accuracy tracking.
- **Deterministic Risk Gate (`agents/risk_agent.py`)**: Hard caps on position allocation, per-trade stop-loss risk, total portfolio heat, sector concentration, volatility, liquidity, and automatic blocking on high-severity adverse news events.
- **Plain-English Notes & `"What Would Prove It Wrong"` (`agents/report_agent.py`)**: Writes candid suggestion briefs (`outputs/reports/ideas/`) only when Risk Gate passes.
- **Monitor Heartbeats & Retrospective Scorecard (`agents/monitor_agent.py`, `core/storage.py`)**: Tracks per-agent heartbeats, scores **both Approved and Risk-Rejected ideas** once their `horizon_days` pass so you can test whether Risk is filtering well against a 50% coin-flip baseline, writes `outputs/reports/daily/`, and supports optional Telegram alerts.
- **Live Paper Portfolio & `SPY` Benchmark (`core/portfolio.py`, `core/exports.py`)**: Manual paper position ledger, advisory stop/target/exit alerts, `SPY` alpha equity curve comparison, and one-click Markdown/CSV exports.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full architecture details and the team's next-steps roadmap.

## Data-Source Notice

`yfinance` is an unofficial Yahoo Finance research interface and may fail or be rate-limited; consult [ranaroussi/yfinance](https://github.com/ranaroussi/yfinance). Offline synthetic price generation is disabled so all analyses run exclusively on real historical market data.

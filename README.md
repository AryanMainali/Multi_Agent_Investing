# Multi-Agent Stock Suggestion & Backtesting Lab

An educational, **suggestion-only** multi-agent stock discovery and backtesting system built on real Yahoo Finance market data. **Agents only suggest — humans decide and trade manually.** There are **no** brokerage integrations, credentials, or automated order-execution calls anywhere in this codebase.

## Architecture & Data Flow

```text
[S&P 100 / S&P 500 Universe]
             │
             ▼
[1. Market Data Batch Scanner] ──┐  (filters affordable whole-share fits & ranks Top 5)
                                 ├──► [3. Strategy] ──► [4. Risk Gate]
[2. Research (AI) News & Info] ──┘                         │
                                                   ┌───────┴───────┐
                                                 (No)            (Yes)
                                                   ▼               ▼
                                         [Drop & Log Reason]  [5. Report (AI)] ──► Human Decision
                                                                   ▲
                                                                   │ (checks every idea & agent)
                                                          [6. Monitor Agent]
```

## Quickstart (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py               # interactive dashboard (asks only for paper capital & risk, then scans S&P)
python run_demo.py                 # CLI runner (scans S&P 100 for Top 5 fits, or --scan sp500 / --ticker AAPL)
pytest -q                          # unit, scanner, workflow, monitor, and Streamlit UI tests
```

Optional AI keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`) can be set in your environment or entered in the Streamlit sidebar for LLM-backed `[2. Research (AI)]` and `[5. Report (AI)]` generation. When no key is configured, both stages automatically use deterministic plain-English synthesis over real Yahoo Finance headlines, fundamentals, and technicals.

## Structure

- `core/universe.py` & `core/scanner.py`: S&P 100 (fast default) and full S&P 500 ticker universes + two-stage batch OHLCV & capital-aware market scanner surfacing the Top 5 fits.
- `agents/data_agent.py` & `agents/market_agent.py` (**1. Market Data**): real Yahoo Finance OHLCV validation, technical indicators, trend, volume, and volatility state.
- `agents/research_agent.py` (**2. Research (AI)**): Yahoo Finance news & company fundamentals gatherer with LLM / rule-assisted sentiment, catalysts, and risk synthesis.
- `agents/strategy_agent.py` & `agents/analyst_agent.py` (**3. Strategy**): combines Market Data + AI Research + multi-strategy signals into a scored candidate (`SUGGEST_LONG` or `AVOID_OR_WAIT`).
- `agents/risk_agent.py` (**4. Risk Gate**): deterministic hard caps on position allocation, per-trade nominal loss, whole shares, volatility, and liquidity. Rejects failing ideas to `dropped_ideas` in SQLite.
- `agents/report_agent.py` (**5. Report (AI)**): generates a plain-English suggestion brief (only when Risk Gate passes) for manual human review.
- `agents/monitor_agent.py` (**6. Monitor Agent**): audits every upstream agent, verifies risk-cap invariants, flags cross-agent divergences or stale bars, and enforces manual-only report compliance.
- `core/strategies.py` & `core/backtest.py`: 7 past-only strategies (`SMA 20/50`, `EMA 12/26`, `MACD`, `RSI 14`, `Bollinger Bands`, `Donchian 20/10`, `Buy & Hold`) and side-by-side historical backtester.
- `core/portfolio.py` & `core/storage.py`: live paper portfolio tracker (available cash, unrealized/realized P&L, and advisory stop/target/exit alerts) plus SQLite audit trail (`outputs/decisions.db`) for `paper_positions`, `decisions`, `dropped_ideas`, and `workflow_audits`.

## Data-source notice

`yfinance` is an unofficial Yahoo Finance research interface and may fail or be rate-limited; consult https://github.com/ranaroussi/yfinance . Offline synthetic data is disabled so all analyses run exclusively on real historical market data.

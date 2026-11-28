# System Architecture, Safety Rules & Project Roadmap

## Non-Negotiable Core Rule

**Agents only suggest. Humans decide and trade manually.**
- No broker order-placing client, execution endpoint, or automated trading call exists anywhere in this repository (`tests/test_no_order_code.py` fails the build immediately if order-submission code is ever introduced).
- Failures default to **"no idea"**: missing/stale bars, insufficient share affordability, portfolio heat caps, sector concentration caps, or high-severity adverse news events automatically trigger rejection before any report is written.
- LLMs only read and write text (event-flag extraction, research synthesis, and plain-English reporting). Directional statistics come from traditional rule-based & walk-forward ML models, and all risk limits live in deterministic plain code.
- Every rejection is saved with its exact reason, and **rejected ideas are scored retrospectively alongside approved ideas** so the team can verify empirically whether `[4. Risk Gate]` is filtering well.

---

## Six-Agent Pipeline & Shared Database Flow

```text
[S&P 100 / S&P 500 / Watchlist]
             │
             ▼
[1. Market Data] ──┐
                   ├──► [3. Strategy (Rules + Walk-Forward ML)] ──► [4. Risk Gate]
[2. Research (AI)] ┘                                                      │
                                                                  ┌───────┴───────┐
                                                                (No)            (Yes)
                                                                  ▼               ▼
                                                        [Drop & Log Reason]  [5. Report (AI)] ──► Human Decision
                                                                  │               │
                                                                  └───────┬───────┘
                                                                          ▼
                                                                 [6. Monitor Agent]
                                          (audits stages, checks heartbeats, scores past approved & rejected ideas)
```

Shared SQLite tables (`outputs/decisions.db`):
```text
research_flags + market bars  ->  proposals  ->  dropped_ideas / workflow_audits / idea notes
                              Monitor reads heartbeats & proposals, writes idea_scores & daily summaries
```

| Stage | Module | Technical Type | Responsibility |
|---|---|---|---|
| **1. Market Data** | `agents/data_agent.py`, `agents/market_agent.py`, `core/scanner.py` | Scripted pipeline (no AI) | Fetch & validate real Yahoo Finance daily OHLCV bars; compute past-only moving averages, momentum, volume ratio, and annualized volatility. |
| **2. Research (AI)** | `agents/research_agent.py`, `core/llm.py` | LLM NLP (tool JSON schema) + keyword fallback | Read real Yahoo Finance (and optional read-only Alpaca) headlines & fundamentals; flag structured events (`earnings`, `guidance`, `legal`, `regulatory`, `merger`, `analyst_rating`, `product`, `macro`) with `low`/`medium`/`high` severity; synthesize sentiment, catalysts, and risks. |
| **3. Strategy** | `agents/strategy_agent.py`, `core/ml_model.py`, `core/strategies.py` | Multi-strategy rules + Walk-Forward Logistic Regression ML | Combine 7 active technical/ML strategies (`SMA`, `EMA`, `MACD`, `RSI`, `Bollinger`, `Donchian`, `ML Walk-Forward Logistic`) and measure held-out `backtest_accuracy` using an `N`-day embargo gap so training labels never peek into the test window. |
| **4. Risk Gate** | `agents/risk_agent.py`, `config/settings.yaml` | Deterministic plain code | Enforce whole-share allocation caps, per-trade stop-loss risk, total open portfolio heat, sector concentration caps, volatility/liquidity bounds, and **high-severity adverse news blocks**. Never overridden by AI. |
| **5. Report (AI)** | `agents/report_agent.py` | LLM text generation + deterministic template | Runs **only** when Risk Gate passes; writes a plain-English note (`outputs/reports/ideas/`) covering the thesis, position plan, risks, candid model uncertainty, and **"What Would Prove It Wrong"**. |
| **6. Monitor Agent** | `agents/monitor_agent.py`, `core/storage.py` | Rule-based auditor + scorecard + optional LLM/Telegram | Audits every stage, checks agent `heartbeats`, **scores past proposals (both Approved and Risk-Rejected)** once their `horizon_days` elapse, writes `outputs/reports/daily/summary_<date>.md`, and optionally sends Telegram health alerts. |

---

## Shared Team Configuration (`config/settings.yaml` & `.env`)

- **`config/settings.yaml`**: Team-reviewed defaults for watchlist symbols, ML `horizon_days`, walk-forward `min_history` / `test_window`, minimum ML confidence/accuracy floors, data freshness limits, high-severity news lookback, fractional Kelly reference, and Monitor heartbeat staleness thresholds. Agents only read this file; update it together via pull request.
- **`.env` (from `.env.example`)**: Optional keys for `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (`CLAUDE_MODEL`), `GEMINI_API_KEY`, read-only `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` news enrichment, and `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` Monitor notifications.

---

## Academic Integrity of Backtests & Walk-Forward ML

1. **Next-Bar Open Execution**: Signals are computed strictly from end-of-day bars through day `T`; simulated fills execute no earlier than day `T+1` open with 10 bps fees, 5 bps slippage, whole-share constraints, and protective stops.
2. **Walk-Forward Embargo**: When fitting the 5-day forward Logistic Regression classifier at bar `T`, only rows at or before `T - 5` have settled forward labels, preventing any future price leakage into training or held-out accuracy measurement.

---

## Project Roadmap & Next Steps

1. **Richer Fundamental & Calendar Context**: Expand `[2. Research]` with upcoming earnings announcement dates, ex-dividend dates, insider filings, and macro economic calendar releases (CPI/FOMC) so `[4. Risk Gate]` can optionally block entries immediately ahead of binary earnings gaps.
2. **Gradient Boosting & Multi-Window Walk-Forward Models**: Extend `core/ml_model.py` beyond baseline L2 Logistic Regression to regularized gradient-boosted trees (e.g., LightGBM / XGBoost) evaluated across rolling multi-year walk-forward windows and market regimes.
3. **Honest Multi-Week Scorecard Evaluation**: Run the `[6. Monitor Agent]` retrospective scorecard across several weeks of live market closes. Compare `Approved by Risk Gate` vs. `Rejected by Risk Gate` hit rates and average 5-day moves — a hit rate near **50%** indicates no statistical edge yet regardless of any single winning idea.
4. **Shared Team Database (PostgreSQL Option)**: When moving from single-machine SQLite (`outputs/decisions.db`) to a shared team research server, point storage at a shared PostgreSQL instance so scheduled runs and the Streamlit dashboard read the same live tables.
5. **Isolated Per-Agent Deployment**: Deploy scheduled post-close weekday runs (`python run_demo.py --mode schedule` at 16:45 ET + hourly Monitor checks) as lightweight isolated containers on a small cloud VM.

# System Architecture & Evaluation

## Non-Negotiable Core Rule

**Agents only suggest. Humans decide and trade manually.**
- No broker credentials, order endpoints, or automated execution exist in this repository.
- Every passing idea terminates in a plain-English report for a human to review and either log a paper approval or skip.

## Six-Agent Pipeline

```text
[1. Market Data] ──┐
                   ├──► [3. Strategy] ──► [4. Risk Gate]
[2. Research (AI)] ┘                         │
                                     ┌───────┴───────┐
                                   (No)            (Yes)
                                     ▼               ▼
                           [Drop & Log Reason]  [5. Report (AI)] ──► Human Decision
                                                     ▲
                                                     │ (checks every idea & agent)
                                            [6. Monitor Agent]
```

| Stage | Module | Responsibility |
|---|---|---|
| 1. Market Data | `agents/data_agent.py`, `agents/market_agent.py` | Fetch & validate real Yahoo Finance daily OHLCV bars; compute past-only moving averages, momentum, volume ratio, and annualized volatility. |
| 2. Research (AI) | `agents/research_agent.py`, `core/llm.py` | Gather recent Yahoo Finance headlines and company fundamentals; synthesize sentiment (`BULLISH` / `NEUTRAL` / `BEARISH`), catalysts, and risks via configured LLM or deterministic fallback. |
| 3. Strategy | `agents/strategy_agent.py`, `core/strategies.py` | Combine Market Data and AI Research with active strategy signals (`SMA`, `EMA`, `MACD`, `RSI`, `Bollinger`, `Donchian`) into a scored candidate (`SUGGEST_LONG` or `AVOID_OR_WAIT`). |
| 4. Risk Gate | `agents/risk_agent.py` | Deterministic hard caps on whole-share allocation, nominal stop-loss budget, annualized volatility ceiling, liquidity floor, and severe bearish headwinds. Never overridden by AI. |
| Drop & Log | `core/storage.py` (`dropped_ideas`) | When Risk Gate returns `No`, drops the idea before report generation and logs exact rejection reasons to SQLite. |
| 5. Report (AI) | `agents/report_agent.py` | When Risk Gate returns `Yes`, produces a plain-English suggestion brief detailing rationale, exact share cap, protective stop price, nominal risk, and risks. |
| 6. Monitor Agent | `agents/monitor_agent.py`, `core/storage.py` (`workflow_audits`) | Audits every upstream stage for data staleness, cross-agent sentiment/signal divergence, mathematical risk invariants, and manual-only report compliance. |

## Academic Integrity of Backtests

Signals are computed strictly from end-of-day bars through day `T` and simulated fills occur no earlier than day `T+1` open, incorporating estimated fees, slippage, whole-share constraints, and protective stops.

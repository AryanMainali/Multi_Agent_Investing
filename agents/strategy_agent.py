"""Stage 3 [Strategy]: Synthesizes [1. Market Data], [2. Research (AI)], multi-strategy signals,
and a walk-forward Logistic Regression direction model into a structured trade candidate.
"""
from dataclasses import dataclass
import pandas as pd
from agents.analyst_agent import AnalystAgent
from core.config import load_settings
from core.ml_model import predict_walk_forward
from core.strategies import STRATEGIES, _compute_rsi, generate_signals

ACTIVE_STRATEGY_KEYS = [k for k, spec in STRATEGIES.items() if spec.uses_risk_policy]


def _evaluate_strategy_stance(prices: pd.DataFrame, key: str) -> dict:
    df, spec = generate_signals(prices, strategy=key)
    last = df.iloc[-1]
    entry_now = bool(last['ENTRY_SIGNAL'])
    exit_now = bool(last['EXIT_SIGNAL'])

    if entry_now:
        stance = 'BULLISH_SIGNAL'
    elif exit_now:
        stance = 'BEARISH_SIGNAL'
    elif key == 'sma_crossover':
        stance = 'BULLISH_TREND' if last['SMA_FAST'] > last['SMA_SLOW'] else 'BEARISH_TREND'
    elif key == 'ema_crossover':
        ema12 = prices['Close'].ewm(span=12, min_periods=12, adjust=False).mean().iloc[-1]
        ema26 = prices['Close'].ewm(span=26, min_periods=26, adjust=False).mean().iloc[-1]
        stance = 'BULLISH_TREND' if ema12 > ema26 else 'BEARISH_TREND'
    elif key == 'macd_crossover':
        ema12 = prices['Close'].ewm(span=12, min_periods=12, adjust=False).mean()
        ema26 = prices['Close'].ewm(span=26, min_periods=26, adjust=False).mean()
        macd = ema12 - ema26
        sig = macd.ewm(span=9, min_periods=9, adjust=False).mean()
        stance = 'BULLISH_TREND' if macd.iloc[-1] > sig.iloc[-1] else 'BEARISH_TREND'
    elif key == 'rsi_mean_reversion':
        rsi_val = float(_compute_rsi(prices['Close'], 14).iloc[-1])
        if rsi_val < 40:
            stance = 'BULLISH_OVERSOLD_ZONE'
        elif rsi_val > 68:
            stance = 'BEARISH_OVERBOUGHT_ZONE'
        else:
            stance = 'NEUTRAL'
    elif key == 'bollinger_reversion':
        mid = float(prices['Close'].rolling(20, min_periods=20).mean().iloc[-1])
        stance = 'BULLISH_TREND' if float(last['Close']) >= mid else 'NEUTRAL_PULLBACK'
    elif key == 'donchian_breakout':
        mid_20 = float(prices['Close'].rolling(20, min_periods=20).mean().iloc[-1])
        stance = 'BULLISH_TREND' if float(last['Close']) >= mid_20 else 'BEARISH_TREND'
    elif key == 'ml_logistic_walkforward':
        p_val = last.get('ML_P_UP')
        if pd.notna(p_val) and float(p_val) >= 0.52:
            stance = 'BULLISH_ML_FORECAST'
        elif pd.notna(p_val) and float(p_val) <= 0.48:
            stance = 'BEARISH_ML_FORECAST'
        else:
            stance = 'NEUTRAL'
    else:
        stance = 'NEUTRAL'

    return {
        'strategy_key': key,
        'strategy_name': spec.name,
        'entry_signal_today': entry_now,
        'exit_signal_today': exit_now,
        'stance': stance,
        'bullish': stance.startswith('BULLISH'),
    }


@dataclass
class StrategyAgent:
    primary_strategy: str = 'sma_crossover'

    def formulate(self, df: pd.DataFrame, market: dict, research: dict) -> dict:
        key = self.primary_strategy if self.primary_strategy in ACTIVE_STRATEGY_KEYS else 'sma_crossover'
        spec = STRATEGIES[key]
        analyst = AnalystAgent().evaluate(df, market)

        strat_cfg = load_settings().get('strategy', {})
        horizon_days = int(strat_cfg.get('horizon_days', 5))
        ml_forecast = predict_walk_forward(
            df,
            horizon_days=horizon_days,
            min_history=int(strat_cfg.get('min_history', 90)),
            test_window=int(strat_cfg.get('test_window', 40)),
        )

        votes = {k: _evaluate_strategy_stance(df, k) for k in ACTIVE_STRATEGY_KEYS}
        primary_vote = votes[key]
        bullish_count = sum(1 for v in votes.values() if v['bullish'])

        conviction = 0
        rationale = []

        # 1. Primary strategy stance & fresh signal
        if primary_vote['entry_signal_today']:
            conviction += 2
            rationale.append(f"{spec.name} triggered a fresh entry signal on latest close ({spec.entry_reason}).")
        elif primary_vote['exit_signal_today']:
            conviction -= 2
            rationale.append(f"{spec.name} triggered an exit signal on latest close ({spec.exit_reason}).")
        elif primary_vote['bullish']:
            conviction += 1
            rationale.append(f"{spec.name} remains in a constructive bullish regime ({primary_vote['stance']}).")
        else:
            conviction -= 1
            rationale.append(f"{spec.name} is currently in a non-bullish / defensive posture ({primary_vote['stance']}).")

        # 2. Broad market/analyst technical score
        h_score = int(analyst['heuristic_score'])
        if h_score >= 2:
            conviction += 1
            rationale.append(f"Technical screening score is supportive ({h_score:+d}; trend={market['trend']}).")
        elif h_score <= -2:
            conviction -= 1
            rationale.append(f"Technical screening score is weak ({h_score:+d}; trend={market['trend']}).")

        # 3. Multi-strategy agreement across all active strategies
        majority_threshold = max(4, (len(ACTIVE_STRATEGY_KEYS) + 1) // 2)
        if bullish_count >= majority_threshold:
            conviction += 1
            rationale.append(f"{bullish_count}/{len(ACTIVE_STRATEGY_KEYS)} active strategies align constructively.")
        elif bullish_count <= 1:
            conviction -= 1
            rationale.append(f"Only {bullish_count}/{len(ACTIVE_STRATEGY_KEYS)} active strategies show bullish posture.")

        # 4. Walk-forward ML direction model (from team Strategy design)
        if ml_forecast is not None:
            p_up = float(ml_forecast['p_up'])
            acc = float(ml_forecast['backtest_accuracy'])
            if p_up >= 0.56 and acc >= 0.52:
                conviction += 1
            elif p_up <= 0.42 and acc >= 0.52:
                conviction -= 1
            rationale.append(
                f"Walk-forward ML ({horizon_days}d logistic model): P(up)={p_up:.0%}, "
                f"held-out accuracy={acc:.0%} over last {ml_forecast['test_window']} settled bars."
            )

        # 5. AI Research sentiment contribution
        sentiment = str(research.get('sentiment', 'NEUTRAL')).upper()
        if sentiment == 'BULLISH':
            conviction += 1
            rationale.append(f"Research (AI) sentiment is BULLISH ({research.get('sentiment_score', 0):+.2f}).")
        elif sentiment == 'BEARISH':
            conviction -= 1
            rationale.append(f"Research (AI) sentiment is BEARISH ({research.get('sentiment_score', 0):+.2f}).")
        else:
            rationale.append('Research (AI) sentiment is NEUTRAL; technicals carry primary weight.')

        conviction = max(-5, min(5, conviction))
        if conviction >= 1 and not primary_vote['exit_signal_today']:
            action = 'SUGGEST_LONG'
            direction = 'LONG'
            summary = f"Constructive paper long candidate via {spec.name} (conviction {conviction:+d}/5)."
        else:
            action = 'AVOID_OR_WAIT'
            direction = 'FLAT'
            summary = f"No long suggestion — conditions favor staying flat / waiting (conviction {conviction:+d}/5)."

        return {
            'strategy_key': key,
            'strategy_name': spec.name,
            'strategy_category': spec.category,
            'direction': direction,
            'action': action,
            'conviction_score': conviction,
            'horizon_days': horizon_days,
            'ml_forecast': ml_forecast,
            'entry_reference_price': analyst['price'],
            'analyst_screening': analyst,
            'bullish_strategies_count': bullish_count,
            'total_active_strategies': len(ACTIVE_STRATEGY_KEYS),
            'strategy_votes': votes,
            'rationale': rationale,
            'summary': summary,
        }

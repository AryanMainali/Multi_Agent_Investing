"""Stage 4 [Risk Gate]: Deterministic hard numerical caps, news/freshness blocks, and pass/drop gate.

Enforces single-position caps (allocation, per-trade nominal risk, stop-loss),
portfolio-level caps (total open portfolio heat and max positions per sector),
and rule-based safety blocks (high-severity adverse news events, stale/missing bars,
and optional ML confidence/accuracy floors). Never overridden by AI models.
"""
from dataclasses import dataclass
from datetime import date
from math import floor
from core.config import load_settings


@dataclass(frozen=True)
class RiskPolicy:
    max_allocation: float = 0.20
    max_risk_fraction: float = 0.01
    stop_loss_fraction: float = 0.05
    max_portfolio_heat: float = 0.04
    max_positions_per_sector: int = 2

    def __post_init__(self):
        if not 0 < self.max_allocation <= 1:
            raise ValueError('max_allocation must be in (0, 1]')
        if not 0 < self.max_risk_fraction <= 1:
            raise ValueError('max_risk_fraction must be in (0, 1]')
        if not 0 < self.stop_loss_fraction < 1:
            raise ValueError('stop_loss_fraction must be in (0, 1)')
        if not 0 < self.max_portfolio_heat <= 1:
            raise ValueError('max_portfolio_heat must be in (0, 1]')
        if int(self.max_positions_per_sector) < 1:
            raise ValueError('max_positions_per_sector must be >= 1')


RISK_PROFILES: dict[str, RiskPolicy] = {
    'low': RiskPolicy(
        max_allocation=0.10,
        max_risk_fraction=0.005,
        stop_loss_fraction=0.04,
        max_portfolio_heat=0.02,
        max_positions_per_sector=1,
    ),
    'medium': RiskPolicy(
        max_allocation=0.20,
        max_risk_fraction=0.01,
        stop_loss_fraction=0.05,
        max_portfolio_heat=0.04,
        max_positions_per_sector=2,
    ),
    'high': RiskPolicy(
        max_allocation=0.35,
        max_risk_fraction=0.025,
        stop_loss_fraction=0.07,
        max_portfolio_heat=0.08,
        max_positions_per_sector=3,
    ),
}

RISK_ALIASES: dict[str, str] = {
    'low': 'low',
    'conservative': 'low',
    'medium': 'medium',
    'moderate': 'medium',
    'high': 'high',
    'aggressive': 'high',
}


def get_risk_policy(level: str = 'medium') -> RiskPolicy:
    """Resolve a named risk tolerance ('low', 'medium', 'high') into a RiskPolicy."""
    key = RISK_ALIASES.get(level.strip().lower())
    if key is None:
        raise ValueError('Risk level must be one of: low, medium, high')
    return RISK_PROFILES[key]


def evaluate_proposal_rules(proposal: dict, ctx: dict, cfg: dict | None = None) -> tuple[bool, list[str], float]:
    """Pure deterministic proposal rule evaluator (from team Risk agent design).

    ctx keys: today (date), last_bar_day (date|None), high_flags (list[str]), approved_today (int)
    Returns (approved, reasons, suggested_max_pct).
    """
    active_cfg = cfg or load_settings().get('risk', {})
    reasons: list[str] = []

    prob = float(proposal.get('probability', 0.0))
    acc = float(proposal.get('backtest_accuracy', 0.0))
    min_conf = float(active_cfg.get('min_confidence', 0.53))
    min_acc = float(active_cfg.get('min_backtest_accuracy', 0.51))

    if prob < min_conf:
        reasons.append(f"confidence {prob:.2f} below {min_conf}")
    if acc < min_acc:
        reasons.append(f"backtest accuracy {acc:.2f} below {min_acc}")

    last_day = ctx.get('last_bar_day')
    if last_day is None:
        reasons.append('no price data')
    elif (ctx['today'] - last_day).days > int(active_cfg.get('max_data_age_days', 5)):
        reasons.append(f"price data is stale (latest {last_day})")

    if ctx.get('high_flags'):
        reasons.append('high-severity news: ' + '; '.join(ctx['high_flags'][:3]))

    max_ideas = int(active_cfg.get('max_ideas_per_day', 10))
    if int(ctx.get('approved_today', 0)) >= max_ideas:
        reasons.append(f"daily limit of {max_ideas} ideas reached")

    approved = not reasons
    kelly_frac = float(active_cfg.get('kelly_fraction', 0.25))
    max_pos_pct = float(active_cfg.get('max_position_pct', 20.0))
    kelly_pct = max(0.0, 2.0 * prob - 1.0) * kelly_frac * 100.0
    size_pct = round(min(kelly_pct, max_pos_pct), 2) if approved else 0.0
    if approved:
        reasons.append('passed all checks')
    return approved, reasons, size_pct


def size_position(
    price: float,
    cash: float,
    equity: float,
    policy: RiskPolicy = RiskPolicy(),
    cost_fraction: float = 0.0,
    open_portfolio_risk: float = 0.0,
) -> dict:
    """Whole shares with caps on notional allocation, per-trade loss, cash, and remaining portfolio heat."""
    if price <= 0 or cash < 0 or equity <= 0 or cost_fraction < 0 or open_portfolio_risk < 0:
        raise ValueError('Invalid price, capital, or cost')
    allowed_notional = min(equity * policy.max_allocation, cash / (1 + cost_fraction))
    qty_allocation = floor((allowed_notional + 1e-12) / price)

    per_share_stop_risk = price * policy.stop_loss_fraction
    qty_trade_risk = floor((equity * policy.max_risk_fraction + 1e-12) / per_share_stop_risk)

    max_heat_dollars = equity * policy.max_portfolio_heat
    remaining_heat_dollars = max(0.0, max_heat_dollars - open_portfolio_risk)
    qty_heat_risk = floor((remaining_heat_dollars + 1e-12) / per_share_stop_risk)

    shares = max(0, min(qty_allocation, qty_trade_risk, qty_heat_risk))
    nominal_risk = round(shares * per_share_stop_risk, 2)
    return {
        'shares': shares,
        'notional': round(shares * price, 2),
        'stop_price': round(price * (1 - policy.stop_loss_fraction), 4),
        'nominal_risk': nominal_risk,
        'max_allocation_pct': round(policy.max_allocation * 100, 2),
        'max_risk_pct': round(policy.max_risk_fraction * 100, 2),
        'stop_loss_pct': round(policy.stop_loss_fraction * 100, 2),
        'max_portfolio_heat_pct': round(policy.max_portfolio_heat * 100, 2),
        'max_positions_per_sector': int(policy.max_positions_per_sector),
        'open_portfolio_risk': round(open_portfolio_risk, 2),
        'remaining_heat_budget': round(remaining_heat_dollars, 2),
        'projected_portfolio_heat_pct': round(((open_portfolio_risk + nominal_risk) / equity) * 100, 2),
        'eligible': shares > 0,
        'warning': 'Stops may fill below the trigger after price gaps; actual losses can exceed nominal risk.',
    }


def evaluate_risk_gate(
    strategy_idea: dict,
    market: dict,
    research: dict,
    bankroll: float,
    policy: RiskPolicy = RiskPolicy(),
    max_annual_volatility_pct: float = 85.0,
    min_volume_ratio: float = 0.30,
    available_cash: float | None = None,
    already_held: bool = False,
    open_portfolio_risk: float = 0.0,
    sector_counts: dict[str, int] | None = None,
    candidate_sector: str | None = None,
    reference_date: date | None = None,
    max_data_age_days: int | None = None,
) -> dict:
    """Deterministic Yes/No gate deciding whether a Strategy idea reaches Report (AI) or gets Dropped."""
    risk_cfg = load_settings().get('risk', {})
    price = float(strategy_idea['entry_reference_price'])
    cash_budget = max(0.0, float(available_cash)) if available_cash is not None else float(bankroll)
    sized = size_position(
        price=price,
        cash=cash_budget,
        equity=bankroll,
        policy=policy,
        open_portfolio_risk=open_portfolio_risk,
    )

    sector = (
        candidate_sector
        or str((research.get('fundamentals') or {}).get('sector') or '').strip()
        or 'Unknown'
    )
    active_sector_counts = sector_counts or {}

    checks = []
    drop_reasons = []

    if already_held:
        checks.append({'check': 'No duplicate open position', 'passed': False, 'detail': 'Already open in paper portfolio'})
        drop_reasons.append('Ticker already has an active open position in your paper portfolio.')

    # Check 1: Strategy must propose an actionable long candidate with positive conviction
    actionable = strategy_idea.get('action') == 'SUGGEST_LONG' and int(strategy_idea.get('conviction_score', 0)) >= 1
    action_detail = f"Action={strategy_idea.get('action')}, conviction={strategy_idea.get('conviction_score')}"
    checks.append({'check': 'Actionable strategy conviction', 'passed': actionable, 'detail': action_detail})
    if not actionable:
        drop_reasons.append(
            f"Strategy did not produce an actionable long setup ({action_detail}; {strategy_idea.get('summary', 'unfavorable conditions')})."
        )

    # Check 1B (when ML strategy is primary): require minimum walk-forward confidence & held-out accuracy
    ml_forecast = strategy_idea.get('ml_forecast')
    if strategy_idea.get('strategy_key') == 'ml_logistic_walkforward' and isinstance(ml_forecast, dict):
        min_conf = float(risk_cfg.get('min_confidence', 0.53))
        min_acc = float(risk_cfg.get('min_backtest_accuracy', 0.51))
        p_up = float(ml_forecast.get('p_up', 0.5))
        acc = float(ml_forecast.get('backtest_accuracy', 0.5))
        ml_ok = p_up >= min_conf and acc >= min_acc
        ml_detail = f"ML P(up)={p_up:.2f} (min {min_conf:.2f}), held-out acc={acc:.2f} (min {min_acc:.2f})"
        checks.append({'check': 'ML confidence & backtest accuracy', 'passed': ml_ok, 'detail': ml_detail})
        if not ml_ok:
            drop_reasons.append(f"ML walk-forward check failed ({ml_detail}).")

    # Check 2: Portfolio heat budget check
    max_heat_dollars = bankroll * policy.max_portfolio_heat
    per_share_risk = price * policy.stop_loss_fraction
    heat_can_fit_one_share = sized['remaining_heat_budget'] + 1e-9 >= per_share_risk
    heat_detail = (
        f"Open risk ${open_portfolio_risk:,.2f} + new ${sized['nominal_risk']:,.2f} / "
        f"${max_heat_dollars:,.2f} max heat ({policy.max_portfolio_heat * 100:g}%)"
    )
    checks.append({'check': 'Total portfolio heat cap', 'passed': heat_can_fit_one_share, 'detail': heat_detail})
    if not heat_can_fit_one_share:
        drop_reasons.append(
            f"Portfolio heat cap reached: open stop-loss risk is ${open_portfolio_risk:,.2f}, leaving "
            f"${sized['remaining_heat_budget']:,.2f} remaining under your {policy.max_portfolio_heat * 100:g}% "
            f"(${max_heat_dollars:,.2f}) portfolio heat limit (1 share requires ${per_share_risk:,.2f} stop risk)."
        )

    # Check 3: Sector diversification cap
    curr_sector_count = int(active_sector_counts.get(sector, 0)) if sector != 'Unknown' else 0
    sector_ok = True if sector == 'Unknown' else (curr_sector_count < int(policy.max_positions_per_sector))
    sector_detail = (
        f"Sector={sector} ({curr_sector_count}/{policy.max_positions_per_sector} open)"
        if sector != 'Unknown'
        else 'Sector=Unknown (unclassified; not blocked)'
    )
    checks.append({'check': 'Sector diversification cap', 'passed': sector_ok, 'detail': sector_detail})
    if not sector_ok:
        drop_reasons.append(
            f"Sector concentration cap reached: already holding {curr_sector_count} open position(s) in {sector} "
            f"(maximum {policy.max_positions_per_sector} per sector)."
        )

    # Check 4: Whole-share affordability within max_allocation, max_risk_fraction, and available cash
    can_size = sized['shares'] >= 1
    size_detail = (
        f"{sized['shares']} whole shares (${sized['notional']:,.2f} notional) within "
        f"${cash_budget:,.2f} available cash, {sized['max_allocation_pct']:g}% allocation "
        f"(${bankroll * policy.max_allocation:,.2f}), and {sized['max_risk_pct']:g}% risk cap "
        f"(${bankroll * policy.max_risk_fraction:,.2f})"
    )
    checks.append({'check': 'Position sizing & capital limits', 'passed': can_size, 'detail': size_detail})
    if not can_size and heat_can_fit_one_share:
        drop_reasons.append(
            f"Position sizing yielded 0 whole shares at ${price:,.2f}/share under ${cash_budget:,.2f} available cash and risk caps."
        )

    # Check 5: Volatility ceiling
    vol_pct = float(market.get('volatility_20_pct', 0.0))
    vol_ok = vol_pct <= max_annual_volatility_pct
    vol_detail = f"20d annualized volatility {vol_pct:.2f}% (ceiling {max_annual_volatility_pct:.1f}%)"
    checks.append({'check': 'Volatility ceiling', 'passed': vol_ok, 'detail': vol_detail})
    if not vol_ok:
        drop_reasons.append(f"20-day annualized volatility ({vol_pct:.2f}%) exceeds maximum ceiling ({max_annual_volatility_pct:.1f}%).")

    # Check 6: Relative volume / liquidity sanity
    vol_ratio = float(market.get('volume_ratio', 0.0))
    liq_ok = vol_ratio >= min_volume_ratio
    liq_detail = f"Volume ratio {vol_ratio:.2f}x 20d avg (minimum {min_volume_ratio:.2f}x)"
    checks.append({'check': 'Liquidity / relative volume', 'passed': liq_ok, 'detail': liq_detail})
    if not liq_ok:
        drop_reasons.append(f"Relative volume ({vol_ratio:.2f}x 20-day average) is below liquidity floor ({min_volume_ratio:.2f}x).")

    # Check 7: High-severity adverse news event block (from team Risk & Research design)
    high_flags = list(research.get('high_severity_flags') or [])
    news_flags_ok = len(high_flags) == 0
    news_detail = 'No high-severity adverse news flagged' if news_flags_ok else '; '.join(high_flags[:2])
    checks.append({'check': 'High-severity news block', 'passed': news_flags_ok, 'detail': news_detail})
    if not news_flags_ok:
        drop_reasons.append(f"High-severity adverse news event flagged: {'; '.join(high_flags[:3])}")

    # Check 8: Severe bearish research headwind check
    sentiment = str(research.get('sentiment', 'NEUTRAL')).upper()
    sent_score = float(research.get('sentiment_score', 0.0))
    conviction = int(strategy_idea.get('conviction_score', 0))
    research_ok = not (sentiment == 'BEARISH' and sent_score <= -0.5 and conviction <= 2)
    res_detail = f"Research sentiment={sentiment} ({sent_score:+.2f}), conviction={conviction}"
    checks.append({'check': 'Research headwind check', 'passed': research_ok, 'detail': res_detail})
    if not research_ok:
        drop_reasons.append(
            f"Strong bearish research sentiment ({sent_score:+.2f}) conflicts with moderate technical conviction ({conviction})."
        )

    # Check 9 (when reference_date supplied): Price bar freshness check
    if reference_date is not None and market.get('as_of'):
        max_age = int(max_data_age_days if max_data_age_days is not None else risk_cfg.get('max_data_age_days', 5))
        try:
            bar_dt = date.fromisoformat(str(market['as_of'])[:10])
            age_days = (reference_date - bar_dt).days
            fresh_ok = age_days <= max_age
            fresh_detail = f"Latest bar {bar_dt} ({age_days}d old; max {max_age}d)"
            checks.append({'check': 'Price data freshness', 'passed': fresh_ok, 'detail': fresh_detail})
            if not fresh_ok:
                drop_reasons.append(f"Price data is stale (latest bar {bar_dt} is {age_days} days old, max {max_age}).")
        except ValueError:
            pass

    passed = len(drop_reasons) == 0
    stop_price = sized['stop_price']
    target_price = round(price * (1 + 2.0 * policy.stop_loss_fraction), 2)
    risk_per_share = max(price - stop_price, 1e-6)
    reward_risk_ratio = round((target_price - price) / risk_per_share, 2)

    # Informational quarter-Kelly sizing reference alongside hard stop-risk share sizing
    kelly_frac = float(risk_cfg.get('kelly_fraction', 0.25))
    p_est = float(ml_forecast['p_up']) if isinstance(ml_forecast, dict) else 0.55
    kelly_pct = round(min(max(0.0, 2.0 * p_est - 1.0) * kelly_frac * 100.0, policy.max_allocation * 100.0), 2)

    return {
        **sized,
        'sector': sector,
        'sector_open_count': curr_sector_count,
        'kelly_reference_pct': kelly_pct,
        'passed': passed,
        'decision': 'PASS_TO_REPORT' if passed else 'DROP_AND_LOG',
        'target_price': target_price,
        'reward_risk_ratio': reward_risk_ratio,
        'gate_checks': checks,
        'drop_reasons': drop_reasons,
    }

"""Stage 4 [Risk Gate]: Deterministic hard numerical caps and pass/drop gate.

Risk limits are never overridden by AI models. If an idea fails any hard gate,
it is dropped and its exact rejection reasons are logged.
"""
from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class RiskPolicy:
    max_allocation: float = 0.20
    max_risk_fraction: float = 0.01
    stop_loss_fraction: float = 0.05

    def __post_init__(self):
        if not 0 < self.max_allocation <= 1:
            raise ValueError('max_allocation must be in (0, 1]')
        if not 0 < self.max_risk_fraction <= 1:
            raise ValueError('max_risk_fraction must be in (0, 1]')
        if not 0 < self.stop_loss_fraction < 1:
            raise ValueError('stop_loss_fraction must be in (0, 1)')


RISK_PROFILES: dict[str, RiskPolicy] = {
    'low': RiskPolicy(max_allocation=0.10, max_risk_fraction=0.005, stop_loss_fraction=0.04),
    'medium': RiskPolicy(max_allocation=0.20, max_risk_fraction=0.01, stop_loss_fraction=0.05),
    'high': RiskPolicy(max_allocation=0.35, max_risk_fraction=0.025, stop_loss_fraction=0.07),
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


def size_position(
    price: float,
    cash: float,
    equity: float,
    policy: RiskPolicy = RiskPolicy(),
    cost_fraction: float = 0.0,
) -> dict:
    """Whole shares with caps on notional allocation, per-trade nominal loss, and cash."""
    if price <= 0 or cash < 0 or equity <= 0 or cost_fraction < 0:
        raise ValueError('Invalid price, capital, or cost')
    allowed_notional = min(equity * policy.max_allocation, cash / (1 + cost_fraction))
    qty_allocation = floor((allowed_notional + 1e-12) / price)
    qty_risk = floor((equity * policy.max_risk_fraction + 1e-12) / (price * policy.stop_loss_fraction))
    shares = max(0, min(qty_allocation, qty_risk))
    return {
        'shares': shares,
        'notional': round(shares * price, 2),
        'stop_price': round(price * (1 - policy.stop_loss_fraction), 4),
        'nominal_risk': round(shares * price * policy.stop_loss_fraction, 2),
        'max_allocation_pct': round(policy.max_allocation * 100, 2),
        'max_risk_pct': round(policy.max_risk_fraction * 100, 2),
        'stop_loss_pct': round(policy.stop_loss_fraction * 100, 2),
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
) -> dict:
    """Deterministic Yes/No gate deciding whether a Strategy idea reaches Report (AI) or gets Dropped."""
    price = float(strategy_idea['entry_reference_price'])
    cash_budget = max(0.0, float(available_cash)) if available_cash is not None else float(bankroll)
    sized = size_position(price=price, cash=cash_budget, equity=bankroll, policy=policy)

    checks = []
    drop_reasons = []

    if already_held:
        checks.append({'check': 'No duplicate open position', 'passed': False, 'detail': 'Already open in paper portfolio'})
        drop_reasons.append('Ticker already has an active open position in your paper portfolio.')

    # Check 1: Strategy must propose an actionable long candidate with positive conviction
    actionable = strategy_idea.get('action') == 'SUGGEST_LONG' and int(strategy_idea.get('conviction_score', 0)) >= 1
    action_detail = (
        f"Action={strategy_idea.get('action')}, conviction={strategy_idea.get('conviction_score')}"
    )
    checks.append({'check': 'Actionable strategy conviction', 'passed': actionable, 'detail': action_detail})
    if not actionable:
        drop_reasons.append(
            f"Strategy did not produce an actionable long setup ({action_detail}; {strategy_idea.get('summary', 'unfavorable conditions')})."
        )

    # Check 2: Whole-share affordability within max_allocation and max_risk_fraction
    can_size = sized['shares'] >= 1
    size_detail = (
        f"{sized['shares']} whole shares (${sized['notional']:,.2f} notional) within "
        f"${cash_budget:,.2f} available cash, {sized['max_allocation_pct']:g}% allocation "
        f"(${bankroll * policy.max_allocation:,.2f}), and {sized['max_risk_pct']:g}% risk cap "
        f"(${bankroll * policy.max_risk_fraction:,.2f})"
    )
    checks.append({'check': 'Position sizing & capital limits', 'passed': can_size, 'detail': size_detail})
    if not can_size:
        drop_reasons.append(
            f"Position sizing yielded 0 whole shares at ${price:,.2f}/share under ${cash_budget:,.2f} available cash and risk caps."
        )

    # Check 3: Volatility ceiling
    vol_pct = float(market.get('volatility_20_pct', 0.0))
    vol_ok = vol_pct <= max_annual_volatility_pct
    vol_detail = f"20d annualized volatility {vol_pct:.2f}% (ceiling {max_annual_volatility_pct:.1f}%)"
    checks.append({'check': 'Volatility ceiling', 'passed': vol_ok, 'detail': vol_detail})
    if not vol_ok:
        drop_reasons.append(f"20-day annualized volatility ({vol_pct:.2f}%) exceeds maximum ceiling ({max_annual_volatility_pct:.1f}%).")

    # Check 4: Relative volume / liquidity sanity
    vol_ratio = float(market.get('volume_ratio', 0.0))
    liq_ok = vol_ratio >= min_volume_ratio
    liq_detail = f"Volume ratio {vol_ratio:.2f}x 20d avg (minimum {min_volume_ratio:.2f}x)"
    checks.append({'check': 'Liquidity / relative volume', 'passed': liq_ok, 'detail': liq_detail})
    if not liq_ok:
        drop_reasons.append(f"Relative volume ({vol_ratio:.2f}x 20-day average) is below liquidity floor ({min_volume_ratio:.2f}x).")

    # Check 5: Severe bearish research headwind check
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

    passed = len(drop_reasons) == 0
    stop_price = sized['stop_price']
    target_price = round(price * (1 + 2.0 * policy.stop_loss_fraction), 2)
    risk_per_share = max(price - stop_price, 1e-6)
    reward_risk_ratio = round((target_price - price) / risk_per_share, 2)

    return {
        **sized,
        'passed': passed,
        'decision': 'PASS_TO_REPORT' if passed else 'DROP_AND_LOG',
        'target_price': target_price,
        'reward_risk_ratio': reward_risk_ratio,
        'gate_checks': checks,
        'drop_reasons': drop_reasons,
    }

"""Stage 4 [Risk Gate]: Deterministic hard numerical caps and pass/drop gate.

Enforces both single-position caps (allocation, per-trade nominal risk, stop-loss)
and portfolio-level caps (total open portfolio heat and max positions per sector).
Risk limits are never overridden by AI models.
"""
from dataclasses import dataclass
from math import floor


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
) -> dict:
    """Deterministic Yes/No gate deciding whether a Strategy idea reaches Report (AI) or gets Dropped."""
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

    # Check 7: Severe bearish research headwind check
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
        'sector': sector,
        'sector_open_count': curr_sector_count,
        'passed': passed,
        'decision': 'PASS_TO_REPORT' if passed else 'DROP_AND_LOG',
        'target_price': target_price,
        'reward_risk_ratio': reward_risk_ratio,
        'gate_checks': checks,
        'drop_reasons': drop_reasons,
    }

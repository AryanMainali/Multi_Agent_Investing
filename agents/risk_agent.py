"""Simple reflex agent: hard numerical caps, no natural-language discretion."""
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
        raise ValueError("Risk level must be one of: low, medium, high")
    return RISK_PROFILES[key]


def size_position(price: float, cash: float, equity: float, policy: RiskPolicy = RiskPolicy(), cost_fraction: float = 0.0) -> dict:
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

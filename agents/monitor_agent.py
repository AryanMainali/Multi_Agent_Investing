"""Stage 6 [Monitor Agent]: Independent auditor checking every idea and every agent.

Inspects [1. Market Data], [2. Research (AI)], [3. Strategy], [4. Risk Gate],
and [5. Report (AI)] for data freshness, cross-agent consistency, risk invariants,
and strict adherence to manual-only human decision rules.
"""
from dataclasses import dataclass
from datetime import date, datetime, timezone
import numpy as np
import pandas as pd
from agents.risk_agent import RiskPolicy

FORBIDDEN_EXECUTION_PHRASES = (
    'order submitted',
    'order executed',
    'broker order placed',
    'trade has been placed',
    'guaranteed profit',
)


@dataclass
class MonitorAgent:
    stale_days_warning: int = 5

    def audit(
        self,
        df: pd.DataFrame,
        market: dict,
        research: dict,
        strategy_idea: dict,
        risk_gate: dict,
        bankroll: float,
        policy: RiskPolicy,
        report: dict | None = None,
        reference_date: date | None = None,
    ) -> dict:
        warnings: list[str] = []
        critical_errors: list[str] = []
        agent_checks: dict[str, dict] = {}

        # 1. Check [1. Market Data]
        today = reference_date or datetime.now(timezone.utc).date()
        last_bar_date = df.index[-1].date()
        age_days = (today - last_bar_date).days
        finite_ok = bool(np.isfinite(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5)).all().all())
        bars_ok = len(df) >= 60 and finite_ok
        if not bars_ok:
            critical_errors.append('Market Data failed bar count or finite OHLCV verification.')
        if age_days > self.stale_days_warning:
            warnings.append(f"Latest daily bar ({last_bar_date}) is {age_days} calendar days old.")
        agent_checks['1_market_data'] = {
            'status': 'CRITICAL' if not bars_ok else ('WARN' if age_days > self.stale_days_warning else 'OK'),
            'bars_checked': len(df),
            'latest_bar': str(last_bar_date),
            'age_days': age_days,
        }

        # 2. Check [2. Research (AI)]
        valid_sentiment = research.get('sentiment') in ('BULLISH', 'NEUTRAL', 'BEARISH')
        has_summary = bool(str(research.get('summary', '')).strip())
        headlines_count = len(research.get('headlines', []))
        if not (valid_sentiment and has_summary):
            critical_errors.append('Research Agent output missing valid sentiment or summary.')
        if headlines_count == 0:
            warnings.append('Research Agent received 0 recent headlines from Yahoo Finance; relied on price & fundamentals.')
        agent_checks['2_research_ai'] = {
            'status': 'CRITICAL' if not (valid_sentiment and has_summary) else ('WARN' if headlines_count == 0 else 'OK'),
            'sentiment': research.get('sentiment'),
            'headlines_reviewed': headlines_count,
            'ai_mode': research.get('ai_mode'),
        }

        # 3. Check [3. Strategy]
        price_match = abs(float(strategy_idea.get('entry_reference_price', -1)) - float(market.get('price', -2))) < 1e-4
        if not price_match:
            critical_errors.append('Strategy reference price does not match Market Data latest close.')
        divergence = (
            strategy_idea.get('action') == 'SUGGEST_LONG'
            and research.get('sentiment') == 'BEARISH'
        )
        if divergence:
            warnings.append('Cross-agent divergence: Strategy suggests LONG while Research sentiment is BEARISH.')
        low_consensus = (
            strategy_idea.get('action') == 'SUGGEST_LONG'
            and int(strategy_idea.get('bullish_strategies_count', 0)) < 3
        )
        if low_consensus:
            warnings.append(
                f"Low multi-strategy consensus: {strategy_idea.get('bullish_strategies_count')}/"
                f"{strategy_idea.get('total_active_strategies')} active strategies are bullish."
            )
        agent_checks['3_strategy'] = {
            'status': 'CRITICAL' if not price_match else ('WARN' if (divergence or low_consensus) else 'OK'),
            'action': strategy_idea.get('action'),
            'conviction': strategy_idea.get('conviction_score'),
            'strategy_consensus': f"{strategy_idea.get('bullish_strategies_count')}/{strategy_idea.get('total_active_strategies')}",
        }

        # 4. Check [4. Risk Gate] mathematical invariants (single-trade + portfolio heat & sector caps)
        notional = float(risk_gate.get('notional', 0.0))
        nom_risk = float(risk_gate.get('nominal_risk', 0.0))
        shares = int(risk_gate.get('shares', 0))
        stop_price = float(risk_gate.get('stop_price', 0.0))
        entry_price = float(strategy_idea.get('entry_reference_price', 0.0))
        open_risk = float(risk_gate.get('open_portfolio_risk', 0.0))
        sector = str(risk_gate.get('sector', 'Unknown'))
        sector_open_count = int(risk_gate.get('sector_open_count', 0))

        alloc_cap_ok = notional <= bankroll * policy.max_allocation + 0.05
        risk_cap_ok = nom_risk <= bankroll * policy.max_risk_fraction + 0.05
        heat_cap_ok = (
            not risk_gate.get('passed')
            or (open_risk + nom_risk) <= bankroll * policy.max_portfolio_heat + 0.05
        )
        sector_cap_ok = (
            not risk_gate.get('passed')
            or sector == 'Unknown'
            or sector_open_count < int(policy.max_positions_per_sector)
        )
        stop_ok = 0 < stop_price < entry_price
        gate_consistent = (
            (risk_gate['passed'] and shares >= 1 and len(risk_gate.get('drop_reasons', [])) == 0)
            or (not risk_gate['passed'] and len(risk_gate.get('drop_reasons', [])) >= 1)
        )
        risk_invariants_ok = alloc_cap_ok and risk_cap_ok and heat_cap_ok and sector_cap_ok and stop_ok and gate_consistent
        if not risk_invariants_ok:
            critical_errors.append('Risk Gate mathematical cap, portfolio heat, or sector diversification invariant failed.')
        agent_checks['4_risk_gate'] = {
            'status': 'OK' if risk_invariants_ok else 'CRITICAL',
            'passed_gate': bool(risk_gate.get('passed')),
            'alloc_cap_verified': alloc_cap_ok,
            'risk_cap_verified': risk_cap_ok,
            'heat_cap_verified': heat_cap_ok,
            'sector_cap_verified': sector_cap_ok,
        }

        # 5. Check [5. Report (AI)] (when Risk Gate passed and report was generated)
        if report is not None:
            md = str(report.get('markdown_report', ''))
            lower_md = md.lower()
            has_manual_notice = 'manual' in lower_md or 'humans decide' in lower_md
            has_stop_mention = f"{stop_price:.2f}" in md or 'stop' in lower_md
            no_forbidden = not any(phrase in lower_md for phrase in FORBIDDEN_EXECUTION_PHRASES)
            report_ok = bool(md.strip()) and has_manual_notice and has_stop_mention and no_forbidden
            if not report_ok:
                critical_errors.append('Report Agent output failed plain-English risk disclosure or manual-only compliance check.')
            agent_checks['5_report_ai'] = {
                'status': 'OK' if report_ok else 'CRITICAL',
                'ai_mode': report.get('ai_mode'),
                'manual_only_verified': has_manual_notice and no_forbidden,
            }
        else:
            agent_checks['5_report_ai'] = {
                'status': 'SKIPPED_DROPPED' if not risk_gate.get('passed') else 'PENDING',
                'manual_only_verified': True,
            }

        if critical_errors:
            overall = 'CRITICAL_ALERT'
        elif warnings:
            overall = 'PASS_WITH_WARNINGS'
        else:
            overall = 'HEALTHY'

        return {
            'status': overall,
            'checked_at_utc': datetime.now(timezone.utc).isoformat(),
            'agent_checks': agent_checks,
            'warnings': warnings,
            'critical_errors': critical_errors,
        }

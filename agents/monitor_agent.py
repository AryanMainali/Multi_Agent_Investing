"""Stage 6 [Monitor Agent]: Independent auditor, heartbeat watcher, and retrospective idea scorer.

1. Per-Idea Audit: Inspects [1. Market Data], [2. Research (AI)], [3. Strategy],
   [4. Risk Gate], and [5. Report (AI)] for data freshness, cross-agent consistency,
   mathematical risk invariants, and strict manual-only report compliance.
2. Agent Heartbeats: Verifies all upstream agents ran recently without errors.
3. Retrospective Scorecard: Once an idea's N-day horizon passes, scores BOTH
   Risk-Approved and Risk-Rejected proposals so the team can verify whether the
   system has a >50% edge and whether Risk is filtering effectively.
4. Daily Summary & Optional Alerts: Writes daily markdown summaries and sends
   optional Telegram notifications when configured in .env.
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from agents.risk_agent import RiskPolicy
from core.config import EnvConfig, REPORTS_DIR, load_env, load_settings
from core.llm import LLMConfig, generate_text
import core.storage as storage_mod
from core.storage import (
    get_idea_scorecard,
    list_heartbeats,
    list_proposals,
    record_heartbeat,
    score_past_proposals,
)

FORBIDDEN_EXECUTION_PHRASES = (
    'order submitted',
    'order executed',
    'broker order placed',
    'trade has been placed',
    'guaranteed profit',
)

WATCHED_AGENTS = ('market_data', 'research', 'strategy', 'risk', 'report')


def send_telegram_alert(text: str, env: EnvConfig | None = None) -> bool:
    """Send an optional read-only health/advisory alert via Telegram bot if configured in .env."""
    active_env = env or load_env()
    if not active_env.telegram_enabled:
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{active_env.telegram_token}/sendMessage",
            json={'chat_id': active_env.telegram_chat_id, 'text': text[:4000]},
            timeout=10.0,
        )
        resp.raise_for_status()
        return True
    except Exception:
        return False


def check_agent_heartbeats(
    stale_after_hours: int | None = None,
    db_path: Path | None = None,
    now_utc: datetime | None = None,
) -> list[str]:
    """Check that all 5 upstream pipeline stages ran recently without error."""
    settings = load_settings()
    max_hours = int(stale_after_hours if stale_after_hours is not None else settings.get('monitor', {}).get('stale_after_hours', 30))
    now = now_utc or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max_hours)

    beats = {row['agent']: row for row in list_heartbeats(db_path=db_path)}
    issues: list[str] = []

    for agent_name in WATCHED_AGENTS:
        b = beats.get(agent_name)
        if b is None:
            issues.append(f"{agent_name} has never recorded a run")
            continue
        if b['status'] != 'ok':
            issues.append(f"{agent_name} last run failed: {b['detail'][:200]}")
            continue
        try:
            last_dt = datetime.fromisoformat(str(b['last_run_utc']))
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            if last_dt < cutoff:
                issues.append(f"{agent_name} has not run since {last_dt:%Y-%m-%d %H:%M} UTC")
        except ValueError:
            issues.append(f"{agent_name} has an invalid heartbeat timestamp")

    return issues


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
        high_flags = research.get('high_severity_flags') or []
        if not (valid_sentiment and has_summary):
            critical_errors.append('Research Agent output missing valid sentiment or summary.')
        if headlines_count == 0:
            warnings.append('Research Agent received 0 recent headlines from Yahoo Finance; relied on price & fundamentals.')
        if high_flags and risk_gate.get('passed'):
            critical_errors.append('High-severity adverse news flag was present on an approved idea.')
        agent_checks['2_research_ai'] = {
            'status': 'CRITICAL' if not (valid_sentiment and has_summary) else ('WARN' if headlines_count == 0 else 'OK'),
            'sentiment': research.get('sentiment'),
            'headlines_reviewed': headlines_count,
            'event_flags_count': len(research.get('event_flags') or []),
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

    def run_health_and_scorecard(
        self,
        bars_by_ticker: dict[str, pd.DataFrame] | None = None,
        db_path: Path | None = None,
        llm_config: LLMConfig | None = None,
        output_dir: Path | None = None,
        send_alerts: bool = False,
    ) -> dict:
        """Score past ideas (approved & rejected), check agent heartbeats, and write daily summary."""
        target_db = Path(db_path) if db_path is not None else storage_mod.DEFAULT_DB
        issues = check_agent_heartbeats(db_path=target_db)
        newly_scored = score_past_proposals(bars_by_ticker or {}, db_path=target_db)
        card_df = get_idea_scorecard(db_path=target_db)
        proposals_list = list_proposals(db_path=target_db)

        status_counts: dict[str, int] = {}
        for p in proposals_list:
            st_key = str(p['status'])
            status_counts[st_key] = status_counts.get(st_key, 0) + 1

        health_md = '\n'.join(f"- {i}" for i in issues) if issues else '- All upstream agents healthy'
        counts_md = (
            '\n'.join(f"- **{k}**: {v}" for k, v in sorted(status_counts.items()))
            if status_counts
            else '- None recorded yet'
        )
        card_md = (
            f"```\n{card_df.to_string(index=False)}\n```"
            if not card_df.empty
            else 'No proposals have reached their forward evaluation horizon yet.'
        )

        facts_md = (
            f"## Agent Heartbeat Health\n{health_md}\n\n"
            f"## Proposals by Risk Gate Status\n{counts_md}\n\n"
            f"## Retrospective Scorecard (Settled Horizon Ideas)\n{card_md}\n"
        )

        llm_summary, _ = generate_text(
            system_prompt=(
                'Summarize this trading-research system status for a non-expert team in 4 short bullet points. '
                'Use only the facts given. Be candid if the hit rate is near 50% (no edge yet). '
                'Never suggest automated trading.'
            ),
            user_prompt=facts_md,
            config=llm_config,
            max_tokens=350,
        )

        daily_dir = output_dir or (target_db.parent / 'reports' / 'daily' if db_path else REPORTS_DIR / 'daily')
        daily_dir.mkdir(parents=True, exist_ok=True)
        summary_path = daily_dir / f"summary_{date.today().isoformat()}.md"
        full_md = (
            f"# Daily Monitor Summary ({date.today().isoformat()})\n\n"
            + (f"{llm_summary}\n\n" if llm_summary else '')
            + facts_md
            + '\n_Agents only suggest. Humans decide and trade manually._\n'
        )
        summary_path.write_text(full_md, encoding='utf-8')

        record_heartbeat(
            'monitor',
            status='ok',
            detail=f"scored {newly_scored} proposals; {len(issues)} health issue(s)",
            db_path=target_db,
        )

        if send_alerts and issues:
            send_telegram_alert('Trading research agents need attention:\n' + health_md)

        return {
            'issues': issues,
            'newly_scored': newly_scored,
            'scorecard_df': card_df,
            'proposal_counts': status_counts,
            'summary_markdown': full_md,
            'summary_path': str(summary_path),
        }

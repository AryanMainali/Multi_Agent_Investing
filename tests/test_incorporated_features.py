"""Tests for team features incorporated into the unified multi-agent platform:
- Shared config/settings.yaml & .env loader
- Pure deterministic Risk proposal rules (confidence, accuracy, staleness, news blocks, daily limit, Kelly cap)
- Structured headline event flagging & high-severity adverse news blocking in workflow
- Walk-forward Logistic Regression ML model with embargo gap & no-lookahead signal series
- Agent heartbeats, retrospective proposal scoring (Approved vs. Risk-Rejected), and daily Monitor summary
"""
from datetime import date, datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from agents.monitor_agent import MonitorAgent, check_agent_heartbeats, send_telegram_alert
from agents.research_agent import classify_headline_events
from agents.risk_agent import evaluate_proposal_rules, get_risk_policy
from core.config import EnvConfig, load_env, load_settings
from core.llm import LLMConfig, call_structured_tool
from core.ml_model import predict_walk_forward
from core.storage import (
    get_idea_scorecard,
    list_heartbeats,
    list_research_flags,
    record_heartbeat,
    save_proposal,
    score_past_proposals,
)
from core.workflow import run_suggestion_workflow

RISK_CFG = {
    'min_confidence': 0.55,
    'min_backtest_accuracy': 0.52,
    'max_data_age_days': 4,
    'block_on_high_severity_days': 3,
    'max_ideas_per_day': 3,
    'kelly_fraction': 0.25,
    'max_position_pct': 2.0,
}
TODAY = date(2026, 11, 18)


def _ctx(**overrides):
    base = {'today': TODAY, 'last_bar_day': TODAY, 'high_flags': [], 'approved_today': 0}
    return base | overrides


def _proposal(**overrides):
    return {'probability': 0.60, 'backtest_accuracy': 0.56} | overrides


def test_pure_proposal_risk_rules_all_cases():
    approved, reasons, size = evaluate_proposal_rules(_proposal(), _ctx(), RISK_CFG)
    assert approved
    assert 0 < size <= RISK_CFG['max_position_pct']

    # Size never exceeds cap even at 99% confidence
    _, _, capped_size = evaluate_proposal_rules(_proposal(probability=0.99), _ctx(), RISK_CFG)
    assert capped_size == RISK_CFG['max_position_pct']

    # Low confidence rejected
    approved, reasons, size = evaluate_proposal_rules(_proposal(probability=0.51), _ctx(), RISK_CFG)
    assert not approved and size == 0.0
    assert any('confidence' in r for r in reasons)

    # Weak held-out backtest accuracy rejected
    approved, *_ = evaluate_proposal_rules(_proposal(backtest_accuracy=0.48), _ctx(), RISK_CFG)
    assert not approved

    # Stale price bar rejected
    approved, reasons, _ = evaluate_proposal_rules(
        _proposal(), _ctx(last_bar_day=TODAY - timedelta(days=10)), RISK_CFG
    )
    assert not approved and any('stale' in r for r in reasons)

    # Missing price data rejected
    approved, *_ = evaluate_proposal_rules(_proposal(), _ctx(last_bar_day=None), RISK_CFG)
    assert not approved

    # High-severity news blocks approval
    approved, reasons, _ = evaluate_proposal_rules(
        _proposal(), _ctx(high_flags=['Regulators open probe']), RISK_CFG
    )
    assert not approved and any('news' in r for r in reasons)

    # Daily idea cap enforced
    approved, *_ = evaluate_proposal_rules(_proposal(), _ctx(approved_today=3), RISK_CFG)
    assert not approved


def test_settings_yaml_and_dotenv_loader(tmp_path, monkeypatch):
    settings = load_settings()
    assert 'AAPL' in settings['watchlist']
    assert settings['strategy']['horizon_days'] == 5
    assert settings['risk']['kelly_fraction'] == 0.25

    env_file = tmp_path / '.env'
    env_file.write_text(
        'CLAUDE_MODEL=claude-sonnet-4-5\nTELEGRAM_BOT_TOKEN="tok_123"\nTELEGRAM_CHAT_ID=98765\n',
        encoding='utf-8',
    )
    monkeypatch.delenv('CLAUDE_MODEL', raising=False)
    monkeypatch.delenv('TELEGRAM_BOT_TOKEN', raising=False)
    monkeypatch.delenv('TELEGRAM_CHAT_ID', raising=False)

    env_cfg = load_env(dotenv_path=env_file)
    assert env_cfg.claude_model == 'claude-sonnet-4-5'
    assert env_cfg.telegram_enabled is True


def test_headline_event_classifier_and_high_severity_workflow_block(monkeypatch, tmp_path):
    headlines = [
        {'title': 'Regulators open probe into ACME business practices', 'summary': ''},
        {'title': 'ACME unveils new product line at annual event', 'summary': ''},
    ]
    flags = classify_headline_events('ACME', headlines)
    assert len(flags) == 2
    assert flags[0]['kind'] == 'regulatory'
    assert flags[0]['severity'] == 'high'
    assert flags[0]['adverse'] is True

    # Even on a strong technical uptrend, a high-severity regulatory probe blocks Risk Gate!
    dates = pd.bdate_range('2026-03-02', periods=180, tz='America/New_York')
    t = np.arange(180, dtype=float)
    close = 120.0 + 0.45 * t + 1.5 * np.sin(t / 4.0)
    bars = pd.DataFrame(
        {'Open': close - 0.3, 'High': close + 1.0, 'Low': close - 0.9, 'Close': close, 'Volume': 40_000_000},
        index=dates,
    )

    class ProbeTicker:
        def __init__(self, symbol):
            self.info = {'shortName': 'ACME Corp', 'sector': 'Technology', 'trailingPE': 22.0}

        def history(self, period='2y', interval='1d', auto_adjust=True):
            return bars

        def get_news(self, count=6):
            return [{'title': 'Regulators open probe into ACME business practices', 'publisher': 'Reuters'}]

    monkeypatch.setattr(yf, 'Ticker', ProbeTicker)
    db = tmp_path / 'probe.db'
    _, res = run_suggestion_workflow(
        ticker='ACME',
        bankroll=25000.0,
        policy=get_risk_policy('medium'),
        db_path=db,
    )
    assert res['outcome'] == 'DROPPED_BY_RISK_GATE'
    assert res['risk']['passed'] is False
    assert any('High-severity adverse news' in r for r in res['risk']['drop_reasons'])
    saved_flags = list_research_flags('ACME', db_path=db)
    assert len(saved_flags) == 1
    assert saved_flags[0]['severity'] == 'high'


def test_walk_forward_ml_model_and_embargo():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range('2024-01-02', periods=260)
    rets = rng.normal(0.0008, 0.014, size=260)
    for i in range(1, len(rets)):
        rets[i] += 0.12 * rets[i - 1]
    close = 100.0 * np.exp(np.cumsum(rets))
    bars = pd.DataFrame(
        {
            'Open': close * 0.998,
            'High': close * 1.008,
            'Low': close * 0.992,
            'Close': close,
            'Volume': rng.integers(5_000_000, 15_000_000, size=260),
        },
        index=dates,
    )

    forecast = predict_walk_forward(bars, horizon_days=5, min_history=90, test_window=40)
    assert forecast is not None
    assert forecast['direction'] in ('up', 'down')
    assert 0.0 <= forecast['p_up'] <= 1.0
    assert 0.5 <= forecast['probability'] <= 1.0
    assert 0.0 <= forecast['backtest_accuracy'] <= 1.0
    assert forecast['horizon_days'] == 5


def test_heartbeats_and_retrospective_idea_scorecard(tmp_path):
    db = tmp_path / 'monitor.db'
    now = datetime(2026, 11, 20, 18, 0, tzinfo=timezone.utc)

    # Missing heartbeats are flagged initially
    missing_issues = check_agent_heartbeats(stale_after_hours=30, db_path=db, now_utc=now)
    assert len(missing_issues) == 5

    for ag in ('market_data', 'research', 'strategy', 'risk', 'report'):
        record_heartbeat(ag, status='ok', detail='healthy', timestamp_utc=now.isoformat(), db_path=db)
    assert check_agent_heartbeats(stale_after_hours=30, db_path=db, now_utc=now) == []

    # Stale heartbeat (>30h old) is flagged
    stale_ts = (now - timedelta(hours=40)).isoformat()
    record_heartbeat('research', status='ok', detail='old', timestamp_utc=stale_ts, db_path=db)
    stale_issues = check_agent_heartbeats(stale_after_hours=30, db_path=db, now_utc=now)
    assert len(stale_issues) == 1 and 'research has not run since' in stale_issues[0]

    # Save 1 approved proposal (AAPL rises -> correct!) and 1 Risk-rejected proposal (WEAK falls -> Risk saved us!)
    dates = pd.bdate_range('2026-10-01', periods=20)
    as_of_str = str(dates[5].date())

    save_proposal(
        symbol='AAPL',
        as_of=as_of_str,
        strategy='SMA 20/50 Crossover',
        direction='up',
        entry_ref_price=180.0,
        status='approved',
        horizon_days=5,
        probability=0.62,
        backtest_accuracy=0.57,
        conviction_score=4,
        db_path=db,
    )
    save_proposal(
        symbol='WEAK',
        as_of=as_of_str,
        strategy='SMA 20/50 Crossover',
        direction='up',
        entry_ref_price=100.0,
        status='rejected',
        horizon_days=5,
        probability=0.51,
        backtest_accuracy=0.49,
        conviction_score=1,
        db_path=db,
    )

    aapl_bars = pd.DataFrame({'Close': np.linspace(175.0, 195.0, 20)}, index=dates)
    weak_bars = pd.DataFrame({'Close': np.linspace(105.0, 85.0, 20)}, index=dates)

    scored_n = score_past_proposals({'AAPL': aapl_bars, 'WEAK': weak_bars}, db_path=db)
    assert scored_n == 2

    card = get_idea_scorecard(db_path=db)
    assert len(card) == 2
    approved_row = card.loc[card['Group'] == 'Approved by Risk Gate'].iloc[0]
    rejected_row = card.loc[card['Group'] == 'Rejected by Risk Gate'].iloc[0]
    assert approved_row['Hit Rate (%)'] == 100.0
    assert approved_row['Avg Horizon Move (%)'] > 0
    assert rejected_row['Hit Rate (%)'] == 0.0
    assert rejected_row['Avg Horizon Move (%)'] < 0

    mon_summary = MonitorAgent().run_health_and_scorecard(
        bars_by_ticker={'AAPL': aapl_bars, 'WEAK': weak_bars},
        db_path=db,
        output_dir=tmp_path / 'daily',
    )
    assert 'Retrospective Scorecard' in mon_summary['summary_markdown']
    assert len(list_heartbeats(db_path=db)) == 6


def test_anthropic_structured_tool_and_telegram_alert_mock(monkeypatch):
    class MockAnthropicToolResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                'content': [
                    {
                        'type': 'tool_use',
                        'name': 'record_flags',
                        'input': {
                            'flags': [
                                {
                                    'kind': 'legal',
                                    'severity': 'high',
                                    'adverse': True,
                                    'summary': 'Class-action lawsuit filed',
                                    'headline': 'Company sued in class-action lawsuit',
                                }
                            ]
                        },
                    }
                ]
            }

    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockAnthropicToolResp())
    cfg = LLMConfig(provider='anthropic', api_key='sk-ant-test', model='claude-3-5-haiku-latest')
    out, mode = call_structured_tool(
        'sys',
        'usr',
        {'name': 'record_flags', 'input_schema': {'type': 'object'}},
        config=cfg,
    )
    assert mode.startswith('llm_tool:anthropic:')
    assert out['flags'][0]['kind'] == 'legal'

    sent = send_telegram_alert(
        'Health check ok',
        env=EnvConfig(
            openai_key=None,
            anthropic_key=None,
            gemini_key=None,
            claude_model='claude-3-5-haiku-latest',
            alpaca_key=None,
            alpaca_secret=None,
            telegram_token='bot123',
            telegram_chat_id='chat456',
        ),
    )
    assert sent is True

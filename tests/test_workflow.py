from datetime import date
import numpy as np
import pandas as pd
import pytest
import requests
import yfinance as yf
from agents.monitor_agent import MonitorAgent
from agents.report_agent import ReportAgent
from agents.risk_agent import get_risk_policy
from core.llm import LLMConfig, generate_text, parse_json_object
from core.storage import list_dropped_ideas, list_workflow_audits
from core.workflow import run_suggestion_workflow


def _make_uptrend_bars(n: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range('2026-03-02', periods=n, tz='America/New_York')
    t = np.arange(n, dtype=float)
    # Steady uptrend so 20d SMA > 50d SMA, positive 20d momentum, moderate volatility
    close = 120.0 + 0.45 * t + 1.5 * np.sin(t / 4.0)
    open_ = close - 0.3
    high = close + 1.0
    low = open_ - 0.8
    volume = np.full(n, 45_000_000, dtype=int)
    return pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume}, index=dates)


def _make_downtrend_bars(n: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range('2026-03-02', periods=n, tz='America/New_York')
    t = np.arange(n, dtype=float)
    close = 260.0 - 0.65 * t + 1.2 * np.sin(t / 4.0)
    open_ = close + 0.3
    high = open_ + 0.8
    low = close - 1.0
    volume = np.full(n, 45_000_000, dtype=int)
    return pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume}, index=dates)


def test_workflow_yes_branch_generates_plain_english_report_and_monitor_audit(monkeypatch, tmp_path):
    bars = _make_uptrend_bars()
    db = tmp_path / 'audit.db'

    class BullishTicker:
        def __init__(self, symbol):
            self.symbol = symbol
            self.info = {
                'shortName': 'Apple Inc.',
                'sector': 'Technology',
                'trailingPE': 24.0,
                'fiftyTwoWeekHigh': 210.0,
                'fiftyTwoWeekLow': 120.0,
            }

        def history(self, period='2y', interval='1d', auto_adjust=True):
            return bars

        def get_news(self, count=6):
            return [
                {
                    'id': '1',
                    'content': {
                        'title': 'Apple beats quarterly revenue growth expectations on strong demand',
                        'summary': 'Strong profit and record services growth reported.',
                        'pubDate': '2026-11-06T14:00:00Z',
                        'provider': {'displayName': 'Yahoo Finance'},
                    },
                }
            ]

    monkeypatch.setattr(yf, 'Ticker', BullishTicker)

    _, res = run_suggestion_workflow(
        ticker='AAPL',
        bankroll=20000.0,
        policy=get_risk_policy('medium'),
        strategy='sma_crossover',
        db_path=db,
    )

    assert res['outcome'] == 'SUGGESTION_READY'
    assert res['risk']['passed'] is True
    assert res['risk']['shares'] >= 1
    assert res['report'] is not None
    md = res['report']['markdown_report']
    assert 'Agents only suggest' in md
    assert str(res['risk']['shares']) in md
    assert res['monitor']['agent_checks']['5_report_ai']['status'] == 'OK'
    assert res['monitor']['critical_errors'] == []

    # Logged to workflow_audits, NOT dropped_ideas
    assert len(list_dropped_ideas(db)) == 0
    audits = list_workflow_audits(db)
    assert len(audits) == 1
    assert audits[0][3] == 'SUGGESTION_READY'


def test_workflow_no_branch_drops_and_logs_reason_without_report(monkeypatch, tmp_path):
    bars = _make_downtrend_bars()
    db = tmp_path / 'dropped.db'

    class BearishTicker:
        def __init__(self, symbol):
            self.symbol = symbol
            self.info = {'shortName': 'Weak Corp', 'trailingPE': 65.0}

        def history(self, period='2y', interval='1d', auto_adjust=True):
            return bars

        def get_news(self, count=6):
            return [
                {
                    'id': '2',
                    'content': {
                        'title': 'Weak Corp plunges after earnings miss and guidance cut',
                        'summary': 'Analysts downgrade stock amid weak demand.',
                        'pubDate': '2026-11-06T14:00:00Z',
                        'provider': {'displayName': 'Yahoo Finance'},
                    },
                }
            ]

    monkeypatch.setattr(yf, 'Ticker', BearishTicker)

    _, res = run_suggestion_workflow(
        ticker='WEAK',
        bankroll=10000.0,
        policy=get_risk_policy('medium'),
        strategy='sma_crossover',
        db_path=db,
    )

    assert res['outcome'] == 'DROPPED_BY_RISK_GATE'
    assert res['risk']['passed'] is False
    assert len(res['risk']['drop_reasons']) >= 1
    assert res['report'] is None
    assert res['monitor']['agent_checks']['5_report_ai']['status'] == 'SKIPPED_DROPPED'

    dropped = list_dropped_ideas(db)
    assert len(dropped) == 1
    assert dropped[0][1] == 'WEAK'
    assert 'Strategy did not produce an actionable long setup' in dropped[0][5]


def test_risk_gate_drops_when_capital_cannot_afford_one_whole_share(monkeypatch, tmp_path):
    bars = _make_uptrend_bars()
    db = tmp_path / 'small_cash.db'

    class ExpensiveTicker:
        def __init__(self, symbol):
            self.symbol = symbol
            self.info = {'shortName': 'HighPrice Inc'}

        def history(self, period='2y', interval='1d', auto_adjust=True):
            return bars

        def get_news(self, count=6):
            return [{'title': 'Strong record growth and profit rally', 'publisher': 'Reuters'}]

    monkeypatch.setattr(yf, 'Ticker', ExpensiveTicker)

    # $500 bankroll with Low risk (10% max allocation = $50) cannot buy a ~$200 share
    _, res = run_suggestion_workflow(
        ticker='HIGHP',
        bankroll=500.0,
        policy=get_risk_policy('low'),
        db_path=db,
    )
    assert res['outcome'] == 'DROPPED_BY_RISK_GATE'
    assert res['risk']['shares'] == 0
    assert any('0 whole shares' in r for r in res['risk']['drop_reasons'])
    assert res['report'] is None
    assert len(list_dropped_ideas(db)) == 1


def test_report_agent_refuses_unapproved_risk_gate():
    with pytest.raises(ValueError, match='only run for ideas that pass the Risk Gate'):
        ReportAgent().generate(
            ticker='AAPL',
            market={},
            research={},
            strategy_idea={},
            risk_gate={'passed': False},
            monitor_precheck={},
            bankroll=10000.0,
        )


def test_monitor_agent_catches_forbidden_execution_claims(monkeypatch, tmp_path):
    bars = _make_uptrend_bars()

    class SimpleTicker:
        def __init__(self, symbol):
            self.info = {'shortName': 'Test Co'}

        def history(self, period='2y', interval='1d', auto_adjust=True):
            return bars

        def get_news(self, count=6):
            return [{'title': 'Record growth beats estimates', 'publisher': 'News'}]

    monkeypatch.setattr(yf, 'Ticker', SimpleTicker)
    data, res = run_suggestion_workflow(ticker='TST', bankroll=25000.0, log_to_db=False)
    assert res['risk']['passed'] is True

    # Healthy when checked on bar's own date with compliant report
    last_date = date.fromisoformat(res['as_of'])
    clean_audit = MonitorAgent().audit(
        df=data,
        market=res['market'],
        research=res['research'],
        strategy_idea=res['strategy'],
        risk_gate=res['risk'],
        bankroll=25000.0,
        policy=get_risk_policy('medium'),
        report=res['report'],
        reference_date=last_date,
    )
    assert clean_audit['status'] == 'HEALTHY'

    # Inject forbidden broker execution language -> Monitor flags CRITICAL_ALERT
    bad_report = {**res['report'], 'markdown_report': 'Broker order placed with guaranteed profit!'}
    bad_audit = MonitorAgent().audit(
        df=data,
        market=res['market'],
        research=res['research'],
        strategy_idea=res['strategy'],
        risk_gate=res['risk'],
        bankroll=25000.0,
        policy=get_risk_policy('medium'),
        report=bad_report,
        reference_date=last_date,
    )
    assert bad_audit['status'] == 'CRITICAL_ALERT'
    assert bad_audit['agent_checks']['5_report_ai']['status'] == 'CRITICAL'


def test_llm_client_json_parsing_and_mock_openai(monkeypatch):
    parsed = parse_json_object('```json\n{"sentiment": "BULLISH", "sentiment_score": 0.75}\n```')
    assert parsed == {'sentiment': 'BULLISH', 'sentiment_score': 0.75}
    assert parse_json_object('not json') is None

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {'choices': [{'message': {'content': '{"sentiment": "BULLISH", "sentiment_score": 0.8}'}}]}

    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: DummyResponse())
    cfg = LLMConfig(provider='openai', api_key='sk-test-dummy', model='gpt-4o-mini')
    text, mode = generate_text('sys', 'usr', cfg)
    assert mode == 'llm:openai:gpt-4o-mini'
    assert parse_json_object(text)['sentiment'] == 'BULLISH'

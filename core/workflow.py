"""Multi-agent trading suggestion orchestration.

Implements the 6-stage suggestion-only pipeline:
  [1. Market Data] ─┐
                    ├─► [3. Strategy] ─► [4. Risk Gate] ─► (No)  [Drop & Log Reason]
  [2. Research AI] ─┘                                   ─► (Yes) [5. Report AI] ◄─ [6. Monitor Agent] ─► Human Decision

Also records per-agent health heartbeats, deduplicated research event flags, and
proposals (both approved and Risk-rejected) so [6. Monitor Agent] can score both
groups once their forward horizon passes.

Agents only suggest. Humans decide and trade manually.
"""
from pathlib import Path
from agents.analyst_agent import AnalystAgent
from agents.data_agent import DataAgent
from agents.market_agent import MarketAgent
from agents.monitor_agent import MonitorAgent
from agents.report_agent import ReportAgent
from agents.research_agent import ResearchAgent
from agents.risk_agent import RiskPolicy, evaluate_risk_gate, size_position
from agents.strategy_agent import StrategyAgent
from core.llm import LLMConfig
import core.storage as storage_mod
from core.storage import (
    record_heartbeat,
    save_dropped_idea,
    save_proposal,
    save_research_flags,
    save_workflow_audit,
    score_past_proposals,
)


def analyze(
    ticker: str = 'AAPL',
    bankroll: float = 10000.0,
    policy: RiskPolicy | None = None,
    period: str = '2y',
    source: str = 'yahoo',
):
    """Core four-agent snapshot preserved for backward compatibility."""
    if bankroll <= 0:
        raise ValueError('Bankroll must be positive')
    if source != 'yahoo':
        raise ValueError("Only real market data ('yahoo') is supported; offline synthetic data is disabled.")
    active_policy = policy or RiskPolicy()
    clean_ticker = ticker.strip().upper()
    data = DataAgent(source=source).collect(ticker=clean_ticker, period=period)
    market = MarketAgent().scan(data)
    analyst = AnalystAgent().evaluate(data, market)
    risk = size_position(analyst['price'], bankroll, bankroll, active_policy)
    return data, {
        'ticker': clean_ticker,
        'source': source,
        'market': market,
        'analyst': analyst,
        'risk': risk,
        'as_of': str(data.index[-1].date()),
    }


def run_suggestion_workflow(
    ticker: str = 'AAPL',
    bankroll: float = 10000.0,
    policy: RiskPolicy | None = None,
    strategy: str = 'sma_crossover',
    period: str = '2y',
    source: str = 'yahoo',
    llm_config: LLMConfig | None = None,
    db_path: Path | None = None,
    log_to_db: bool = True,
    available_cash: float | None = None,
    already_held: bool = False,
    open_portfolio_risk: float = 0.0,
    sector_counts: dict[str, int] | None = None,
):
    """Run the full 6-agent suggestion workflow ending in either Drop & Log or a Plain-English Report."""
    if bankroll <= 0:
        raise ValueError('Bankroll must be positive')
    if source != 'yahoo':
        raise ValueError("Only real market data ('yahoo') is supported; offline synthetic data is disabled.")

    active_policy = policy or RiskPolicy()
    clean_ticker = ticker.strip().upper()
    target_db = Path(db_path) if db_path is not None else storage_mod.DEFAULT_DB

    # [1. Market Data]
    data = DataAgent(source=source).collect(ticker=clean_ticker, period=period)
    market = MarketAgent().scan(data)
    if log_to_db:
        record_heartbeat('market_data', 'ok', f"{clean_ticker}: {len(data)} bars through {market['as_of']}", db_path=target_db)

    # [2. Research (AI)]
    research = ResearchAgent(llm_config=llm_config).research(ticker=clean_ticker, market=market)
    if log_to_db:
        save_research_flags(
            symbol=clean_ticker,
            flags=research.get('event_flags') or [],
            source=str(research.get('flag_source') or 'keywords'),
            db_path=target_db,
        )
        record_heartbeat(
            'research',
            'ok',
            f"{clean_ticker}: {len(research.get('headlines', []))} headlines, {len(research.get('event_flags', []))} flag(s)",
            db_path=target_db,
        )

    # [3. Strategy]
    strategy_idea = StrategyAgent(primary_strategy=strategy).formulate(
        df=data,
        market=market,
        research=research,
    )
    if log_to_db:
        record_heartbeat(
            'strategy',
            'ok',
            f"{clean_ticker}: {strategy_idea['action']} (conviction {strategy_idea['conviction_score']:+d}/5)",
            db_path=target_db,
        )

    # [4. Risk Gate]
    risk_gate = evaluate_risk_gate(
        strategy_idea=strategy_idea,
        market=market,
        research=research,
        bankroll=bankroll,
        policy=active_policy,
        available_cash=available_cash,
        already_held=already_held,
        open_portfolio_risk=open_portfolio_risk,
        sector_counts=sector_counts,
    )
    if log_to_db:
        record_heartbeat(
            'risk',
            'ok',
            f"{clean_ticker}: {'APPROVED' if risk_gate['passed'] else 'REJECTED'}",
            db_path=target_db,
        )

    monitor_agent = MonitorAgent()
    ml_info = strategy_idea.get('ml_forecast') or {}
    ml_prob = float(ml_info.get('probability', 0.5)) if isinstance(ml_info, dict) else 0.5
    ml_acc = float(ml_info.get('backtest_accuracy', 0.5)) if isinstance(ml_info, dict) else 0.5

    # Branch on [4. Risk Gate]
    if not risk_gate['passed']:
        # (No) -> [6. Monitor Agent] verifies upstream agents + drop reasons -> [Drop & Log Reason]
        monitor = monitor_agent.audit(
            df=data,
            market=market,
            research=research,
            strategy_idea=strategy_idea,
            risk_gate=risk_gate,
            bankroll=bankroll,
            policy=active_policy,
            report=None,
        )
        outcome = 'DROPPED_BY_RISK_GATE'
        headline = f"Dropped {clean_ticker} ({strategy_idea['strategy_name']}): {'; '.join(risk_gate['drop_reasons'])}"
        if log_to_db:
            save_dropped_idea(
                ticker=clean_ticker,
                strategy=strategy_idea['strategy_name'],
                price=strategy_idea['entry_reference_price'],
                drop_reasons=risk_gate['drop_reasons'],
                conviction_score=strategy_idea['conviction_score'],
                monitor_status=monitor['status'],
                db_path=target_db,
            )
            save_proposal(
                symbol=clean_ticker,
                as_of=market['as_of'],
                strategy=strategy_idea['strategy_name'],
                direction='up' if strategy_idea['action'] == 'SUGGEST_LONG' else 'down',
                entry_ref_price=strategy_idea['entry_reference_price'],
                status='rejected',
                horizon_days=int(strategy_idea.get('horizon_days', 5)),
                probability=ml_prob,
                backtest_accuracy=ml_acc,
                conviction_score=strategy_idea['conviction_score'],
                db_path=target_db,
            )
            score_past_proposals({clean_ticker: data}, db_path=target_db)
            save_workflow_audit(
                ticker=clean_ticker,
                strategy=strategy_idea['strategy_name'],
                outcome=outcome,
                monitor_status=monitor['status'],
                warnings=monitor['warnings'],
                headline=headline,
                db_path=target_db,
            )
            record_heartbeat('monitor', 'ok', f"{clean_ticker}: {monitor['status']}", db_path=target_db)
        report = None
    else:
        # (Yes) -> [6. Monitor Agent] pre-checks -> [5. Report (AI)] -> [6. Monitor Agent] final report check
        precheck = monitor_agent.audit(
            df=data,
            market=market,
            research=research,
            strategy_idea=strategy_idea,
            risk_gate=risk_gate,
            bankroll=bankroll,
            policy=active_policy,
            report=None,
        )
        ideas_dir = (target_db.parent / 'reports' / 'ideas') if log_to_db else None
        report = ReportAgent(llm_config=llm_config).generate(
            ticker=clean_ticker,
            market=market,
            research=research,
            strategy_idea=strategy_idea,
            risk_gate=risk_gate,
            monitor_precheck=precheck,
            bankroll=bankroll,
            output_dir=ideas_dir,
        )
        monitor = monitor_agent.audit(
            df=data,
            market=market,
            research=research,
            strategy_idea=strategy_idea,
            risk_gate=risk_gate,
            bankroll=bankroll,
            policy=active_policy,
            report=report,
        )
        outcome = 'SUGGESTION_READY'
        headline = report['headline']
        if log_to_db:
            record_heartbeat('report', 'ok', f"{clean_ticker}: wrote suggestion brief", db_path=target_db)
            save_proposal(
                symbol=clean_ticker,
                as_of=market['as_of'],
                strategy=strategy_idea['strategy_name'],
                direction='up',
                entry_ref_price=strategy_idea['entry_reference_price'],
                status='approved',
                horizon_days=int(strategy_idea.get('horizon_days', 5)),
                probability=ml_prob,
                backtest_accuracy=ml_acc,
                conviction_score=strategy_idea['conviction_score'],
                db_path=target_db,
            )
            score_past_proposals({clean_ticker: data}, db_path=target_db)
            save_workflow_audit(
                ticker=clean_ticker,
                strategy=strategy_idea['strategy_name'],
                outcome=outcome,
                monitor_status=monitor['status'],
                warnings=monitor['warnings'],
                headline=headline,
                db_path=target_db,
            )
            record_heartbeat('monitor', 'ok', f"{clean_ticker}: {monitor['status']}", db_path=target_db)

    return data, {
        'ticker': clean_ticker,
        'source': source,
        'as_of': market['as_of'],
        'outcome': outcome,
        'market': market,
        'analyst': strategy_idea['analyst_screening'],
        'research': research,
        'strategy': strategy_idea,
        'risk': risk_gate,
        'report': report,
        'monitor': monitor,
    }

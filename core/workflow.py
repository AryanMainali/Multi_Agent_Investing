"""Straight-line four-agent orchestration; LangGraph is an optional future extension."""
from agents.market_agent import MarketAgent
from agents.data_agent import DataAgent
from agents.analyst_agent import AnalystAgent
from agents.risk_agent import RiskPolicy, size_position


def analyze(
    ticker: str = 'AAPL',
    bankroll: float = 10000.0,
    policy: RiskPolicy | None = None,
    period: str = '2y',
    source: str = 'yahoo',
):
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

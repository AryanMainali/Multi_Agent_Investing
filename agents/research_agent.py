"""Stage 2 [Research (AI)]: Qualitative news & fundamental context synthesizer.

Fetches real company metadata and recent news headlines from Yahoo Finance.
Uses an LLM when configured via LLMConfig / environment variables, and falls back
to transparent rule-assisted synthesis over the real fetched data otherwise.
"""
from dataclasses import dataclass
import yfinance as yf
from core.llm import LLMConfig, generate_text, parse_json_object

POSITIVE_WORDS = {
    'beat', 'beats', 'surge', 'surges', 'rally', 'upgrade', 'upgraded', 'growth',
    'record', 'strong', 'bullish', 'outperform', 'profit', 'gain', 'gains', 'launch',
    'partnership', 'demand', 'buyback', 'dividend', 'optimism', 'breakthrough',
}
NEGATIVE_WORDS = {
    'miss', 'misses', 'drop', 'drops', 'plunge', 'downgrade', 'downgraded', 'lawsuit',
    'probe', 'investigation', 'weak', 'bearish', 'underperform', 'loss', 'cut', 'cuts',
    'layoff', 'layoffs', 'recall', 'tariff', 'warning', 'delay', 'slump', 'decline',
}


def _extract_headlines(raw_news: list | None, limit: int = 6) -> list[dict]:
    items = []
    for entry in (raw_news or [])[:limit]:
        if not isinstance(entry, dict):
            continue
        content = entry.get('content') if isinstance(entry.get('content'), dict) else entry
        title = str(content.get('title') or entry.get('title') or '').strip()
        if not title:
            continue
        summary = str(content.get('summary') or entry.get('summary') or '').strip()
        provider = content.get('provider')
        if isinstance(provider, dict):
            publisher = str(provider.get('displayName') or 'Yahoo Finance')
        else:
            publisher = str(entry.get('publisher') or 'Yahoo Finance')
        pub_date = str(content.get('pubDate') or entry.get('providerPublishTime') or '')
        items.append({
            'title': title,
            'summary': summary[:240],
            'publisher': publisher,
            'published_at': pub_date,
        })
    return items


def _extract_fundamentals(raw_info: dict | None) -> dict:
    info = raw_info if isinstance(raw_info, dict) else {}
    fields = {}
    for key in (
        'shortName',
        'sector',
        'industry',
        'marketCap',
        'trailingPE',
        'forwardPE',
        'beta',
        'fiftyTwoWeekHigh',
        'fiftyTwoWeekLow',
    ):
        val = info.get(key)
        if val is not None and val != '':
            fields[key] = val
    return fields


def _deterministic_synthesis(
    ticker: str,
    market: dict,
    headlines: list[dict],
    fundamentals: dict,
) -> dict:
    pos_hits = 0
    neg_hits = 0
    for h in headlines:
        blob = f"{h['title']} {h['summary']}".lower()
        tokens = {tok.strip('.,:;!?()[]{}"\'') for tok in blob.split()}
        pos_hits += len(tokens & POSITIVE_WORDS)
        neg_hits += len(tokens & NEGATIVE_WORDS)

    score = 0.0
    if pos_hits + neg_hits > 0:
        score += 0.6 * ((pos_hits - neg_hits) / (pos_hits + neg_hits))

    mom = float(market.get('momentum_20_pct', 0.0))
    if mom > 3.0:
        score += 0.25
    elif mom < -3.0:
        score -= 0.25

    trailing_pe = fundamentals.get('trailingPE')
    if isinstance(trailing_pe, (int, float)):
        if 0 < trailing_pe < 25:
            score += 0.10
        elif trailing_pe > 55:
            score -= 0.10

    score = max(-1.0, min(1.0, round(score, 2)))
    if score >= 0.20:
        sentiment = 'BULLISH'
    elif score <= -0.20:
        sentiment = 'BEARISH'
    else:
        sentiment = 'NEUTRAL'

    catalysts = []
    risks = []

    if mom > 0:
        catalysts.append(f"Positive 20-day price momentum ({mom:+.2f}%).")
    else:
        risks.append(f"Negative 20-day price momentum ({mom:+.2f}%).")

    high_52 = fundamentals.get('fiftyTwoWeekHigh')
    low_52 = fundamentals.get('fiftyTwoWeekLow')
    price = float(market.get('price', 0.0))
    if isinstance(high_52, (int, float)) and high_52 > 0 and price > 0:
        pct_from_high = (price / high_52 - 1.0) * 100
        if pct_from_high >= -8.0:
            catalysts.append(f"Trading within {abs(pct_from_high):.1f}% of its 52-week high (${high_52:.2f}).")
        elif pct_from_high <= -25.0:
            risks.append(f"Trading {abs(pct_from_high):.1f}% below its 52-week high (${high_52:.2f}).")

    if isinstance(trailing_pe, (int, float)) and trailing_pe > 45:
        risks.append(f"Elevated trailing P/E multiple ({trailing_pe:.1f}x) increases sensitivity to earnings misses.")
    elif isinstance(trailing_pe, (int, float)) and 0 < trailing_pe <= 25:
        catalysts.append(f"Moderate trailing P/E valuation ({trailing_pe:.1f}x).")

    vol_pct = float(market.get('volatility_20_pct', 0.0))
    if vol_pct >= 35.0:
        risks.append(f"Elevated 20-day annualized volatility ({vol_pct:.1f}%) raises stop-out and gap risk.")
    else:
        catalysts.append(f"Contained 20-day annualized volatility ({vol_pct:.1f}%).")

    if headlines:
        catalysts.append(f"Reviewed {len(headlines)} recent Yahoo Finance headline(s); latest: \"{headlines[0]['title']}\".")
    else:
        risks.append('No recent ticker news headlines returned by feed; sentiment relies on market & fundamental context.')

    if not catalysts:
        catalysts.append('Price and volume data validated without structural anomalies.')
    if not risks:
        risks.append('Daily protective stops remain exposed to overnight earnings or macro price gaps.')

    company_name = fundamentals.get('shortName') or ticker
    sector = fundamentals.get('sector')
    sector_label = f" ({sector})" if sector else ''
    summary = (
        f"{company_name}{sector_label} carries a {sentiment.lower()} research assessment (score {score:+.2f}) "
        f"with {len(headlines)} recent news item(s), {mom:+.2f}% 20-day momentum, and {vol_pct:.1f}% annualized volatility."
    )

    return {
        'sentiment': sentiment,
        'sentiment_score': score,
        'catalysts': catalysts[:4],
        'risks': risks[:4],
        'summary': summary,
    }


@dataclass
class ResearchAgent:
    llm_config: LLMConfig | None = None

    def research(self, ticker: str, market: dict) -> dict:
        symbol = ticker.strip().upper()
        raw_news = []
        raw_info = {}
        feed_notes = []

        try:
            tkr = yf.Ticker(symbol)
            try:
                if hasattr(tkr, 'get_news'):
                    raw_news = tkr.get_news(count=6) or []
                else:
                    raw_news = getattr(tkr, 'news', []) or []
            except Exception as exc:
                feed_notes.append(f'News unavailable ({type(exc).__name__})')
            try:
                info_attr = getattr(tkr, 'info', None)
                if isinstance(info_attr, dict):
                    raw_info = info_attr
            except Exception as exc:
                feed_notes.append(f'Company metadata unavailable ({type(exc).__name__})')
        except Exception as exc:
            feed_notes.append(f'Ticker metadata lookup skipped ({type(exc).__name__})')

        headlines = _extract_headlines(raw_news)
        fundamentals = _extract_fundamentals(raw_info)
        baseline = _deterministic_synthesis(symbol, market, headlines, fundamentals)

        system_prompt = (
            'You are the Research (AI) Agent in a strictly educational, paper-only stock analysis workflow. '
            'Agents only suggest; humans decide and trade manually. Never recommend automated execution or guarantee returns. '
            'Return ONLY valid JSON with keys: sentiment (BULLISH, NEUTRAL, or BEARISH), sentiment_score (float -1.0 to 1.0), '
            'catalysts (list of up to 4 short strings), risks (list of up to 4 short strings), summary (2-3 plain-English sentences).'
        )
        user_prompt = (
            f"Ticker: {symbol}\n"
            f"Market snapshot: {market}\n"
            f"Fundamentals: {fundamentals}\n"
            f"Recent headlines: {[h['title'] for h in headlines]}\n"
        )
        llm_raw, ai_mode = generate_text(system_prompt, user_prompt, self.llm_config)
        parsed = parse_json_object(llm_raw)

        if parsed and str(parsed.get('sentiment', '')).upper() in ('BULLISH', 'NEUTRAL', 'BEARISH'):
            sentiment = str(parsed['sentiment']).upper()
            try:
                score = max(-1.0, min(1.0, round(float(parsed.get('sentiment_score', baseline['sentiment_score'])), 2)))
            except (TypeError, ValueError):
                score = baseline['sentiment_score']
            catalysts = [str(x) for x in (parsed.get('catalysts') or baseline['catalysts'])][:4]
            risks = [str(x) for x in (parsed.get('risks') or baseline['risks'])][:4]
            summary = str(parsed.get('summary') or baseline['summary']).strip()
        else:
            sentiment = baseline['sentiment']
            score = baseline['sentiment_score']
            catalysts = baseline['catalysts']
            risks = baseline['risks']
            summary = baseline['summary']

        return {
            'ticker': symbol,
            'sentiment': sentiment,
            'sentiment_score': score,
            'catalysts': catalysts,
            'risks': risks,
            'summary': summary,
            'headlines': headlines,
            'fundamentals': fundamentals,
            'ai_mode': ai_mode,
            'feed_notes': feed_notes,
        }

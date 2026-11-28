"""Stage 2 [Research (AI)]: Qualitative news, event-flagging & fundamental synthesizer.

Fetches real company metadata and recent news headlines from Yahoo Finance (plus optional
read-only Alpaca News when ALPACA_API_KEY / ALPACA_SECRET_KEY are set in .env).
Flags structured market-moving events (earnings, guidance, legal, regulatory, merger,
analyst_rating, product, management, macro) with low/medium/high severity so the
Risk Gate can automatically block high-severity adverse news events.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
import requests
import yfinance as yf
from core.config import load_env, load_settings
from core.llm import LLMConfig, call_structured_tool, generate_text, parse_json_object
from core.universe import get_known_sector

EVENT_KINDS = (
    'earnings',
    'guidance',
    'legal',
    'regulatory',
    'analyst_rating',
    'merger',
    'management',
    'product',
    'macro',
    'other',
)

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

# Ordered rules: (regex_pattern, kind, severity, is_adverse)
HEADLINE_EVENT_RULES: list[tuple[str, str, str, bool]] = [
    (r'\b(lawsuit|sued|litigation|class[- ]action|subpoena|indictment|fraud)\b', 'legal', 'high', True),
    (r'\b(probe|sec investigation|doj investigation|antitrust probe|regulator[s]? open|fda reject\w*)\b', 'regulatory', 'high', True),
    (r'\b(recall|recalled|safety halt|data breach)\b', 'product', 'high', True),
    (r'\b(guidance cut|cuts guidance|lowers guidance|slashes guidance|profit warning|weak guidance)\b', 'guidance', 'high', True),
    (r'\b(earnings miss|missed estimates|misses estimates|revenue miss|wider loss)\b', 'earnings', 'high', True),
    (r'\b(raises guidance|lifts guidance|boosts outlook)\b', 'guidance', 'medium', False),
    (r'\b(earnings|quarterly results|eps|q[1-4] results)\b', 'earnings', 'medium', False),
    (r'\b(downgrade|downgraded|cuts price target|underperform)\b', 'analyst_rating', 'medium', True),
    (r'\b(upgrade|upgraded|raises price target|outperform)\b', 'analyst_rating', 'low', False),
    (r'\b(acquire|acquires|acquisition|merger|buyout|takeover)\b', 'merger', 'medium', False),
    (r'\b(ceo|cfo|resigns|steps down|executive)\b', 'management', 'medium', False),
    (r'\b(unveils|launch|launches|product line|breakthrough)\b', 'product', 'low', False),
    (r'\b(fed|interest rate|inflation|cpi|tariff|tariffs)\b', 'macro', 'low', False),
]

FLAG_TOOL_SCHEMA = {
    'name': 'record_flags',
    'description': 'Record structured market-relevant events found in recent headlines.',
    'input_schema': {
        'type': 'object',
        'properties': {
            'flags': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'kind': {'type': 'string', 'enum': list(EVENT_KINDS)},
                        'severity': {'type': 'string', 'enum': ['low', 'medium', 'high']},
                        'adverse': {'type': 'boolean', 'description': 'True if negative/risk event for longs'},
                        'summary': {'type': 'string', 'description': 'Plain words, 25 words max'},
                        'headline': {'type': 'string'},
                    },
                    'required': ['kind', 'severity', 'adverse', 'summary', 'headline'],
                },
            },
        },
        'required': ['flags'],
    },
}


def classify_headline_events(symbol: str, headlines: list[dict]) -> list[dict]:
    """Deterministically flag material corporate/market events and severity from headlines."""
    clean_sym = symbol.strip().upper()
    flags: list[dict] = []
    seen_titles: set[str] = set()

    for item in headlines:
        title = str(item.get('title') or item.get('headline') or '').strip()
        summary = str(item.get('summary') or '').strip()
        if not title or title in seen_titles:
            continue
        blob = f"{title} {summary}".lower()
        for pattern, kind, severity, adverse in HEADLINE_EVENT_RULES:
            if re.search(pattern, blob):
                # Check if a generic earnings/management match actually contains strong negative words
                is_neg = adverse or any(
                    w in blob for w in ('miss', 'plunge', 'resigns', 'investigation', 'cut', 'warning', 'slump')
                )
                eff_severity = 'high' if (kind in ('guidance', 'earnings') and 'miss' in blob) else severity
                flags.append({
                    'symbol': clean_sym,
                    'kind': kind,
                    'severity': eff_severity,
                    'adverse': bool(is_neg),
                    'summary': title[:220],
                    'headline': title[:300],
                })
                seen_titles.add(title)
                break
    return flags


def _fetch_optional_alpaca_headlines(symbol: str, lookback_days: int = 3) -> list[dict]:
    """Read-only market news lookup if ALPACA_API_KEY & ALPACA_SECRET_KEY are set in .env."""
    env = load_env()
    if not env.alpaca_news_enabled:
        return []
    start_iso = (datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))).strftime('%Y-%m-%dT%H:%M:%SZ')
    try:
        resp = requests.get(
            'https://data.alpaca.markets/v1beta1/news',
            headers={
                'APCA-API-KEY-ID': env.alpaca_key or '',
                'APCA-API-SECRET-KEY': env.alpaca_secret or '',
            },
            params={'symbols': symbol, 'start': start_iso, 'limit': 6},
            timeout=8.0,
        )
        resp.raise_for_status()
        news_items = resp.json().get('news', [])
        out = []
        for n in news_items:
            title = str(n.get('headline') or '').strip()
            if title:
                out.append({
                    'title': title,
                    'summary': str(n.get('summary') or '')[:240],
                    'publisher': str(n.get('source') or 'Alpaca News'),
                    'published_at': str(n.get('created_at') or ''),
                })
        return out
    except Exception:
        return []


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
    event_flags: list[dict],
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

    adverse_high = [f for f in event_flags if f.get('severity') == 'high' and f.get('adverse')]
    if adverse_high:
        score -= 0.45

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

    for af in adverse_high[:2]:
        risks.append(f"High-severity {af['kind']} flag: {af['summary']}")

    if mom > 0:
        catalysts.append(f"Positive 20-day price momentum ({mom:+.2f}%).")
    else:
        risks.append(f"Negative 20-day price momentum ({mom:+.2f}%).")

    high_52 = fundamentals.get('fiftyTwoWeekHigh')
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
        catalysts.append(f"Reviewed {len(headlines)} recent headline(s); latest: \"{headlines[0]['title']}\".")
    else:
        risks.append('No recent ticker news headlines returned by feed; sentiment relies on market & fundamental context.')

    if not catalysts:
        catalysts.append('Price and volume data validated without structural anomalies.')
    if not risks:
        risks.append('Daily protective stops remain exposed to overnight earnings or macro price gaps.')

    company_name = fundamentals.get('shortName') or ticker
    sector = fundamentals.get('sector')
    sector_label = f" ({sector})" if sector else ''
    flag_note = f", {len(event_flags)} event flag(s) ({len(adverse_high)} high-severity risk)" if event_flags else ''
    summary = (
        f"{company_name}{sector_label} carries a {sentiment.lower()} research assessment (score {score:+.2f}) "
        f"with {len(headlines)} recent news item(s){flag_note}, {mom:+.2f}% 20-day momentum, and {vol_pct:.1f}% annualized volatility."
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
        settings = load_settings()
        lookback_days = int(settings.get('research', {}).get('news_lookback_days', 3))

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
        extra_alpaca = _fetch_optional_alpaca_headlines(symbol, lookback_days=lookback_days)
        if extra_alpaca:
            existing_titles = {h['title'].lower() for h in headlines}
            for item in extra_alpaca:
                if item['title'].lower() not in existing_titles:
                    headlines.append(item)
                    existing_titles.add(item['title'].lower())

        fundamentals = _extract_fundamentals(raw_info)
        if 'sector' not in fundamentals:
            known_sec = get_known_sector(symbol)
            if known_sec != 'Unknown':
                fundamentals['sector'] = known_sec

        event_flags = classify_headline_events(symbol, headlines)
        flag_source = 'keywords'

        # When headlines exist and an Anthropic tool-capable key is configured, refine event flags via schema
        if headlines and self.llm_config and self.llm_config.provider == 'anthropic':
            flag_sys = (
                'You are a financial news analyst on a research team. '
                'Only report events actually stated in the headlines. Never invent facts or predict prices. '
                'severity: high = sharp adverse shock (earnings miss, lawsuit, probe, fraud, FDA rejection, recall, guidance cut); '
                'medium = notable; low = routine.'
            )
            flag_usr = f"Symbol: {symbol}\nHeadlines:\n" + '\n'.join(f"- {h['title']}" for h in headlines)
            tool_out, tool_mode = call_structured_tool(flag_sys, flag_usr, FLAG_TOOL_SCHEMA, config=self.llm_config)
            if isinstance(tool_out, dict) and isinstance(tool_out.get('flags'), list):
                parsed_flags = []
                for f in tool_out['flags']:
                    if isinstance(f, dict) and f.get('kind') in EVENT_KINDS and f.get('severity') in ('low', 'medium', 'high'):
                        parsed_flags.append({
                            'symbol': symbol,
                            'kind': str(f['kind']),
                            'severity': str(f['severity']),
                            'adverse': bool(f.get('adverse', False)),
                            'summary': str(f.get('summary') or f.get('headline') or '')[:220],
                            'headline': str(f.get('headline') or f.get('summary') or '')[:300],
                        })
                if parsed_flags:
                    event_flags = parsed_flags
                    flag_source = tool_mode

        baseline = _deterministic_synthesis(symbol, market, headlines, fundamentals, event_flags)

        system_prompt = (
            'You are the Research (AI) Agent in a strictly educational, paper-only stock analysis workflow. '
            'Agents only suggest; humans decide and trade manually. Never recommend automated execution or guarantee returns. '
            'Only report facts stated in the provided snapshot and headlines. '
            'Return ONLY valid JSON with keys: sentiment (BULLISH, NEUTRAL, or BEARISH), sentiment_score (float -1.0 to 1.0), '
            'catalysts (list of up to 4 short strings), risks (list of up to 4 short strings), summary (2-3 plain-English sentences).'
        )
        user_prompt = (
            f"Ticker: {symbol}\n"
            f"Market snapshot: {market}\n"
            f"Fundamentals: {fundamentals}\n"
            f"Recent headlines: {[h['title'] for h in headlines]}\n"
            f"Structured event flags: {event_flags}\n"
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

        high_severity_flags = [
            f['summary'] for f in event_flags if f.get('severity') == 'high' and f.get('adverse')
        ]

        return {
            'ticker': symbol,
            'sentiment': sentiment,
            'sentiment_score': score,
            'catalysts': catalysts,
            'risks': risks,
            'summary': summary,
            'headlines': headlines,
            'event_flags': event_flags,
            'high_severity_flags': high_severity_flags,
            'flag_source': flag_source,
            'fundamentals': fundamentals,
            'ai_mode': ai_mode,
            'feed_notes': feed_notes,
        }

"""Real S&P 100 and S&P 500 ticker universes (Yahoo Finance symbol format).

Contains ticker symbols ONLY — zero synthetic price data. All OHLCV bars,
fundamentals, and news are fetched live from Yahoo Finance.
"""
from io import StringIO
import re
import pandas as pd
import requests

# S&P 100 liquid large-cap constituents (Yahoo Finance symbol format: '.' -> '-')
SP100_TICKERS: list[str] = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT', 'AMZN',
    'AVGO', 'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BMY', 'BRK-B', 'C',
    'CAT', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS',
    'CVX', 'DE', 'DHR', 'DIS', 'DUK', 'EMR', 'FDX', 'GD', 'GE', 'GILD',
    'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'INTU', 'ISRG',
    'JNJ', 'JPM', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'MA', 'MCD', 'MDLZ',
    'MDT', 'MET', 'META', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'NEE', 'NFLX',
    'NKE', 'NOW', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PLTR', 'PM', 'PYPL',
    'QCOM', 'RTX', 'SBUX', 'SCHW', 'SO', 'SPG', 'T', 'TGT', 'TMO', 'TMUS',
    'TSLA', 'TXN', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'VZ', 'WFC', 'WMT', 'XOM',
]

# Full S&P 500 constituent fallback symbol list (Yahoo Finance format)
SP500_FALLBACK_TICKERS: list[str] = sorted(set(SP100_TICKERS + [
    'A', 'ABNB', 'ACGL', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES',
    'AFL', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR',
    'AME', 'AMP', 'ANET', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APO', 'APTV',
    'ARE', 'ATO', 'AVB', 'AVY', 'AWK', 'AXON', 'AZO', 'BALL', 'BAX', 'BBY',
    'BDX', 'BEN', 'BF-B', 'BG', 'BIIB', 'BLDR', 'BKR', 'BR', 'BRO', 'BSX',
    'BX', 'BXP', 'CAG', 'CAH', 'CARR', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL',
    'CDNS', 'CDW', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CI', 'CINF', 'CLX',
    'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COO', 'COR', 'CPAY', 'CPB', 'CPRT',
    'CPT', 'CRL', 'CRWD', 'CSGP', 'CSX', 'CTAS', 'CTRA', 'CTSH', 'CTVA', 'D',
    'DAL', 'DAY', 'DD', 'DECK', 'DELL', 'DG', 'DGX', 'DHI', 'DLR', 'DLTR',
    'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DVA', 'DVN', 'DXCM', 'EA',
    'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN', 'ENPH',
    'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ERIE', 'ES', 'ESS', 'ETN', 'ETR',
    'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCX',
    'FDS', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB', 'FOX', 'FOXA', 'FRT',
    'FSLR', 'FTNT', 'FTV', 'GDDY', 'GEHC', 'GEN', 'GEV', 'GIS', 'GL', 'GLW',
    'GNRC', 'GPC', 'GPN', 'GRMN', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HES',
    'HIG', 'HII', 'HLT', 'HOLX', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY',
    'HUBB', 'HUM', 'HWM', 'ICE', 'IDXX', 'IEX', 'IFF', 'INCY', 'INVH', 'IP',
    'IPG', 'IQV', 'IR', 'IRM', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL',
    'JCI', 'JKHY', 'JNPR', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KKR',
    'KLAC', 'KMB', 'KMI', 'KMX', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH',
    'LHX', 'LKQ', 'LNT', 'LRCX', 'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYV',
    'MAA', 'MAR', 'MAS', 'MCHP', 'MCK', 'MCO', 'MGM', 'MHK', 'MKC', 'MKTX',
    'MLM', 'MMC', 'MNST', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRNA', 'MSI', 'MTB',
    'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEM', 'NI', 'NOC', 'NRG',
    'NSC', 'NTAP', 'NTRS', 'NUE', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL',
    'OKE', 'OMC', 'ON', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PARA', 'PAYC', 'PAYX',
    'PCAR', 'PCG', 'PEG', 'PFG', 'PH', 'PHM', 'PKG', 'PLD', 'PNC', 'PNR',
    'PNW', 'PODD', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR',
    'Q', 'RCL', 'REG', 'REGN', 'RF', 'RJF', 'RL', 'RMD', 'ROK', 'ROL',
    'ROP', 'ROST', 'RSG', 'RVTY', 'SBAC', 'SJM', 'SLB', 'SMCI', 'SNA', 'SNPS',
    'SOLV', 'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SW', 'SWK',
    'SWKS', 'SYF', 'SYK', 'SYY', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER',
    'TFC', 'TFX', 'TJX', 'TKO', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO',
    'TSN', 'TT', 'TTWO', 'TYL', 'UAL', 'UBER', 'UDR', 'UHS', 'ULTA', 'URI',
    'VICI', 'VLO', 'VLTO', 'VMC', 'VRSK', 'VRSN', 'VRTX', 'VST', 'VTR', 'VTRS',
    'WAB', 'WAT', 'WBA', 'WBD', 'WDAY', 'WDC', 'WEC', 'WELL', 'WM', 'WMB',
    'WRB', 'WSM', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBH',
    'ZBRA', 'ZTS',
]))

WIKIPEDIA_SP500_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'


def normalize_yahoo_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace('.', '-')


def fetch_sp500_tickers_live(timeout: float = 6.0) -> list[str]:
    """Fetch live S&P 500 constituent symbols from Wikipedia, falling back to SP500_FALLBACK_TICKERS."""
    try:
        resp = requests.get(
            WIKIPEDIA_SP500_URL,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; MultiAgentStockLab/1.0)'},
            timeout=timeout,
        )
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        for tbl in tables:
            if 'Symbol' in tbl.columns:
                syms = [
                    normalize_yahoo_symbol(str(s))
                    for s in tbl['Symbol'].dropna().tolist()
                ]
                valid = [s for s in syms if re.fullmatch(r'[A-Z0-9\-]{1,10}', s)]
                if len(valid) >= 450:
                    return sorted(set(valid))
    except Exception:
        pass
    return list(SP500_FALLBACK_TICKERS)


def get_universe_tickers(scope: str = 'sp100', refresh_web: bool = True) -> list[str]:
    """Return real ticker symbols for 'sp100' or 'sp500'."""
    key = scope.strip().lower()
    if key == 'sp100':
        return list(SP100_TICKERS)
    if key == 'sp500':
        return fetch_sp500_tickers_live() if refresh_web else list(SP500_FALLBACK_TICKERS)
    raise ValueError("Universe scope must be 'sp100' or 'sp500'")

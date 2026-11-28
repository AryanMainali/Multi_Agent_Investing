"""Shared team configuration (config/settings.yaml), .env loader, and structured logger.

Adapted from the team's agent configuration design:
- Humans edit config/settings.yaml; agents only read it.
- Optional API keys and alert credentials load cleanly from .env or environment variables.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / 'config' / 'settings.yaml'
ENV_PATH = ROOT_DIR / '.env'
REPORTS_DIR = ROOT_DIR / 'outputs' / 'reports'

DEFAULT_SETTINGS: dict = {
    'watchlist': ['AAPL', 'MSFT', 'NVDA', 'JPM', 'SPY'],
    'market_data': {
        'lookback_days': 730,
    },
    'research': {
        'news_lookback_days': 3,
    },
    'strategy': {
        'horizon_days': 5,
        'min_history': 90,
        'test_window': 40,
    },
    'risk': {
        'min_confidence': 0.53,
        'min_backtest_accuracy': 0.51,
        'max_data_age_days': 5,
        'block_on_high_severity_days': 3,
        'max_ideas_per_day': 10,
        'kelly_fraction': 0.25,
        'max_position_pct': 20.0,
    },
    'monitor': {
        'stale_after_hours': 30,
    },
}


def load_dotenv_file(dotenv_path: Path | None = None, override: bool = False) -> None:
    """Load key=value pairs from .env into os.environ without external dependencies."""
    target = dotenv_path or ENV_PATH
    if not target.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(target, override=override)
        return
    except ImportError:
        pass

    for raw_line in target.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, val = line.split('=', 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val = val[1:-1]
        elif ' #' in val:
            val = val.split(' #', 1)[0].strip()
        if key and (override or key not in os.environ):
            os.environ[key] = val


def _parse_scalar(raw: str):
    val = raw.strip()
    if val.startswith('[') and val.endswith(']'):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in inner.split(',')]
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    low = val.lower()
    if low in ('true', 'yes', 'on'):
        return True
    if low in ('false', 'no', 'off'):
        return False
    if low in ('null', 'none', '~'):
        return None
    try:
        if '.' in val or 'e' in low:
            return float(val)
        return int(val)
    except ValueError:
        return val


def _parse_simple_yaml(text: str) -> dict:
    """Parse flat/two-level settings.yaml files cleanly when PyYAML is not installed."""
    result: dict = {}
    current_section: dict | None = None

    for raw_line in text.splitlines():
        # Strip inline comments outside quotes
        in_single = False
        in_double = False
        cleaned_chars = []
        for ch in raw_line:
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif ch == '#' and not in_single and not in_double:
                break
            cleaned_chars.append(ch)
        line = ''.join(cleaned_chars).rstrip()
        if not line.strip() or ':' not in line:
            continue

        indent = len(line) - len(line.lstrip(' '))
        key_part, val_part = line.strip().split(':', 1)
        key = key_part.strip()
        val_str = val_part.strip()

        if indent == 0:
            if val_str == '':
                current_section = {}
                result[key] = current_section
            else:
                current_section = None
                result[key] = _parse_scalar(val_str)
        elif current_section is not None and val_str != '':
            current_section[key] = _parse_scalar(val_str)

    return result


def load_settings(settings_path: Path | None = None) -> dict:
    """Load shared agent settings from config/settings.yaml merged over safe defaults."""
    path = settings_path or CONFIG_PATH
    merged = {
        k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
        for k, v in DEFAULT_SETTINGS.items()
    }
    if not path.exists():
        return merged

    text = path.read_text(encoding='utf-8')
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(text) or {}
    except ImportError:
        loaded = _parse_simple_yaml(text)

    if isinstance(loaded, dict):
        for k, v in loaded.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k].update(v)
            else:
                merged[k] = v
    return merged


def _clean_env(name: str) -> str | None:
    val = (os.environ.get(name) or '').strip()
    return val or None


@dataclass(frozen=True)
class EnvConfig:
    openai_key: str | None
    anthropic_key: str | None
    gemini_key: str | None
    claude_model: str
    alpaca_key: str | None
    alpaca_secret: str | None
    telegram_token: str | None
    telegram_chat_id: str | None

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_key or self.anthropic_key or self.gemini_key)

    @property
    def alpaca_news_enabled(self) -> bool:
        return bool(self.alpaca_key and self.alpaca_secret)

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat_id)


def load_env(dotenv_path: Path | None = None) -> EnvConfig:
    """Load optional environment credentials (.env + process environment)."""
    load_dotenv_file(dotenv_path=dotenv_path)
    return EnvConfig(
        openai_key=_clean_env('OPENAI_API_KEY'),
        anthropic_key=_clean_env('ANTHROPIC_API_KEY'),
        gemini_key=_clean_env('GEMINI_API_KEY'),
        claude_model=_clean_env('CLAUDE_MODEL') or 'claude-3-5-haiku-latest',
        alpaca_key=_clean_env('ALPACA_API_KEY'),
        alpaca_secret=_clean_env('ALPACA_SECRET_KEY'),
        telegram_token=_clean_env('TELEGRAM_BOT_TOKEN'),
        telegram_chat_id=_clean_env('TELEGRAM_CHAT_ID'),
    )


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter('%(asctime)s  %(levelname)-7s  %(name)-14s  %(message)s', '%H:%M:%S')
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

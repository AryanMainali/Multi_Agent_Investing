"""Guardrail: this project is suggestion-only and self-contained.

1. Fail the build immediately if any broker order-placing symbol or endpoint appears.
2. Fail the build if any root module imports from or references `trading-agents`, so
   the legacy subfolder can be safely deleted without affecting runtime or tests.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_ORDER_TOKENS = (
    'TradingClient',
    'submit_order',
    'MarketOrderRequest',
    'LimitOrderRequest',
    'StopOrderRequest',
    'place_order',
    '/v2/orders',
)


def _iter_project_python_files() -> list[Path]:
    files: list[Path] = []
    for folder in ('agents', 'core'):
        files.extend(sorted((ROOT / folder).rglob('*.py')))
    for top_file in ('app.py', 'run_demo.py'):
        p = ROOT / top_file
        if p.exists():
            files.append(p)
    return files


def test_no_order_placing_code():
    hits = [
        f"{path.relative_to(ROOT)}: {token}"
        for path in _iter_project_python_files()
        for token in FORBIDDEN_ORDER_TOKENS
        if token in path.read_text(encoding='utf-8')
    ]
    assert not hits, 'Order-placing code found (this system must never trade):\n' + '\n'.join(hits)


def test_no_references_to_legacy_trading_agents_folder():
    checked = _iter_project_python_files() + [
        p for p in sorted((ROOT / 'tests').rglob('*.py')) if p.name != 'test_no_order_code.py'
    ]
    hits = [
        str(path.relative_to(ROOT))
        for path in checked
        if 'trading-agents' in path.read_text(encoding='utf-8') or 'trading_agents' in path.read_text(encoding='utf-8')
    ]
    assert not hits, 'Found runtime/test dependency on legacy trading-agents folder:\n' + '\n'.join(hits)

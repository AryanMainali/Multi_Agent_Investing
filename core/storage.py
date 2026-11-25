"""Local audit trail for human-approved/rejected *suggestions*. Never submits orders."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[1] / 'outputs' / 'decisions.db'


def save_decision(ticker: str, source: str, decision: str, price: float, reason: str = '', db_path: Path = DEFAULT_DB):
    if decision not in ('APPROVE_PAPER', 'REJECT'):
        raise ValueError('Decision must be APPROVE_PAPER or REJECT')
    if price <= 0:
        raise ValueError('Price must be positive')
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_utc TEXT NOT NULL, ticker TEXT NOT NULL,
            source TEXT NOT NULL, decision TEXT NOT NULL,
            observed_price REAL NOT NULL, reason TEXT NOT NULL
        )''')
        conn.execute('''INSERT INTO decisions
            (timestamp_utc, ticker, source, decision, observed_price, reason)
            VALUES (?, ?, ?, ?, ?, ?)''',
            (datetime.now(timezone.utc).isoformat(), ticker, source, decision, float(price), reason.strip()))


def list_decisions(db_path: Path = DEFAULT_DB):
    path = Path(db_path)
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        return conn.execute('''SELECT timestamp_utc, ticker, source, decision,
          observed_price, reason FROM decisions ORDER BY id DESC LIMIT 100''').fetchall()

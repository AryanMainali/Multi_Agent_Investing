"""Local SQLite audit trail and manual paper-portfolio ledger.

Strictly records human-initiated paper positions, decisions, risk-dropped ideas,
and Monitor Agent audits. Never submits orders or connects to any brokerage.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

DEFAULT_DB = Path(__file__).resolve().parents[1] / 'outputs' / 'decisions.db'
VALID_HUMAN_DECISIONS = ('APPROVE_PAPER', 'REJECT', 'SKIP_IDEA', 'CLOSE_PAPER')


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute('''CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_utc TEXT NOT NULL,
        ticker TEXT NOT NULL,
        source TEXT NOT NULL,
        decision TEXT NOT NULL,
        observed_price REAL NOT NULL,
        reason TEXT NOT NULL
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS dropped_ideas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_utc TEXT NOT NULL,
        ticker TEXT NOT NULL,
        strategy TEXT NOT NULL,
        observed_price REAL NOT NULL,
        conviction_score INTEGER NOT NULL,
        drop_reasons TEXT NOT NULL,
        monitor_status TEXT NOT NULL
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS workflow_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_utc TEXT NOT NULL,
        ticker TEXT NOT NULL,
        strategy TEXT NOT NULL,
        outcome TEXT NOT NULL,
        monitor_status TEXT NOT NULL,
        warnings TEXT NOT NULL,
        headline TEXT NOT NULL
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS paper_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        opened_at_utc TEXT NOT NULL,
        closed_at_utc TEXT,
        status TEXT NOT NULL,
        ticker TEXT NOT NULL,
        strategy TEXT NOT NULL,
        shares INTEGER NOT NULL,
        entry_price REAL NOT NULL,
        stop_price REAL NOT NULL,
        target_price REAL NOT NULL,
        exit_price REAL,
        realized_pnl REAL,
        notes TEXT NOT NULL
    )''')


def save_decision(
    ticker: str,
    source: str,
    decision: str,
    price: float,
    reason: str = '',
    db_path: Path | None = None,
) -> None:
    if decision not in VALID_HUMAN_DECISIONS:
        raise ValueError(f'Decision must be one of {VALID_HUMAN_DECISIONS}')
    if price <= 0:
        raise ValueError('Price must be positive')
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        conn.execute(
            '''INSERT INTO decisions
            (timestamp_utc, ticker, source, decision, observed_price, reason)
            VALUES (?, ?, ?, ?, ?, ?)''',
            (datetime.now(timezone.utc).isoformat(), ticker.strip().upper(), source, decision, float(price), reason.strip()),
        )


def list_decisions(db_path: Path | None = None) -> list[tuple]:
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        return conn.execute(
            '''SELECT timestamp_utc, ticker, source, decision,
            observed_price, reason FROM decisions ORDER BY id DESC LIMIT 100'''
        ).fetchall()


def save_dropped_idea(
    ticker: str,
    strategy: str,
    price: float,
    drop_reasons: list[str],
    conviction_score: int = 0,
    monitor_status: str = 'HEALTHY',
    db_path: Path | None = None,
) -> None:
    if price <= 0:
        raise ValueError('Price must be positive')
    if not drop_reasons:
        raise ValueError('Dropped idea must record at least one reason')
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        conn.execute(
            '''INSERT INTO dropped_ideas
            (timestamp_utc, ticker, strategy, observed_price, conviction_score, drop_reasons, monitor_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (
                datetime.now(timezone.utc).isoformat(),
                ticker.strip().upper(),
                strategy,
                float(price),
                int(conviction_score),
                ' | '.join(drop_reasons),
                monitor_status,
            ),
        )


def list_dropped_ideas(db_path: Path | None = None) -> list[tuple]:
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        return conn.execute(
            '''SELECT timestamp_utc, ticker, strategy, observed_price,
            conviction_score, drop_reasons, monitor_status
            FROM dropped_ideas ORDER BY id DESC LIMIT 100'''
        ).fetchall()


def save_workflow_audit(
    ticker: str,
    strategy: str,
    outcome: str,
    monitor_status: str,
    warnings: list[str],
    headline: str,
    db_path: Path | None = None,
) -> None:
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        conn.execute(
            '''INSERT INTO workflow_audits
            (timestamp_utc, ticker, strategy, outcome, monitor_status, warnings, headline)
            VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (
                datetime.now(timezone.utc).isoformat(),
                ticker.strip().upper(),
                strategy,
                outcome,
                monitor_status,
                json.dumps(warnings),
                headline.strip(),
            ),
        )


def list_workflow_audits(db_path: Path | None = None) -> list[tuple]:
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        return conn.execute(
            '''SELECT timestamp_utc, ticker, strategy, outcome, monitor_status, warnings, headline
            FROM workflow_audits ORDER BY id DESC LIMIT 100'''
        ).fetchall()


def open_paper_position(
    ticker: str,
    strategy: str,
    shares: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
    notes: str = '',
    db_path: Path | None = None,
) -> int:
    """Record a manual paper long position in local SQLite."""
    symbol = ticker.strip().upper()
    if int(shares) < 1:
        raise ValueError('Shares must be at least 1 whole share')
    if entry_price <= 0:
        raise ValueError('Entry price must be positive')
    if not 0 < stop_price < entry_price:
        raise ValueError('Stop price must be positive and below entry price')
    if target_price <= entry_price:
        raise ValueError('Target price must be above entry price')

    path = Path(db_path) if db_path is not None else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        existing = conn.execute(
            "SELECT id FROM paper_positions WHERE ticker = ? AND status = 'OPEN'",
            (symbol,),
        ).fetchone()
        if existing is not None:
            raise ValueError(f'An open paper position for {symbol} already exists (ID #{existing[0]}).')

        cur = conn.execute(
            '''INSERT INTO paper_positions
            (opened_at_utc, closed_at_utc, status, ticker, strategy, shares,
             entry_price, stop_price, target_price, exit_price, realized_pnl, notes)
            VALUES (?, NULL, 'OPEN', ?, ?, ?, ?, ?, ?, NULL, NULL, ?)''',
            (
                datetime.now(timezone.utc).isoformat(),
                symbol,
                strategy,
                int(shares),
                round(float(entry_price), 4),
                round(float(stop_price), 4),
                round(float(target_price), 4),
                notes.strip(),
            ),
        )
        return int(cur.lastrowid)


def close_paper_position(
    position_id: int,
    exit_price: float,
    close_note: str = '',
    db_path: Path | None = None,
) -> dict:
    """Manually close an open paper position and record realized paper P&L."""
    if exit_price <= 0:
        raise ValueError('Exit price must be positive')
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    if not path.exists():
        raise ValueError(f'Position #{position_id} not found')

    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        row = conn.execute(
            '''SELECT id, ticker, strategy, shares, entry_price, notes, status
            FROM paper_positions WHERE id = ?''',
            (int(position_id),),
        ).fetchone()
        if row is None:
            raise ValueError(f'Position #{position_id} not found')
        if row[6] != 'OPEN':
            raise ValueError(f'Position #{position_id} is already closed')

        _, ticker, strategy, shares, entry_price, existing_notes, _ = row
        realized_pnl = round((float(exit_price) - float(entry_price)) * int(shares), 2)
        combined_notes = (
            f"{existing_notes} | Closed: {close_note.strip()}".strip(' |')
            if close_note.strip()
            else existing_notes
        )
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            '''UPDATE paper_positions
            SET closed_at_utc = ?, status = 'CLOSED', exit_price = ?, realized_pnl = ?, notes = ?
            WHERE id = ?''',
            (now_iso, round(float(exit_price), 4), realized_pnl, combined_notes, int(position_id)),
        )

    save_decision(
        ticker=ticker,
        source='yahoo',
        decision='CLOSE_PAPER',
        price=float(exit_price),
        reason=f"[{strategy}] Closed {shares} sh @ ${exit_price:.2f} (Realized PnL ${realized_pnl:+.2f}) {close_note}".strip(),
        db_path=path,
    )
    return {
        'id': int(position_id),
        'ticker': ticker,
        'shares': int(shares),
        'entry_price': float(entry_price),
        'exit_price': float(exit_price),
        'realized_pnl': realized_pnl,
    }


def list_paper_positions(status: str | None = None, db_path: Path | None = None) -> list[dict]:
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    if not path.exists():
        return []
    cols = [
        'id', 'opened_at_utc', 'closed_at_utc', 'status', 'ticker', 'strategy',
        'shares', 'entry_price', 'stop_price', 'target_price', 'exit_price',
        'realized_pnl', 'notes',
    ]
    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        if status in ('OPEN', 'CLOSED'):
            rows = conn.execute(
                f"SELECT {', '.join(cols)} FROM paper_positions WHERE status = ? ORDER BY id DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {', '.join(cols)} FROM paper_positions ORDER BY id DESC"
            ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def clear_paper_positions(db_path: Path | None = None) -> None:
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    if not path.exists():
        return
    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        conn.execute('DELETE FROM paper_positions')

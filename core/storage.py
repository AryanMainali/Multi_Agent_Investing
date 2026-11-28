"""Local SQLite audit trail, manual paper-portfolio ledger, agent heartbeats, and idea scorecard.

Strictly records human-initiated paper positions, decisions, risk-dropped ideas,
structured research flags, agent health heartbeats, and retrospective idea scores
(evaluating both Approved and Risk-Rejected ideas after their N-day horizon passes).
Never submits orders or connects to any brokerage.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import pandas as pd

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
        sector TEXT NOT NULL DEFAULT 'Unknown',
        strategy TEXT NOT NULL,
        shares INTEGER NOT NULL,
        entry_price REAL NOT NULL,
        stop_price REAL NOT NULL,
        target_price REAL NOT NULL,
        exit_price REAL,
        realized_pnl REAL,
        notes TEXT NOT NULL
    )''')
    existing_cols = {row[1] for row in conn.execute('PRAGMA table_info(paper_positions)').fetchall()}
    if 'sector' not in existing_cols:
        conn.execute("ALTER TABLE paper_positions ADD COLUMN sector TEXT NOT NULL DEFAULT 'Unknown'")

    # Incorporated from team architecture: structured research flags, agent heartbeats, proposals & scorecard
    conn.execute('''CREATE TABLE IF NOT EXISTS research_flags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at_utc TEXT NOT NULL,
        symbol TEXT NOT NULL,
        kind TEXT NOT NULL,
        severity TEXT NOT NULL,
        adverse INTEGER NOT NULL DEFAULT 0,
        summary TEXT NOT NULL,
        headline TEXT NOT NULL,
        source TEXT NOT NULL
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS heartbeats (
        agent TEXT PRIMARY KEY,
        last_run_utc TEXT NOT NULL,
        status TEXT NOT NULL,
        detail TEXT NOT NULL
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS proposals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at_utc TEXT NOT NULL,
        as_of TEXT NOT NULL,
        symbol TEXT NOT NULL,
        strategy TEXT NOT NULL,
        direction TEXT NOT NULL,
        probability REAL NOT NULL,
        backtest_accuracy REAL NOT NULL,
        conviction_score INTEGER NOT NULL,
        horizon_days INTEGER NOT NULL,
        entry_ref_price REAL NOT NULL,
        status TEXT NOT NULL
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS idea_scores (
        proposal_id INTEGER PRIMARY KEY,
        scored_at_utc TEXT NOT NULL,
        exit_day TEXT NOT NULL,
        exit_price REAL NOT NULL,
        return_pct REAL NOT NULL,
        correct INTEGER NOT NULL
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


def record_heartbeat(
    agent: str,
    status: str = 'ok',
    detail: str = '',
    timestamp_utc: str | None = None,
    db_path: Path | None = None,
) -> None:
    """Record the latest execution status and timestamp for one of the 6 pipeline agents."""
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = timestamp_utc or datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        conn.execute(
            '''INSERT INTO heartbeats (agent, last_run_utc, status, detail)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(agent) DO UPDATE SET
                last_run_utc = excluded.last_run_utc,
                status = excluded.status,
                detail = excluded.detail''',
            (agent.strip().lower(), ts, status.strip().lower(), detail[:2000]),
        )


def list_heartbeats(db_path: Path | None = None) -> list[dict]:
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    if not path.exists():
        return []
    cols = ['agent', 'last_run_utc', 'status', 'detail']
    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        rows = conn.execute(
            'SELECT agent, last_run_utc, status, detail FROM heartbeats ORDER BY agent ASC'
        ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def save_research_flags(
    symbol: str,
    flags: list[dict],
    source: str = 'keywords',
    db_path: Path | None = None,
) -> int:
    """Persist deduplicated structured news event flags for a symbol."""
    if not flags:
        return 0
    clean_sym = symbol.strip().upper()
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()
    inserted = 0
    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        seen = {
            row[0]
            for row in conn.execute(
                'SELECT headline FROM research_flags WHERE symbol = ?',
                (clean_sym,),
            ).fetchall()
        }
        for flag in flags:
            headline = str(flag.get('headline') or flag.get('summary') or '')[:500].strip()
            if not headline or headline in seen:
                continue
            seen.add(headline)
            conn.execute(
                '''INSERT INTO research_flags
                (created_at_utc, symbol, kind, severity, adverse, summary, headline, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    now_iso,
                    clean_sym,
                    str(flag.get('kind') or 'other')[:32],
                    str(flag.get('severity') or 'low')[:8],
                    1 if flag.get('adverse') else 0,
                    str(flag.get('summary') or headline)[:500],
                    headline,
                    source[:32],
                ),
            )
            inserted += 1
    return inserted


def list_research_flags(symbol: str | None = None, limit: int = 50, db_path: Path | None = None) -> list[dict]:
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    if not path.exists():
        return []
    cols = ['id', 'created_at_utc', 'symbol', 'kind', 'severity', 'adverse', 'summary', 'headline', 'source']
    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        if symbol:
            rows = conn.execute(
                f"SELECT {', '.join(cols)} FROM research_flags WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                (symbol.strip().upper(), int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {', '.join(cols)} FROM research_flags ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
    return [{**dict(zip(cols, r)), 'adverse': bool(r[5])} for r in rows]


def save_proposal(
    symbol: str,
    as_of: str,
    strategy: str,
    direction: str,
    entry_ref_price: float,
    status: str,
    horizon_days: int = 5,
    probability: float = 0.5,
    backtest_accuracy: float = 0.5,
    conviction_score: int = 0,
    db_path: Path | None = None,
) -> int:
    """Save an evaluated proposal (approved or rejected by Risk Gate) so Monitor can score both."""
    clean_sym = symbol.strip().upper()
    norm_dir = 'up' if str(direction).lower() in ('up', 'long', 'suggest_long') else 'down'
    norm_status = 'approved' if str(status).lower() in ('approved', 'suggestion_ready', 'pass_to_report') else 'rejected'
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        existing = conn.execute(
            'SELECT id FROM proposals WHERE symbol = ? AND as_of = ? AND strategy = ?',
            (clean_sym, str(as_of), strategy),
        ).fetchone()
        if existing is not None:
            conn.execute(
                'UPDATE proposals SET status = ?, conviction_score = ?, probability = ?, backtest_accuracy = ? WHERE id = ?',
                (norm_status, int(conviction_score), float(probability), float(backtest_accuracy), int(existing[0])),
            )
            return int(existing[0])

        cur = conn.execute(
            '''INSERT INTO proposals
            (created_at_utc, as_of, symbol, strategy, direction, probability,
             backtest_accuracy, conviction_score, horizon_days, entry_ref_price, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                datetime.now(timezone.utc).isoformat(),
                str(as_of),
                clean_sym,
                strategy,
                norm_dir,
                round(float(probability), 4),
                round(float(backtest_accuracy), 4),
                int(conviction_score),
                max(1, int(horizon_days)),
                round(float(entry_ref_price), 4),
                norm_status,
            ),
        )
        return int(cur.lastrowid)


def list_proposals(db_path: Path | None = None) -> list[dict]:
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    if not path.exists():
        return []
    cols = [
        'id', 'created_at_utc', 'as_of', 'symbol', 'strategy', 'direction',
        'probability', 'backtest_accuracy', 'conviction_score', 'horizon_days',
        'entry_ref_price', 'status',
    ]
    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        rows = conn.execute(f"SELECT {', '.join(cols)} FROM proposals ORDER BY id DESC LIMIT 250").fetchall()
    return [dict(zip(cols, r)) for r in rows]


def score_past_proposals(
    bars_by_ticker: dict[str, pd.DataFrame],
    db_path: Path | None = None,
) -> int:
    """Score any past proposal (approved OR Risk-rejected) whose horizon_days have elapsed."""
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    if not path.exists() or not bars_by_ticker:
        return 0

    norm_bars: dict[str, pd.DataFrame] = {k.strip().upper(): v for k, v in bars_by_ticker.items() if v is not None and not v.empty}
    added = 0

    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        scored_ids = {row[0] for row in conn.execute('SELECT proposal_id FROM idea_scores').fetchall()}
        pending = conn.execute(
            'SELECT id, as_of, symbol, direction, horizon_days, entry_ref_price FROM proposals'
        ).fetchall()

        for prop_id, as_of, sym, direction, horizon_days, entry_price in pending:
            if prop_id in scored_ids:
                continue
            df = norm_bars.get(sym.upper())
            if df is None or df.empty:
                continue

            close_col = 'Close' if 'Close' in df.columns else 'close'
            bar_dates = [str(idx.date()) if hasattr(idx, 'date') else str(idx)[:10] for idx in df.index]
            later_indices = [i for i, d in enumerate(bar_dates) if d > str(as_of)[:10]]
            if len(later_indices) < int(horizon_days):
                continue

            exit_idx = later_indices[int(horizon_days) - 1]
            exit_day = bar_dates[exit_idx]
            exit_price = float(df[close_col].iloc[exit_idx])
            ret_pct = (exit_price / float(entry_price) - 1.0) * 100.0
            correct = 1 if ((ret_pct > 0.0) == (str(direction).lower() == 'up')) else 0

            conn.execute(
                '''INSERT INTO idea_scores
                (proposal_id, scored_at_utc, exit_day, exit_price, return_pct, correct)
                VALUES (?, ?, ?, ?, ?, ?)''',
                (
                    int(prop_id),
                    datetime.now(timezone.utc).isoformat(),
                    exit_day,
                    round(exit_price, 4),
                    round(ret_pct, 3),
                    correct,
                ),
            )
            added += 1

    return added


def get_idea_scorecard(db_path: Path | None = None) -> pd.DataFrame:
    """Compare realized hit rate & average N-day return of Approved vs. Risk-Rejected ideas."""
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    empty = pd.DataFrame(columns=['Group', 'Ideas Scored', 'Hit Rate (%)', 'Avg Horizon Move (%)'])
    if not path.exists():
        return empty

    with sqlite3.connect(path) as conn:
        _ensure_tables(conn)
        rows = conn.execute(
            '''SELECT p.status, s.correct, s.return_pct
            FROM proposals p
            JOIN idea_scores s ON s.proposal_id = p.id'''
        ).fetchall()

    if not rows:
        return empty

    df = pd.DataFrame(rows, columns=['status', 'correct', 'return_pct'])
    df['Group'] = df['status'].map(
        lambda s: 'Rejected by Risk Gate' if s == 'rejected' else 'Approved by Risk Gate'
    )
    grouped = (
        df.groupby('Group', as_index=False)
        .agg(
            ideas=('correct', 'size'),
            hit_rate=('correct', 'mean'),
            avg_move=('return_pct', 'mean'),
        )
    )
    grouped['Hit Rate (%)'] = (grouped['hit_rate'] * 100.0).round(1)
    grouped['Avg Horizon Move (%)'] = grouped['avg_move'].round(2)
    return grouped.rename(columns={'ideas': 'Ideas Scored'})[
        ['Group', 'Ideas Scored', 'Hit Rate (%)', 'Avg Horizon Move (%)']
    ]


def open_paper_position(
    ticker: str,
    strategy: str,
    shares: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
    notes: str = '',
    sector: str = 'Unknown',
    db_path: Path | None = None,
) -> int:
    """Record a manual paper long position in local SQLite."""
    symbol = ticker.strip().upper()
    clean_sector = (sector or 'Unknown').strip() or 'Unknown'
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
            (opened_at_utc, closed_at_utc, status, ticker, sector, strategy, shares,
             entry_price, stop_price, target_price, exit_price, realized_pnl, notes)
            VALUES (?, NULL, 'OPEN', ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)''',
            (
                datetime.now(timezone.utc).isoformat(),
                symbol,
                clean_sector,
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
        'id', 'opened_at_utc', 'closed_at_utc', 'status', 'ticker', 'sector', 'strategy',
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

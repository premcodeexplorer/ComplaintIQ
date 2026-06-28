"""Database persistence layer for ComplaintIQ.

Supports both local SQLite (default) and cloud PostgreSQL (when DATABASE_URL is set).
Schema is created on first call to `init_db()`.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

DATABASE_URL = os.getenv("DATABASE_URL")
IS_POSTGRES = bool(DATABASE_URL)

if IS_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor
else:
    import sqlite3

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "complaintiq.sqlite"

SCHEMA_COMMON = """
CREATE TABLE IF NOT EXISTS complaints (
    id              TEXT PRIMARY KEY,
    customer_name   TEXT NOT NULL,
    channel         TEXT NOT NULL,
    complaint_text  TEXT NOT NULL,
    language        TEXT,
    date            TEXT NOT NULL,
    location        TEXT,
    account_type    TEXT,
    amount_involved REAL,
    category        TEXT,
    severity        TEXT,
    sentiment       TEXT,
    intake_json     TEXT,
    duplicate_of    TEXT,
    similarity      REAL,
    draft_response  TEXT,
    sla_due_date    TEXT,
    sla_breach_prob REAL,
    risk_score      INTEGER,
    risk_ombudsman  INTEGER,
    risk_churn      INTEGER,
    risk_social     INTEGER,
    cluster_id      INTEGER,
    status          TEXT DEFAULT 'open',
    processed_at    TEXT,
    FOREIGN KEY(duplicate_of) REFERENCES complaints(id)
);

CREATE INDEX IF NOT EXISTS idx_complaints_customer ON complaints(customer_name);
CREATE INDEX IF NOT EXISTS idx_complaints_date     ON complaints(date);
CREATE INDEX IF NOT EXISTS idx_complaints_category ON complaints(category);
CREATE INDEX IF NOT EXISTS idx_complaints_location ON complaints(location);
"""

if IS_POSTGRES:
    SCHEMA_SPECIFIC = """
    CREATE TABLE IF NOT EXISTS root_cause_alerts (
        id          SERIAL PRIMARY KEY,
        cluster_id  INTEGER,
        category    TEXT,
        location    TEXT,
        count       INTEGER,
        summary     TEXT,
        created_at  TEXT
    );

    CREATE TABLE IF NOT EXISTS feedback (
        id              SERIAL PRIMARY KEY,
        complaint_id    TEXT NOT NULL,
        field           TEXT NOT NULL,
        original_value  TEXT,
        corrected_value TEXT,
        is_correct      INTEGER NOT NULL,
        created_at      TEXT NOT NULL,
        FOREIGN KEY(complaint_id) REFERENCES complaints(id)
    );
    CREATE INDEX IF NOT EXISTS idx_feedback_complaint ON feedback(complaint_id);
    """
else:
    SCHEMA_SPECIFIC = """
    CREATE TABLE IF NOT EXISTS root_cause_alerts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        cluster_id  INTEGER,
        category    TEXT,
        location    TEXT,
        count       INTEGER,
        summary     TEXT,
        created_at  TEXT
    );

    CREATE TABLE IF NOT EXISTS feedback (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        complaint_id    TEXT NOT NULL,
        field           TEXT NOT NULL,
        original_value  TEXT,
        corrected_value TEXT,
        is_correct      INTEGER NOT NULL,
        created_at      TEXT NOT NULL,
        FOREIGN KEY(complaint_id) REFERENCES complaints(id)
    );
    CREATE INDEX IF NOT EXISTS idx_feedback_complaint ON feedback(complaint_id);
    """

SCHEMA = SCHEMA_COMMON + SCHEMA_SPECIFIC


@contextmanager
def connect():
    if IS_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    else:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def _exec(conn, query: str, params: Iterable[Any] = ()) -> Any:
    """Helper to abstract ? vs %s and execute the query."""
    if IS_POSTGRES:
        # Simple naive replacement for our specific queries
        query = query.replace("?", "%s")
    
    cur = conn.cursor()
    cur.execute(query, tuple(params))
    return cur


def init_db() -> None:
    with connect() as c:
        if IS_POSTGRES:
            cur = c.cursor()
            cur.execute(SCHEMA)
        else:
            c.executescript(SCHEMA)
    ensure_schema()


# --- ingestion ----------------------------------------------------------------

INSERT_COLS = (
    "id", "customer_name", "channel", "complaint_text", "language",
    "date", "location", "account_type", "amount_involved",
)


def upsert_raw_complaint(row: dict[str, Any]) -> None:
    """Insert a raw complaint from complaints.json. Existing rows are ignored
    (their enriched columns are preserved)."""
    cols = ", ".join(INSERT_COLS)
    vals = [row.get(c) for c in INSERT_COLS]
    with connect() as c:
        if IS_POSTGRES:
            qs_pg = ", ".join(["%s"] * len(INSERT_COLS))
            c.cursor().execute(
                f"INSERT INTO complaints ({cols}) VALUES ({qs_pg}) ON CONFLICT (id) DO NOTHING",
                vals
            )
        else:
            qs = ", ".join(["?"] * len(INSERT_COLS))
            c.execute(f"INSERT OR IGNORE INTO complaints ({cols}) VALUES ({qs})", vals)


def load_raw_from_json(json_path: str | Path) -> int:
    """Bulk-load the seed complaints.json. Returns number of rows inserted."""
    init_db()
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    n = 0
    for row in data:
        upsert_raw_complaint(row)
        n += 1
    return n


# --- enrichment updates -------------------------------------------------------

UPDATABLE_FIELDS = {
    "date",                            # writable so the data-remix script can rebalance
    "channel", "complaint_text",       # writable so we can flip 50 rows to mobile_app
    "category", "severity", "sentiment", "intake_json",
    "duplicate_of", "similarity", "draft_response",
    "sla_due_date", "sla_breach_prob", "risk_score",
    "risk_ombudsman", "risk_churn", "risk_social",
    "cluster_id", "status", "resolved_at", "processed_at",
    # ML second-opinions + composite
    "ml_category", "ml_category_prob", "category_confidence",
    "ml_sentiment", "ml_sentiment_prob", "sentiment_confidence",
    "priority_score",
}


def update_complaint(complaint_id: str, **fields: Any) -> None:
    bad = set(fields) - UPDATABLE_FIELDS
    if bad:
        raise ValueError(f"Not updatable: {bad}")
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [complaint_id]
    with connect() as c:
        _exec(c, f"UPDATE complaints SET {set_clause} WHERE id = ?", vals)


# --- reads --------------------------------------------------------------------

def get_complaint(complaint_id: str) -> dict[str, Any] | None:
    with connect() as c:
        row = _exec(c, "SELECT * FROM complaints WHERE id = ?", (complaint_id,)).fetchone()
    return dict(row) if row else None


def list_complaints(limit: int | None = None, where: str | None = None,
                    params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    q = "SELECT * FROM complaints"
    if where:
        q += " WHERE " + where
    q += " ORDER BY date DESC, id DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    with connect() as c:
        rows = _exec(c, q, params).fetchall()
    return [dict(r) for r in rows]


def list_unprocessed(limit: int | None = None) -> list[dict[str, Any]]:
    return list_complaints(limit=limit, where="processed_at IS NULL")


def customer_history(customer_name: str) -> list[dict[str, Any]]:
    return list_complaints(where="customer_name = ?", params=(customer_name,))


# --- alerts -------------------------------------------------------------------

def insert_root_cause_alert(cluster_id: int, category: str, location: str | None,
                            count: int, summary: str, created_at: str) -> None:
    with connect() as c:
        _exec(c,
            "INSERT INTO root_cause_alerts (cluster_id, category, location, count, summary, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cluster_id, category, location, count, summary, created_at),
        )


def list_root_cause_alerts(limit: int = 20) -> list[dict[str, Any]]:
    with connect() as c:
        rows = _exec(c, "SELECT * FROM root_cause_alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def clear_alerts() -> None:
    with connect() as c:
        _exec(c, "DELETE FROM root_cause_alerts")


# --- feedback ----------------------------------------------------------------

def record_feedback(complaint_id: str, field: str, original_value: str | None,
                    corrected_value: str | None, is_correct: bool) -> None:
    with connect() as c:
        _exec(c,
            "INSERT INTO feedback (complaint_id, field, original_value, "
            "corrected_value, is_correct, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (complaint_id, field, original_value, corrected_value,
             1 if is_correct else 0, datetime.utcnow().isoformat(timespec="seconds") + "Z"),
        )
    # If a correction was supplied, also overwrite the complaint row so the
    # dashboard reflects the human's call immediately.
    if not is_correct and corrected_value and field in UPDATABLE_FIELDS:
        update_complaint(complaint_id, **{field: corrected_value})


def list_feedback(complaint_id: str | None = None) -> list[dict[str, Any]]:
    q = "SELECT * FROM feedback"
    params: tuple = ()
    if complaint_id:
        q += " WHERE complaint_id = ?"
        params = (complaint_id,)
    q += " ORDER BY id DESC"
    with connect() as c:
        rows = _exec(c, q, params).fetchall()
    return [dict(r) for r in rows]


def feedback_stats() -> dict[str, Any]:
    with connect() as c:
        total = _exec(c, "SELECT COUNT(*) FROM feedback").fetchone()
        total = total['count'] if IS_POSTGRES else total[0]
        
        correct = _exec(c, "SELECT COUNT(*) FROM feedback WHERE is_correct = 1").fetchone()
        correct = correct['count'] if IS_POSTGRES else correct[0]
        
        corrections = total - correct
        by_field = _exec(c, "SELECT field, COUNT(*) FROM feedback WHERE is_correct = 0 GROUP BY field").fetchall()
        
        if IS_POSTGRES:
            corrections_dict = {row['field']: row['count'] for row in by_field}
        else:
            corrections_dict = {row[0]: row[1] for row in by_field}
            
    return {
        "total": total,
        "correct": correct,
        "corrections": corrections,
        "accuracy_rate": (correct / total) if total else None,
        "corrections_by_field": corrections_dict,
    }


# --- schema migration --------------------------------------------------------

def ensure_schema() -> None:
    """Apply additive schema migrations (idempotent). Called after init_db()."""
    with connect() as c:
        if IS_POSTGRES:
            cur = c.cursor()
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='complaints'")
            existing = {r['column_name'] for r in cur.fetchall()}
        else:
            existing = {r[1] for r in c.execute("PRAGMA table_info(complaints)").fetchall()}
            
        for col, ddl in (
            ("risk_ombudsman", "INTEGER"),
            ("risk_churn",     "INTEGER"),
            ("risk_social",    "INTEGER"),
            ("ml_category",          "TEXT"),
            ("ml_category_prob",     "REAL"),
            ("category_confidence",  "TEXT"),   # 'High Confidence' / 'Needs Review'
            ("ml_sentiment",         "TEXT"),   # positive / neutral / negative
            ("ml_sentiment_prob",    "REAL"),
            ("sentiment_confidence", "TEXT"),
            ("priority_score",       "INTEGER"),
            ("resolved_at",          "TEXT"),
        ):
            if col not in existing:
                if IS_POSTGRES:
                    c.cursor().execute(f"ALTER TABLE complaints ADD COLUMN {col} {ddl}")
                else:
                    c.execute(f"ALTER TABLE complaints ADD COLUMN {col} {ddl}")


if __name__ == "__main__":
    # Quick smoke test: load seed data and report counts.
    seed = Path(__file__).resolve().parent.parent / "data" / "complaints.json"
    n = load_raw_from_json(seed)
    print(f"Seeded {n} rows. Total now: {len(list_complaints())}")

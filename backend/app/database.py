"""Lightweight SQLite storage for logged PPE violations.

A single-file SQLite database is used deliberately instead of a full
ORM/server database: this project runs as a self-contained demo and
does not need a separate database service to be useful.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    compliance_score REAL NOT NULL,
    missing_gear TEXT NOT NULL,
    image_path TEXT NOT NULL
);
"""


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with row access by column name."""
    Path(settings.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the violations table if it does not already exist."""
    with get_connection() as conn:
        conn.execute(_SCHEMA)


def insert_violation(
    timestamp: str, source: str, compliance_score: float, missing_gear: list[str], image_path: str
) -> int:
    """Insert a violation record and return its new row id."""
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO violations (timestamp, source, compliance_score, missing_gear, image_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (timestamp, source, compliance_score, json.dumps(missing_gear), image_path),
        )
        return int(cursor.lastrowid)


def list_violations(limit: int = 50, offset: int = 0) -> tuple[int, list[dict]]:
    """Return (total_count, page_of_records) ordered by most recent first."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM violations ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        items = [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "source": row["source"],
                "compliance_score": row["compliance_score"],
                "missing_gear": json.loads(row["missing_gear"]),
                "image_path": row["image_path"],
            }
            for row in rows
        ]
        return total, items

"""SQLite persistence for URL scan results."""

import json
import sqlite3
from pathlib import Path
from typing import Any


def initialize_database(database_path: str | Path) -> None:
    """Create the scans table if needed, keeping setup safe to repeat."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                score REAL NOT NULL,
                level TEXT NOT NULL,
                reasons TEXT NOT NULL,
                model TEXT NOT NULL,
                scanned_at TEXT NOT NULL
            )"""
        )


def insert_scan(database_path: str | Path, scan: dict[str, Any]) -> None:
    """Insert one scan, serializing its reason list as JSON text."""
    initialize_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO scans (url, score, level, reasons, model, scanned_at) VALUES (?, ?, ?, ?, ?, ?)",
            (scan["url"], scan["score"], scan["level"], json.dumps(scan["reasons"]),
             scan["model"], scan["scanned_at"]),
        )


def list_scans(database_path: str | Path, limit: int = 20) -> list[dict[str, Any]]:
    """Return newest scans first, restoring reason lists from their JSON."""
    initialize_database(database_path)
    safe_limit = max(1, min(int(limit), 1000))
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id, url, score, level, reasons, model, scanned_at "
            "FROM scans ORDER BY id DESC LIMIT ?", (safe_limit,)
        ).fetchall()
    return [
        {"id": row["id"], "url": row["url"], "score": row["score"], "level": row["level"],
         "reasons": json.loads(row["reasons"]), "model": row["model"], "scanned_at": row["scanned_at"]}
        for row in rows
    ]

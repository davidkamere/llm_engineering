import json
import sqlite3
from datetime import datetime, timezone

from config import DB_PATH


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(cur, table, col_name, col_type):
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
    if col_name not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS deals_seen (
            deal_id TEXT PRIMARY KEY,
            title TEXT,
            source TEXT,
            seen_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS opportunities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deal_id TEXT,
            title TEXT,
            listed_price REAL,
            estimated_price REAL,
            discount_pct REAL,
            confidence REAL,
            rationale TEXT,
            url TEXT,
            created_at TEXT
        )
        """
    )
    _ensure_column(cur, "opportunities", "llm_price", "REAL")
    _ensure_column(cur, "opportunities", "rag_price", "REAL")
    _ensure_column(cur, "opportunities", "ensemble_price", "REAL")
    _ensure_column(cur, "opportunities", "planner_score", "REAL")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deal_id TEXT,
            message TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_trace (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            agent_name TEXT,
            event TEXT,
            payload TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def trace(run_id, agent_name, event, payload):
    conn = get_conn()
    conn.execute(
        "INSERT INTO agent_trace(run_id, agent_name, event, payload, created_at) VALUES (?, ?, ?, ?, ?)",
        (run_id, agent_name, event, json.dumps(payload, ensure_ascii=False), utc_now_iso()),
    )
    conn.commit()
    conn.close()

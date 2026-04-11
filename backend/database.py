"""
database.py — SQLite persistence layer
=======================================
Tables:
  users          — registered accounts (email, name, password hash)
  predictions    — every AI prediction run (symbol, days, results JSON)
  portfolio_data — saved user portfolio holdings (JSON blob per user)
  price_cache    — historical price snapshots for audit trail
  audit_log      — key system events

Usage:
    from database import db
    db.save_prediction("RELIANCE", 7, result_dict)
    rows = db.get_predictions("RELIANCE", limit=10)
"""

import sqlite3
import json
import os
import time
import threading
import logging

DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "aifin.db"))

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection (creates if needed)."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")   # better concurrency
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


class Database:
    """Thin wrapper around SQLite with all table operations."""

    def __init__(self):
        self._lock = threading.Lock()
        self._init_tables()

    # ── Schema ─────────────────────────────────────────────────────────────

    def _init_tables(self):
        conn = _get_conn()
        with self._lock:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    email      TEXT    UNIQUE NOT NULL,
                    name       TEXT    NOT NULL,
                    pwd_hash   TEXT    NOT NULL,
                    created_at REAL    NOT NULL DEFAULT (unixepoch())
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol     TEXT    NOT NULL,
                    days       INTEGER NOT NULL,
                    result     TEXT    NOT NULL,   -- JSON blob
                    rmse       REAL,
                    mae        REAL,
                    signal     TEXT,
                    created_at REAL    NOT NULL DEFAULT (unixepoch())
                );

                CREATE TABLE IF NOT EXISTS portfolio_data (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_email TEXT    NOT NULL,
                    holdings   TEXT    NOT NULL,   -- JSON blob
                    updated_at REAL    NOT NULL DEFAULT (unixepoch())
                );

                CREATE TABLE IF NOT EXISTS price_cache (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol     TEXT    NOT NULL,
                    price      REAL    NOT NULL,
                    change_pct REAL,
                    recorded_at REAL   NOT NULL DEFAULT (unixepoch())
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    event      TEXT    NOT NULL,
                    detail     TEXT,
                    created_at REAL    NOT NULL DEFAULT (unixepoch())
                );

                CREATE INDEX IF NOT EXISTS idx_pred_symbol  ON predictions(symbol);
                CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at);
                CREATE INDEX IF NOT EXISTS idx_price_symbol ON price_cache(symbol);
            """)
            conn.commit()
        logging.info(f"[db] SQLite initialised at {DB_PATH}")

    # ── Users ───────────────────────────────────────────────────────────────

    def create_user(self, email: str, name: str, pwd_hash: str) -> bool:
        """Insert new user. Returns True on success, False if email exists."""
        try:
            conn = _get_conn()
            with self._lock:
                conn.execute(
                    "INSERT INTO users (email, name, pwd_hash) VALUES (?,?,?)",
                    (email.lower(), name, pwd_hash)
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_user(self, email: str) -> dict | None:
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM users WHERE email=?", (email.lower(),)
        ).fetchone()
        return dict(row) if row else None

    def user_exists(self, email: str) -> bool:
        conn = _get_conn()
        row = conn.execute(
            "SELECT id FROM users WHERE email=?", (email.lower(),)
        ).fetchone()
        return row is not None

    # ── Predictions ─────────────────────────────────────────────────────────

    def save_prediction(self, symbol: str, days: int, result: dict) -> int:
        """Persist a prediction result. Returns the new row id."""
        conn = _get_conn()
        metrics = result.get("metrics", {})
        with self._lock:
            cur = conn.execute(
                """INSERT INTO predictions (symbol, days, result, rmse, mae, signal)
                   VALUES (?,?,?,?,?,?)""",
                (
                    symbol, days,
                    json.dumps(result),
                    metrics.get("RMSE"),
                    metrics.get("MAE"),
                    result.get("signal"),
                )
            )
            conn.commit()
        self.log_event("prediction", f"{symbol} {days}d signal={result.get('signal')}")
        return cur.lastrowid

    def get_predictions(self, symbol: str = None, limit: int = 20) -> list:
        conn = _get_conn()
        if symbol:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE symbol=? ORDER BY created_at DESC LIMIT ?",
                (symbol, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["result"] = json.loads(d["result"])
            out.append(d)
        return out

    def get_prediction_stats(self, symbol: str) -> dict:
        """Aggregate RMSE/MAE stats for a symbol from history."""
        conn = _get_conn()
        row = conn.execute(
            """SELECT COUNT(*) as n, AVG(rmse) as avg_rmse, AVG(mae) as avg_mae,
                      MIN(rmse) as best_rmse
               FROM predictions WHERE symbol=? AND rmse IS NOT NULL""",
            (symbol,)
        ).fetchone()
        return dict(row) if row else {}

    # ── Portfolio ───────────────────────────────────────────────────────────

    def save_portfolio(self, user_email: str, holdings: list) -> None:
        conn = _get_conn()
        with self._lock:
            # Upsert: delete old, insert new
            conn.execute(
                "DELETE FROM portfolio_data WHERE user_email=?", (user_email,)
            )
            conn.execute(
                "INSERT INTO portfolio_data (user_email, holdings) VALUES (?,?)",
                (user_email, json.dumps(holdings))
            )
            conn.commit()

    def get_portfolio(self, user_email: str) -> list | None:
        conn = _get_conn()
        row = conn.execute(
            "SELECT holdings FROM portfolio_data WHERE user_email=? ORDER BY updated_at DESC LIMIT 1",
            (user_email,)
        ).fetchone()
        return json.loads(row["holdings"]) if row else None

    # ── Price Cache ─────────────────────────────────────────────────────────

    def record_price(self, symbol: str, price: float, change_pct: float = None) -> None:
        conn = _get_conn()
        with self._lock:
            conn.execute(
                "INSERT INTO price_cache (symbol, price, change_pct) VALUES (?,?,?)",
                (symbol, price, change_pct)
            )
            conn.commit()

    def get_price_history(self, symbol: str, limit: int = 100) -> list:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT price, change_pct, recorded_at FROM price_cache WHERE symbol=? ORDER BY recorded_at DESC LIMIT ?",
            (symbol, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Audit Log ───────────────────────────────────────────────────────────

    def log_event(self, event: str, detail: str = None) -> None:
        try:
            conn = _get_conn()
            with self._lock:
                conn.execute(
                    "INSERT INTO audit_log (event, detail) VALUES (?,?)",
                    (event, detail)
                )
                conn.commit()
        except Exception:
            pass  # audit must never crash the main flow

    def get_audit_log(self, limit: int = 50) -> list:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Admin stats ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        conn = _get_conn()
        return {
            "users":       conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
            "predictions": conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
            "portfolios":  conn.execute("SELECT COUNT(*) FROM portfolio_data").fetchone()[0],
            "price_records": conn.execute("SELECT COUNT(*) FROM price_cache").fetchone()[0],
            "db_path":     DB_PATH,
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
db = Database()

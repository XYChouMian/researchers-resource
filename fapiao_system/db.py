"""数据库层：SQLite 路径、连接、表结构（schema 唯一定义处）。"""
import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("FAPIAO_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = Path(os.environ.get("FAPIAO_UPLOADS_DIR", BASE_DIR / "uploads"))
DB_PATH = DATA_DIR / "fapiao.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id    TEXT    NOT NULL UNIQUE,
    name          TEXT    NOT NULL DEFAULT '',
    password_hash TEXT    NOT NULL,
    role          TEXT    NOT NULL CHECK (role IN ('admin', 'user')),
    created_at    TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id),
    product_name  TEXT    NOT NULL,
    channel       TEXT    NOT NULL CHECK (channel IN ('taobao', 'jd', 'other')),
    channel_note  TEXT    NOT NULL DEFAULT '',
    amount        REAL    NOT NULL CHECK (amount > 0),
    paid_at       TEXT    NOT NULL,
    payer         TEXT    NOT NULL CHECK (payer IN ('self', 'tang')),
    invoice_count INTEGER,
    status        TEXT    NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'invoiced', 'reimbursed')),
    reimbursed_at TEXT,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at    TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS invoices (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id   INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    orig_name   TEXT    NOT NULL,
    stored_name TEXT    NOT NULL UNIQUE,
    category    TEXT    NOT NULL DEFAULT 'invoice'
                CHECK (category IN ('invoice', 'attachment')),
    uploaded_at TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_records_user ON records(user_id);
CREATE INDEX IF NOT EXISTS idx_invoices_record ON invoices(record_id);
"""


def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def get_db():
    """获取当前 Flask 请求内共享的连接。"""
    from flask import g

    if "db" not in g:
        g.db = connect()
    return g.db


def close_db(_exc=None):
    from flask import g

    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect()
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
    conn.close()


def _migrate(conn):
    """对已有旧库做增量迁移（只加列不改数据，存量发票自动归为 invoice）。"""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(invoices)")}
    if "category" not in columns:
        conn.execute(
            "ALTER TABLE invoices ADD COLUMN category TEXT NOT NULL DEFAULT 'invoice'"
        )

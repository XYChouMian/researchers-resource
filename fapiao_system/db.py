"""数据库层：SQLite 路径、连接、表结构（schema 唯一定义处）。"""
import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("FAPIAO_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = Path(os.environ.get("FAPIAO_UPLOADS_DIR", BASE_DIR / "uploads"))
DB_PATH = DATA_DIR / "fapiao.db"

RECORDS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id),
    product_name  TEXT    NOT NULL,
    channel       TEXT    NOT NULL CHECK (channel IN ('taobao', 'jd', 'other')),
    channel_note  TEXT    NOT NULL DEFAULT '',
    remark        TEXT    NOT NULL DEFAULT '',
    amount        REAL    NOT NULL CHECK (amount > 0),
    paid_at       TEXT    NOT NULL,
    payer         TEXT    NOT NULL CHECK (payer IN ('self', 'tang')),
    invoice_count INTEGER,
    status        TEXT    NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'invoiced', 'reimbursed', 'rejected', 'processed')),
    reimbursed_at TEXT,
    process_note  TEXT,
    reject_note   TEXT,
    reimburse_note TEXT,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at    TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id    TEXT    NOT NULL UNIQUE,
    name          TEXT    NOT NULL DEFAULT '',
    password_hash TEXT    NOT NULL,
    role          TEXT    NOT NULL CHECK (role IN ('admin', 'user')),
    created_at    TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS invoices (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id   INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    orig_name   TEXT    NOT NULL,
    stored_name TEXT    NOT NULL UNIQUE,
    category    TEXT    NOT NULL DEFAULT 'invoice'
                CHECK (category IN ('invoice', 'attachment')),
    remark      TEXT    NOT NULL DEFAULT '',
    uploaded_at TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);
""" + RECORDS_TABLE_SQL + """
CREATE TABLE IF NOT EXISTS audit_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id   INTEGER NOT NULL,
    action      TEXT    NOT NULL,
    operator_id INTEGER NOT NULL REFERENCES users(id),
    note        TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_records_user ON records(user_id);
CREATE INDEX IF NOT EXISTS idx_invoices_record ON invoices(record_id);
CREATE INDEX IF NOT EXISTS idx_audit_record ON audit_logs(record_id);
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
    """对已有旧库做增量迁移（只加列/放宽约束，不改数据）。"""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(invoices)")}
    if "category" not in columns:
        conn.execute(
            "ALTER TABLE invoices ADD COLUMN category TEXT NOT NULL DEFAULT 'invoice'"
        )
    if "remark" not in columns:
        conn.execute("ALTER TABLE invoices ADD COLUMN remark TEXT NOT NULL DEFAULT ''")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(records)")}
    if "process_note" not in columns:
        conn.execute("ALTER TABLE records ADD COLUMN process_note TEXT")
    if "reimburse_note" not in columns:
        conn.execute("ALTER TABLE records ADD COLUMN reimburse_note TEXT")
    if "remark" not in columns:
        conn.execute("ALTER TABLE records ADD COLUMN remark TEXT NOT NULL DEFAULT ''")
    _rebuild_records_if_needed(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(records)")}
    if "reject_note" not in columns:
        conn.execute("ALTER TABLE records ADD COLUMN reject_note TEXT")


def _rebuild_records_if_needed(conn):
    """SQLite 不能修改 CHECK 约束：状态约束缺少新状态时按官方流程重建 records 表。"""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='records'"
    ).fetchone()
    if row is None or "rejected" in row[0]:
        return
    new_table_sql = RECORDS_TABLE_SQL.replace(
        "CREATE TABLE IF NOT EXISTS records", "CREATE TABLE records_new"
    )
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.executescript("BEGIN;" + new_table_sql + """
        INSERT INTO records_new (id, user_id, product_name, channel, channel_note,
                                 remark, amount, paid_at, payer, invoice_count, status,
                                 reimbursed_at, process_note, reject_note,
                                 reimburse_note, created_at, updated_at)
        SELECT id, user_id, product_name, channel, channel_note,
               remark, amount, paid_at, payer, invoice_count, status,
               reimbursed_at, process_note, NULL, reimburse_note,
               created_at, updated_at
        FROM records;
        DROP TABLE records;
        ALTER TABLE records_new RENAME TO records;
        CREATE INDEX IF NOT EXISTS idx_records_user ON records(user_id);
        COMMIT;
    """)
    conn.execute("PRAGMA foreign_keys = ON")

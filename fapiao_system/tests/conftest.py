"""测试夹具：临时目录数据库 + 空子路径应用。"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_tmp = tempfile.mkdtemp(prefix="fapiao-test-")
os.environ["FAPIAO_DATA_DIR"] = os.path.join(_tmp, "data")
os.environ["FAPIAO_UPLOADS_DIR"] = os.path.join(_tmp, "uploads")
os.environ["FAPIAO_PREFIX"] = ""

import pytest
from werkzeug.security import generate_password_hash

import db
from app import create_app

ADMIN_ID = "admin"
ADMIN_PASSWORD = "admin"


@pytest.fixture()
def client():
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(db.DB_PATH) + suffix)
        if path.exists():
            path.unlink()
    db.init_db()
    conn = db.connect()
    conn.execute(
        "INSERT INTO users (student_id, name, password_hash, role) VALUES (?, '管理员', ?, 'admin')",
        (ADMIN_ID, generate_password_hash(ADMIN_PASSWORD)),
    )
    conn.commit()
    conn.close()
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def login(client, student_id, password):
    return client.post("/api/login", json={"student_id": student_id, "password": password})


def make_user(client, student_id="2023123", name="测试同学"):
    assert login(client, ADMIN_ID, ADMIN_PASSWORD).status_code == 200
    resp = client.post(
        "/api/admin/users",
        json={"student_id": student_id, "name": name, "role": "user"},
    )
    assert resp.status_code == 201
    return student_id

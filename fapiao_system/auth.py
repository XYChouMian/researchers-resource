"""认证：登录/登出/改密接口与权限装饰器。"""
from functools import wraps

from flask import Blueprint, g, jsonify, request, session
from werkzeug.security import check_password_hash, generate_password_hash

import db

bp = Blueprint("auth", __name__)


def public_user(row):
    return {
        "id": row["id"],
        "student_id": row["student_id"],
        "name": row["name"],
        "role": row["role"],
    }


def current_user():
    uid = session.get("user_id")
    if uid is None:
        return None
    return db.get_db().execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = current_user()
        if user is None:
            return jsonify(error="未登录或会话已过期"), 401
        g.user = user
        return fn(*args, **kwargs)

    return wrapper


def admin_required(fn):
    @wraps(fn)
    @login_required
    def wrapper(*args, **kwargs):
        if g.user["role"] != "admin":
            return jsonify(error="需要管理员权限"), 403
        return fn(*args, **kwargs)

    return wrapper


@bp.post("/api/login")
def login():
    data = request.get_json(silent=True) or {}
    student_id = str(data.get("student_id", "")).strip()
    password = str(data.get("password", ""))
    row = db.get_db().execute(
        "SELECT * FROM users WHERE student_id=?", (student_id,)
    ).fetchone()
    if row is None or not check_password_hash(row["password_hash"], password):
        return jsonify(error="学号或密码错误"), 401
    session.permanent = True
    session["user_id"] = row["id"]
    return jsonify(user=public_user(row))


@bp.post("/api/logout")
def logout():
    session.clear()
    return jsonify(ok=True)


@bp.get("/api/me")
@login_required
def me():
    return jsonify(user=public_user(g.user))


@bp.post("/api/password")
@login_required
def change_password():
    data = request.get_json(silent=True) or {}
    old_password = str(data.get("old_password", ""))
    new_password = str(data.get("new_password", ""))
    if not check_password_hash(g.user["password_hash"], old_password):
        return jsonify(error="原密码错误"), 400
    if len(new_password) < 6:
        return jsonify(error="新密码至少 6 位"), 400
    conn = db.get_db()
    conn.execute(
        "UPDATE users SET password_hash=? WHERE id=?",
        (generate_password_hash(new_password), g.user["id"]),
    )
    conn.commit()
    return jsonify(ok=True)

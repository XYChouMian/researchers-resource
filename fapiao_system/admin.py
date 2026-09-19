"""管理员接口：用户管理、记录查看、报销材料导出、报销标记。"""
import io
import re
from datetime import datetime

from flask import Blueprint, g, jsonify, request, send_file
from werkzeug.security import generate_password_hash

import db
import export
from auth import admin_required
from records import record_to_dict

bp = Blueprint("admin", __name__, url_prefix="/api/admin")

ROLES = ("user", "admin")
RECORD_STATUSES = ("pending", "invoiced", "reimbursed")
STUDENT_ID_RE = re.compile(r"^[A-Za-z0-9]{3,20}$")


@bp.get("/users")
@admin_required
def list_users():
    rows = db.get_db().execute(
        """SELECT u.id, u.student_id, u.name, u.role, u.created_at,
                  COUNT(r.id) AS record_count,
                  COALESCE(SUM(r.status != 'reimbursed'), 0) AS open_count
           FROM users u LEFT JOIN records r ON r.user_id = u.id
           GROUP BY u.id ORDER BY u.role ASC, u.student_id"""
    ).fetchall()
    return jsonify(users=[dict(r) for r in rows])


@bp.post("/users")
@admin_required
def create_user():
    data = request.get_json(silent=True) or {}
    student_id = str(data.get("student_id", "")).strip()
    name = str(data.get("name", "")).strip() or student_id
    role = data.get("role", "user")
    if not STUDENT_ID_RE.fullmatch(student_id):
        return jsonify(error="学号只能是 3~20 位字母或数字"), 400
    if len(name) > 50:
        return jsonify(error="姓名过长"), 400
    if role not in ROLES:
        return jsonify(error="角色不合法"), 400
    conn = db.get_db()
    if conn.execute("SELECT 1 FROM users WHERE student_id=?", (student_id,)).fetchone():
        return jsonify(error="该学号已存在"), 400
    cur = conn.execute(
        "INSERT INTO users (student_id, name, password_hash, role) VALUES (?, ?, ?, ?)",
        (student_id, name, generate_password_hash(student_id), role),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM users WHERE id=?", (cur.lastrowid,)).fetchone()
    return jsonify(user={
        "id": row["id"], "student_id": row["student_id"],
        "name": row["name"], "role": row["role"],
    }), 201


@bp.put("/users/<int:uid>")
@admin_required
def update_user(uid):
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    role = data.get("role")
    if not name or len(name) > 50:
        return jsonify(error="姓名不能为空且不超过 50 字"), 400
    if role not in ROLES:
        return jsonify(error="角色不合法"), 400
    if uid == g.user["id"] and role != "admin":
        return jsonify(error="不能修改自己的角色"), 400
    conn = db.get_db()
    if not conn.execute("SELECT 1 FROM users WHERE id=?", (uid,)).fetchone():
        return jsonify(error="用户不存在"), 404
    conn.execute("UPDATE users SET name=?, role=? WHERE id=?", (name, role, uid))
    conn.commit()
    return jsonify(ok=True)


@bp.post("/users/<int:uid>/reset_password")
@admin_required
def reset_password(uid):
    row = db.get_db().execute(
        "SELECT student_id FROM users WHERE id=?", (uid,)
    ).fetchone()
    if row is None:
        return jsonify(error="用户不存在"), 404
    conn = db.get_db()
    conn.execute(
        "UPDATE users SET password_hash=? WHERE id=?",
        (generate_password_hash(row["student_id"]), uid),
    )
    conn.commit()
    return jsonify(ok=True)


@bp.delete("/users/<int:uid>")
@admin_required
def delete_user(uid):
    if uid == g.user["id"]:
        return jsonify(error="不能删除自己"), 400
    conn = db.get_db()
    if not conn.execute("SELECT 1 FROM users WHERE id=?", (uid,)).fetchone():
        return jsonify(error="用户不存在"), 404
    if conn.execute("SELECT 1 FROM records WHERE user_id=? LIMIT 1", (uid,)).fetchone():
        return jsonify(error="该用户名下存在购买记录，不能删除"), 400
    conn.execute("DELETE FROM users WHERE id=?", (uid,))
    conn.commit()
    return jsonify(ok=True)


def _status_filter(default=None):
    raw = request.args.get("status", "")
    statuses = [s for s in raw.split(",") if s in RECORD_STATUSES]
    return statuses or list(default or RECORD_STATUSES)


@bp.get("/records")
@admin_required
def list_records():
    statuses = _status_filter()
    sql = """SELECT r.*, u.student_id, u.name AS user_name
             FROM records r JOIN users u ON u.id = r.user_id"""
    params = []
    clauses = []
    if len(statuses) < len(RECORD_STATUSES):
        clauses.append("r.status IN (%s)" % ",".join("?" * len(statuses)))
        params += statuses
    user_id = request.args.get("user_id", type=int)
    if user_id:
        clauses.append("r.user_id = ?")
        params.append(user_id)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY r.id DESC"
    rows = db.get_db().execute(sql, params).fetchall()
    return jsonify(records=[record_to_dict(r) for r in rows])


@bp.get("/export")
@admin_required
def export_records():
    raw_ids = request.args.get("ids", "")
    try:
        ids = list(dict.fromkeys(int(i) for i in raw_ids.split(",") if i.strip()))
    except ValueError:
        return jsonify(error="记录 id 不合法"), 400
    if not ids:
        return jsonify(error="请先勾选要导出的记录"), 400

    placeholders = ",".join("?" * len(ids))
    rows = db.get_db().execute(
        f"""SELECT r.*, u.student_id, u.name AS user_name
            FROM records r JOIN users u ON u.id = r.user_id
            WHERE r.id IN ({placeholders}) ORDER BY u.student_id, u.name, r.id""",
        ids,
    ).fetchall()
    if not rows:
        return jsonify(error="选中的记录不存在"), 404
    invoices = db.get_db().execute(
        f"""SELECT i.record_id, i.orig_name, i.stored_name, i.category
            FROM invoices i WHERE i.record_id IN ({placeholders}) ORDER BY i.id""",
        ids,
    ).fetchall()
    by_record = {}
    for inv in invoices:
        by_record.setdefault(inv["record_id"], []).append(dict(inv))

    def read_file(student_id, stored_name):
        path = db.UPLOAD_DIR / student_id / stored_name
        return path.read_bytes() if path.is_file() else None

    records = []
    for r in rows:
        item = dict(r)
        item["invoices"] = by_record.get(r["id"], [])
        records.append(item)

    filename = "报销材料-%s.zip" % datetime.now().strftime("%Y%m%d")
    return send_file(
        io.BytesIO(export.build_export_zip(records, read_file)),
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename,
    )


@bp.post("/reimburse")
@admin_required
def reimburse():
    data = request.get_json(silent=True) or {}
    ids = data.get("ids")
    if not isinstance(ids, list) or not ids:
        return jsonify(error="请提供要报销的记录 id 列表"), 400
    try:
        ids = [int(i) for i in ids]
    except (TypeError, ValueError):
        return jsonify(error="记录 id 不合法"), 400
    conn = db.get_db()
    cur = conn.execute(
        """UPDATE records SET status='reimbursed',
               reimbursed_at=datetime('now','localtime')
           WHERE status != 'reimbursed' AND id IN (%s)""" % ",".join("?" * len(ids)),
        ids,
    )
    conn.commit()
    return jsonify(updated=cur.rowcount)

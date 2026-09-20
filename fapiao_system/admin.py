"""管理员接口：用户管理、记录查看、报销材料导出、报销操作（报销/驳回/恢复/已处理）。

四类报销操作全部写入 audit_logs（只增不删）并记录后台日志文件（JSON 行），
操作人、物品信息、状态流转、备注永久留痕，任何接口都无法删除。
"""
import io
import json
import logging
import re
from datetime import datetime

from flask import Blueprint, g, jsonify, request, send_file
from werkzeug.security import generate_password_hash

import db
import export
from auth import admin_required
from records import batch_ids, load_export_records, record_to_dict

bp = Blueprint("admin", __name__, url_prefix="/api/admin")

logger = logging.getLogger("fapiao.admin")

ROLES = ("user", "admin")
RECORD_STATUSES = ("pending", "invoiced", "reimbursed", "rejected", "processed")
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
    records, read_file = load_export_records(ids)
    if not records:
        return jsonify(error="选中的记录不存在"), 404

    filename = "报销材料-%s.zip" % datetime.now().strftime("%Y%m%d")
    return send_file(
        io.BytesIO(export.build_export_zip(records, read_file)),
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename,
    )


def _required_note(data, label):
    note = str(data.get("note", "")).strip()
    if not note or len(note) > 200:
        raise ValueError(f"请填写{label}（200 字以内）")
    return note


def _optional_note(data):
    note = str(data.get("note", "")).strip()
    if len(note) > 200:
        raise ValueError("备注不能超过 200 字")
    return note


ACTION_TEXT = {"reimburse": "标记已报销", "reject": "驳回", "revert": "恢复未报销", "process": "标记已处理"}
TO_STATUS = {"reimburse": "reimbursed", "reject": "rejected", "process": "processed"}


def _audit(rec, action, note=""):
    """写入审计表并输出 JSON 日志行（操作人/物品信息/状态流转/备注），只增不删。"""
    conn = db.get_db()
    conn.execute(
        "INSERT INTO audit_logs (record_id, action, operator_id, note) VALUES (?, ?, ?, ?)",
        (rec["id"], action, g.user["id"], note),
    )
    to_status = TO_STATUS.get(action) or ("invoiced" if rec["invoice_count"] else "pending")
    payload = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "action_text": ACTION_TEXT.get(action, action),
        "operator": g.user["name"],
        "operator_id": g.user["student_id"],
        "record_id": rec["id"],
        "student_id": rec["student_id"],
        "name": rec["user_name"],
        "product_name": rec["product_name"],
        "amount": rec["amount"],
        "paid_at": rec["paid_at"],
        "channel": export.CHANNEL_TEXT.get(rec["channel"], rec["channel"]),
        "payer": export.PAYER_TEXT.get(rec["payer"], rec["payer"]),
        "from_status": export.STATUS_TEXT.get(rec["status"], rec["status"]),
        "to_status": export.STATUS_TEXT.get(to_status, to_status),
        "note": note,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))


def _read_log_entries(limit=1000):
    """读取后台日志末尾若干行：JSON 行解析为对象，历史格式行以 raw 原样返回（倒序）。"""
    path = db.DATA_DIR / "admin.log"
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            entries.append({"raw": line})
    entries.reverse()
    return entries


def _changed_records(status_sql, ids):
    """查出批量操作中实际会变更的记录（含变更前状态与归属人），用于精准留痕。"""
    placeholders = ",".join("?" * len(ids))
    return db.get_db().execute(
        f"""SELECT r.*, u.student_id, u.name AS user_name
            FROM records r JOIN users u ON u.id = r.user_id
            WHERE {status_sql} AND r.id IN ({placeholders})""",
        ids,
    ).fetchall()


@bp.post("/reimburse")
@admin_required
def reimburse():
    """仅已开票的记录可标记为已报销（核对开票），可填写选填备注。"""
    data = request.get_json(silent=True) or {}
    try:
        ids = batch_ids(data)
        note = _optional_note(data)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    conn = db.get_db()
    changed = _changed_records("status = 'invoiced'", ids)
    if changed:
        conn.execute(
            """UPDATE records SET status='reimbursed',
                   reimbursed_at=datetime('now','localtime'), reimburse_note=?
               WHERE status='invoiced' AND id IN (%s)""" % ",".join("?" * len(ids)),
            [note] + ids,
        )
        for rec in changed:
            _audit(rec, "reimburse", note)
        conn.commit()
    return jsonify(updated=len(changed))


@bp.post("/reject")
@admin_required
def reject():
    """仅已开票的记录可驳回，必须填写理由；用户修改单据后自动回到已开票。"""
    data = request.get_json(silent=True) or {}
    try:
        ids = batch_ids(data)
        note = _required_note(data, "驳回理由")
    except ValueError as e:
        return jsonify(error=str(e)), 400
    conn = db.get_db()
    changed = _changed_records("status = 'invoiced'", ids)
    if changed:
        conn.execute(
            """UPDATE records SET status='rejected', reject_note=?,
                   updated_at=datetime('now','localtime')
               WHERE status='invoiced' AND id IN (%s)""" % ",".join("?" * len(ids)),
            [note] + ids,
        )
        for rec in changed:
            _audit(rec, "reject", note)
        conn.commit()
    return jsonify(updated=len(changed))


@bp.post("/revert")
@admin_required
def revert():
    """已报销/已处理/已驳回恢复为未报销：按发票数自动回到已开票或待开票。"""
    try:
        ids = batch_ids(request.get_json(silent=True) or {})
    except ValueError as e:
        return jsonify(error=str(e)), 400
    conn = db.get_db()
    changed = _changed_records(
        "status IN ('reimbursed', 'rejected', 'processed')", ids
    )
    if changed:
        conn.execute(
            """UPDATE records SET status = CASE WHEN invoice_count > 0
                       THEN 'invoiced' ELSE 'pending' END,
                   reimbursed_at = NULL, process_note = NULL, reject_note = NULL,
                   reimburse_note = NULL, updated_at = datetime('now','localtime')
               WHERE status IN ('reimbursed', 'rejected', 'processed')
                 AND id IN (%s)""" % ",".join("?" * len(ids)),
            ids,
        )
        for rec in changed:
            _audit(rec, "revert")
        conn.commit()
    return jsonify(updated=len(changed))


@bp.post("/process")
@admin_required
def process():
    """待开票/已开票的记录标记为已处理（经其他途径处理，必须填写说明）。"""
    data = request.get_json(silent=True) or {}
    try:
        ids = batch_ids(data)
        note = _required_note(data, "处理说明")
    except ValueError as e:
        return jsonify(error=str(e)), 400
    conn = db.get_db()
    changed = _changed_records("status IN ('pending', 'invoiced')", ids)
    if changed:
        conn.execute(
            """UPDATE records SET status='processed', process_note=?,
                   updated_at=datetime('now','localtime')
               WHERE status IN ('pending', 'invoiced') AND id IN (%s)"""
            % ",".join("?" * len(ids)),
            [note] + ids,
        )
        for rec in changed:
            _audit(rec, "process", note)
        conn.commit()
    return jsonify(updated=len(changed))


@bp.get("/logs")
@admin_required
def list_logs():
    """后台操作日志（末尾 1000 行，倒序；非 JSON 历史行原样返回）。"""
    return jsonify(entries=_read_log_entries())

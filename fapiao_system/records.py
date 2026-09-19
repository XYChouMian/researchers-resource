"""用户侧接口：购买记录增删改查、发票上传与下载。"""
import uuid
from datetime import datetime
from pathlib import Path

from flask import Blueprint, g, jsonify, request, send_file

import db
from auth import login_required

bp = Blueprint("records", __name__)

CHANNELS = ("taobao", "jd", "other")
PAYERS = ("self", "tang")
INVOICE_CATEGORIES = ("invoice", "attachment")
ALLOWED_EXTS = {".pdf", ".jpg", ".jpeg", ".png"}
MAX_INVOICE_SIZE = 10 * 1024 * 1024

_RECORD_SQL = """
SELECT r.*, u.student_id, u.name AS user_name
FROM records r JOIN users u ON u.id = r.user_id
"""


def get_record(rid):
    return db.get_db().execute(_RECORD_SQL + " WHERE r.id=?", (rid,)).fetchone()


def record_to_dict(row):
    invoices = db.get_db().execute(
        "SELECT id, orig_name, category, uploaded_at FROM invoices WHERE record_id=? ORDER BY id",
        (row["id"],),
    ).fetchall()
    result = dict(row)
    result["invoices"] = [dict(i) for i in invoices]
    return result


def _refresh_invoice_state(rid):
    """开票张数与状态由发票类文件数自动决定：≥1 张发票为已开票，否则回到待开票。"""
    conn = db.get_db()
    count = conn.execute(
        "SELECT COUNT(*) FROM invoices WHERE record_id=? AND category='invoice'",
        (rid,),
    ).fetchone()[0]
    status = "invoiced" if count else "pending"
    conn.execute(
        """UPDATE records SET invoice_count=?, status=?,
               updated_at=datetime('now','localtime') WHERE id=?""",
        (count, status, rid),
    )
    return count, status


def parse_record_payload(data):
    """校验并清洗记录字段；不合法时抛 ValueError，消息可直接展示。"""
    product_name = str(data.get("product_name", "")).strip()
    if not product_name or len(product_name) > 200:
        raise ValueError("商品名不能为空，且不超过 200 字")
    channel = data.get("channel")
    if channel not in CHANNELS:
        raise ValueError("购买渠道不合法")
    channel_note = str(data.get("channel_note", "")).strip()
    if channel == "other" and not channel_note:
        raise ValueError("购买渠道为“其他”时，必须填写渠道说明")
    try:
        amount = round(float(data.get("amount")), 2)
    except (TypeError, ValueError):
        raise ValueError("付款金额不合法")
    if not 0 < amount <= 1_000_000:
        raise ValueError("付款金额必须在 0 ~ 100 万之间")
    paid_at = str(data.get("paid_at", "")).strip()
    try:
        datetime.strptime(paid_at, "%Y-%m-%d")
    except ValueError:
        raise ValueError("付款时间格式不正确")
    payer = data.get("payer")
    if payer not in PAYERS:
        raise ValueError("付款人不合法")
    return {
        "product_name": product_name,
        "channel": channel,
        "channel_note": channel_note,
        "amount": amount,
        "paid_at": paid_at,
        "payer": payer,
    }


@bp.get("/api/records")
@login_required
def list_my_records():
    rows = db.get_db().execute(
        "SELECT * FROM records WHERE user_id=? ORDER BY id DESC", (g.user["id"],)
    ).fetchall()
    return jsonify(records=[record_to_dict(r) for r in rows])


@bp.post("/api/records")
@login_required
def create_record():
    try:
        fields = parse_record_payload(request.get_json(silent=True) or {})
    except ValueError as e:
        return jsonify(error=str(e)), 400
    conn = db.get_db()
    cur = conn.execute(
        """INSERT INTO records (user_id, product_name, channel, channel_note,
                                amount, paid_at, payer)
           VALUES (:user_id, :product_name, :channel, :channel_note,
                   :amount, :paid_at, :payer)""",
        {**fields, "user_id": g.user["id"]},
    )
    conn.commit()
    return jsonify(record=record_to_dict(get_record(cur.lastrowid))), 201


@bp.put("/api/records/<int:rid>")
@login_required
def update_record(rid):
    rec = get_record(rid)
    if rec is None or rec["user_id"] != g.user["id"]:
        return jsonify(error="记录不存在"), 404
    if rec["status"] == "reimbursed":
        return jsonify(error="已报销的记录不能修改，请联系管理员"), 400
    try:
        fields = parse_record_payload(request.get_json(silent=True) or {})
    except ValueError as e:
        return jsonify(error=str(e)), 400
    conn = db.get_db()
    conn.execute(
        """UPDATE records SET product_name=:product_name, channel=:channel,
               channel_note=:channel_note, amount=:amount, paid_at=:paid_at,
               payer=:payer, updated_at=datetime('now','localtime')
           WHERE id=:id""",
        {**fields, "id": rid},
    )
    conn.commit()
    return jsonify(record=record_to_dict(get_record(rid)))


@bp.delete("/api/records/<int:rid>")
@login_required
def delete_record(rid):
    rec = get_record(rid)
    if rec is None or rec["user_id"] != g.user["id"]:
        return jsonify(error="记录不存在"), 404
    if rec["status"] == "reimbursed":
        return jsonify(error="已报销的记录不能删除，请联系管理员"), 400
    conn = db.get_db()
    for inv in conn.execute(
        "SELECT stored_name FROM invoices WHERE record_id=?", (rid,)
    ):
        path = db.UPLOAD_DIR / rec["student_id"] / inv["stored_name"]
        if path.is_file():
            path.unlink()
    conn.execute("DELETE FROM records WHERE id=?", (rid,))
    conn.commit()
    return jsonify(ok=True)


@bp.post("/api/records/<int:rid>/invoices")
@login_required
def upload_invoices(rid):
    rec = get_record(rid)
    if rec is None or rec["user_id"] != g.user["id"]:
        return jsonify(error="记录不存在"), 404
    if rec["status"] == "reimbursed":
        return jsonify(error="该记录已报销，不能再上传文件"), 400
    files = request.files.getlist("files")
    categories = request.form.getlist("categories")
    if not files:
        return jsonify(error="请选择要上传的文件"), 400
    if len(categories) != len(files):
        return jsonify(error="每个文件都必须指定类型（发票/附件）"), 400
    if any(c not in INVOICE_CATEGORIES for c in categories):
        return jsonify(error="文件类型不合法"), 400

    prepared = []
    for f, category in zip(files, categories):
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTS:
            return jsonify(error=f"文件 {f.filename} 格式不支持，仅支持 PDF/JPG/PNG"), 400
        content = f.read()
        if not content:
            return jsonify(error=f"文件 {f.filename} 内容为空"), 400
        if len(content) > MAX_INVOICE_SIZE:
            return jsonify(error=f"文件 {f.filename} 超过 10MB 限制"), 400
        prepared.append((f.filename, uuid.uuid4().hex + ext, content, category))

    invoice_dir = db.UPLOAD_DIR / rec["student_id"]
    invoice_dir.mkdir(parents=True, exist_ok=True)
    conn = db.get_db()
    try:
        for orig_name, stored_name, content, category in prepared:
            (invoice_dir / stored_name).write_bytes(content)
            conn.execute(
                "INSERT INTO invoices (record_id, orig_name, stored_name, category) VALUES (?, ?, ?, ?)",
                (rid, orig_name, stored_name, category),
            )
        _refresh_invoice_state(rid)
        conn.commit()
    except Exception:
        for _, stored_name, _, _ in prepared:
            path = invoice_dir / stored_name
            if path.exists():
                path.unlink()
        raise
    return jsonify(record=record_to_dict(get_record(rid)))


@bp.delete("/api/invoices/<int:iid>")
@login_required
def delete_invoice(iid):
    inv = db.get_db().execute(
        """SELECT i.id, i.stored_name, r.id AS record_id, r.status, u.student_id, r.user_id
           FROM invoices i
           JOIN records r ON r.id = i.record_id
           JOIN users u ON u.id = r.user_id
           WHERE i.id=?""",
        (iid,),
    ).fetchone()
    if inv is None:
        return jsonify(error="文件不存在"), 404
    if g.user["role"] != "admin" and inv["user_id"] != g.user["id"]:
        return jsonify(error="无权删除该文件"), 403
    if inv["status"] == "reimbursed":
        return jsonify(error="该记录已报销，文件不能删除，请联系管理员"), 400

    path = db.UPLOAD_DIR / inv["student_id"] / inv["stored_name"]
    if path.is_file():
        path.unlink()
    conn = db.get_db()
    conn.execute("DELETE FROM invoices WHERE id=?", (iid,))
    _refresh_invoice_state(inv["record_id"])
    conn.commit()
    return jsonify(record=record_to_dict(get_record(inv["record_id"])))


@bp.get("/api/invoices/<int:iid>/download")
@login_required
def download_invoice(iid):
    inv = db.get_db().execute(
        """SELECT i.orig_name, i.stored_name, u.student_id, r.user_id
           FROM invoices i
           JOIN records r ON r.id = i.record_id
           JOIN users u ON u.id = r.user_id
           WHERE i.id=?""",
        (iid,),
    ).fetchone()
    if inv is None:
        return jsonify(error="发票不存在"), 404
    if g.user["role"] != "admin" and inv["user_id"] != g.user["id"]:
        return jsonify(error="无权下载该发票"), 403
    path = db.UPLOAD_DIR / inv["student_id"] / inv["stored_name"]
    if not path.is_file():
        return jsonify(error="发票文件已丢失，请联系管理员"), 404
    return send_file(path, download_name=inv["orig_name"], as_attachment=True)

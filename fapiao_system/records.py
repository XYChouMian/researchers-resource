"""用户侧接口：购买记录增删改查、发票上传与下载。"""
import io
import uuid
from datetime import datetime
from pathlib import Path

from flask import Blueprint, g, jsonify, request, send_file

import db
import export
from auth import login_required

bp = Blueprint("records", __name__)

CHANNELS = ("taobao", "jd", "other")
PAYERS = ("self", "tang")
INVOICE_CATEGORIES = ("invoice", "attachment")
TERMINAL_STATUSES = ("reimbursed", "processed")
ALLOWED_EXTS = {".pdf", ".jpg", ".jpeg", ".png"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MAX_INVOICE_SIZE = 20 * 1024 * 1024

_RECORD_SQL = """
SELECT r.*, u.student_id, u.name AS user_name
FROM records r JOIN users u ON u.id = r.user_id
"""


def get_record(rid):
    return db.get_db().execute(_RECORD_SQL + " WHERE r.id=?", (rid,)).fetchone()


def record_to_dict(row):
    invoices = db.get_db().execute(
        "SELECT id, orig_name, category, remark, uploaded_at FROM invoices WHERE record_id=? ORDER BY id",
        (row["id"],),
    ).fetchall()
    result = dict(row)
    result["invoices"] = [dict(i) for i in invoices]
    return result


def _refresh_invoice_state(rid):
    """开票张数与状态由发票类文件数自动决定；从已驳回修改后自动回流并清空理由；终态保持不变。"""
    conn = db.get_db()
    status = conn.execute("SELECT status FROM records WHERE id=?", (rid,)).fetchone()[0]
    if status in TERMINAL_STATUSES:
        return
    count = conn.execute(
        "SELECT COUNT(*) FROM invoices WHERE record_id=? AND category='invoice'",
        (rid,),
    ).fetchone()[0]
    conn.execute(
        """UPDATE records SET invoice_count=?, status=?,
               reject_note = CASE WHEN ? = 'rejected' THEN NULL ELSE reject_note END,
               updated_at = datetime('now', 'localtime')
           WHERE id=?""",
        (count, "invoiced" if count else "pending", status, rid),
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
    if len(channel_note) > 100:
        raise ValueError("渠道说明不能超过 100 字")
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
    remark = str(data.get("remark", "")).strip()
    if len(remark) > 200:
        raise ValueError("备注不能超过 200 字")
    return {
        "product_name": product_name,
        "channel": channel,
        "channel_note": channel_note,
        "remark": remark,
        "amount": amount,
        "paid_at": paid_at,
        "payer": payer,
    }


def batch_ids(data):
    """解析批量操作的记录 id 列表，不合法时抛 ValueError。"""
    ids = data.get("ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("请提供记录 id 列表")
    try:
        return [int(i) for i in ids]
    except (TypeError, ValueError):
        raise ValueError("记录 id 不合法")


def load_export_records(ids, user_id=None):
    """按 id 列表取出导出所需记录与发票明细（user_id 用于限定本人）。"""
    placeholders = ",".join("?" * len(ids))
    sql = f"""SELECT r.*, u.student_id, u.name AS user_name
              FROM records r JOIN users u ON u.id = r.user_id
              WHERE r.id IN ({placeholders})"""
    params = list(ids)
    if user_id is not None:
        sql += " AND r.user_id = ?"
        params.append(user_id)
    sql += " ORDER BY u.student_id, u.name, r.id"
    rows = db.get_db().execute(sql, params).fetchall()
    invoices = db.get_db().execute(
        f"""SELECT i.record_id, i.orig_name, i.stored_name, i.category
            FROM invoices i WHERE i.record_id IN ({placeholders}) ORDER BY i.id""",
        ids,
    ).fetchall()
    by_record = {}
    for inv in invoices:
        by_record.setdefault(inv["record_id"], []).append(dict(inv))
    records = []
    for row in rows:
        item = dict(row)
        item["invoices"] = by_record.get(row["id"], [])
        records.append(item)

    def read_file(student_id, stored_name):
        path = db.UPLOAD_DIR / student_id / stored_name
        return path.read_bytes() if path.is_file() else None

    return records, read_file


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
                                remark, amount, paid_at, payer)
           VALUES (:user_id, :product_name, :channel, :channel_note,
                   :remark, :amount, :paid_at, :payer)""",
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
    if rec["status"] in TERMINAL_STATUSES:
        return jsonify(error="已报销或已处理的记录不能修改，请联系管理员"), 400
    try:
        fields = parse_record_payload(request.get_json(silent=True) or {})
    except ValueError as e:
        return jsonify(error=str(e)), 400
    conn = db.get_db()
    conn.execute(
        """UPDATE records SET product_name=:product_name, channel=:channel,
               channel_note=:channel_note, remark=:remark, amount=:amount,
               paid_at=:paid_at, payer=:payer,
               updated_at=datetime('now','localtime')
           WHERE id=:id""",
        {**fields, "id": rid},
    )
    if rec["status"] == "rejected":
        _refresh_invoice_state(rid)
    conn.commit()
    return jsonify(record=record_to_dict(get_record(rid)))


@bp.delete("/api/records/<int:rid>")
@login_required
def delete_record(rid):
    rec = get_record(rid)
    if rec is None or rec["user_id"] != g.user["id"]:
        return jsonify(error="记录不存在"), 404
    if rec["status"] in TERMINAL_STATUSES:
        return jsonify(error="已报销或已处理的记录不能删除，请联系管理员"), 400
    conn = db.get_db()
    for inv in conn.execute(
        "SELECT stored_name FROM invoices WHERE record_id=?", (rid,)
    ):
        _unlink_invoice_file(db.UPLOAD_DIR / rec["student_id"] / inv["stored_name"])
    conn.execute("DELETE FROM records WHERE id=?", (rid,))
    conn.commit()
    return jsonify(ok=True)


@bp.post("/api/records/<int:rid>/invoices")
@login_required
def upload_invoices(rid):
    rec = get_record(rid)
    if rec is None or rec["user_id"] != g.user["id"]:
        return jsonify(error="记录不存在"), 404
    if rec["status"] in TERMINAL_STATUSES:
        return jsonify(error="该记录已报销或已处理，不能再上传文件"), 400
    files = request.files.getlist("files")
    categories = request.form.getlist("categories")
    remarks = request.form.getlist("remarks")
    if not files:
        return jsonify(error="请选择要上传的文件"), 400
    if len(categories) != len(files):
        return jsonify(error="每个文件都必须指定类型（发票/附件）"), 400
    if any(c not in INVOICE_CATEGORIES for c in categories):
        return jsonify(error="文件类型不合法"), 400
    if not remarks:
        remarks = [""] * len(files)
    if len(remarks) != len(files):
        return jsonify(error="备注数量必须与文件一一对应"), 400
    remarks = [str(r).strip() for r in remarks]
    if any(len(r) > 100 for r in remarks):
        return jsonify(error="文件备注不能超过 100 字"), 400

    prepared = []
    for f, category, remark in zip(files, categories, remarks):
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTS:
            return jsonify(error=f"文件 {f.filename} 格式不支持，仅支持 PDF/JPG/PNG"), 400
        content = f.read()
        if not content:
            return jsonify(error=f"文件 {f.filename} 内容为空"), 400
        if len(content) > MAX_INVOICE_SIZE:
            return jsonify(error=f"文件 {f.filename} 超过 20MB 限制"), 400
        prepared.append((f.filename, uuid.uuid4().hex + ext, content, category, remark))

    invoice_dir = db.UPLOAD_DIR / rec["student_id"]
    invoice_dir.mkdir(parents=True, exist_ok=True)
    conn = db.get_db()
    try:
        for orig_name, stored_name, content, category, remark in prepared:
            (invoice_dir / stored_name).write_bytes(content)
            conn.execute(
                "INSERT INTO invoices (record_id, orig_name, stored_name, category, remark) VALUES (?, ?, ?, ?, ?)",
                (rid, orig_name, stored_name, category, remark),
            )
        _refresh_invoice_state(rid)
        conn.commit()
    except Exception:
        for _, stored_name, _, _, _ in prepared:
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
    if inv["status"] in TERMINAL_STATUSES:
        return jsonify(error="该记录已报销或已处理，文件不能删除，请联系管理员"), 400

    _unlink_invoice_file(db.UPLOAD_DIR / inv["student_id"] / inv["stored_name"])
    conn = db.get_db()
    conn.execute("DELETE FROM invoices WHERE id=?", (iid,))
    _refresh_invoice_state(inv["record_id"])
    conn.commit()
    return jsonify(record=record_to_dict(get_record(inv["record_id"])))


def _preview_pdf_path(stored_path):
    """图片发票的派生预览 PDF 路径（与原图同目录）。"""
    return stored_path.with_name(stored_path.stem + ".preview.pdf")


def _unlink_invoice_file(stored_path):
    """删除发票原文件及其派生预览 PDF（若存在）。"""
    for path in (stored_path, _preview_pdf_path(stored_path)):
        if path.is_file():
            path.unlink()


def _ensure_image_preview_pdf(stored_path):
    """把图片发票转换为同比例 PDF 供浏览器内嵌预览，返回派生文件路径；失败返回 None。

    派生文件首次预览时生成并缓存（uuid 文件名不变，缓存永不失效）。
    """
    derived = _preview_pdf_path(stored_path)
    if derived.is_file():
        return derived
    try:
        from PIL import Image, ImageOps

        with Image.open(stored_path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                background = Image.new("RGB", im.size, (255, 255, 255))
                background.paste(im, mask=im.split()[-1])
                im = background
            elif im.mode != "RGB":
                im = im.convert("RGB")
            im.save(derived, "PDF", resolution=96, quality=90)
    except Exception:
        if derived.exists():
            derived.unlink()
        return None
    return derived


@bp.get("/api/records/<int:rid>/detail")
@login_required
def record_detail(rid):
    """记录详情：含发票列表与管理员操作历史（本人或管理员可见）。"""
    rec = get_record(rid)
    if rec is None:
        return jsonify(error="记录不存在"), 404
    if g.user["role"] != "admin" and rec["user_id"] != g.user["id"]:
        return jsonify(error="无权查看该记录"), 403
    history = db.get_db().execute(
        """SELECT a.action, a.note, a.created_at, u.name AS operator
           FROM audit_logs a JOIN users u ON u.id = a.operator_id
           WHERE a.record_id=? ORDER BY a.id""",
        (rid,),
    ).fetchall()
    return jsonify(record=record_to_dict(rec), history=[dict(h) for h in history])


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
    inline = request.args.get("inline") == "1"
    if inline and path.suffix.lower() in IMAGE_EXTS:
        derived = _ensure_image_preview_pdf(path)
        if derived is not None:
            return send_file(derived, mimetype="application/pdf", as_attachment=False)
    return send_file(path, download_name=inv["orig_name"], as_attachment=not inline)


@bp.post("/api/records/batch-delete")
@login_required
def batch_delete_records():
    """批量删除本人的可删记录（待开票/已开票/已驳回）；终态自动跳过。"""
    try:
        ids = batch_ids(request.get_json(silent=True) or {})
    except ValueError as e:
        return jsonify(error=str(e)), 400
    conn = db.get_db()
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""SELECT r.id, u.student_id
            FROM records r JOIN users u ON u.id = r.user_id
            WHERE r.user_id = ? AND r.status IN ('pending', 'invoiced', 'rejected')
              AND r.id IN ({placeholders})""",
        [g.user["id"]] + ids,
    ).fetchall()
    for row in rows:
        for inv in conn.execute(
            "SELECT stored_name FROM invoices WHERE record_id=?", (row["id"],)
        ):
            _unlink_invoice_file(db.UPLOAD_DIR / row["student_id"] / inv["stored_name"])
        conn.execute("DELETE FROM records WHERE id=?", (row["id"],))
    conn.commit()
    return jsonify(deleted=len(rows))


@bp.get("/api/records/export")
@login_required
def export_my_records():
    """导出本人选中的记录为报销材料包 zip（与管理员导出同格式）。"""
    raw_ids = request.args.get("ids", "")
    try:
        ids = list(dict.fromkeys(int(i) for i in raw_ids.split(",") if i.strip()))
    except ValueError:
        return jsonify(error="记录 id 不合法"), 400
    if not ids:
        return jsonify(error="请先勾选要导出的记录"), 400
    records, read_file = load_export_records(ids, user_id=g.user["id"])
    if not records:
        return jsonify(error="选中的记录不存在"), 404
    filename = "我的报销材料-%s.zip" % datetime.now().strftime("%Y%m%d")
    return send_file(
        io.BytesIO(export.build_export_zip(records, read_file)),
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename,
    )

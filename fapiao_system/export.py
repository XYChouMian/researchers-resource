"""报销材料打包：把选中记录导出为 zip（根目录 Excel 清单 + 按用户分文件夹的重命名文件）。

本模块为纯函数集合，不依赖 Flask，便于独立测试与复用。
"""
import io
import re
import zipfile
from pathlib import Path

from openpyxl import Workbook

ILLEGAL_CHARS_RE = re.compile(r'[\\/:*?"<>|\r\n\t]')
CHANNEL_TEXT = {"taobao": "淘宝", "jd": "京东", "other": "其他"}
PAYER_TEXT = {"self": "本人", "tang": "唐老师"}
STATUS_TEXT = {"pending": "待开票", "invoiced": "已开票", "reimbursed": "已报销", "rejected": "已驳回", "processed": "已处理"}
FILE_TYPE_TEXT = {"invoice": "发票", "attachment": "附件"}

SHEET_HEADERS = ["学号", "姓名", "商品名", "购买渠道", "渠道说明", "备注", "付款金额(元)",
                 "付款时间", "付款人", "开票张数", "状态", "文件名"]
COLUMN_WIDTHS = (12, 10, 30, 10, 18, 20, 12, 18, 10, 10, 10, 40)


def sanitize_filename(text, limit=80):
    """把任意文本转成安全的文件名片段。"""
    cleaned = ILLEGAL_CHARS_RE.sub("_", str(text)).strip().strip(".")
    return (cleaned or "未命名")[:limit]


def unique_name(used, base, ext):
    """base.ext 被占用时依次改用 base(1).ext、base(2).ext…"""
    name = f"{base}{ext}"
    n = 1
    while name in used:
        name = f"{base}({n}){ext}"
        n += 1
    used.add(name)
    return name


def build_export_zip(records, read_file):
    """把记录打包为 zip 字节流。

    records: 每项为记录字典，需含记录字段与 student_id、user_name、invoices
             （invoices 每项含 orig_name、stored_name、category）。
    read_file(student_id, stored_name) -> bytes | None，None 表示磁盘文件缺失。
    """
    sheet_rows = []
    entries = []
    folders = {}
    used_names = {}

    for rec in records:
        renamed = []
        for inv in rec["invoices"]:
            content = read_file(rec["student_id"], inv["stored_name"])
            if content is None:
                renamed.append("（文件缺失）")
                continue
            folder = _folder_for(rec["user_id"], rec["user_name"], folders, used_names)
            base = "{}_{:.2f}_{}".format(
                sanitize_filename(rec["product_name"]),
                float(rec["amount"]),
                FILE_TYPE_TEXT.get(inv["category"], "附件"),
            )
            name = unique_name(used_names[folder], base, _ext_of(inv))
            entries.append((f"{folder}/{name}", content))
            renamed.append(name)
        sheet_rows.append(_sheet_row(rec, renamed))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("报销清单.xlsx", _xlsx_bytes(sheet_rows))
        for path, content in entries:
            zf.writestr(path, content)
    return buf.getvalue()


def _folder_for(user_id, user_name, folders, used_names):
    """按姓名建文件夹；不同用户重名时依次 张三(2)、张三(3)…（按人的序号，从 2 起）"""
    if user_id not in folders:
        base = sanitize_filename(user_name, limit=50)
        name, n = base, 2
        while name in folders.values():
            name = f"{base}({n})"
            n += 1
        folders[user_id] = name
        used_names[name] = set()
    return folders[user_id]


def _ext_of(inv):
    return Path(inv["stored_name"]).suffix.lower() or Path(inv["orig_name"]).suffix.lower()


def _sheet_row(rec, renamed):
    return [
        rec["student_id"], rec["user_name"], rec["product_name"],
        CHANNEL_TEXT[rec["channel"]], rec["channel_note"], rec.get("remark", ""),
        float(rec["amount"]),
        rec["paid_at"], PAYER_TEXT[rec["payer"]],
        rec["invoice_count"] or 0, STATUS_TEXT[rec["status"]],
        ";".join(renamed) if renamed else "未开票",
    ]


def _xlsx_bytes(rows):
    wb = Workbook()
    ws = wb.active
    ws.title = "报销清单"
    ws.append(SHEET_HEADERS)
    for row in rows:
        ws.append(row)
    for col, width in zip("ABCDEFGHIJKL", COLUMN_WIDTHS):
        ws.column_dimensions[col].width = width
    for cells in ws.iter_rows(min_row=2, min_col=7, max_col=7):
        for cell in cells:
            cell.number_format = "0.00"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

"""API 冒烟测试：登录、记录增删改、发票/附件上传删除、选中导出 zip、报销、用户管理、子路径。"""
import io
import zipfile

import db

from conftest import ADMIN_ID, ADMIN_PASSWORD, make_user

PDF_BYTES = b"%PDF-1.4\n fake invoice"


def login(client, student_id, password):
    return client.post("/api/login", json={"student_id": student_id, "password": password})


def create_record(client, **overrides):
    payload = {
        "product_name": "移液枪枪头",
        "channel": "taobao",
        "channel_note": "",
        "amount": "128.5",
        "paid_at": "2026-09-01",
        "payer": "self",
    }
    payload.update(overrides)
    return client.post("/api/records", json=payload)


def upload_files(client, rid, items=(("fapiao1.pdf", "invoice"), ("fapiao2.jpg", "invoice"))):
    return client.post(
        f"/api/records/{rid}/invoices",
        data={
            "files": [(io.BytesIO(PDF_BYTES), name) for name, _ in items],
            "categories": [category for _, category in items],
        },
        content_type="multipart/form-data",
    )


def test_login_wrong_password(client):
    assert login(client, "admin", "wrong").status_code == 401


def test_unauthorized_access(client):
    assert client.get("/api/records").status_code == 401
    assert client.get("/api/admin/users").status_code == 401


def test_user_cannot_access_admin_api(client):
    make_user(client)
    login(client, "2023123", "2023123")
    assert client.get("/api/admin/users").status_code == 403
    assert client.post("/api/admin/reimburse", json={"ids": [1]}).status_code == 403


def test_record_validation(client):
    make_user(client)
    login(client, "2023123", "2023123")
    assert create_record(client, channel="other", channel_note="").status_code == 400
    assert create_record(client, amount="abc").status_code == 400
    assert create_record(client, amount="0").status_code == 400
    assert create_record(client, paid_at="2026/09/01").status_code == 400
    assert create_record(client, payer="boss").status_code == 400
    assert client.get("/api/records").get_json()["records"] == []


def test_full_reimburse_flow(client):
    make_user(client)
    login(client, "2023123", "2023123")

    resp = create_record(client)
    assert resp.status_code == 201
    rid = resp.get_json()["record"]["id"]

    resp = client.put(f"/api/records/{rid}", json={
        "product_name": "移液枪枪头（改）", "channel": "jd", "channel_note": "",
        "amount": "99.9", "paid_at": "2026-09-02", "payer": "tang",
    })
    assert resp.status_code == 200
    assert resp.get_json()["record"]["channel"] == "jd"

    resp = upload_files(client, rid)
    assert resp.status_code == 200
    record = resp.get_json()["record"]
    assert record["status"] == "invoiced"
    assert record["invoice_count"] == 2
    assert len(record["invoices"]) == 2

    # 已开票的记录仍然可以修改
    resp = client.put(f"/api/records/{rid}", json={
        "product_name": "移液枪枪头（改2）", "channel": "jd", "channel_note": "",
        "amount": "88", "paid_at": "2026-09-02", "payer": "tang",
    })
    assert resp.status_code == 200

    # 已开票后可以补充附件，开票张数不受影响
    resp = upload_files(client, rid, (("hetong.pdf", "attachment"),))
    assert resp.status_code == 200
    record = resp.get_json()["record"]
    assert record["status"] == "invoiced"
    assert record["invoice_count"] == 2
    assert len(record["invoices"]) == 3

    inv_id = record["invoices"][0]["id"]
    resp = client.get(f"/api/invoices/{inv_id}/download")
    assert resp.status_code == 200
    assert resp.data.startswith(b"%PDF")

    login(client, ADMIN_ID, ADMIN_PASSWORD)
    resp = client.get(f"/api/admin/export?ids={rid}")
    assert resp.status_code == 200
    assert resp.mimetype == "application/zip"
    assert resp.data[:2] == b"PK"
    with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
        assert "报销清单.xlsx" in zf.namelist()

    resp = client.post("/api/admin/reimburse", json={"ids": [rid]})
    assert resp.status_code == 200
    assert resp.get_json()["updated"] == 1

    login(client, "2023123", "2023123")
    record = client.get("/api/records").get_json()["records"][0]
    assert record["status"] == "reimbursed"
    # 已报销后：上传、改、删记录、删文件全部被锁
    assert upload_files(client, rid, (("x.pdf", "invoice"),)).status_code == 400
    assert client.put(f"/api/records/{rid}", json={
        "product_name": "x", "channel": "jd", "channel_note": "",
        "amount": "1", "paid_at": "2026-09-02", "payer": "tang",
    }).status_code == 400
    assert client.delete(f"/api/records/{rid}").status_code == 400
    assert client.delete(f"/api/invoices/{inv_id}").status_code == 400


def test_invoice_count_auto_and_revert(client):
    make_user(client)
    login(client, "2023123", "2023123")
    rid = create_record(client).get_json()["record"]["id"]

    resp = upload_files(client, rid, (
        ("f1.pdf", "invoice"), ("a1.jpg", "attachment"), ("f2.png", "invoice"),
    ))
    record = resp.get_json()["record"]
    assert record["status"] == "invoiced"
    assert record["invoice_count"] == 2
    assert [i["category"] for i in record["invoices"]] == [
        "invoice", "attachment", "invoice",
    ]

    resp = client.delete(f"/api/invoices/{record['invoices'][0]['id']}")
    record = resp.get_json()["record"]
    assert record["invoice_count"] == 1
    assert record["status"] == "invoiced"

    att_id = next(i["id"] for i in record["invoices"] if i["category"] == "attachment")
    record = client.delete(f"/api/invoices/{att_id}").get_json()["record"]
    assert record["invoice_count"] == 1
    assert record["status"] == "invoiced"

    last_inv_id = next(i["id"] for i in record["invoices"] if i["category"] == "invoice")
    record = client.delete(f"/api/invoices/{last_inv_id}").get_json()["record"]
    assert record["invoice_count"] == 0
    assert record["status"] == "pending"


def test_upload_categories_validation(client):
    make_user(client)
    login(client, "2023123", "2023123")
    rid = create_record(client).get_json()["record"]["id"]

    resp = client.post(
        f"/api/records/{rid}/invoices",
        data={
            "files": [(io.BytesIO(PDF_BYTES), "a.pdf")],
            "categories": ["invoice", "attachment"],
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400

    resp = client.post(
        f"/api/records/{rid}/invoices",
        data={
            "files": [(io.BytesIO(PDF_BYTES), "a.pdf")],
            "categories": ["bogus"],
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400

    resp = client.post(
        f"/api/records/{rid}/invoices",
        data={"files": [(io.BytesIO(PDF_BYTES), "a.pdf")]},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert client.get("/api/records").get_json()["records"][0]["invoices"] == []


def test_upload_rejects_bad_extension(client):
    make_user(client)
    login(client, "2023123", "2023123")
    rid = create_record(client).get_json()["record"]["id"]
    resp = client.post(
        f"/api/records/{rid}/invoices",
        data={
            "files": [(io.BytesIO(b"MZxxx"), "virus.exe")],
            "categories": ["invoice"],
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert client.get("/api/records").get_json()["records"][0]["status"] == "pending"


def test_delete_record_allowed_until_reimbursed(client):
    make_user(client)
    login(client, "2023123", "2023123")
    rid = create_record(client).get_json()["record"]["id"]
    upload_files(client, rid, (("a.pdf", "invoice"),))
    rid2 = create_record(client).get_json()["record"]["id"]
    assert client.delete(f"/api/records/{rid2}").status_code == 200
    assert client.delete(f"/api/records/{rid}").status_code == 200
    assert client.get("/api/records").get_json()["records"] == []


def test_user_cannot_touch_others_record(client):
    make_user(client, "2023123")
    make_user(client, "2023456")
    login(client, "2023123", "2023123")
    rid = create_record(client).get_json()["record"]["id"]
    upload_files(client, rid, (("a.pdf", "invoice"),))
    inv_id = client.get("/api/records").get_json()["records"][0]["invoices"][0]["id"]
    login(client, "2023456", "2023456")
    assert client.delete(f"/api/records/{rid}").status_code == 404
    assert upload_files(client, rid, (("x.pdf", "invoice"),)).status_code == 404
    assert client.get(f"/api/invoices/{inv_id}/download").status_code == 403
    assert client.delete(f"/api/invoices/{inv_id}").status_code == 403


def test_change_password(client):
    make_user(client)
    login(client, "2023123", "2023123")
    resp = client.post("/api/password", json={"old_password": "bad", "new_password": "654321"})
    assert resp.status_code == 400
    resp = client.post("/api/password", json={"old_password": "2023123", "new_password": "654321"})
    assert resp.status_code == 200
    assert login(client, "2023123", "2023123").status_code == 401
    assert login(client, "2023123", "654321").status_code == 200


def test_admin_user_management(client):
    login(client, ADMIN_ID, ADMIN_PASSWORD)
    assert client.post("/api/admin/users", json={"student_id": "2023123"}).status_code == 201
    assert client.post("/api/admin/users", json={"student_id": "2023123"}).status_code == 400
    assert client.post("/api/admin/users", json={"student_id": "非法!"}).status_code == 400

    users = client.get("/api/admin/users").get_json()["users"]
    uid = next(u["id"] for u in users if u["student_id"] == "2023123")
    admin_id = next(u["id"] for u in users if u["student_id"] == ADMIN_ID)

    assert client.post(f"/api/admin/users/{uid}", json={}).status_code == 405
    assert client.post(f"/api/admin/users/{uid}/reset_password", json={}).status_code == 200
    assert client.delete(f"/api/admin/users/{admin_id}").status_code == 400

    resp = client.put(f"/api/admin/users/{uid}", json={"name": "新名字", "role": "user"})
    assert resp.status_code == 200

    resp = client.post("/api/admin/users", json={"student_id": "2099999"})
    uid2 = resp.get_json()["user"]["id"]
    assert client.delete(f"/api/admin/users/{uid2}").status_code == 200


def test_admin_cannot_delete_user_with_records(client):
    make_user(client)
    login(client, "2023123", "2023123")
    create_record(client)
    login(client, ADMIN_ID, ADMIN_PASSWORD)
    users = client.get("/api/admin/users").get_json()["users"]
    uid = next(u["id"] for u in users if u["student_id"] == "2023123")
    resp = client.delete(f"/api/admin/users/{uid}")
    assert resp.status_code == 400
    assert "购买记录" in resp.get_json()["error"]


def test_admin_records_filter(client):
    make_user(client)
    login(client, "2023123", "2023123")
    create_record(client)
    login(client, ADMIN_ID, ADMIN_PASSWORD)
    data = client.get("/api/admin/records?status=invoiced").get_json()
    assert data["records"] == []
    data = client.get("/api/admin/records").get_json()
    assert len(data["records"]) == 1
    assert data["records"][0]["student_id"] == "2023123"


def test_reimburse_invalid_ids(client):
    login(client, ADMIN_ID, ADMIN_PASSWORD)
    assert client.post("/api/admin/reimburse", json={"ids": []}).status_code == 400
    assert client.post("/api/admin/reimburse", json={"ids": ["abc"]}).status_code == 400


def _export(client, ids):
    return client.get(f"/api/admin/export?ids={','.join(map(str, ids))}")


def _zip_entries(data):
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def _sheet_rows(xlsx_bytes):
    from openpyxl import load_workbook

    ws = load_workbook(io.BytesIO(xlsx_bytes)).active
    return [list(row) for row in ws.iter_rows(min_row=2, values_only=True)]


def test_export_ids_validation(client):
    login(client, ADMIN_ID, ADMIN_PASSWORD)
    assert client.get("/api/admin/export").status_code == 400
    assert client.get("/api/admin/export?ids=abc").status_code == 400
    assert client.get("/api/admin/export?ids=999").status_code == 404


def test_export_selected_zip(client):
    make_user(client, "2023123", "张三")
    make_user(client, "2023456", "张三")
    make_user(client, "2023789", "李四")

    login(client, "2023123", "2023123")
    ra = create_record(client).get_json()["record"]["id"]
    upload_files(client, ra, (("f1.pdf", "invoice"), ("a1.jpg", "attachment")))
    rb = create_record(client).get_json()["record"]["id"]
    upload_files(client, rb, (("f2.pdf", "invoice"),))

    login(client, "2023456", "2023456")
    rc = create_record(client, product_name="离心管", amount="9.9").get_json()["record"]["id"]
    upload_files(client, rc, (("f3.pdf", "invoice"),))

    login(client, "2023789", "2023789")
    rd = create_record(client, product_name="耗材", amount="5").get_json()["record"]["id"]

    login(client, ADMIN_ID, ADMIN_PASSWORD)
    resp = _export(client, [ra, rb, rc, rd])
    assert resp.status_code == 200
    entries = _zip_entries(resp.data)
    assert set(entries) == {
        "报销清单.xlsx",
        "张三/移液枪枪头_128.50_发票.pdf",
        "张三/移液枪枪头_128.50_附件.jpg",
        "张三/移液枪枪头_128.50_发票(1).pdf",
        "张三(2)/离心管_9.90_发票.pdf",
    }

    rows = _sheet_rows(entries["报销清单.xlsx"])
    assert len(rows) == 4
    assert rows[0][-1] == "移液枪枪头_128.50_发票.pdf;移液枪枪头_128.50_附件.jpg"
    assert rows[1][-1] == "移液枪枪头_128.50_发票(1).pdf"
    assert rows[2][-1] == "离心管_9.90_发票.pdf"
    assert rows[2][0] == "2023456"
    assert rows[3][9] == "待开票"
    assert rows[3][-1] == "未开票"


def test_export_marks_missing_file(client):
    make_user(client)
    login(client, "2023123", "2023123")
    rid = create_record(client).get_json()["record"]["id"]
    upload_files(client, rid, (("f.pdf", "invoice"),))
    conn = db.connect()
    stored = conn.execute("SELECT stored_name FROM invoices").fetchone()[0]
    conn.close()
    (db.UPLOAD_DIR / "2023123" / stored).unlink()

    login(client, ADMIN_ID, ADMIN_PASSWORD)
    entries = _zip_entries(_export(client, [rid]).data)
    assert set(entries) == {"报销清单.xlsx"}
    rows = _sheet_rows(entries["报销清单.xlsx"])
    assert rows[0][-1] == "（文件缺失）"


def test_prefix_routing(client, monkeypatch):
    monkeypatch.setenv("FAPIAO_PREFIX", "/fapiao")
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    c = app.test_client()
    assert c.get("/fapiao/api/me").status_code == 401
    assert c.get("/fapiao/").status_code == 200
    assert c.get("/fapiao/static/css/style.css").status_code == 200
    assert c.get("/api/me").status_code == 404
    assert c.get("/other/api/me").status_code == 404

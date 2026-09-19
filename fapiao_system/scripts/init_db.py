"""初始化数据库：建表 + 创建默认管理员（admin/admin）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db
from werkzeug.security import generate_password_hash


def main():
    db.init_db()
    conn = db.connect()
    try:
        exists = conn.execute("SELECT 1 FROM users WHERE student_id='admin'").fetchone()
        if exists:
            print("管理员已存在，跳过创建")
            return
        conn.execute(
            "INSERT INTO users (student_id, name, password_hash, role) VALUES (?, ?, ?, ?)",
            ("admin", "管理员", generate_password_hash("admin"), "admin"),
        )
        conn.commit()
        print("已创建默认管理员：学号 admin，密码 admin（请登录后尽快修改密码）")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

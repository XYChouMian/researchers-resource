"""发票管理系统入口：创建应用、注册蓝图、子路径支持与启动。"""
import logging
import os
import secrets
from datetime import timedelta

from flask import Flask, render_template

import db
from admin import bp as admin_bp
from auth import bp as auth_bp
from records import bp as records_bp

PREFIX = os.environ.get("FAPIAO_PREFIX", "/fapiao")
PORT = int(os.environ.get("FAPIAO_PORT", "8085"))


class PrefixMiddleware:
    """让应用挂在反向代理的子路径下（如 /fapiao/）。"""

    def __init__(self, wsgi_app, prefix):
        self.wsgi_app = wsgi_app
        self.prefix = prefix.rstrip("/")

    def __call__(self, environ, start_response):
        if not self.prefix:
            return self.wsgi_app(environ, start_response)
        path = environ.get("PATH_INFO", "")
        if path == self.prefix or path.startswith(self.prefix + "/"):
            environ["SCRIPT_NAME"] = self.prefix
            environ["PATH_INFO"] = path[len(self.prefix):] or "/"
            return self.wsgi_app(environ, start_response)
        body = f"请通过 {self.prefix}/ 访问本系统\n".encode()
        start_response("404 Not Found", [
            ("Content-Type", "text/plain; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ])
        return [body]


def _load_secret_key():
    db.DATA_DIR.mkdir(parents=True, exist_ok=True)
    key_file = db.DATA_DIR / "secret_key"
    if not key_file.exists():
        key_file.write_text(secrets.token_hex(32), encoding="utf-8")
    return key_file.read_text(encoding="utf-8").strip()


def _setup_admin_log():
    """管理员操作日志落盘 data/admin.log（每行一个 JSON 对象，应用内不可删除）。"""
    admin_logger = logging.getLogger("fapiao.admin")
    if admin_logger.handlers:
        return
    handler = logging.FileHandler(db.DATA_DIR / "admin.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    admin_logger.addHandler(handler)
    admin_logger.setLevel(logging.INFO)
    admin_logger.propagate = False


def create_app():
    app = Flask(__name__)
    prefix = os.environ.get("FAPIAO_PREFIX", PREFIX)
    app.secret_key = _load_secret_key()
    app.config.update(
        MAX_CONTENT_LENGTH=25 * 1024 * 1024,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        TEMPLATES_AUTO_RELOAD=True,
    )
    app.register_blueprint(auth_bp)
    app.register_blueprint(records_bp)
    app.register_blueprint(admin_bp)
    app.teardown_appcontext(db.close_db)
    _setup_admin_log()

    @app.get("/")
    def index_page():
        return render_template("index.html")

    @app.get("/user")
    def user_page():
        return render_template("user.html")

    @app.get("/reimburse")
    def reimburse_page():
        return render_template("reimburse.html")

    @app.get("/admin")
    def admin_page():
        return render_template("admin.html")

    @app.get("/logs")
    def logs_page():
        return render_template("logs.html")

    @app.errorhandler(404)
    def not_found(_e):
        return {"error": "接口或页面不存在"}, 404

    @app.errorhandler(413)
    def too_large(_e):
        return {"error": "上传内容过大（单个发票文件限 20MB）"}, 413

    if prefix:
        app.wsgi_app = PrefixMiddleware(app.wsgi_app, prefix)
    return app


app = create_app()

if __name__ == "__main__":
    db.init_db()
    print(f"发票管理系统运行于 http://127.0.0.1:{PORT}{PREFIX}/")
    app.run(host="127.0.0.1", port=PORT, threaded=True)

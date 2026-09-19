# 发票管理系统

Flask + SQLite 的实验室报销系统。监听 `127.0.0.1:8085`，通过 nginx 子路径 `/fapiao/` 对外提供
（nginx 配置由服务器管理员维护，`proxy_pass http://127.0.0.1:8085;` 不带尾部 URI）。

项目约定、数据模型、API 契约见 **AGENTS.md**（AI 维护的唯一权威说明）。

## 快速开始

```bash
python scripts/init_db.py   # 建库 + 创建默认管理员 admin/admin（登录后请改密）
python app.py               # 监听 127.0.0.1:8085，前缀 /fapiao
```

本地调试不带前缀：`FAPIAO_PREFIX= python app.py`

## 生产部署（systemd）

```bash
sudo cp fapiao.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fapiao
```

## 备份

crontab（每天 02:30，保留最近 30 份）：

```
30 2 * * * /mnt/data1/resources/fapiao_system/scripts/backup.sh
```

## 测试

```bash
pytest
```

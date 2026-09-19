#!/usr/bin/env bash
# SQLite 在线备份数据库到 data/backups/，保留最近 30 份。
# crontab 示例（每天 02:30 备份）：
#   30 2 * * * /mnt/data1/resources/fapiao_system/scripts/backup.sh
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
python3 - "$DIR/data" <<'PY'
import datetime
import pathlib
import sqlite3
import sys

data = pathlib.Path(sys.argv[1])
backups = data / "backups"
backups.mkdir(parents=True, exist_ok=True)

name = backups / f"fapiao-{datetime.date.today():%Y%m%d}.db"
src = sqlite3.connect(data / "fapiao.db")
dst = sqlite3.connect(name)
src.backup(dst)
dst.close()
src.close()

for old in sorted(backups.glob("fapiao-*.db"))[:-30]:
    old.unlink()
print(f"已备份到 {name}")
PY

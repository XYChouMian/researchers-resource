# AGENTS.md — AI 维护手册

本文件是本项目的**唯一权威说明**。任何 AI agent 修改代码前必须先读完本文件；
修改代码后如涉及架构、接口、数据结构的变化，**必须同步更新本文件**。

## 1. 项目目标

实验室内部发票报销系统，用户少于 20 人，无并发压力。

角色与三步工作流（状态机）：

| 步骤 | 操作者 | 动作 | 记录状态 |
|---|---|---|---|
| 1 | 用户 | 购买后首次填报：商品名、购买渠道（淘宝/京东/其他+说明）、付款金额、付款时间（默认当前，可改过去）、付款人（本人/唐老师） | `pending` 待开票 |
| 2 | 用户 | 开票后第二次上传：发票文件与辅助附件（PDF/JPG/PNG，可多个，单个≤10MB），每个文件标注类型（发票/附件），开票张数=发票类文件数（自动） | `invoiced` 已开票 |
| 3 | 管理员 | 勾选记录导出报销材料包（zip：根目录 Excel 清单 + 按用户分文件夹的重命名发票/附件；待开票记录只进表格），线下报销成功后批量标记 | `reimbursed` 已报销（终态） |

业务规则：

- 用户名 = 学号，初始密码 = 学号，**不能自助注册**，管理员全权管理用户
- `pending`/`invoiced` 状态的记录用户均可修改/删除，也可单独增删文件；发票类文件被删光时记录自动回到 `pending`；`reimbursed` 为终态，记录与文件全部锁定
- 不强制改密，页面提供修改密码入口；管理员可把任意用户密码重置为学号
- `pending`/`invoiced` 的记录均可补充上传发票；已报销（`reimbursed`）不可再传
- 默认管理员：学号 `admin` 密码 `admin`（由 scripts/init_db.py 创建，部署后人工改密）

## 2. 技术栈（刻意保持最小，禁止随意加依赖）

| 层 | 选择 | 说明 |
|---|---|---|
| 后端 | Flask 3.x + sqlite3 标准库 | 服务器已预装，无并发压力 |
| 前端 | 原生 HTML/CSS/JS | 无框架、无构建步骤，内容(HTML)与样式(CSS)分离 |
| 导出 | openpyxl | 生成 .xlsx |
| 测试 | pytest | 全流程 API 冒烟测试 |

## 3. 目录结构与文件职责

```
fapiao_system/
├── AGENTS.md            本文件（AI 维护契约）
├── app.py               入口：create_app、子路径中间件、页面路由、启动
├── db.py                数据库层：路径、连接、建表 SQL（唯一 schema 定义处）
├── auth.py              认证：登录/登出/改密接口 + 权限装饰器
├── records.py           用户侧 API：购买记录增删改查、发票上传/下载
├── admin.py             管理员 API：用户管理、记录查看、报销材料导出、报销标记
├── export.py            报销材料打包：文件名清洗、重名去重、zip 构建（纯函数，无 Flask 依赖）
├── templates/           Jinja2 模板，只用 url_for 引静态资源（保证子路径正确）
│   ├── base.html        公共骨架
│   ├── index.html       登录页
│   ├── user.html        用户页
│   └── admin.html       管理页
├── static/
│   ├── css/style.css    全部样式（内容与样式分离的唯一样式文件）
│   └── js/
│       ├── app.js       公共：api() 封装、esc()、toast、弹窗、改密弹窗、顶栏事件
│       ├── login.js     登录页逻辑
│       ├── user.js      用户页逻辑
│       └── admin.js     管理页逻辑
├── scripts/
│   ├── init_db.py       初始化数据库 + 创建默认管理员
│   └── backup.sh        SQLite 在线备份（供 crontab，保留最近 30 份）
├── tests/
│   ├── conftest.py      夹具：临时目录数据库、空前缀应用
│   └── test_api.py      API 冒烟测试
├── fapiao.service       systemd 单元文件（部署时人工安装）
├── requirements.txt     flask、openpyxl
├── data/                运行时生成：fapiao.db、secret_key、backups/（勿提交）
└── uploads/             运行时生成：发票文件，按学号分目录（勿提交）
```

## 4. 数据模型（定义在 db.py 的 SCHEMA，唯一出处）

- `users(id, student_id UNIQUE, name, password_hash, role admin|user, created_at)`
- `records(id, user_id→users, product_name, channel taobao|jd|other, channel_note,
  amount>0, paid_at 'YYYY-MM-DD', payer self|tang, invoice_count?,
  status pending|invoiced|reimbursed, reimbursed_at?, created_at, updated_at)`
- `invoices(id, record_id→records ON DELETE CASCADE, orig_name, stored_name(uuid),
  category invoice|attachment, uploaded_at)`

约定：时间统一文本 `YYYY-MM-DD HH:MM`（datetime('now','localtime')）；金额 REAL 元。

## 5. API 契约（全部 JSON；错误返回 `{"error": "中文原因"}` + 4xx）

| 方法 | 路径 | 权限 | 说明 |
|---|---|---|---|
| POST | /api/login | 匿名 | `{student_id, password}` |
| POST | /api/logout | 匿名 | 清除会话 |
| GET | /api/me | 登录 | 当前用户 |
| POST | /api/password | 登录 | `{old_password, new_password}` ≥6位 |
| GET | /api/records | 登录 | 本人全部记录（含 invoices 列表） |
| POST | /api/records | 登录 | 新建记录（校验见 records.parse_record_payload） |
| PUT | /api/records/\<id\> | 本人 | `pending`/`invoiced` 可改；`reimbursed` 拒绝 |
| DELETE | /api/records/\<id\> | 本人 | `pending`/`invoiced` 可删，磁盘文件一并清理；`reimbursed` 拒绝 |
| POST | /api/records/\<id\>/invoices | 本人 | multipart：`files`多文件 + `categories`逐文件 invoice\|attachment；`pending`/`invoiced` 可传；张数与状态自动重算 |
| DELETE | /api/invoices/\<id\> | 本人或管理员 | 删除单个文件；记录未报销才允许；发票删光自动回到待开票 |
| GET | /api/invoices/\<id\>/download | 本人或管理员 | 附件下载 |
| GET | /api/admin/users | 管理员 | 用户列表（含记录数统计） |
| POST | /api/admin/users | 管理员 | `{student_id, name?, role?}` 建号，初始密码=学号 |
| PUT | /api/admin/users/\<id\> | 管理员 | 改 name/role；不能改自己的角色 |
| POST | /api/admin/users/\<id\>/reset_password | 管理员 | 重置为学号 |
| DELETE | /api/admin/users/\<id\> | 管理员 | 不能删自己；名下有记录则拒绝（保护数据） |
| GET | /api/admin/records?status=a,b&user_id= | 管理员 | 全部记录（含学号/姓名） |
| GET | /api/admin/export?ids=1,2,3 | 管理员 | 导出选中记录的报销材料包 zip：根目录\`报销清单.xlsx\`（含“文件名”列，分号分隔）+ 按姓名分文件夹（重名加(2)）+ \`商品名_金额_发票/附件.扩展名\`（冲突加(1)）；待开票记录只进表格、文件名列标“未开票”；磁盘缺失文件标“（文件缺失）” |
| POST | /api/admin/reimburse | 管理员 | `{ids:[...]}` 批量标记已报销，返回 `{updated}` |

页面路由：`/` 登录页、`/user` 用户页、`/admin` 管理页（页面级权限由 JS 调 /api/me 把关）。

## 6. 编码规范

- 后端与前端**完全隔离**：后端只输出 JSON 和模板页面，业务逻辑禁止写进前端 JS
  或 Jinja 模板；前端只通过 `api()` 调接口
- 每个文件顶部一句模块 docstring；函数一般不写注释，除非逻辑不明显
- SQL 必须**参数化**（`?` 占位），禁止字符串拼接用户输入；仅 `IN (...)` 占位符个数
  可以动态拼接（个数来自服务端 int 校验后的列表长度）
- 所有用户输入在**服务端**校验（前端校验仅用于体验）；错误消息用中文，可直接展示
- 前端插入用户数据的唯一方式是 `esc()`（app.js），防止 XSS
- 时间在服务端一律 `datetime('now','localtime')`；用户提交付款时间格式 `YYYY-MM-DD`（只精确到天）
- UI 文案中文；状态枚举与中文映射只在 export.py（导出用）与 app.js（展示用）

## 7. 安全清单（修改代码时必须维持）

- [ ] 密码只用 werkzeug pbkdf2 哈希，任何地方不得出现明文
- [ ] SECRET_KEY 存 `data/secret_key`，首次启动自动生成，不入库不提交
- [ ] 会话 Cookie：HttpOnly + SameSite=Lax，有效期 12h
- [ ] 上传：扩展名白名单 pdf/jpg/jpeg/png、单文件 ≤10MB（全局 20MB）、
      磁盘文件名一律 uuid 重命名，按学号目录存放（学号建号时已校验为字母数字）
- [ ] 权限双层：装饰器拦截 + 数据属主比对（records/invoices 均校验 user_id）
- [ ] SQLite：外键开启、WAL 模式、每日备份脚本

## 8. 常见维护任务

- **给记录加字段**：db.py SCHEMA（新字段带 DEFAULT）→ records.parse_record_payload
  校验 → INSERT/UPDATE 语句 → user.html 表单与 user.js → 本文件第 4/5 节
- **重置管理员密码**（忘记时）：`python -c "import sys; sys.path.insert(0,'.'); from werkzeug.security import generate_password_hash; import db; c=db.connect(); c.execute('UPDATE users SET password_hash=? WHERE student_id=?',(generate_password_hash('新密码'),'admin')); c.commit()"`
- **跑测试**：`pytest`（tests 用临时目录，不碰真实 data/）
- **本地运行**：`python scripts/init_db.py && python app.py`
  （FAPIAO_PREFIX 默认 /fapiao；本地调试可设 `FAPIAO_PREFIX=`）

## 9. 禁区

- 不要修改 nginx 配置（由服务器管理员人工维护）
- 不要引入前端框架、构建工具、ORM、新的重型依赖
- 不要破坏 API 返回结构（前端与 AGENTS.md 依赖它）；改动必须三处同步
- 不要把发票文件存进数据库、不要改 uuid 命名规则
- 不要删除或绕过任何安全清单（第 7 节）中的措施

## 10. 部署事实

- 监听 `127.0.0.1:8085`（FAPIAO_PORT），子路径前缀 `/fapiao`（FAPIAO_PREFIX）
- nginx 由管理员配置：`location /fapiao/ { proxy_pass http://127.0.0.1:8085; }`
  （proxy_pass 不带尾部 URI，全路径透传，由 PrefixMiddleware 剥前缀）
- 进程管理：systemd 单元见 fapiao.service；备份 crontab 见 scripts/backup.sh 头注

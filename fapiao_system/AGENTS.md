# AGENTS.md — AI 维护手册

本文件是本项目的**唯一权威说明**。任何 AI agent 修改代码前必须先读完本文件；
修改代码后如涉及架构、接口、数据结构的变化，**必须同步更新本文件**。

## 1. 项目目标

实验室内部发票报销系统，用户少于 20 人，无并发压力。

角色与三步工作流（状态机）：

| 步骤 | 操作者 | 动作 | 记录状态 |
|---|---|---|---|
| 1 | 用户 | 购买后首次填报：商品名、购买渠道（淘宝/京东/其他+说明）、付款金额、付款时间（默认当前，可改过去）、付款人（本人/唐老师）、备注（选填） | `pending` 待开票 |
| 2 | 用户 | 开票后第二次上传：发票文件与辅助附件（PDF/JPG/PNG，可多个，单个≤20MB），每个文件标注类型（发票/附件）与选填备注，开票张数=发票类文件数（自动） | `invoiced` 已开票 |
| 3 | 管理员 | 发票报销页：勾选已开票记录导出材料包，线下报销成功后标记"已报销"；不符合要求的可"驳回"（必填理由，用户修改单据后自动回到已开票）；无法正常报销的可标记"已处理"（必填处理说明，等同已报销锁定）；以上均可一键"恢复未报销" | `reimbursed`/`processed`（终态）/ `rejected` 已驳回（用户修改后自动回流） |

业务规则：

- 用户名 = 学号，初始密码 = 学号，**不能自助注册**，管理员全权管理用户
- `pending`/`invoiced` 状态的记录用户均可修改/删除，也可单独增删文件；发票类文件被删光时记录自动回到 `pending`；`reimbursed`/`processed` 为终态，记录与文件全部锁定
- 报销仅对"已开票"记录生效；已报销/已处理可由管理员"恢复未报销"（按发票数自动回到已开票/待开票，清空报销时间/处理说明）
- `processed`（已处理）仅管理员可设置且必须填写处理说明（≤200 字）
- 驳回仅对"已开票"生效且必填理由（≤200 字）；被驳回记录用户可编辑/增删附件/删除重报，任意修改后自动回到已开票（发票删光回待开票）并清空理由
- 用户可勾选多条记录批量删除（规则同单条删除：终态跳过）并导出选中记录的报销材料包 zip（与管理员导出同格式，仅限本人记录）
- 管理员本人也可提交发票；三个页面：`/user` 发票提交（所有登录用户）、`/reimburse` 发票报销、`/admin` 后台管理（后两者管理员专属）
- 不强制改密，页面提供修改密码入口；管理员可把任意用户密码重置为学号
- `pending`/`invoiced` 的记录均可补充上传发票；已报销（`reimbursed`）不可再传
- 默认管理员：学号 `admin` 密码 `admin`（由 scripts/init_db.py 创建，部署后人工改密）

## 2. 技术栈（刻意保持最小，禁止随意加依赖）

| 层 | 选择 | 说明 |
|---|---|---|
| 后端 | Flask 3.x + sqlite3 标准库 | 服务器已预装，无并发压力 |
| 前端 | 原生 HTML/CSS/JS | 无框架、无构建步骤，内容(HTML)与样式(CSS)分离 |
| 导出 | openpyxl | 生成 .xlsx |
| 预览转换 | Pillow | 图片发票预览时自动转 PDF（环境自带，非新安装） |
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
│   ├── _topbar.html     顶栏导航（含管理员专属链接，由 CSS+JS 控制显隐）
│   ├── index.html       登录页
│   ├── user.html        发票提交页（所有登录用户）
│   ├── reimburse.html   发票报销页（管理员）
│   ├── admin.html       后台管理页（用户管理，管理员）
│   └── logs.html        系统日志页（管理员，由后台管理页进入）
├── static/
│   ├── css/style.css    全部样式（内容与样式分离的唯一样式文件）
│   └── js/
│       ├── app.js       公共：api() 封装、esc()、toast、弹窗、改密弹窗、文件链接工具、表格分页组件、顶栏事件
│       ├── login.js     登录页逻辑
│       ├── user.js      发票提交页逻辑
│       ├── reimburse.js 发票报销页逻辑
│       ├── admin.js     后台管理页逻辑
│       └── logs.js      系统日志页逻辑
├── scripts/
│   ├── init_db.py       初始化数据库 + 创建默认管理员
│   └── backup.sh        SQLite 在线备份（供 crontab，保留最近 30 份）
├── tests/
│   ├── conftest.py      夹具：临时目录数据库、空前缀应用
│   └── test_api.py      API 冒烟测试
├── fapiao.service       systemd 单元文件（部署时人工安装）
├── requirements.txt     flask、openpyxl
├── data/                运行时生成：fapiao.db、secret_key、admin.log、backups/（勿提交）
└── uploads/             运行时生成：发票文件，按学号分目录（含 .preview.pdf 派生预览文件，勿提交）
```

## 4. 数据模型（定义在 db.py 的 SCHEMA，唯一出处）

- `users(id, student_id UNIQUE, name, password_hash, role admin|user, created_at)`
- `records(id, user_id→users, product_name, channel taobao|jd|other, channel_note,
  remark'', amount>0, paid_at 'YYYY-MM-DD', payer self|tang, invoice_count?,
  status pending|invoiced|reimbursed|rejected|processed,
  reimbursed_at?, process_note?, reject_note?, reimburse_note?,
  created_at, updated_at)
- `audit_logs(id, record_id, action reimburse|reject|revert|process,
  operator_id→users, note, created_at)`
  —— 管理员操作审计（append-only：仅有 INSERT 入口，record_id 无外键级联，任何接口不可删除）`
- `invoices(id, record_id→records ON DELETE CASCADE, orig_name, stored_name(uuid),
  category invoice|attachment, remark'', uploaded_at)`

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
| POST | /api/records/batch-delete | 本人 | `{ids:[...]}` 批量删除本人可删记录（待开票/已开票/已驳回；终态跳过），返回 `{deleted}` |
| GET | /api/records/export?ids= | 本人 | 导出本人选中记录的报销材料包 zip（与管理员导出同格式） |
| POST | /api/records/\<id\>/invoices | 本人 | multipart：`files`多文件 + `categories`逐文件 invoice\|attachment + `remarks`逐文件选填备注(≤100字)；`pending`/`invoiced` 可传；张数与状态自动重算 |
| DELETE | /api/invoices/\<id\> | 本人或管理员 | 删除单个文件；记录未报销才允许；发票删光自动回到待开票 |
| GET | /api/invoices/\<id\>/download | 本人或管理员 | 附件下载；`?inline=1` 为浏览器内嵌预览（图片自动转 PDF，转换失败回退原图） |
| GET | /api/admin/users | 管理员 | 用户列表（含记录数统计） |
| POST | /api/admin/users | 管理员 | `{student_id, name?, role?}` 建号，初始密码=学号 |
| PUT | /api/admin/users/\<id\> | 管理员 | 改 name/role；不能改自己的角色 |
| POST | /api/admin/users/\<id\>/reset_password | 管理员 | 重置为学号 |
| DELETE | /api/admin/users/\<id\> | 管理员 | 不能删自己；名下有记录则拒绝（保护数据） |
| GET | /api/admin/records?status=a,b&user_id= | 管理员 | 全部记录（含学号/姓名） |
| GET | /api/records/\<id\>/detail | 本人或管理员 | 记录详情：记录 + 发票列表 + 操作历史（审计） |
| GET | /api/admin/export?ids=1,2,3 | 管理员 | 导出选中记录的报销材料包 zip：根目录\`报销清单.xlsx\`（含“备注”列与“文件名”列，分号分隔）+ 按姓名分文件夹（重名加(2)）+ \`商品名_金额_发票/附件.扩展名\`（冲突加(1)）；待开票记录只进表格、文件名列标“未开票”；磁盘缺失文件标“（文件缺失）” |
| POST | /api/admin/reimburse | 管理员 | `{ids:[...]}` 批量标记已报销（仅已开票生效），返回 `{updated}` |
| POST | /api/admin/reject | 管理员 | `{ids:[...], note}` 仅已开票可驳回，note 必填 ≤200 字；用户修改单据后自动回到已开票 |
| POST | /api/admin/revert | 管理员 | `{ids:[...]}` 已报销/已处理/已驳回恢复为未报销（按发票数自动回已开票/待开票），返回 `{updated}` |
| POST | /api/admin/process | 管理员 | `{ids:[...], note}` 待开票/已开票标记为已处理，note 必填 ≤200 字，返回 `{updated}` |
| GET | /api/admin/logs | 管理员 | 后台操作日志（`data/admin.log` 末尾 1000 行，倒序；JSON 行解析为对象，历史格式行返回 `{raw}`） |

页面路由：`/` 登录页、`/user` 发票提交页、`/reimburse` 发票报销页、`/admin` 后台管理页、`/logs` 系统日志页（后三者为管理员专属，页面级权限由 JS 调 /api/me 把关）。

## 6. 编码规范

- 后端与前端**完全隔离**：后端只输出 JSON 和模板页面，业务逻辑禁止写进前端 JS
  或 Jinja 模板；前端只通过 `api()` 调接口
- 每个文件顶部一句模块 docstring；函数一般不写注释，除非逻辑不明显
- SQL 必须**参数化**（`?` 占位），禁止字符串拼接用户输入；仅 `IN (...)` 占位符个数
  可以动态拼接（个数来自服务端 int 校验后的列表长度）
- 所有用户输入在**服务端**校验（前端校验仅用于体验）；错误消息用中文，可直接展示
- 前端插入用户数据的唯一方式是 `esc()`（app.js），防止 XSS
- 时间在服务端一律 `datetime('now','localtime')`；用户提交付款时间格式 `YYYY-MM-DD`（只精确到天）
- UI 文案中文；状态/渠道/付款人中文映射在 export.py（导出与日志共用）与 app.js（展示用）；操作动作映射在 admin.py（日志用）与 app.js（操作历史展示用）

## 7. 安全清单（修改代码时必须维持）

- [ ] 密码只用 werkzeug 哈希（scrypt/pbkdf2），任何地方不得出现明文
- [ ] SECRET_KEY 存 `data/secret_key`，首次启动自动生成，不入库不提交
- [ ] 会话 Cookie：HttpOnly + SameSite=Lax，有效期 12h
- [ ] 上传：扩展名白名单 pdf/jpg/jpeg/png、单文件 ≤20MB（全局 25MB，含 multipart 开销余量）、
      磁盘文件名一律 uuid 重命名，按学号目录存放（学号建号时已校验为字母数字）
- [ ] 权限双层：装饰器拦截 + 数据属主比对（records/invoices 均校验 user_id）
- [ ] SQLite：外键开启、WAL 模式、每日备份脚本
- [ ] 管理员四类报销操作（报销/驳回/恢复/已处理）写 audit_logs + data/admin.log，
      只增不删，任何接口不可清除（操作留痕）

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
  **必配** `client_max_body_size 25m;`（nginx 默认仅 1MB，缺失会导致上传 413）
- 进程管理：systemd 单元见 fapiao.service；备份 crontab 见 scripts/backup.sh 头注

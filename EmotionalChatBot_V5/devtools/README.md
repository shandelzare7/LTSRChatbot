# Devtools

开发与运维脚本，按需手动执行。运行前请配置 `DATABASE_URL` 或使用项目根目录下的 `.env`。从项目根目录执行时可用 `python EmotionalChatBot_V5/devtools/xxx.py`。

## 运维与数据修复（原 tools）

| 脚本 | 用途 |
|------|------|
| `clear_memory.py` | 清理指定用户/关系的记忆与转录 |
| `fix_bot_ages.py` | 批量修正 Bot 的 age 字段 |
| `fix_user_dimensions.py` | 修正用户关系维度（0–1 归一化等） |
| `push_openai_key_to_server.py` | 将 OpenAI API Key 推送到远程环境 |
| `update_relationship_defaults.py` | 更新关系维度默认值/模板 |

## 数据库与迁移

| 脚本 | 用途 |
|------|------|
| `ensure_schema.py` | 确保 DB 表结构存在（Render 启动时常用） |
| `create_new_database.py` | 创建新数据库与表结构 |
| `create_two_bots_for_render.py` | 为 Render 环境创建两个 Bot |
| `reset_and_seed_local_postgres.py` / `seed_local_postgres.py` | 重置并种子化本地 Postgres |
| `migrate_*.py` / `*.sql` | 各类迁移脚本 |

## 查询与日志

| 脚本 | 用途 |
|------|------|
| `query_*.py` | 按用户/ Bot/ 维度等查询 |
| `download_*.py` | 从 Render 下载日志/会话 |
| `fetch_recent_render_logs_and_check_failures.py` | 拉取近期 Render 日志并检查失败 |

## 其他

| 脚本 | 用途 |
|------|------|
| `bot_to_bot_chat.py` | 双 Bot 对话测试/压测 |
| `add_bot_name_to_users.py` / `add_users_display_view.py` | 用户表与展示字段 |
| `generate_vapid_keys.py` | 生成 VAPID 密钥 |
| `update_bot_lore_from_sidewrite.py` | 从 sidewrite 更新 Bot 设定 |

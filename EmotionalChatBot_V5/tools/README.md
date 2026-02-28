# 运维与数据工具

本目录为迁移、修复与运维脚本，按需手动执行。

| 脚本 | 用途 |
|------|------|
| `clear_memory.py` | 清理指定用户/关系的记忆与转录 |
| `fix_bot_ages.py` | 批量修正 Bot 的 age 字段 |
| `fix_user_dimensions.py` | 修正用户关系维度（0–1 归一化等） |
| `push_openai_key_to_server.py` | 将 OpenAI API Key 推送到远程环境 |
| `update_relationship_defaults.py` | 更新关系维度默认值/模板 |

运行前请配置 `DATABASE_URL` 或使用项目根目录下的 `.env`。

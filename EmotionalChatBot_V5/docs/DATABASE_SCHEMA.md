# 数据库结构说明（嵌套：Bot → User → 消息/记忆）

## 结构

- **bots**：顶层，一个 Bot 一行。含 `mood_state`（PAD(B) 情绪，该 bot 下所有用户共享）和 `urgent_tasks`（Bot 级别紧急任务）。
- **users**：挂在 bot 下，`(bot_id, external_id)` 唯一；同一人在不同 Bot 下是不同行。含 `basic_info` 及关系状态（`current_stage`, `dimensions`, `inferred_profile`, `assets`, `spt_info`, `conversation_summary`）和 `urgent_tasks`（User 级别紧急任务）。
- **messages / memories / transcripts / derived_notes**：外键为 `user_id`（→ users.id），表示「某 Bot 下某用户」的数据。

## 新库

**方式一：一键创建新库并播种（推荐）**

在 `.env` 中已配置 `DATABASE_URL`（可指向任意已存在库，如 `postgres`）时：

```bash
# 创建新库 ltsrchatbot_v5，执行 init_schema，写入示例数据，并可选更新 .env
python devtools/create_new_database.py

# 指定新库名
NEW_DB_NAME=my_chatbot python devtools/create_new_database.py

# 创建后把 .env 里的 DATABASE_URL 改为指向新库
python devtools/create_new_database.py --update-env
```

**方式二：在已有空库上建表并播种**

```bash
# 将 .env 的 DATABASE_URL 指向目标空库后执行
python devtools/seed_local_postgres.py   # 会按需建表并写入一条示例
```

## 已有旧库（含 relationships、旧 users）

先备份，再执行一次性迁移：

```bash
python devtools/migrate_user_bound_to_bot.py
```

若 **users 表仍有 mood_state 列**（PAD(B) 尚未迁至 bot），需执行：

```bash
python devtools/migrate_mood_state_to_bot.py
```

若 **bots/users 表缺少 urgent_tasks 列**，需执行：

```bash
python devtools/migrate_add_urgent_tasks.py
```

或使用「重置并播种」脚本（会先跑迁移再清空并写入示例）：

```bash
python devtools/reset_and_seed_local_postgres.py
```

## 紧急任务（urgent_tasks）

紧急任务存储在 `bots.urgent_tasks`（Bot 级别，所有用户共享）和 `users.urgent_tasks`（User 级别，仅针对特定 bot-user 关系）。

**来源**：
1. **开发者写入 DB**：通过 SQL 直接写入 `bots.urgent_tasks` 或 `users.urgent_tasks`
2. **Detection 节点生成**：Detection LLM 在极端情况下（如用户即将愤怒离开、自伤暗示等）产出紧急任务

**格式**（JSONB 数组）：
```json
[{"description": "立即安抚用户情绪", "importance": 0.9, "source": "developer"}]
```

**行为**：
- 紧急任务绕过 TaskPlanner 的 LLM 打分，直接注入 `tasks_for_lats`
- LATS 生成回复时必须完成紧急任务（不受 `task_budget_max` 限制）
- 执行后自动从 DB 中清空（一次性消费），并在日志中高亮报告

## 说明

- 对外接口不变：仍传 `(user_id, bot_id)`，其中 `user_id` 即 external_id；内部按 `(bot_id, external_id)` 定位唯一 User。
- `app/services/db_service.py` 使用另一套同步模型（BotModel、RelationshipModel 等），与当前 `app/core/database.py` 的异步 ORM 不一致；若使用该服务需单独对齐或迁移。

-- 补齐 bots / users 与当前 init_schema 一致（Render 若从旧 schema 建库会缺这些列）
-- 执行：psql $RENDER_DATABASE_URL -f devtools/migrate_bots_users_urgent_mood.sql
-- 或：在 Render Shell / 本地用 RENDER_DATABASE_URL 连接后执行本文件

-- bots: PAD(B) 与 Bot 级紧急任务
ALTER TABLE bots ADD COLUMN IF NOT EXISTS mood_state JSONB DEFAULT '{"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0}'::jsonb;
ALTER TABLE bots ADD COLUMN IF NOT EXISTS urgent_tasks JSONB DEFAULT '[]'::jsonb;

-- users: 用户级紧急任务（users 上若还有旧列 mood_state 可保留不删，应用已不再使用）
ALTER TABLE users ADD COLUMN IF NOT EXISTS urgent_tasks JSONB DEFAULT '[]'::jsonb;

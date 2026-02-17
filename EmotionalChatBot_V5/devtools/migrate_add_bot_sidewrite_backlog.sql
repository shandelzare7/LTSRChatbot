-- 为 bots 表增加人物侧写与个性任务库字段（创建 bot 时由 LLM 生成）
-- 执行：psql $DATABASE_URL -f devtools/migrate_add_bot_sidewrite_backlog.sql
ALTER TABLE bots ADD COLUMN IF NOT EXISTS character_sidewrite TEXT;
ALTER TABLE bots ADD COLUMN IF NOT EXISTS backlog_tasks JSONB;

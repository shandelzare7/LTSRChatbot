-- 启用 UUID 生成扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. 定义 Knapp 关系阶段枚举
CREATE TYPE knapp_stage AS ENUM (
    'initiating', 'experimenting', 'intensifying', 'integrating', 'bonding',
    'differentiating', 'circumscribing', 'stagnating', 'avoiding', 'terminating'
);

-- 2. 机器人表 (Bots) - 顶层
CREATE TABLE bots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    basic_info JSONB DEFAULT '{}'::jsonb,
    big_five JSONB DEFAULT '{}'::jsonb,
    persona JSONB DEFAULT '{}'::jsonb,
    character_sidewrite TEXT,
    backlog_tasks JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. 用户表 (Users) - 挂在 bot 下，每个 user 绑定一个 bot
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    bot_id UUID NOT NULL REFERENCES bots(id) ON DELETE CASCADE,
    bot_name TEXT,
    external_id TEXT NOT NULL,
    basic_info JSONB DEFAULT '{}'::jsonb,
    current_stage knapp_stage DEFAULT 'initiating',
    dimensions JSONB DEFAULT '{"closeness": 0.3, "trust": 0.3, "liking": 0.3, "respect": 0.3, "warmth": 0.3, "power": 0.5}'::jsonb,
    mood_state JSONB DEFAULT '{"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0}'::jsonb,
    inferred_profile JSONB DEFAULT '{}'::jsonb,
    assets JSONB DEFAULT '{}'::jsonb,
    spt_info JSONB DEFAULT '{}'::jsonb,
    conversation_summary TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(bot_id, external_id)
);

-- 4. 消息流水表 (Messages) - 挂在 user 下
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role TEXT CHECK (role IN ('user', 'ai', 'system')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. 记忆表 (Memories) - 挂在 user 下
CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 6. Memory Store A：Raw Transcript Store - 挂在 user 下
CREATE TABLE transcripts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id TEXT,
    thread_id TEXT,
    turn_index INTEGER,
    user_text TEXT NOT NULL DEFAULT '',
    bot_text TEXT NOT NULL DEFAULT '',
    entities JSONB DEFAULT '{}'::jsonb,
    topic TEXT,
    importance DOUBLE PRECISION,
    short_context TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 7. Memory Store B：Derived Notes Store - 挂在 user 下
CREATE TABLE derived_notes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    transcript_id UUID NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    note_type TEXT,
    content TEXT NOT NULL,
    importance DOUBLE PRECISION,
    source_pointer TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 8. Bot 任务清单（按 user+bot 维度）
CREATE TABLE bot_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    bot_id UUID NOT NULL REFERENCES bots(id) ON DELETE CASCADE,
    task_type TEXT NOT NULL DEFAULT 'custom',
    description TEXT NOT NULL DEFAULT '',
    importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    last_attempt_at TIMESTAMPTZ,
    attempt_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_bot_tasks_user_bot ON bot_tasks(user_id, bot_id);

-- 9. Web 对话日志快照（用于 Render 等无持久磁盘环境）
CREATE TABLE web_chat_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    bot_id UUID NOT NULL REFERENCES bots(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    filename TEXT,
    content TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, session_id)
);

-- 索引
CREATE INDEX idx_users_bot_external ON users(bot_id, external_id);
CREATE INDEX idx_messages_user_time ON messages(user_id, created_at DESC);
CREATE INDEX idx_transcripts_user_time ON transcripts(user_id, created_at DESC);
CREATE INDEX idx_notes_user_time ON derived_notes(user_id, created_at DESC);
CREATE INDEX idx_notes_transcript ON derived_notes(transcript_id);
CREATE INDEX idx_web_chat_logs_user_time ON web_chat_logs(user_id, updated_at DESC);

-- 便于在库中查看「哪个 User 属于哪个 Bot」的视图（避免只看 uuid 难以辨认）
CREATE OR REPLACE VIEW users_with_bot_names AS
SELECT
  u.id AS user_id,
  u.bot_id,
  b.name AS bot_name,
  u.external_id,
  COALESCE(u.basic_info->>'name', u.basic_info->>'nickname', u.external_id) AS user_name
FROM users u
JOIN bots b ON b.id = u.bot_id;

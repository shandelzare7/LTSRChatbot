-- 启用 UUID 生成扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. 定义 Knapp 关系阶段枚举
CREATE TYPE knapp_stage AS ENUM (
    'initiating', 'experimenting', 'intensifying', 'integrating', 'bonding',
    'differentiating', 'circumscribing', 'stagnating', 'avoiding', 'terminating'
);

-- 2. 机器人表 (Bots)
CREATE TABLE bots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    basic_info JSONB DEFAULT '{}'::jsonb,  -- 对应 BotBasicInfo
    big_five JSONB DEFAULT '{}'::jsonb,    -- 对应 BotBigFive
    persona JSONB DEFAULT '{}'::jsonb,     -- 对应 BotPersona
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. 用户表 (Users)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id TEXT UNIQUE,               -- 外部ID (如微信ID)
    basic_info JSONB DEFAULT '{}'::jsonb,  -- 对应 UserBasicInfo
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. 关系状态表 (Relationships) - 核心引擎
CREATE TABLE relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    bot_id UUID REFERENCES bots(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,

    -- 核心状态字段 (对应 state.py)
    current_stage knapp_stage DEFAULT 'initiating',
    dimensions JSONB DEFAULT '{"closeness": 0, "trust": 0, "liking": 0, "respect": 0, "warmth": 0, "power": 50}'::jsonb, -- RelationshipState
    mood_state JSONB DEFAULT '{"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0}'::jsonb,     -- MoodState
    inferred_profile JSONB DEFAULT '{}'::jsonb,  -- UserInferredProfile
    assets JSONB DEFAULT '{}'::jsonb,            -- RelationshipAssets
    spt_info JSONB DEFAULT '{}'::jsonb,          -- SPTInfo

    conversation_summary TEXT,                   -- 长期记忆摘要

    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(bot_id, user_id) -- 确保一个Bot对应一个User只有一条关系
);

-- 5. 消息流水表 (Messages)
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    relationship_id UUID REFERENCES relationships(id) ON DELETE CASCADE,
    role TEXT CHECK (role IN ('user', 'ai', 'system')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb, -- 存 detection_result, latency, intent 等
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 6. 记忆表 (Memories) - 暂时仅存文本，为未来留接口
CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    relationship_id UUID REFERENCES relationships(id) ON DELETE CASCADE,
    content TEXT NOT NULL,              -- 记忆内容
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 建立索引优化查询速度
CREATE INDEX idx_relationships_lookup ON relationships(bot_id, user_id);
CREATE INDEX idx_messages_rel_time ON messages(relationship_id, created_at DESC);


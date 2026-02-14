"""
为 users 表增加 bot_name 列（明文存 bot 名称），并回填；且将 bot_name 放在 bot_id 后面（列顺序）。
执行一次即可；新库由 init_schema.sql 已包含 bot_name 且顺序正确。

  python devtools/add_bot_name_to_users.py
"""

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from sqlalchemy import text
from app.core.database import DBManager


async def main() -> None:
    if not os.getenv("DATABASE_URL"):
        print("DATABASE_URL 未设置。")
        sys.exit(1)
    db = DBManager.from_env()
    async with db.engine.connect() as conn:
        await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS bot_name TEXT"))
        await conn.execute(
            text("UPDATE users SET bot_name = (SELECT name FROM bots WHERE bots.id = users.bot_id) WHERE bot_name IS NULL")
        )
        await conn.commit()

    # 将 bot_name 移到 bot_id 后面：重建 users 表列顺序（PostgreSQL 无法直接改列序）
    async with db.engine.connect() as conn:
        await conn.execute(text("""
            CREATE TABLE users_new (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                bot_id UUID NOT NULL REFERENCES bots(id) ON DELETE CASCADE,
                bot_name TEXT,
                external_id TEXT NOT NULL,
                basic_info JSONB DEFAULT '{}'::jsonb,
                current_stage knapp_stage DEFAULT 'initiating',
                dimensions JSONB DEFAULT '{"closeness": 0, "trust": 0, "liking": 0, "respect": 0, "warmth": 0, "power": 50}'::jsonb,
                mood_state JSONB DEFAULT '{"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0}'::jsonb,
                inferred_profile JSONB DEFAULT '{}'::jsonb,
                assets JSONB DEFAULT '{}'::jsonb,
                spt_info JSONB DEFAULT '{}'::jsonb,
                conversation_summary TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(bot_id, external_id)
            )
        """))
        await conn.execute(text("""
            INSERT INTO users_new (id, bot_id, bot_name, external_id, basic_info, current_stage, dimensions, mood_state, inferred_profile, assets, spt_info, conversation_summary, created_at, updated_at)
            SELECT id, bot_id, bot_name, external_id, basic_info, current_stage, dimensions, mood_state, inferred_profile, assets, spt_info, conversation_summary, created_at, updated_at
            FROM users
        """))
        await conn.execute(text("DROP TABLE users CASCADE"))
        await conn.execute(text("ALTER TABLE users_new RENAME TO users"))
        await conn.execute(text("CREATE INDEX idx_users_bot_external ON users(bot_id, external_id)"))
        await conn.execute(text("ALTER TABLE messages ADD CONSTRAINT fk_messages_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE"))
        await conn.execute(text("ALTER TABLE memories ADD CONSTRAINT fk_memories_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE"))
        await conn.execute(text("ALTER TABLE transcripts ADD CONSTRAINT fk_transcripts_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE"))
        await conn.execute(text("ALTER TABLE derived_notes ADD CONSTRAINT fk_derived_notes_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE"))
        await conn.commit()
    print("已为 users 表添加 bot_name 并置于 bot_id 后，打开 users 表即可看到列顺序：id, bot_id, bot_name, external_id, ...")


if __name__ == "__main__":
    asyncio.run(main())

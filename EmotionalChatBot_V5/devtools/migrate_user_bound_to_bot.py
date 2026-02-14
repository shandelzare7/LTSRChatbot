"""
迁移脚本：从「全局 users + relationships」改为「users 挂在 bot 下」的嵌套结构。

用法：
  cd EmotionalChatBot_V5
  python devtools/migrate_user_bound_to_bot.py

- 若当前库中不存在表 relationships，视为已是新结构，跳过迁移。
- 若存在 relationships，则：建 users_new(bot_id, external_id, ...)，从 relationships+users 回填；
  给 messages/memories/transcripts/derived_notes 增加 user_id 并回填；删旧 FK 与 relationships/users，重命名 users_new。
- 执行前请备份数据库。
"""

from __future__ import annotations

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


async def _table_exists(conn, name: str) -> bool:
    r = await conn.execute(
        text(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = :t"
        ),
        {"t": name},
    )
    return r.scalar() is not None


async def run_migration():
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL 未设置")

    db = DBManager.from_env()
    async with db.engine.connect() as conn:
        ac = await conn.execution_options(isolation_level="AUTOCOMMIT")
        if not await _table_exists(ac, "relationships"):
            print("未发现 relationships 表，视为已是新结构，跳过迁移。")
            return
        if not await _table_exists(ac, "users"):
            print("未发现 users 表，无法迁移。")
            return

        print("开始迁移：users 绑定到 bot …")
        # 1. 创建 users_new（新结构：bot_id + external_id + 关系状态）
        await ac.execute(text("""
            CREATE TABLE IF NOT EXISTS users_new (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                bot_id UUID NOT NULL REFERENCES bots(id) ON DELETE CASCADE,
                external_id TEXT NOT NULL,
                basic_info JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                current_stage TEXT DEFAULT 'initiating',
                dimensions JSONB DEFAULT '{"closeness": 0, "trust": 0, "liking": 0, "respect": 0, "warmth": 0, "power": 50}'::jsonb,
                mood_state JSONB DEFAULT '{"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0}'::jsonb,
                inferred_profile JSONB DEFAULT '{}'::jsonb,
                assets JSONB DEFAULT '{}'::jsonb,
                spt_info JSONB DEFAULT '{}'::jsonb,
                conversation_summary TEXT,
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(bot_id, external_id)
            )
        """))
        # 2. 从 relationships + users 回填 users_new，建立 rel.id -> users_new.id 映射即用 users_new.id 同序生成
        await ac.execute(text("""
            INSERT INTO users_new (id, bot_id, external_id, basic_info, created_at, current_stage, dimensions, mood_state, inferred_profile, assets, spt_info, conversation_summary, updated_at)
            SELECT
                r.id,
                r.bot_id,
                COALESCE(u.external_id, 'migrated_' || r.id::text),
                COALESCE(u.basic_info, '{}'::jsonb),
                NOW(),
                r.current_stage::text,
                r.dimensions,
                r.mood_state,
                r.inferred_profile,
                r.assets,
                r.spt_info,
                r.conversation_summary,
                r.updated_at
            FROM relationships r
            JOIN users u ON u.id = r.user_id
        """))
        # 3. 子表加 user_id（与 relationship_id 同源，即 users_new.id = 原 rel.id）
        for tbl in ["messages", "memories", "transcripts", "derived_notes"]:
            try:
                await ac.execute(text(f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users_new(id) ON DELETE CASCADE"))
            except Exception:
                pass
            await ac.execute(text(f"UPDATE {tbl} SET user_id = relationship_id WHERE user_id IS NULL AND relationship_id IS NOT NULL"))
        # 4. 删旧 FK、删 relationship_id 列、删 relationships 与旧 users
        for tbl in ["messages", "memories", "transcripts"]:
            await ac.execute(text(f"ALTER TABLE {tbl} DROP CONSTRAINT IF EXISTS {tbl}_relationship_id_fkey"))
            await ac.execute(text(f"ALTER TABLE {tbl} DROP COLUMN IF EXISTS relationship_id"))
        await ac.execute(text("ALTER TABLE derived_notes DROP CONSTRAINT IF EXISTS derived_notes_relationship_id_fkey"))
        await ac.execute(text("ALTER TABLE derived_notes DROP COLUMN IF EXISTS relationship_id"))
        await ac.execute(text("DROP TABLE IF EXISTS relationships CASCADE"))
        await ac.execute(text("DROP TABLE IF EXISTS users CASCADE"))
        # 5. 重命名 users_new -> users
        await ac.execute(text("ALTER TABLE users_new RENAME TO users"))
        # 6. 重建索引（按需）
        await ac.execute(text("CREATE INDEX IF NOT EXISTS idx_users_bot_external ON users(bot_id, external_id)"))
        await ac.execute(text("CREATE INDEX IF NOT EXISTS idx_messages_user_time ON messages(user_id, created_at DESC)"))
        await ac.execute(text("CREATE INDEX IF NOT EXISTS idx_transcripts_user_time ON transcripts(user_id, created_at DESC)"))
        await ac.execute(text("CREATE INDEX IF NOT EXISTS idx_notes_user_time ON derived_notes(user_id, created_at DESC)"))
        print("迁移完成。")
        return


if __name__ == "__main__":
    asyncio.run(run_migration())

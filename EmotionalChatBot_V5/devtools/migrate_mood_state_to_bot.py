"""
迁移：将 PAD(B) mood_state 从 users 表移到 bots 表。
- 给 bots 表增加 mood_state 列（若不存在），默认值 PAD(B) 全 0。
- 从 users 表删除 mood_state 列（若存在）。

运行前请备份数据库。适用于已有库（新库直接用 init_schema.sql 即可）。
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from sqlalchemy import text


PADB_DEFAULT = '{"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0}'


async def run_migration():
    from app.core.database import _create_async_engine_from_database_url, Base
    url = os.getenv("DATABASE_URL")
    if not url:
        print("DATABASE_URL 未设置")
        sys.exit(1)
    engine = _create_async_engine_from_database_url(url)

    async with engine.begin() as conn:
        # 1) bots 表增加 mood_state（若不存在）
        try:
            await conn.execute(text("""
                ALTER TABLE bots
                ADD COLUMN IF NOT EXISTS mood_state JSONB
                DEFAULT '""" + PADB_DEFAULT + """'::jsonb
            """))
            print("bots: mood_state 列已存在或已添加")
        except Exception as e:
            print("bots 添加 mood_state 失败:", e)
            raise

        # 2) 已有 bot 行若 mood_state 为 NULL，设为默认
        await conn.execute(text("""
            UPDATE bots
            SET mood_state = '""" + PADB_DEFAULT + """'::jsonb
            WHERE mood_state IS NULL
        """))
        print("bots: 已为 NULL 的 mood_state 填默认值")

        # 3) users 表删除 mood_state
        try:
            await conn.execute(text("ALTER TABLE users DROP COLUMN IF EXISTS mood_state"))
            print("users: mood_state 列已删除（若存在）")
        except Exception as e:
            print("users 删除 mood_state 失败:", e)
            raise

    print("迁移完成。PAD(B) 现存储在 bots 表，该 bot 下所有用户共享。")


if __name__ == "__main__":
    asyncio.run(run_migration())

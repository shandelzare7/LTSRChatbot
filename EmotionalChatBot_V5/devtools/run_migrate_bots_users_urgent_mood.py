"""
在 Render（或 DATABASE_URL）上执行 migrate_bots_users_urgent_mood.sql，补齐 bots/users 列。

使用：
  RENDER_DATABASE_URL=postgresql+asyncpg://... python -m devtools.run_migrate_bots_users_urgent_mood
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
from app.core.database import _create_async_engine_from_database_url


MIGRATION_SQL = """
ALTER TABLE bots ADD COLUMN IF NOT EXISTS mood_state JSONB DEFAULT '{"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0}'::jsonb;
ALTER TABLE bots ADD COLUMN IF NOT EXISTS urgent_tasks JSONB DEFAULT '[]'::jsonb;
ALTER TABLE users ADD COLUMN IF NOT EXISTS urgent_tasks JSONB DEFAULT '[]'::jsonb;
"""


async def main() -> int:
    url = os.getenv("RENDER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: 请设置 RENDER_DATABASE_URL 或 DATABASE_URL")
        return 1
    engine = _create_async_engine_from_database_url(url)
    try:
        async with engine.connect() as conn:
            async with conn.begin():
                for line in MIGRATION_SQL.strip().split(";"):
                    line = line.strip()
                    if not line:
                        continue
                    await conn.execute(text(line))
        print("已执行迁移: bots.mood_state, bots.urgent_tasks, users.urgent_tasks")
    except Exception as e:
        print(f"执行失败: {e}")
        return 1
    finally:
        await engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

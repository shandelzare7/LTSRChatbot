"""
迁移：为 bots 表和 users 表添加 urgent_tasks 列。
- bots.urgent_tasks: Bot 级别紧急任务，该 Bot 下所有用户共享
- users.urgent_tasks: User 级别紧急任务，仅针对特定 bot-user 关系

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


async def run_migration():
    from app.core.database import _create_async_engine_from_database_url

    url = os.getenv("DATABASE_URL")
    if not url:
        print("DATABASE_URL 未设置")
        sys.exit(1)
    engine = _create_async_engine_from_database_url(url)

    async with engine.begin() as conn:
        # 1) bots 表增加 urgent_tasks（若不存在）
        try:
            await conn.execute(text(
                "ALTER TABLE bots ADD COLUMN IF NOT EXISTS urgent_tasks JSONB DEFAULT '[]'::jsonb"
            ))
            print("bots: urgent_tasks 列已存在或已添加")
        except Exception as e:
            print("bots 添加 urgent_tasks 失败:", e)
            raise

        # 2) 已有 bot 行若 urgent_tasks 为 NULL，设为默认空数组
        await conn.execute(text(
            "UPDATE bots SET urgent_tasks = '[]'::jsonb WHERE urgent_tasks IS NULL"
        ))
        print("bots: 已为 NULL 的 urgent_tasks 填默认值 []")

        # 3) users 表增加 urgent_tasks（若不存在）
        try:
            await conn.execute(text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS urgent_tasks JSONB DEFAULT '[]'::jsonb"
            ))
            print("users: urgent_tasks 列已存在或已添加")
        except Exception as e:
            print("users 添加 urgent_tasks 失败:", e)
            raise

        # 4) 已有 user 行若 urgent_tasks 为 NULL，设为默认空数组
        await conn.execute(text(
            "UPDATE users SET urgent_tasks = '[]'::jsonb WHERE urgent_tasks IS NULL"
        ))
        print("users: 已为 NULL 的 urgent_tasks 填默认值 []")

    print("\n迁移完成。bots 和 users 表均已添加 urgent_tasks 列。")
    print("开发者可通过 SQL 直接向 bots.urgent_tasks 或 users.urgent_tasks 写入紧急任务，")
    print("格式: [{\"description\": \"...\", \"importance\": 0.9, \"source\": \"developer\"}]")
    print("紧急任务将在下一轮对话中被强制执行，执行后自动清空。")


if __name__ == "__main__":
    asyncio.run(run_migration())

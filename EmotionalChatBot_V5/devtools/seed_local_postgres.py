"""
seed_local_postgres.py

用途：
- 连接“本地 PostgreSQL”（通过 env: DATABASE_URL）
- 随机生成一个 bot + 一个 user
- 写入 bots/users/relationships/messages（可选 memories）
- 再把该 bot/user 关联的全部数据查询并打印出来

前置：
1) 启动本地 Postgres
2) 执行 init_schema.sql 初始化表结构
3) 在 EmotionalChatBot_V5/.env 设置 DATABASE_URL（postgresql+asyncpg://...）

运行：
  cd EmotionalChatBot_V5
  python3 devtools/seed_local_postgres.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import select, text

# allow running from devtools/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# load .env (same behavior as main.py)
try:
    from utils.env_loader import load_project_env

    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from app.core.database import Bot, DBManager, Memory, Message, User


def _split_sql_statements(sql: str) -> list[str]:
    """
    Very small SQL splitter: splits by ';' and drops empty chunks.
    Good enough for our init_schema.sql (no stored procedures).
    """
    parts = []
    for chunk in sql.split(";"):
        stmt = chunk.strip()
        if stmt:
            parts.append(stmt)
    return parts


async def _ensure_schema(db: DBManager) -> None:
    """
    使用 SQLAlchemy 直接执行 init_schema.sql（不依赖 psql）。
    - 若对象已存在，会跳过（best-effort）
    """
    schema_path = Path(__file__).resolve().parents[1] / "init_schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    statements = _split_sql_statements(sql)
    # Use AUTOCOMMIT so one failed DDL won't abort the whole sequence.
    async with db.engine.connect() as conn:
        ac = await conn.execution_options(isolation_level="AUTOCOMMIT")
        for stmt in statements:
            try:
                await ac.execute(text(stmt))
            except Exception as e:
                msg = str(e).lower()
                # best-effort ignore for repeated runs
                if "already exists" in msg or "duplicate" in msg:
                    continue
                # CREATE EXTENSION may require superuser; allow continuing
                if "create extension" in stmt.lower():
                    continue
                raise


async def main() -> None:
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL 未设置：请在 .env 里配置本地 PostgreSQL 连接串。")

    db = DBManager.from_env()
    await _ensure_schema(db)

    # 随机生成一个 bot_id（UUID 字符串）和 user_external_id
    bot_id = str(uuid.uuid4())
    user_external_id = f"local_user_{uuid.uuid4().hex[:8]}"

    print("== Seeding local postgres ==")
    print("bot_id:", bot_id)
    print("user_external_id:", user_external_id)

    # 触发 get-or-create：会生成并写入 bot/user/profile/relationship 初始状态
    _ = await db.load_state(user_external_id, bot_id)

    # 写入一轮对话
    now = datetime.now().replace(microsecond=0).isoformat()
    state = {
        "user_id": user_external_id,
        "bot_id": bot_id,
        "current_time": now,
        "user_input": "你好，我想测试一下本地数据库读写。",
        "final_response": "收到，我已经把本地数据库写入跑通了。",
        "detection_category": "NORMAL",
        "humanized_output": {"total_latency_seconds": 1.23},
        # 关键：把 relationships 要更新的字段也带上（最少带 current_stage/relationship_state/mood_state）
        "current_stage": "initiating",
        "relationship_state": {"closeness": 5, "trust": 3, "liking": 6, "respect": 50, "warmth": 10, "power": 50},
        "mood_state": {"pleasure": 0.1, "arousal": 0.2, "dominance": 0.0, "busyness": 0.0},
        # 可选
        "new_memory_content": "这是一次本地数据库写入测试。",
    }
    await db.save_turn(user_external_id, bot_id, state)

    # 查询打印：bot / user（挂在 bot 下）/ messages / memories
    async with db.Session() as session:
        async with session.begin():
            bot_uuid = uuid.UUID(bot_id)
            bot = (await session.execute(select(Bot).where(Bot.id == bot_uuid))).scalars().first()
            user = (
                (await session.execute(select(User).where(User.bot_id == bot_uuid, User.external_id == user_external_id)))
                .scalars()
                .first()
            )

            if not bot or not user:
                raise RuntimeError("bot/user 写入失败：请确认 init_schema.sql 是否已执行，且 DATABASE_URL 指向正确库。")

            msgs = (
                (await session.execute(select(Message).where(Message.user_id == user.id).order_by(Message.created_at.asc())))
                .scalars()
                .all()
            )
            mems = (
                (await session.execute(select(Memory).where(Memory.user_id == user.id).order_by(Memory.created_at.asc())))
                .scalars()
                .all()
            )

            print("\n== BOT ==")
            print({"id": str(bot.id), "name": bot.name, "basic_info": bot.basic_info, "big_five": bot.big_five})

            print("\n== USER (under bot) ==")
            print({
                "id": str(user.id),
                "bot_id": str(user.bot_id),
                "external_id": user.external_id,
                "basic_info": user.basic_info,
                "current_stage": user.current_stage,
                "dimensions": user.dimensions,
                "mood_state": getattr(bot, "mood_state", None) or {},
            })

            print("\n== MESSAGES ==")
            for m in msgs:
                print({"role": m.role, "content": m.content, "metadata": m.meta, "created_at": str(m.created_at)})

            print("\n== MEMORIES ==")
            for mm in mems:
                print({"content": mm.content, "created_at": str(mm.created_at)})

    print("\n✅ Done.")


if __name__ == "__main__":
    asyncio.run(main())


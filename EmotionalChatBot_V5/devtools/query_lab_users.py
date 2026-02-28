"""
查询 visit_source=lab 的用户数量与列表（需 RENDER_DATABASE_URL 或 DATABASE_URL）
"""
import os
import sys
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass


async def main() -> None:
    url = os.getenv("RENDER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: 请设置 RENDER_DATABASE_URL 或 DATABASE_URL")
        sys.exit(1)

    from app.core import DBManager, User, _create_async_engine_from_database_url
    from app.core import Bot
    from sqlalchemy import select

    engine = _create_async_engine_from_database_url(url)
    db = DBManager(engine)

    try:
        async with db.Session() as session:
            # assets->>'visit_source' = 'lab'
            q = (
                select(User, Bot.name.label("bot_name"))
                .join(Bot, Bot.id == User.bot_id)
                .where(User.assets["visit_source"].astext == "lab")
            )
            result = await session.execute(q)
            rows = result.all()

        count = len(rows)
        print(f"visit_source=lab 的用户数: {count}")
        if rows:
            print("\n列表 (bot_name, external_id, user_id):")
            for user, bot_name in rows:
                print(f"  {bot_name or '?'} | {user.external_id} | {user.id}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

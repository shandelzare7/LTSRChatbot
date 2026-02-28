"""
根据 user id (users.id) 或 external_id 查询用户信息，含 assets.visit_source
用法: python devtools/query_user_by_id.py <uuid 或 external_id>
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
    if len(sys.argv) < 2:
        print("用法: python devtools/query_user_by_id.py <uuid 或 external_id>")
        sys.exit(1)
    key = sys.argv[1].strip()

    url = os.getenv("RENDER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: 请设置 RENDER_DATABASE_URL 或 DATABASE_URL")
        sys.exit(1)

    from app.core import DBManager, User, Bot, _create_async_engine_from_database_url
    from sqlalchemy import select

    engine = _create_async_engine_from_database_url(url)
    db = DBManager(engine)

    try:
        async with db.Session() as session:
            # 先按 users.id (UUID) 查
            try:
                import uuid
                uid = uuid.UUID(key)
            except ValueError:
                uid = None
            if uid is not None:
                q = select(User, Bot.name.label("bot_name")).join(Bot, Bot.id == User.bot_id).where(User.id == uid)
                r = await session.execute(q)
                row = r.first()
                if row:
                    user, bot_name = row
                    _print_user(user, bot_name)
                    await engine.dispose()
                    return
            # 再按 external_id 查（可能多条，不同 bot）
            q = (
                select(User, Bot.name.label("bot_name"))
                .join(Bot, Bot.id == User.bot_id)
                .where(User.external_id == key)
            )
            result = await session.execute(q)
            rows = result.all()
        if not rows:
            print(f"未找到 id 或 external_id = {key!r} 的用户")
            return
        for user, bot_name in rows:
            _print_user(user, bot_name)
            print("  ---")
    finally:
        await engine.dispose()


def _print_user(user, bot_name: str) -> None:
    print(f"users.id:     {user.id}")
    print(f"bot_id:       {user.bot_id}  (bot_name: {bot_name or '?'})")
    print(f"external_id:  {user.external_id}")
    print(f"assets:       {user.assets}")
    print(f"visit_source: {(user.assets or {}).get('visit_source')}")


if __name__ == "__main__":
    asyncio.run(main())

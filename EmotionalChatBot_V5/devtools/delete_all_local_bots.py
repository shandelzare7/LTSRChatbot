"""
删除本地数据库（DATABASE_URL）中的全部 bot。
仅使用 DATABASE_URL，不会动 RENDER_DATABASE_URL。
User / Message / Memory 等会因 FK ondelete=CASCADE 被级联删除。

用法:
  cd EmotionalChatBot_V5
  DATABASE_URL=postgresql+asyncpg://... python -m devtools.delete_all_local_bots
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

from sqlalchemy import delete, select
from app.core import Bot, DBManager, _create_async_engine_from_database_url


async def main() -> None:
    url = os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: 请设置 DATABASE_URL（仅操作本地库，不会使用 RENDER_DATABASE_URL）")
        sys.exit(1)
    if os.getenv("RENDER_DATABASE_URL") and url == os.getenv("RENDER_DATABASE_URL"):
        print("ERROR: 当前 DATABASE_URL 与 RENDER_DATABASE_URL 相同，拒绝执行，以免误删 Render 数据")
        sys.exit(1)

    engine = _create_async_engine_from_database_url(url)
    db = DBManager(engine)

    async with db.Session() as session:
        count_result = await session.execute(select(Bot))
        bots = list(count_result.scalars().all())
        n = len(bots)
        if n == 0:
            print("本地数据库当前没有 bot，无需删除。")
            await engine.dispose()
            return
        await session.execute(delete(Bot))
        await session.commit()
        print(f"已删除本地数据库中的 {n} 个 bot（其下 user 及消息等已级联删除）。")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

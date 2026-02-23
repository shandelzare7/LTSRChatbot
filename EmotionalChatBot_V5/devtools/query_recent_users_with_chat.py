"""
查询 Render（或 DATABASE_URL）中「近三天有聊天记录」的用户，并列举每个的 user_name。

user_name 优先取 users.basic_info->>'name'，若无则用 external_id。

用法:
  RENDER_DATABASE_URL=postgresql+asyncpg://... python -m devtools.query_recent_users_with_chat
  或: DATABASE_URL=postgresql+asyncpg://... python -m devtools.query_recent_users_with_chat
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from sqlalchemy import select
from app.core.database import (
    DBManager,
    User,
    Message,
    Bot,
    _create_async_engine_from_database_url,
)


async def main() -> None:
    url = os.getenv("RENDER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: 请设置 RENDER_DATABASE_URL 或 DATABASE_URL")
        sys.exit(1)

    source = "RENDER_DATABASE_URL" if os.getenv("RENDER_DATABASE_URL") else "DATABASE_URL"
    print(f"数据来源: {source}")
    print()

    engine = _create_async_engine_from_database_url(url)
    db = DBManager(engine)

    # 近三天：按 UTC 算
    now = datetime.now(timezone.utc)
    three_days_ago = now - timedelta(days=3)

    async with db.Session() as session:
        # 近三天有消息的 user_id（去重）
        subq = (
            select(Message.user_id)
            .where(Message.created_at >= three_days_ago)
            .distinct()
        )
        result = await session.execute(subq)
        user_ids = [row[0] for row in result.all()]

    if not user_ids:
        print("近三天没有任何聊天记录的用户。")
        await engine.dispose()
        return

    async with db.Session() as session:
        # 拉取这些 User + Bot name
        result = await session.execute(
            select(User, Bot.name.label("bot_name"))
            .join(Bot, Bot.id == User.bot_id)
            .where(User.id.in_(user_ids))
        )
        rows = result.all()

    # user_name: basic_info.name 或 external_id
    out: list[tuple[str, str, str]] = []
    for user, bot_name in rows:
        basic = user.basic_info if isinstance(user.basic_info, dict) else {}
        name_from_basic = (basic.get("name") or "").strip()
        user_name = name_from_basic or (user.external_id or str(user.id))
        bot_name_str = (bot_name or "").strip() or "?"
        out.append((bot_name_str, user_name, user.external_id or ""))

    # 按 bot 名、user_name 排序，便于阅读
    out.sort(key=lambda x: (x[0], x[1]))

    print(f"近三天有聊天记录的用户共 {len(out)} 个：\n")
    print("=" * 80)
    for i, (bot_name, user_name, external_id) in enumerate(out, 1):
        print(f"{i}. user_name = {user_name!r}  (bot: {bot_name}, external_id: {external_id})")
    print("=" * 80)
    print("\n仅列举 user_name 如下：")
    for _, user_name, _ in out:
        print(f"  - {user_name}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

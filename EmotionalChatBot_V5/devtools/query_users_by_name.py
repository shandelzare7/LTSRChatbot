"""
查询本地数据库（DATABASE_URL）中指定姓名的 Bot / User 及其偏好（basic_info、inferred_profile）。

用法:
  DATABASE_URL=postgresql+asyncpg://... python -m devtools.query_users_by_name
  或先设置 .env 再运行。
"""
from __future__ import annotations

import asyncio
import json
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

from sqlalchemy import select
from app.core import DBManager, Bot, User


# 要查询的姓名（Bot.name 或 User.basic_info.name）
SEARCH_NAMES = ["李明", "李静怡", "林静怡"]


async def main() -> None:
    url = os.getenv("DATABASE_URL")
    if not url:
        print("未设置 DATABASE_URL，无法连接数据库。")
        print("本地数据目录 local_data 下未发现包含「李明」「李静怡」的 relationship.json。")
        sys.exit(1)

    db = DBManager.from_env()

    async with db.Session() as session:
        # Bot: 按 name 列查
        r = await session.execute(select(Bot).where(Bot.name.in_(SEARCH_NAMES)))
        bots = list(r.scalars().all())

        # User: basic_info->>'name' 在名单中（JSONB 查询）
        r2 = await session.execute(select(User))
        all_users = list(r2.scalars().all())

    # 过滤出 basic_info.name 在 SEARCH_NAMES 中的 user
    users = [u for u in all_users if isinstance(u.basic_info, dict) and (u.basic_info.get("name") or "").strip() in SEARCH_NAMES]

    print("【按姓名查询：李明 / 李静怡 / 林静怡】\n")
    print("=" * 80)

    if bots:
        for b in bots:
            print(f"\n[Bot] id={b.id} name={b.name}")
            print("  basic_info:", json.dumps(b.basic_info or {}, ensure_ascii=False, indent=4))
            print("  （偏好/风格等通常在 basic_info 如 speaking_style、persona 等）")
            print("-" * 80)
    else:
        print("\n未找到名为 李明/李静怡/林静怡 的 Bot。")

    if users:
        for u in users:
            name = (u.basic_info or {}).get("name", "")
            print(f"\n[User] id={u.id} bot_id={u.bot_id} bot_name={u.bot_name} basic_info.name={name}")
            print("  basic_info:", json.dumps(u.basic_info or {}, ensure_ascii=False, indent=4))
            print("  inferred_profile（推断偏好/画像）:", json.dumps(u.inferred_profile or {}, ensure_ascii=False, indent=4))
            print("-" * 80)
    else:
        print("\n未找到 basic_info.name 为 李明/李静怡/林静怡 的 User。")

    if not bots and not users:
        print("\n本地数据库中没有李明、李静怡或林静怡的相关记录。")
        print("若使用 local_data 而非 PostgreSQL，请确认对应 relationship.json 中 bot_basic_info.name / user_basic_info.name 是否为此类姓名。")

    await db.engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

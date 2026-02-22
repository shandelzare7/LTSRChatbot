"""查询 DB 中「最近一次 bot-to-bot 运行」对应的两个 User 的 basic_info / inferred_profile。
   只取 external_id 以 bot_user_ 开头、按 updated_at 降序的前 2 条，即本轮互相对话的两个 user。"""
from __future__ import annotations
import asyncio
import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass
from sqlalchemy import select
from app.core.database import DBManager, User

async def main():
    if not os.getenv("DATABASE_URL"):
        print("未设置 DATABASE_URL")
        return
    db = DBManager.from_env()
    async with db.Session() as session:
        # 只查最近一次 bot-to-bot 的两个 user：按 updated_at 降序取 2 条
        r = await session.execute(
            select(User)
            .where(User.external_id.like("bot_user_%"))
            .order_by(User.updated_at.desc())
            .limit(2)
        )
        users = list(r.scalars().all())
    if not users:
        print("未找到 bot_user_* 的 User（或未按 updated_at 排序）")
        return
    print("【最近一次 bot-to-bot 运行对应的 2 个 User】\n")
    for u in users:
        basic = u.basic_info or {}
        inferred = u.inferred_profile or {}
        print("---")
        print("bot_id:", u.bot_id, "| external_id:", u.external_id)
        print("basic_info:", basic)
        print("inferred_profile keys:", list(inferred.keys()), "| 条数:", len(inferred))
        if inferred:
            for k, v in list(inferred.items())[:5]:
                print("  ", k, ":", (str(v)[:80] + "..." if len(str(v)) > 80 else v))
    print("\n共", len(users), "个（本轮互相对话的 2 个 User）")

if __name__ == "__main__":
    asyncio.run(main())

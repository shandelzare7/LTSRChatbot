"""
删除数据库中「除新 Bot（李阳、林静怡）以外」的所有 Bot 及其关联数据。
因外键为 ON DELETE CASCADE，删除 Bot 会级联删除其 users、messages、memories、bot_tasks、web_chat_logs 等。

使用：
  DATABASE_URL=postgresql+asyncpg://... python -m devtools.delete_old_bots_keep_new

可选环境变量：
  BOT2BOT_KEEP_NAMES  逗号分隔的要保留的 bot 名称，默认 "李阳,林静怡"
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
from app.core.database import Bot, DBManager


# 默认只保留这两个新 Bot，用于 bot-to-bot
DEFAULT_KEEP_NAMES = ["李阳", "林静怡"]


async def main() -> None:
    if not os.getenv("DATABASE_URL"):
        print("ERROR: DATABASE_URL 未设置")
        sys.exit(1)

    keep_names_str = os.getenv("BOT2BOT_KEEP_NAMES", "李阳,林静怡")
    keep_names = [n.strip() for n in keep_names_str.split(",") if n.strip()]
    if not keep_names:
        keep_names = list(DEFAULT_KEEP_NAMES)

    db = DBManager.from_env()
    print("=" * 60)
    print("删除旧 Bot，仅保留新 Bot（用于 bot-to-bot）")
    print("=" * 60)
    print(f"保留名称: {keep_names}")
    print()

    async with db.Session() as session:
        # 列出将要保留的
        result_keep = await session.execute(select(Bot).where(Bot.name.in_(keep_names)))
        to_keep = list(result_keep.scalars().all())
        # 列出将要删除的（所有不在保留列表中的 bot）
        result_del = await session.execute(select(Bot).where(Bot.name.notin_(keep_names)))
        to_delete = list(result_del.scalars().all())

        if not to_delete:
            print("没有需要删除的 Bot，当前库中仅有保留名单内的 Bot 或为空。")
            for b in to_keep:
                print(f"  保留: {b.name} (ID: {b.id})")
            return

        print(f"将删除以下 {len(to_delete)} 个 Bot（及其关联 users/messages/memories/bot_tasks 等）：")
        for b in to_delete:
            print(f"  - {b.name} (ID: {b.id})")
        print()
        print(f"将保留以下 {len(to_keep)} 个 Bot：")
        for b in to_keep:
            print(f"  - {b.name} (ID: {b.id})")
        if not to_keep:
            print("  （无，仅删除旧 Bot）")
        print()

        async with session.begin():
            n = await session.execute(delete(Bot).where(Bot.name.notin_(keep_names)))
            # delete() 返回 Result; rowcount 在 async 里可能需 fetch
            # 实际删除行数 = len(to_delete)，因我们已查过
        print(f"✓ 已删除 {len(to_delete)} 个旧 Bot 及其关联数据。")
    print()
    print("可运行 bot-to-bot 测试：BOT2BOT_NUM_RUNS=1 BOT2BOT_ROUNDS_PER_RUN=5 python -m devtools.bot_to_bot_chat")


if __name__ == "__main__":
    asyncio.run(main())

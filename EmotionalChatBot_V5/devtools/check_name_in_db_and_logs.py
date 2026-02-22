"""
检查：1) DB 里 user 是否写入了名字  2) 聊天记录里是否真的提问了姓名

用法（在 EmotionalChatBot_V5 目录下）：
  python -m devtools.check_name_in_db_and_logs [日志目录]
  不传参数时默认用 ./logs，并会找最新的 bot_to_bot_chat_*.log 搜索「问姓名」相关回复。
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import uuid
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

from app.core.database import Bot, DBManager, User


# 与 bot_to_bot_chat 一致：用于定位两个 Bot 和两个 User
BOT_A_NAMES = ["李阳", "李浩然"]
BOT_B_NAMES = ["林静怡", "苏雨桐"]


async def main() -> None:
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "logs"

    db = DBManager.from_env()

    print("=" * 60)
    print("1. 数据库 users 表：是否写入了名字（basic_info.name）")
    print("=" * 60)

    async with db.Session() as session:
        result_a = await session.execute(select(Bot).where(Bot.name.in_(BOT_A_NAMES)))
        bot_a = result_a.scalars().first()
        result_b = await session.execute(select(Bot).where(Bot.name.in_(BOT_B_NAMES)))
        bot_b = result_b.scalars().first()

        if not bot_a or not bot_b:
            print("未找到 Bot（李阳/李浩然、林静怡/苏雨桐）。请先运行 bot_to_bot_chat 或 create_two_bots_for_render。")
            return

        bot_a_id = str(bot_a.id)
        bot_b_id = str(bot_b.id)
        user_b_external_id = f"bot_user_{bot_b_id}"
        user_a_external_id = f"bot_user_{bot_a_id}"

        # Bot A 下的 User B（Bot B 在 A 这边当用户）
        u_ab = (
            await session.execute(
                select(User).where(
                    User.bot_id == uuid.UUID(bot_a_id),
                    User.external_id == user_b_external_id,
                )
            )
        ).scalars().first()

        # Bot B 下的 User A
        u_ba = (
            await session.execute(
                select(User).where(
                    User.bot_id == uuid.UUID(bot_b_id),
                    User.external_id == user_a_external_id,
                )
            )
        ).scalars().first()

        for label, u in [("Bot A 下的 User B (external_id=bot_user_<B>)", u_ab), ("Bot B 下的 User A (external_id=bot_user_<A>)", u_ba)]:
            print(f"\n{label}:")
            if not u:
                print("  (无此 User 记录)")
                continue
            basic = u.basic_info or {}
            inferred = u.inferred_profile or {}
            name_basic = basic.get("name")
            name_inferred = inferred.get("name")
            print(f"  basic_info:    {basic}")
            print(f"  inferred_profile (前 200 字): {str(inferred)[:200]}")
            print(f"  → basic_info.name: {repr(name_basic)}")
            if name_inferred is not None:
                print(f"  → inferred_profile.name: {repr(name_inferred)}")
            if name_basic or name_inferred:
                print("  => 已写入名字")
            else:
                print("  => 未写入名字")

    print("\n" + "=" * 60)
    print("2. 聊天记录中是否出现「问姓名」的 Bot 回复")
    print("=" * 60)

    if not log_dir.exists():
        print(f"日志目录不存在: {log_dir}")
        return

    log_files = sorted(log_dir.glob("bot_to_bot_chat_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not log_files:
        print(f"未找到 bot_to_bot_chat_*.log（目录: {log_dir}）")
        return

    latest = log_files[0]
    print(f"使用最新日志: {latest.name}\n")

    # 问姓名相关关键词（Bot 回复里出现即视为“有问”）
    name_ask_pattern = re.compile(
        r"名字|姓名|叫什么|你叫|怎么称呼|如何称呼|怎么叫"
    )

    hits = []
    with open(latest, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[Bot Reply]" not in line:
                continue
            # 格式: "  [Bot Reply] 回复内容..."
            content = line.split("[Bot Reply]", 1)[-1].strip()
            if name_ask_pattern.search(content):
                hits.append(content[:200])

    if hits:
        print(f"共发现 {len(hits)} 条 Bot 回复中包含「问姓名」相关表述，示例：")
        for i, h in enumerate(hits[:10], 1):
            print(f"  {i}. {h}...")
    else:
        print("未发现任何 Bot 回复中包含「名字/姓名/叫什么/你叫/怎么称呼」等表述。")
        print("（即：聊天记录里没有明确提问姓名。）")


if __name__ == "__main__":
    asyncio.run(main())

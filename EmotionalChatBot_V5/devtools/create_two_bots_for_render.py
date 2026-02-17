"""
用「新 bot 创建方式」创建两个新 Bot（LLM 人设 + 人物侧写 + 个性任务库 B1–B6），
并写入 DATABASE_URL 指向的数据库；若设置了 RENDER_DATABASE_URL，则把同一份两个 Bot 再写入 Render 库。

使用：
  # 只写入本地/当前 DATABASE_URL
  DATABASE_URL=postgresql+asyncpg://user:pass@localhost/db python devtools/create_two_bots_for_render.py

  # 同时写入 Render（创建后同步到 Render）
  DATABASE_URL=postgresql+asyncpg://... RENDER_DATABASE_URL=postgresql+asyncpg://... python devtools/create_two_bots_for_render.py

  # 直接只写入 Render
  DATABASE_URL=postgres://user:pass@dpg-xxx/render_db python devtools/create_two_bots_for_render.py

Render 库需已执行 init_schema.sql 及 migrate_add_bot_sidewrite_backlog.sql（含 character_sidewrite、backlog_tasks 列）。

获取 Render 数据库连接串：
  1. 打开 https://dashboard.render.com 进入你的 Postgres 服务（如 dpg-xxx）。
  2. 左侧 Connect -> External Database URL（从本机/CI 连接用）或 Internal（仅 Render 内网）。
  3. 若为 postgres:// 开头，改为 postgresql+asyncpg:// 后填入 RENDER_DATABASE_URL。
"""
from __future__ import annotations

import asyncio
import os
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
from app.core.database import Bot, DBManager
from app.core.database import _create_async_engine_from_database_url


async def create_bot_via_llm(llm, bot_name: str, bot_description: str):
    """复用 bot_to_bot_chat 的 LLM 创建逻辑。"""
    from devtools.bot_to_bot_chat import create_bot_via_llm as _create
    def log_line(msg: str):
        print(msg)
    return await _create(llm, bot_name, bot_description, log_line)


async def main() -> None:
    if not os.getenv("DATABASE_URL"):
        print("ERROR: DATABASE_URL 未设置。请在 .env 中配置或在命令行传入。")
        print("  Render 库连接串可在 Dashboard -> 你的 Postgres -> Connect -> External Database URL 复制。")
        print("  若为 postgres:// 开头，请改为 postgresql+asyncpg:// 后使用。")
        sys.exit(1)

    from app.services.llm import get_llm
    from app.core.bot_creation_llm import generate_sidewrite_and_backlog

    print("=" * 60)
    print("使用新方式创建两个 Bot（人设 + 人物侧写 + 个性任务库）")
    print("=" * 60)

    llm = get_llm()
    print(f"LLM: {getattr(llm, 'model_name', 'unknown')}\n")

    # 1) LLM 生成两个 Bot 的人设 + 侧写 + 任务库
    bot_a_id = str(uuid.uuid4())
    bot_b_id = str(uuid.uuid4())

    print("创建 Bot 1（男，全名，非程序员）...")
    b1_basic, b1_big_five, b1_persona = await create_bot_via_llm(
        llm,
        "Bot 1",
        "请为人设起一个中文全名（姓+名，如陈明轩、林浩然），男性。性格开朗、喜欢交流，对新鲜事物充满好奇。职业不要程序员，请从以下任选其一：产品经理、设计师、教师、插画师、自由撰稿人。",
    )
    b1_sidewrite, b1_backlog = None, None
    try:
        print("  生成人物侧写与个性任务库…")
        b1_sidewrite, b1_backlog = await generate_sidewrite_and_backlog(llm, b1_basic, b1_big_five, b1_persona)
        if b1_backlog:
            print(f"  ✓ 个性任务库 {len(b1_backlog)} 条")
    except Exception as e:
        print(f"  ⚠ 侧写/任务库生成失败: {e}")

    print("\n创建 Bot 2（女，全名，非程序员）...")
    b2_basic, b2_big_five, b2_persona = await create_bot_via_llm(
        llm,
        "Bot 2",
        "请为人设起一个中文全名（姓+名，如苏雨桐、周思琪），女性。性格温和、善于倾听，喜欢深入思考。职业不要程序员，请从以下任选其一：编辑、运营、心理咨询师、策展人、摄影师。",
    )
    b2_sidewrite, b2_backlog = None, None
    try:
        print("  生成人物侧写与个性任务库…")
        b2_sidewrite, b2_backlog = await generate_sidewrite_and_backlog(llm, b2_basic, b2_big_five, b2_persona)
        if b2_backlog:
            print(f"  ✓ 个性任务库 {len(b2_backlog)} 条")
    except Exception as e:
        print(f"  ⚠ 侧写/任务库生成失败: {e}")

    # 2) 写入 DATABASE_URL（主库）
    print("\n" + "=" * 60)
    print("写入主库 (DATABASE_URL)")
    print("=" * 60)
    db = DBManager.from_env()
    async with db.Session() as session:
        async with session.begin():
            bot1 = Bot(
                id=uuid.UUID(bot_a_id),
                name=str(b1_basic.get("name") or "Bot 1"),
                basic_info=b1_basic,
                big_five=b1_big_five,
                persona=b1_persona,
                character_sidewrite=b1_sidewrite,
                backlog_tasks=b1_backlog,
            )
            bot2 = Bot(
                id=uuid.UUID(bot_b_id),
                name=str(b2_basic.get("name") or "Bot 2"),
                basic_info=b2_basic,
                big_five=b2_big_five,
                persona=b2_persona,
                character_sidewrite=b2_sidewrite,
                backlog_tasks=b2_backlog,
            )
            session.add(bot1)
            session.add(bot2)
            await session.flush()
        print(f"  ✓ Bot 1: {bot1.name} (ID: {bot_a_id[:8]}...)")
        print(f"  ✓ Bot 2: {bot2.name} (ID: {bot_b_id[:8]}...)")

    # 3) 若设置了 RENDER_DATABASE_URL，同步同一份两个 Bot 到 Render
    render_url = os.getenv("RENDER_DATABASE_URL", "").strip()
    if render_url:
        print("\n" + "=" * 60)
        print("同步到 Render 库 (RENDER_DATABASE_URL)")
        print("=" * 60)
        try:
            engine_render = _create_async_engine_from_database_url(render_url)
            db_render = DBManager(engine_render)
            r1_name, r2_name = "Bot 1", "Bot 2"
            async with db_render.Session() as session:
                async with session.begin():
                    r1 = Bot(
                        id=uuid.UUID(bot_a_id),
                        name=str(b1_basic.get("name") or "Bot 1"),
                        basic_info=b1_basic,
                        big_five=b1_big_five,
                        persona=b1_persona,
                        character_sidewrite=b1_sidewrite,
                        backlog_tasks=b1_backlog,
                    )
                    r2 = Bot(
                        id=uuid.UUID(bot_b_id),
                        name=str(b2_basic.get("name") or "Bot 2"),
                        basic_info=b2_basic,
                        big_five=b2_big_five,
                        persona=b2_persona,
                        character_sidewrite=b2_sidewrite,
                        backlog_tasks=b2_backlog,
                    )
                    r1_name, r2_name = r1.name, r2.name
                    session.add(r1)
                    session.add(r2)
                    await session.flush()
            print(f"  ✓ Render 已写入 Bot 1: {r1_name}, Bot 2: {r2_name}")
            await engine_render.dispose()
        except Exception as e:
            print(f"  ⚠ 写入 Render 失败: {e}")
            print("  请确认 Render 库已执行 init_schema.sql 与 migrate_add_bot_sidewrite_backlog.sql")
    else:
        print("\n提示: 设置 RENDER_DATABASE_URL 可将同一份两个 Bot 同步到 Render。")

    print("\n✅ 完成。")


if __name__ == "__main__":
    asyncio.run(main())

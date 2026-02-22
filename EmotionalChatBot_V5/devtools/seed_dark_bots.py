"""
用「暗黑性格」预设数据创建两个 Bot 并写入数据库。
数据来自同目录下的 dark_bots_seed_data.json（两套 basic_info + big_five + persona）。

使用：
  # 写入 DATABASE_URL 指向的库
  DATABASE_URL=postgresql+asyncpg://user:pass@localhost/db python devtools/seed_dark_bots.py

  # 同时写入 Render
  DATABASE_URL=postgresql+asyncpg://... RENDER_DATABASE_URL=postgresql+asyncpg://... python devtools/seed_dark_bots.py

  # 仅写入 Render
  DATABASE_URL=postgresql+asyncpg://... python devtools/seed_dark_bots.py

依赖：库已执行 init_schema.sql；若有 character_sidewrite/backlog_tasks 列则需对应迁移。
"""
from __future__ import annotations

import asyncio
import json
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
from app.core.database import Bot, DBManager, _PADB_DEFAULT
from app.core.database import _create_async_engine_from_database_url


def load_dark_bots_data() -> list[dict]:
    """从 devtools/dark_bots_seed_data.json 加载两套 bot 数据。"""
    path = Path(__file__).resolve().parent / "dark_bots_seed_data.json"
    if not path.is_file():
        raise FileNotFoundError(f"未找到 {path}，请确保 dark_bots_seed_data.json 与本脚本同目录。")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bots = data.get("bots")
    if not isinstance(bots, list) or len(bots) < 2:
        raise ValueError("dark_bots_seed_data.json 中需包含至少 2 个 bot 对象（bots 数组）。")
    return bots


async def main() -> None:
    if not os.getenv("DATABASE_URL"):
        print("ERROR: DATABASE_URL 未设置。请在 .env 中配置或在命令行传入。")
        print("  若为 postgres:// 开头，请改为 postgresql+asyncpg:// 后使用。")
        sys.exit(1)

    try:
        raw = load_dark_bots_data()
    except Exception as e:
        print(f"ERROR: 加载暗黑 Bot 数据失败: {e}")
        sys.exit(1)

    # 取前两个，并补齐结构
    bot_a_raw = raw[0]
    bot_b_raw = raw[1]
    bot_a_id = str(uuid.uuid4())
    bot_b_id = str(uuid.uuid4())

    def _ensure_bot(b: dict) -> tuple[dict, dict, dict, str | None, list | None]:
        basic = dict(b.get("basic_info") or {})
        big_five = dict(b.get("big_five") or {})
        persona = dict(b.get("persona") or {})
        if "attributes" not in persona:
            persona["attributes"] = {}
        if "collections" not in persona:
            persona["collections"] = {}
        if "lore" not in persona:
            persona["lore"] = {}
        sidewrite = b.get("character_sidewrite")
        backlog = b.get("backlog_tasks") if isinstance(b.get("backlog_tasks"), list) else None
        return basic, big_five, persona, sidewrite, backlog

    b1_basic, b1_big_five, b1_persona, b1_sidewrite, b1_backlog = _ensure_bot(bot_a_raw)
    b2_basic, b2_big_five, b2_persona, b2_sidewrite, b2_backlog = _ensure_bot(bot_b_raw)

    name_a = str(b1_basic.get("name") or "暗黑A")
    name_b = str(b2_basic.get("name") or "暗黑B")

    print("=" * 60)
    print("写入暗黑性格 Bot（来自 dark_bots_seed_data.json）")
    print("=" * 60)
    print(f"  Bot 1: {name_a}")
    print(f"  Bot 2: {name_b}\n")

    db = DBManager.from_env()
    async with db.Session() as session:
        async with session.begin():
            bot1 = Bot(
                id=uuid.UUID(bot_a_id),
                name=name_a,
                basic_info=b1_basic,
                big_five=b1_big_five,
                persona=b1_persona,
                character_sidewrite=b1_sidewrite,
                backlog_tasks=b1_backlog,
                mood_state=_PADB_DEFAULT(),
            )
            bot2 = Bot(
                id=uuid.UUID(bot_b_id),
                name=name_b,
                basic_info=b2_basic,
                big_five=b2_big_five,
                persona=b2_persona,
                character_sidewrite=b2_sidewrite,
                backlog_tasks=b2_backlog,
                mood_state=_PADB_DEFAULT(),
            )
            session.add(bot1)
            session.add(bot2)
            await session.flush()
        print(f"  ✓ Bot 1: {name_a} (ID: {bot_a_id[:8]}...)")
        print(f"  ✓ Bot 2: {name_b} (ID: {bot_b_id[:8]}...)")

    render_url = os.getenv("RENDER_DATABASE_URL", "").strip()
    if render_url:
        print("\n" + "=" * 60)
        print("同步到 Render 库 (RENDER_DATABASE_URL)")
        print("=" * 60)
        try:
            engine_render = _create_async_engine_from_database_url(render_url)
            db_render = DBManager(engine_render)
            async with db_render.Session() as session:
                async with session.begin():
                    r1 = Bot(
                        id=uuid.UUID(bot_a_id),
                        name=name_a,
                        basic_info=b1_basic,
                        big_five=b1_big_five,
                        persona=b1_persona,
                        character_sidewrite=b1_sidewrite,
                        backlog_tasks=b1_backlog,
                        mood_state=_PADB_DEFAULT(),
                    )
                    r2 = Bot(
                        id=uuid.UUID(bot_b_id),
                        name=name_b,
                        basic_info=b2_basic,
                        big_five=b2_big_five,
                        persona=b2_persona,
                        character_sidewrite=b2_sidewrite,
                        backlog_tasks=b2_backlog,
                        mood_state=_PADB_DEFAULT(),
                    )
                    session.add(r1)
                    session.add(r2)
                    await session.flush()
            print(f"  ✓ Render 已写入: {name_a}, {name_b}")
            await engine_render.dispose()
        except Exception as e:
            print(f"  ⚠ 写入 Render 失败: {e}")
    else:
        print("\n提示: 设置 RENDER_DATABASE_URL 可将同一份两个 Bot 同步到 Render。")

    print("\n✅ 完成。")


if __name__ == "__main__":
    asyncio.run(main())

"""
将 Render 数据库（RENDER_DATABASE_URL）的全部数据【只读】复制到本地（DATABASE_URL）。
- 仅从 Render 读取，不会删除或修改 Render 上任何数据。
- 会先清空本地库相关表，再按依赖顺序把 Render 数据插入本地。

用法:
  cd EmotionalChatBot_V5
  DATABASE_URL=postgresql+asyncpg://... RENDER_DATABASE_URL=postgresql+asyncpg://... python -m devtools.copy_render_to_local
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
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import (
    _create_async_engine_from_database_url,
    Bot,
    User,
    Message,
    Memory,
    Transcript,
    DerivedNote,
    BotTask,
    WebChatLog,
)


def _row_to_dict(row, model_class):
    """把 ORM 行转成可用于新建实例的 dict，保留主键。"""
    return {c.name: getattr(row, c.name) for c in model_class.__table__.columns}


async def main() -> None:
    render_url = os.getenv("RENDER_DATABASE_URL")
    local_url = os.getenv("DATABASE_URL")
    if not render_url or not local_url:
        print("ERROR: 请同时设置 RENDER_DATABASE_URL 和 DATABASE_URL")
        sys.exit(1)
    if render_url.strip() == local_url.strip():
        print("ERROR: RENDER_DATABASE_URL 与 DATABASE_URL 不能相同")
        sys.exit(1)

    engine_render = _create_async_engine_from_database_url(render_url)
    engine_local = _create_async_engine_from_database_url(local_url)
    SessionRender = sessionmaker(engine_render, class_=AsyncSession, expire_on_commit=False)
    SessionLocal = sessionmaker(engine_local, class_=AsyncSession, expire_on_commit=False)

    # 1) 仅从 Render 读取（不写、不删 Render）
    print("从 Render 只读拉取数据...")
    async with SessionRender() as sr:
        bots = (await sr.execute(select(Bot))).scalars().all()
        users = (await sr.execute(select(User))).scalars().all()
        messages = (await sr.execute(select(Message))).scalars().all()
        memories = (await sr.execute(select(Memory))).scalars().all()
        transcripts = (await sr.execute(select(Transcript))).scalars().all()
        derived_notes = (await sr.execute(select(DerivedNote))).scalars().all()
        bot_tasks = (await sr.execute(select(BotTask))).scalars().all()
        web_chat_logs = (await sr.execute(select(WebChatLog))).scalars().all()

    print(f"  bots={len(bots)}, users={len(users)}, messages={len(messages)}, memories={len(memories)}, transcripts={len(transcripts)}, derived_notes={len(derived_notes)}, bot_tasks={len(bot_tasks)}, web_chat_logs={len(web_chat_logs)}")

    # 2) 清空本地（仅动本地库）
    print("清空本地表...")
    async with SessionLocal() as sl:
        await sl.execute(delete(DerivedNote))
        await sl.execute(delete(WebChatLog))
        await sl.execute(delete(BotTask))
        await sl.execute(delete(Message))
        await sl.execute(delete(Memory))
        await sl.execute(delete(Transcript))
        await sl.execute(delete(User))
        await sl.execute(delete(Bot))
        await sl.commit()

    # 3) 写入本地
    print("写入本地...")
    async with SessionLocal() as sl:
        for r in bots:
            sl.add(Bot(**_row_to_dict(r, Bot)))
        await sl.flush()
        for r in users:
            sl.add(User(**_row_to_dict(r, User)))
        await sl.flush()
        for r in messages:
            sl.add(Message(**_row_to_dict(r, Message)))
        for r in memories:
            sl.add(Memory(**_row_to_dict(r, Memory)))
        for r in transcripts:
            sl.add(Transcript(**_row_to_dict(r, Transcript)))
        await sl.flush()
        for r in derived_notes:
            sl.add(DerivedNote(**_row_to_dict(r, DerivedNote)))
        for r in bot_tasks:
            sl.add(BotTask(**_row_to_dict(r, BotTask)))
        for r in web_chat_logs:
            sl.add(WebChatLog(**_row_to_dict(r, WebChatLog)))
        await sl.commit()

    await engine_render.dispose()
    await engine_local.dispose()
    print("完成：Render 数据已复制到本地（Render 未被修改）。")


if __name__ == "__main__":
    asyncio.run(main())

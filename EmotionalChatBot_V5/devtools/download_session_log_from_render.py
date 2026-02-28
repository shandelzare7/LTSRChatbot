"""
从 Render 数据库（RENDER_DATABASE_URL）按 user_id 下载该用户最新的 web_chat_log 到本地文件。
传入的为 User 表主键 id（UUID），会取该 user 下 updated_at 最新的一条 log 内容。

用法:
  cd EmotionalChatBot_V5
  RENDER_DATABASE_URL=... python -m devtools.download_session_log_from_render 89b1f712-7c8f-4f09-ab1d-65dc17f32f4c
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid as uuid_lib
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
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import WebChatLog, _create_async_engine_from_database_url


async def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python -m devtools.download_session_log_from_render <user_id>")
        sys.exit(1)
    raw = sys.argv[1].strip()
    if not raw:
        print("请提供 user_id (UUID)")
        sys.exit(1)
    try:
        user_uuid = uuid_lib.UUID(raw)
    except ValueError:
        print("user_id 不是合法 UUID")
        sys.exit(1)

    url = os.getenv("RENDER_DATABASE_URL")
    if not url:
        print("ERROR: 请设置 RENDER_DATABASE_URL")
        sys.exit(1)

    engine = _create_async_engine_from_database_url(url)
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        r = await session.execute(
            select(WebChatLog)
            .where(WebChatLog.user_id == user_uuid)
            .order_by(WebChatLog.updated_at.desc())
        )
        rows = list(r.scalars().all())
    await engine.dispose()

    if not rows:
        print(f"未找到 user_id={raw} 的 web_chat_log 记录")
        sys.exit(1)
    row = rows[0]
    content = (row.content or "").strip()
    out_dir = PROJECT_ROOT / "devtools" / "downloaded_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = raw.replace(" ", "_")
    out_path = out_dir / f"user_{safe_id}.log"
    out_path.write_text(content, encoding="utf-8")
    print(f"已保存到: {out_path}")
    print(f"session_id: {row.session_id}, updated_at: {row.updated_at}")
    print(f"内容长度: {len(content)} 字符")


if __name__ == "__main__":
    asyncio.run(main())

"""
从 Render 数据库（RENDER_DATABASE_URL）下载全部 web_chat_logs，按 user 分到 data_summary/<user_id>/ 下，
每个 user 一个子文件夹，内含该 user 的所有历史 log 文件。

用法:
  cd EmotionalChatBot_V5
  python -m devtools.download_all_render_logs_by_user
"""
from __future__ import annotations

import asyncio
import os
import re
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
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import WebChatLog, _create_async_engine_from_database_url


def safe_filename(s: str, max_len: int = 120) -> str:
    """去掉不宜做文件名的字符，截断长度。"""
    s = re.sub(r'[<>:"/\\|?*\s]+', "_", str(s).strip())
    s = s.strip("._") or "unnamed"
    return s[:max_len]


async def main() -> None:
    url = os.getenv("RENDER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: 请设置 RENDER_DATABASE_URL 或 DATABASE_URL")
        sys.exit(1)

    out_base = PROJECT_ROOT / "data_summary"
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_base}")
    print("使用:", "RENDER_DATABASE_URL" if os.getenv("RENDER_DATABASE_URL") else "DATABASE_URL")

    engine = _create_async_engine_from_database_url(url)
    from sqlalchemy.ext.asyncio import async_sessionmaker
    async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(WebChatLog)
                .order_by(WebChatLog.user_id.asc(), WebChatLog.updated_at.asc())
            )
            rows = list(result.scalars().all())

        if not rows:
            print("没有 web_chat_log 记录")
            return

        # 按 user_id 分组写入
        per_user: dict[str, list] = {}
        for row in rows:
            uid = str(row.user_id)
            per_user.setdefault(uid, []).append(row)

        total_files = 0
        for user_id, user_rows in per_user.items():
            user_dir = out_base / f"user_{user_id}"
            user_dir.mkdir(parents=True, exist_ok=True)
            for i, row in enumerate(user_rows):
                # 文件名：session 或 updated 时间 + session，避免重名
                sid_safe = safe_filename(row.session_id or "session", 80)
                ts = (row.updated_at or row.created_at)
                ts_str = ts.strftime("%Y%m%d_%H%M%S") if ts else str(i)
                fname = f"{ts_str}_{sid_safe}.log"
                if not fname.endswith(".log"):
                    fname += ".log"
                path = user_dir / fname
                path.write_text(row.content or "", encoding="utf-8")
                total_files += 1
            print(f"  user_{user_id}: {len(user_rows)} 个 log")

        print(f"\n共 {len(per_user)} 个 user，{total_files} 个 log 文件 → {out_base}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

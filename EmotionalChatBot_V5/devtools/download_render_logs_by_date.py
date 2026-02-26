"""
从 Render 数据库（RENDER_DATABASE_URL）按日期下载所有 web_chat_logs 到本地。
分别下载 2 月 24 日、2 月 25 日的所有 log，保存到 devtools/downloaded_logs/render_YYYY-MM-DD/。

用法:
  cd EmotionalChatBot_V5
  # 使用 .env 中的 RENDER_DATABASE_URL
  python -m devtools.download_render_logs_by_date

  或指定日期（可选，默认 2026-02-24 和 2026-02-25）:
  python -m devtools.download_render_logs_by_date 2026-02-24 2026-02-25
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
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import WebChatLog, _create_async_engine_from_database_url


def parse_date(s: str):
    """Parse YYYY-MM-DD to date."""
    return datetime.strptime(s.strip()[:10], "%Y-%m-%d").date()


def day_range_utc(d: datetime.date):
    """Return (start, end) as timezone-aware UTC datetimes for that day."""
    start = datetime(d.year, d.month, d.day, 0, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


async def main() -> None:
    dates_str = ["2026-02-24", "2026-02-25"]
    if len(sys.argv) >= 2:
        dates_str = [sys.argv[1]]
        if len(sys.argv) >= 3:
            dates_str = [sys.argv[1], sys.argv[2]]

    url = os.getenv("RENDER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: 请设置 RENDER_DATABASE_URL 或 DATABASE_URL（可从 .env 读取）")
        sys.exit(1)

    print(f"使用: {'RENDER_DATABASE_URL' if os.getenv('RENDER_DATABASE_URL') else 'DATABASE_URL'}")
    print(f"日期: {dates_str}\n")

    engine = _create_async_engine_from_database_url(url)
    out_base = PROJECT_ROOT / "devtools" / "downloaded_logs"

    from sqlalchemy.ext.asyncio import async_sessionmaker
    async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        for date_str in dates_str:
            try:
                d = parse_date(date_str)
            except ValueError:
                print(f"跳过无效日期: {date_str}")
                continue
            start_utc, end_utc = day_range_utc(d)
            out_dir = out_base / f"render_{d.isoformat()}"
            out_dir.mkdir(parents=True, exist_ok=True)

            async with async_session_factory() as session:
                result = await session.execute(
                    select(WebChatLog)
                    .where(WebChatLog.updated_at >= start_utc, WebChatLog.updated_at < end_utc)
                    .order_by(WebChatLog.updated_at.asc())
                )
                rows = list(result.scalars().all())

            if not rows:
                print(f"{d.isoformat()}: 0 条 log，跳过")
                continue

            for i, row in enumerate(rows):
                safe_sid = (row.session_id or "unknown").replace("/", "_").replace("\\", "_")[:64]
                fname = f"user_{row.user_id}_{safe_sid}.log"
                if row.filename and row.filename.strip():
                    base = row.filename.strip()
                    fname = base if base.endswith(".log") else base + ".log"
                path = out_dir / fname
                if path.exists():
                    path = out_dir / f"user_{row.user_id}_{safe_sid}_{i}.log"
                path.write_text(row.content or "", encoding="utf-8")

            print(f"{d.isoformat()}: {len(rows)} 条 → {out_dir}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

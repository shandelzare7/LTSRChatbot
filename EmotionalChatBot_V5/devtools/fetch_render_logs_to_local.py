"""
从「数据库」表 web_chat_logs 拉取所有记录，保存到本地 render_log 文件夹。
（Render 上你已要求把 log 都存在数据库里，本脚本从该表读出并落盘到本地。）

使用：
  # 必须用 Render 的库连接串，否则会连到本地库（可能没有记录）
  RENDER_DATABASE_URL=postgresql+asyncpg://user:pass@dpg-xxx/render_db python -m devtools.fetch_render_logs_to_local

  若未设置 RENDER_DATABASE_URL 则用 DATABASE_URL（可能是本地库）。
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore[misc, assignment]

# 文件名时间戳使用北京时间（Asia/Shanghai）
BEIJING_TZ = ZoneInfo("Asia/Shanghai") if ZoneInfo else None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from sqlalchemy import select
from app.core.database import WebChatLog, DBManager
from app.core.database import _create_async_engine_from_database_url


def _safe_filename(name: str, max_len: int = 200) -> str:
    """保留扩展名，去掉非法字符，截断过长。"""
    name = name.strip() or "log"
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    if len(name) > max_len:
        base, ext = os.path.splitext(name)
        name = base[: max_len - len(ext)] + ext
    return name or "log"


async def main() -> None:
    # 优先用 Render 数据库连接串（你要的是「Render 上存的 log」）
    url = os.getenv("RENDER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: 请设置 RENDER_DATABASE_URL（Render 库）或 DATABASE_URL")
        sys.exit(1)

    try:
        parsed = urlparse(url.replace("postgresql+asyncpg://", "postgres://"))
        host = parsed.hostname or "(unknown)"
    except Exception:
        host = "(unknown)"
    source = "RENDER_DATABASE_URL" if os.getenv("RENDER_DATABASE_URL") else "DATABASE_URL"
    print(f"数据来源: 数据库表 web_chat_logs（连接: {source}, host: {host}）")
    print()

    engine = _create_async_engine_from_database_url(url)
    db = DBManager(engine)

    out_dir = PROJECT_ROOT / "render_log"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir.resolve()}")
    print()

    async with db.Session() as session:
        result = await session.execute(
            select(WebChatLog).order_by(WebChatLog.updated_at.desc())
        )
        rows = list(result.scalars().all())

    if not rows:
        print("当前连接的库中 web_chat_logs 表暂无记录。")
        if source != "RENDER_DATABASE_URL":
            print("提示: 若要从 Render 数据库拉取，请设置 RENDER_DATABASE_URL 后再运行。")
        return

    print(f"共 {len(rows)} 条日志，开始写入…")
    written = 0
    for i, row in enumerate(rows):
        try:
            ts = row.updated_at or row.created_at
            if ts:
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if BEIJING_TZ:
                    ts = ts.astimezone(BEIJING_TZ)
                ts_str = ts.strftime("%Y%m%d_%H%M%S")
            else:
                ts_str = str(i)
            # 统一用「北京时间」的 session_id + ts_str 作为文件名，不再用 DB 里的 filename（避免时区是美区等问题）
            base = f"{row.session_id}_{ts_str}"
            fname = base + ".log"
            # 避免重名：若已存在则加 id 前缀
            path = out_dir / fname
            if path.exists():
                path = out_dir / f"{row.id}_{fname}"
            path.write_text(row.content or "", encoding="utf-8")
            written += 1
            print(f"  [{written}/{len(rows)}] {path.name}")
        except Exception as e:
            print(f"  ⚠ 写入失败 id={row.id}: {e}")

    print()
    print(f"✅ 已保存 {written} 个文件到 {out_dir.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())

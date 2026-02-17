"""
从 Render 数据库拉取「所有 bot、所有 user」的备份并保存到本地。

包含：
  1. web_chat_logs 表：全部会话 log 快照 → render_backup/web_chat_logs/
  2. 每个 user 的 users.assets（含 log_backup 等）→ render_backup/users/{bot_name}_{external_id}.json

使用：
  RENDER_DATABASE_URL=postgresql+asyncpg://... python -m devtools.download_all_render_backups
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

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
from sqlalchemy.orm import selectinload

from app.core.database import (
    DBManager,
    Bot,
    User,
    WebChatLog,
    _create_async_engine_from_database_url,
)


def _safe_dirname(s: str, max_len: int = 80) -> str:
    s = (s or "unknown").strip()
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s)
    return s[:max_len] if len(s) > max_len else s or "unknown"


async def main() -> None:
    url = os.getenv("RENDER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: 请设置 RENDER_DATABASE_URL 或 DATABASE_URL")
        sys.exit(1)

    try:
        parsed = urlparse(url.replace("postgresql+asyncpg://", "postgres://"))
        host = parsed.hostname or "(unknown)"
    except Exception:
        host = "(unknown)"
    source = "RENDER_DATABASE_URL" if os.getenv("RENDER_DATABASE_URL") else "DATABASE_URL"
    print(f"数据来源: {source} (host: {host})")
    print()

    base_out = PROJECT_ROOT / "render_backup"
    base_out.mkdir(parents=True, exist_ok=True)
    dir_logs = base_out / "web_chat_logs"
    dir_users = base_out / "users"
    dir_logs.mkdir(parents=True, exist_ok=True)
    dir_users.mkdir(parents=True, exist_ok=True)
    print(f"输出根目录: {base_out.resolve()}")
    print()

    engine = _create_async_engine_from_database_url(url)
    db = DBManager(engine)

    # ---------- 1) 全部 web_chat_logs ----------
    async with db.Session() as session:
        r = await session.execute(select(WebChatLog).order_by(WebChatLog.updated_at.desc()))
        log_rows = list(r.scalars().all())

    print(f"[1/2] web_chat_logs: 共 {len(log_rows)} 条")
    written_logs = 0
    for i, row in enumerate(log_rows):
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
            base = f"{row.session_id}_{ts_str}"
            fname = base + ".log"
            path = dir_logs / fname
            if path.exists():
                path = dir_logs / f"{row.id}_{fname}"
            path.write_text(row.content or "", encoding="utf-8")
            written_logs += 1
            print(f"  [{written_logs}/{len(log_rows)}] {path.name}")
        except Exception as e:
            print(f"  ⚠ 写入失败 id={row.id}: {e}")
    print(f"  → 已保存到 {dir_logs.resolve()}")
    print()

    # ---------- 2) 全部 user 的 assets（按 bot + user） ----------
    async with db.Session() as session:
        r = await session.execute(select(User).options(selectinload(User.bot)).order_by(User.bot_id, User.external_id))
        users = list(r.scalars().unique().all())

    print(f"[2/2] users (assets): 共 {len(users)} 个")
    written_users = 0
    for u in users:
        try:
            bot_name = (u.bot.name if u.bot else "") or str(u.bot_id)[:8]
            safe_bot = _safe_dirname(bot_name)
            safe_ext = _safe_dirname(u.external_id)
            fname = f"{safe_bot}_{safe_ext}_{u.id}.json"
            path = dir_users / fname
            payload = {
                "user_id": str(u.id),
                "bot_id": str(u.bot_id),
                "bot_name": bot_name,
                "external_id": u.external_id,
                "updated_at": (u.updated_at.isoformat() if u.updated_at else None),
                "assets": u.assets if isinstance(u.assets, dict) else {},
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            written_users += 1
            has_backup = bool((payload.get("assets") or {}).get("log_backup"))
            print(f"  [{written_users}/{len(users)}] {path.name}" + (" (含 log_backup)" if has_backup else ""))
        except Exception as e:
            print(f"  ⚠ 写入失败 user_id={u.id}: {e}")
    print(f"  → 已保存到 {dir_users.resolve()}")
    print()

    print("✅ 全部备份已下载完成。")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

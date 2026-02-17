"""
本地冒烟测试：验证
  1) web_chat_logs: upsert_web_chat_log 写入后可读回（完整内容不丢）
  2) users.assets["log_backup"]: append_user_log_backup 写入后，save_turn 不会覆盖丢失

默认连接：
  - 优先 RENDER_DATABASE_URL（若你想测 Render 上的库）
  - 否则 DATABASE_URL（本地库）

运行：
  python -m devtools.test_log_db_roundtrip
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone

from sqlalchemy import select, delete

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from app.core.database import DBManager, WebChatLog, User, Bot, _create_async_engine_from_database_url


def _pick_url() -> tuple[str, str]:
    url = (os.getenv("RENDER_DATABASE_URL") or "").strip()
    if url:
        return url, "RENDER_DATABASE_URL"
    url = (os.getenv("DATABASE_URL") or "").strip()
    if url:
        return url, "DATABASE_URL"
    raise RuntimeError("请设置 RENDER_DATABASE_URL 或 DATABASE_URL")


async def main() -> int:
    url, source = _pick_url()
    print(f"[TEST] using {source}")

    engine = _create_async_engine_from_database_url(url)
    db = DBManager(engine)

    # 使用一次性 bot/user，测试完删除 bot（级联清理 user/messages/web_chat_logs）
    bot_uuid = uuid.uuid4()
    user_external_id = f"local_test_user_{uuid.uuid4().hex[:10]}"
    session_id = f"local_test_session_{uuid.uuid4().hex[:10]}"

    marker = f"MARKER::{uuid.uuid4().hex}"
    content = "\n".join(
        [
            "=== TEST WEB CHAT LOG SNAPSHOT ===",
            f"bot={bot_uuid}",
            f"user={user_external_id}",
            f"session={session_id}",
            marker,
            "line1: hello",
            "line2: world",
        ]
    )

    try:
        # 1) upsert web_chat_logs
        await db.upsert_web_chat_log(
            user_external_id=user_external_id,
            bot_id=str(bot_uuid),
            session_id=session_id,
            filename="web_chat_test_roundtrip.log",
            content=content,
        )
        print("[OK] upsert_web_chat_log wrote")

        # 2) append log_backup entry
        payload = {
            "session_id": session_id,
            "user_input": "hi",
            "bot_reply": "hello",
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        await db.append_user_log_backup(
            user_external_id=user_external_id,
            bot_id=str(bot_uuid),
            session_id=session_id,
            kind="web_chat_turn",
            payload=payload,
        )
        print("[OK] append_user_log_backup wrote")

        # 3) call save_turn with *stale* relationship_assets (does NOT include log_backup)
        state = {
            "user_input": "hi",
            "final_response": "hello",
            "current_stage": "initiating",
            "relationship_assets": {"topic_history": [], "breadth_score": 0, "max_spt_depth": 1},
            "user_received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ai_sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        await db.save_turn(user_external_id, str(bot_uuid), state)
        print("[OK] save_turn wrote (should preserve log_backup)")

        # 4) read back and validate
        async with db.Session() as s:
            # web_chat_logs
            res = await s.execute(
                select(WebChatLog).where(WebChatLog.session_id == session_id).order_by(WebChatLog.updated_at.desc())
            )
            row = res.scalars().first()
            if not row:
                raise AssertionError("web_chat_logs missing row after upsert")
            if marker not in (row.content or ""):
                raise AssertionError("web_chat_logs content missing marker (content overwritten/truncated?)")
            print("[OK] web_chat_logs roundtrip read OK")

            # user assets log_backup
            ures = await s.execute(select(User).where(User.bot_id == bot_uuid, User.external_id == user_external_id))
            u = ures.scalars().first()
            if not u:
                raise AssertionError("users missing test user")
            assets = u.assets if isinstance(u.assets, dict) else {}
            lb = (assets.get("log_backup") or {}) if isinstance(assets, dict) else {}
            sessions = (lb.get("sessions") or {}) if isinstance(lb, dict) else {}
            srec = sessions.get(session_id) if isinstance(sessions, dict) else None
            entries = (srec.get("entries") or []) if isinstance(srec, dict) else []
            if not entries:
                raise AssertionError("log_backup entries missing after save_turn (likely overwritten)")
            print(f"[OK] log_backup preserved ({len(entries)} entries)")

        # 5) download from DB to local file for human inspection
        out_dir = PROJECT_ROOT / "render_logs_download"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"roundtrip_{session_id}.log"
        out_path.write_text(row.content or "", encoding="utf-8")
        print(f"[OK] wrote local copy: {out_path.resolve()}")

        return 0
    finally:
        # Cleanup: delete bot (cascades to users/web_chat_logs/messages/etc.)
        try:
            async with db.Session() as s:
                async with s.begin():
                    await s.execute(delete(Bot).where(Bot.id == bot_uuid))
        except Exception as e:
            print(f"[WARN] cleanup failed: {e}")
        await engine.dispose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))


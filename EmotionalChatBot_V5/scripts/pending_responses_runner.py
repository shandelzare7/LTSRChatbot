#!/usr/bin/env python3
"""
pending_responses_runner.py — 延迟回复 cron 执行器

功能：
  - 每次被 cron 调用时，查询 pending_bot_responses 中 deliver_at <= now 的任务。
  - 对每条任务：用存储的 user_message + 最新 bot/user 状态重新触发 LLM 生成（Option B）。
  - 生成成功后通过推送通知送达，并将任务标记为 done。
  - 生成失败则标记为 failed，记录错误。

用法：
  python scripts/pending_responses_runner.py
  # 或 cron：*/5 * * * * python /path/to/scripts/pending_responses_runner.py

环境变量：
  DATABASE_URL         : 数据库连接字符串（必需）
  REAL_MODE_ENABLED    : 必须为 true 才会执行（若为 false 则脚本直接退出）
  PENDING_BATCH_LIMIT  : 单次最多处理多少条任务（默认 10）
  LOG_LEVEL            : 日志级别（默认 INFO）
  VAPID_PRIVATE_KEY    : Web Push VAPID 私钥（Base64url，缺失则跳过推送）
  VAPID_PUBLIC_KEY     : Web Push VAPID 公钥
  VAPID_CLAIM_EMAIL    : VAPID 发件人邮箱（如 mailto:admin@example.com）
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── 项目根目录加入 sys.path ──────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from utils.env_loader import load_project_env
    load_project_env(project_root)
except Exception:
    pass

# ── 日志 ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pending_runner")


def _is_real_mode() -> bool:
    return str(os.getenv("REAL_MODE_ENABLED", "false")).strip().lower() in ("1", "true", "yes", "on")


# ── 推送工具 ──────────────────────────────────────────────────────────────────

def _get_vapid_keys() -> Optional[tuple[str, str, str]]:
    """读取 VAPID 三件套。任一缺失则返回 None（推送不可用）。"""
    priv = (os.getenv("VAPID_PRIVATE_KEY") or "").strip()
    pub  = (os.getenv("VAPID_PUBLIC_KEY")  or "").strip()
    mail = (os.getenv("VAPID_CLAIM_EMAIL") or "").strip()
    if not (priv and pub and mail):
        return None
    if not mail.startswith("mailto:"):
        mail = "mailto:" + mail
    return priv, pub, mail


async def _get_push_subscription_for_user(db: Any, user_external_id: str, bot_id: str) -> Optional[dict]:
    """
    从 push_subscriptions 表查询该 user+bot 最新一条订阅。
    返回 dict（Web Push subscription object）或 None。
    """
    try:
        from sqlalchemy import text
        sql = text("""
            SELECT subscription FROM push_subscriptions
            WHERE user_external_id = :uid
              AND bot_id = CAST(:bid AS uuid)
            ORDER BY updated_at DESC
            LIMIT 1
        """)
        async with db.engine.connect() as conn:
            result = await conn.execute(sql, {"uid": user_external_id, "bid": bot_id})
            row = result.fetchone()
        if row is None:
            return None
        raw = row[0]
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, dict):
            return raw
        return None
    except Exception as e:
        logger.warning("[Runner] push subscription lookup failed: %s", e)
        return None


async def _push_segments(
    db: Any,
    user_external_id: str,
    bot_id: str,
    bot_name: str,
    segments: List[Dict[str, Any]],
) -> None:
    """将生成的气泡通过 Web Push 送达用户。"""
    vapid = _get_vapid_keys()
    if vapid is None:
        logger.info(
            "[Runner] VAPID keys not configured; skipping push for user=%s bot=%s",
            user_external_id, bot_id,
        )
        return

    sub = await _get_push_subscription_for_user(db, user_external_id, bot_id)
    if not sub:
        logger.info(
            "[Runner] No push subscription found for user=%s bot=%s",
            user_external_id, bot_id,
        )
        return

    vapid_private, vapid_public, vapid_email = vapid
    combined = " ".join(
        s.get("content", "") for s in segments if s.get("content")
    )
    title = bot_name or "Chatbot"
    body  = combined[:200]

    try:
        import concurrent.futures
        from pywebpush import webpush, WebPushException

        payload = json.dumps({"title": title, "body": body, "url": "/", "tag": "ltsr-bot-message"})

        def _sync_push():
            webpush(
                subscription_info=sub,
                data=payload,
                vapid_private_key=vapid_private,
                vapid_claims={"sub": vapid_email},
            )

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            await loop.run_in_executor(pool, _sync_push)

        logger.info(
            "[Runner] Push sent: user=%s bot=%s body_len=%d",
            user_external_id, bot_id, len(body),
        )
    except Exception as e:
        logger.warning("[Runner] Push send failed: %s", e)


# ── 任务处理 ──────────────────────────────────────────────────────────────────

async def _process_task(task: Dict[str, Any], db: Any, graph: Any) -> None:
    """处理单条延迟任务：加载状态 → 运行图 → 推送 → 标记 done。"""
    task_id  = task["id"]
    bot_id   = task["bot_id"]
    user_msg = task["user_message"]

    # 通过 DB 找到 user 的 external_id（user_id 列存的是 UUID，需要查 User 行）
    try:
        from app.core.db.database import User
        from sqlalchemy import select as sa_select
        import uuid
        user_uuid = uuid.UUID(task["user_id"])
        async with db.Session() as session:
            result = await session.execute(
                sa_select(User).where(User.id == user_uuid)
            )
            user_row = result.scalar_one_or_none()
        if user_row is None:
            raise ValueError(f"User row not found: {task['user_id']}")
        user_external_id = str(user_row.external_id)
    except Exception as e:
        logger.error("[Runner] Task %s: failed to resolve user: %s", task_id, e)
        await db.mark_pending_response_done(task_id, error=str(e))
        return

    logger.info(
        "[Runner] Task %s: user=%s bot=%s reason=%s sub=%s",
        task_id, user_external_id, bot_id,
        task.get("absence_reason"), task.get("absence_sub_reason"),
    )

    try:
        # Option B: 用最新 bot/user 状态 + 存储的 user_message 重新生成。
        # _scheduled_run=True 告知 absence_gate 透明通过，不重复触发延迟。
        initial_state: Dict[str, Any] = {
            "user_id": user_external_id,
            "bot_id": bot_id,
            "user_input": user_msg,
            "current_time": datetime.now(timezone.utc).isoformat(),
            "_scheduled_run": True,
        }
        result_state = await graph.ainvoke(initial_state)

        segments = (result_state.get("humanized_output") or {}).get("segments") or []
        if segments:
            # 获取 bot 名称用于推送标题
            bot_name = str((result_state.get("bot_basic_info") or {}).get("name") or "")
            await _push_segments(db, user_external_id, bot_id, bot_name, segments)
        else:
            logger.warning("[Runner] Task %s: no segments generated", task_id)

        await db.mark_pending_response_done(task_id)
        logger.info("[Runner] Task %s: done", task_id)

    except Exception as e:
        logger.error("[Runner] Task %s: generation failed: %s", task_id, e, exc_info=True)
        await db.mark_pending_response_done(task_id, error=str(e))


async def main() -> None:
    if not _is_real_mode():
        logger.info("[Runner] REAL_MODE_ENABLED is false — nothing to do, exiting.")
        return

    limit = int(os.getenv("PENDING_BATCH_LIMIT", "10"))

    from app.core.db.database import DBManager
    db = DBManager.from_env()

    # 构建完整图（entry_point=loader，end_at=memory_writer）。
    # absence_gate 节点会因 _scheduled_run=True 直接透传，不会重复触发延迟。
    from app.graph import build_graph
    graph = build_graph(db_manager=db)

    now = datetime.now(timezone.utc)
    tasks = await db.get_due_pending_responses(now=now, limit=limit)

    if not tasks:
        logger.info("[Runner] No due tasks at %s", now.isoformat())
        return

    logger.info("[Runner] Processing %d due task(s)...", len(tasks))
    for task in tasks:
        await _process_task(task, db, graph)


if __name__ == "__main__":
    asyncio.run(main())

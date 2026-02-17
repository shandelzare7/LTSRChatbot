"""
批量清理“系统性/助手味”的 backlog 任务，避免把 LATS 拉回“助手味”。

会做三件事：
1) 从 bot_tasks 表里删除 task_type=backlog 且命中“系统性关键词”的任务
2) 过滤 bots.backlog_tasks（Bot 自带个性任务库）里命中项
3) 过滤 users.assets.current_session_tasks 里命中项（避免 carry 池残留）

使用：
  # 清理本地/默认库（DATABASE_URL）；若也设置了 RENDER_DATABASE_URL，则会一并清理 Render 库
  python -m devtools.purge_systemic_backlog_tasks
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env

    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from sqlalchemy import select
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm.attributes import flag_modified

from app.core.database import Bot, BotTask, DBManager, User, _create_async_engine_from_database_url


def _get_systemic_predicate() -> Callable[[str], bool]:
    """
    优先复用创建期过滤器；若 import 失败则使用本地兜底规则。
    """
    try:
        from app.core.bot_creation_llm import _is_systemic_backlog_task  # type: ignore

        return _is_systemic_backlog_task
    except Exception:
        banned = (
            "写入长期记忆",
            "长期记忆",
            "写入记忆",
            "记忆锚点",
            "锚点",
            "短标签",
            "标签",
            "TTL",
            "待澄清",
            "澄清点",
            "共同叙事小总结",
            "总结一下",
            "总结",
            "记录",
            "写入",
            "持久化",
            "数据库",
            "transcript:",
            "src=",
            "note",
            "derived",
            "memory store",
            "我记住",
            "我会记住",
            "我帮你总结",
            "我给你总结",
        )

        def _fallback(desc: str) -> bool:
            d = str(desc or "").strip()
            if not d:
                return True
            if any(x in d for x in banned):
                return True
            if d.startswith("每轮") and ("识别" in d or "写入" in d or "总结" in d):
                return True
            return False

        return _fallback


def _filter_task_dicts(tasks: Any, is_systemic: Callable[[str], bool]) -> Tuple[List[Dict[str, Any]], int]:
    """
    tasks 期望为 List[Dict]；返回 (filtered, removed_count)
    """
    if not isinstance(tasks, list):
        return [], 0
    kept: List[Dict[str, Any]] = []
    removed = 0
    for t in tasks:
        if not isinstance(t, dict):
            removed += 1
            continue
        desc = str(t.get("description") or "").strip()
        if is_systemic(desc):
            removed += 1
            continue
        kept.append(t)
    return kept, removed


async def _purge_one_db(url: str, *, label: str) -> None:
    is_systemic = _get_systemic_predicate()

    engine = _create_async_engine_from_database_url(url)
    db = DBManager(engine)

    bots_updated = 0
    bots_removed = 0
    users_updated = 0
    users_removed = 0
    tasks_deleted = 0

    async with db.Session() as session:
        # 1) bots.backlog_tasks 清理（独立事务）
        async with session.begin():
            bot_rows = (await session.execute(select(Bot))).scalars().all()
            for bot in bot_rows:
                raw = getattr(bot, "backlog_tasks", None)
                if not isinstance(raw, list):
                    continue
                kept, removed = _filter_task_dicts(raw, is_systemic)
                if removed > 0:
                    bot.backlog_tasks = kept
                    flag_modified(bot, "backlog_tasks")
                    bots_updated += 1
                    bots_removed += removed

        # 2) bot_tasks 表清理（独立事务；表缺失时跳过，且不影响后续步骤）
        try:
            async with session.begin():
                task_rows = (
                    await session.execute(
                        select(BotTask).where(BotTask.task_type == "backlog")
                    )
                ).scalars().all()
                for t in task_rows:
                    desc = str(getattr(t, "description", "") or "").strip()
                    if is_systemic(desc):
                        await session.delete(t)
                        tasks_deleted += 1
        except ProgrammingError:
            tasks_deleted = -1
            try:
                await session.rollback()
            except Exception:
                pass

        # 3) users.assets.current_session_tasks 清理（独立事务）
        async with session.begin():
            user_rows = (await session.execute(select(User))).scalars().all()
            for user in user_rows:
                assets = user.assets or {}
                raw_tasks = assets.get("current_session_tasks")
                if not isinstance(raw_tasks, list):
                    continue
                kept, removed = _filter_task_dicts(raw_tasks, is_systemic)
                if removed > 0:
                    assets = dict(assets)
                    assets["current_session_tasks"] = kept
                    user.assets = assets
                    flag_modified(user, "assets")
                    users_updated += 1
                    users_removed += removed

    print(f"[{label}] ✅ purge done")
    print(f"[{label}] bots.backlog_tasks updated: {bots_updated}, removed items: {bots_removed}")
    if tasks_deleted >= 0:
        print(f"[{label}] bot_tasks backlog deleted: {tasks_deleted}")
    else:
        print(f"[{label}] bot_tasks backlog deleted: (skipped; table missing)")
    print(f"[{label}] users.assets.current_session_tasks updated: {users_updated}, removed items: {users_removed}")
    print()


async def main() -> None:
    targets: List[Tuple[str, str]] = []

    db_url = os.getenv("DATABASE_URL")
    render_url = os.getenv("RENDER_DATABASE_URL")

    if db_url:
        targets.append((db_url, "DATABASE_URL"))
    if render_url:
        targets.append((render_url, "RENDER_DATABASE_URL"))

    if not targets:
        print("ERROR: 请设置 DATABASE_URL（以及可选的 RENDER_DATABASE_URL）后再运行。")
        sys.exit(1)

    for url, label in targets:
        await _purge_one_db(url, label=label)


if __name__ == "__main__":
    asyncio.run(main())


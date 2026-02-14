"""
clear_memory.py

一键清理指定 user_id/bot_id 的所有持久化记忆：
- DB（若配置 DATABASE_URL）：messages / memories / transcripts / derived_notes / conversation_summary，并重置阶段与关系数值
- 本地 local_data：删除 rel__{bot_id}__{user_id} 目录（messages/transcripts/notes/summary 全清空）

用法示例：
  python tools/clear_memory.py --user_id local_user_5128d1c1 --bot_id 4d803b5a-cb30-4d14-89eb-88d259564610

如果只想清本地：
  python tools/clear_memory.py --user_id ... --bot_id ... --local_only
"""

from __future__ import annotations

import argparse
import os


def _run_async(coro):
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Detected running event loop; please run this script in a normal terminal (no running event loop).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_id", required=True)
    ap.add_argument("--bot_id", required=True)
    ap.add_argument("--local_only", action="store_true", help="只清理本地 local_data，不动 DB")
    ap.add_argument("--db_only", action="store_true", help="只清理 DB，不动本地 local_data")
    args = ap.parse_args()

    user_id = str(args.user_id).strip()
    bot_id = str(args.bot_id).strip()
    if not user_id or not bot_id:
        raise SystemExit("user_id/bot_id 不能为空")

    cleared_local = False
    cleared_db = False

    if not args.db_only:
        try:
            from app.core.local_store import LocalStoreManager

            store = LocalStoreManager()
            cleared_local = store.clear_relationship(user_id, bot_id)
        except Exception as e:
            print(f"[clear_memory] 本地清理失败: {e}")

    if not args.local_only:
        if os.getenv("DATABASE_URL"):
            try:
                from app.core.database import DBManager

                db = DBManager.from_env()
                counts = _run_async(db.clear_all_memory_for(user_id, bot_id, reset_profile=True))
                print(f"[clear_memory] DB 清理完成: {counts}")
                cleared_db = True
            except Exception as e:
                print(f"[clear_memory] DB 清理失败: {e}")
        else:
            print("[clear_memory] 未设置 DATABASE_URL，跳过 DB 清理")

    print(f"[clear_memory] local_cleared={cleared_local}, db_cleared={cleared_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


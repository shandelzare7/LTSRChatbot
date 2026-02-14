"""出口写入节点：优先写 DB（Commit Late），无 DB 时回退 Memory Service。"""

from typing import TYPE_CHECKING, Callable

import os
import asyncio
from datetime import datetime, timezone

from app.state import AgentState

if TYPE_CHECKING:
    from app.services.memory.base import MemoryBase


_DB_MANAGER = None


def _get_db_manager():
    global _DB_MANAGER
    if _DB_MANAGER is not None:
        return _DB_MANAGER
    if not os.getenv("DATABASE_URL"):
        return None
    try:
        from app.core.database import DBManager

        _DB_MANAGER = DBManager.from_env()
        return _DB_MANAGER
    except Exception:
        return None


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Detected running event loop; please use an async graph entry (ainvoke) for DB operations.")


def create_memory_writer_node(memory_service: "MemoryBase") -> Callable[[AgentState], dict]:
    """创建 memory_writer 节点。返回 async 节点，与 loader 配合 ainvoke 使用同一事件循环。"""

    async def node(state: AgentState) -> dict:
        user_id = state.get("user_id") or "default_user"
        segments = state.get("final_segments", []) or []
        final_text = " ".join(segments) if segments else state.get("final_response") or state.get("draft_response") or ""
        final_text = (final_text or "").strip()

        # 在真正“准备返回给用户前”的时刻打点（用于 DB created_at）
        state["ai_sent_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        db = _get_db_manager()
        if db:
            bot_id = state.get("bot_id") or (state.get("bot_basic_info") or {}).get("name") or "default_bot"
            new_memory = state.get("generated_new_memory_text") or state.get("new_memory_content")
            await db.save_turn(str(user_id), str(bot_id), dict(state), new_memory=new_memory)
            return {}

        # local store (default)
        try:
            from app.core.local_store import LocalStoreManager

            store = LocalStoreManager()
            bot_id = state.get("bot_id") or (state.get("bot_basic_info") or {}).get("name") or "default_bot"
            new_memory = state.get("generated_new_memory_text") or state.get("new_memory_content")
            store.save_turn(str(user_id), str(bot_id), dict(state), new_memory=new_memory)
            return {}
        except Exception:
            pass

        # fallback: MockMemory
        if final_text:
            memory_service.append_memory(user_id, f"Bot 回复: {final_text[:200]}", meta={"source": "memory_writer"})
        return {}

    return node


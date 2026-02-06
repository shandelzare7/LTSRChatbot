"""入口加载节点：优先从 DB 读取状态（Load Early），无 DB 时回退 Memory Service。"""
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import os
import asyncio

from app.state import AgentState

if TYPE_CHECKING:
    from app.services.memory.base import MemoryBase


_DB_MANAGER = None


def _get_db_manager():
    """
    懒加载 DBManager。
    - 未配置 DATABASE_URL 时返回 None
    - 仅在 loader/writer 触发，符合“早读晚写”
    """
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
    """在同步节点里执行 async DB 调用。若运行在已有 event loop 中，请改用异步入口。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Detected running event loop; please use an async graph entry (ainvoke) for DB operations.")


def create_loader_node(memory_service: "MemoryBase") -> Callable[[AgentState], dict]:
    def loader_node(state: AgentState) -> dict:
        user_id = state.get("user_id") or "default_user"

        # 统一补齐 user_input / chat_buffer，便于 Detection/Reasoner 等节点使用
        messages = state.get("messages", []) or []
        last = messages[-1] if messages else None
        user_input = getattr(last, "content", "") if last else state.get("user_input", "")
        chat_buffer = state.get("chat_buffer") or messages

        db = _get_db_manager()
        if db:
            # bot_id：优先从 configurable/state/bot_basic_info.name 推断
            bot_id = state.get("bot_id") or (state.get("bot_basic_info") or {}).get("name") or "default_bot"
            db_data: Dict[str, Any] = _run_async(db.load_state(str(user_id), str(bot_id)))
            # DB 返回的 chat_buffer 是历史（旧->新），我们把当前 messages 也拼上（避免丢本轮输入）
            history = db_data.get("chat_buffer") or []
            merged_buffer = list(history) + [m for m in chat_buffer if m not in history]
            return {
                # DB state
                "relationship_state": db_data.get("relationship_state") or {},
                "mood_state": db_data.get("mood_state") or {},
                "current_stage": db_data.get("current_stage") or state.get("current_stage") or "initiating",
                "bot_basic_info": db_data.get("bot_basic_info") or state.get("bot_basic_info") or {},
                "bot_big_five": db_data.get("bot_big_five") or state.get("bot_big_five") or {},
                "bot_persona": db_data.get("bot_persona") or state.get("bot_persona") or {},
                "user_basic_info": db_data.get("user_basic_info") or state.get("user_basic_info") or {},
                "user_inferred_profile": db_data.get("user_inferred_profile") or state.get("user_inferred_profile") or {},
                "relationship_assets": db_data.get("relationship_assets") or state.get("relationship_assets") or {},
                "spt_info": db_data.get("spt_info") or state.get("spt_info") or {},
                "conversation_summary": db_data.get("conversation_summary") or state.get("conversation_summary") or "",
                # runtime convenience
                "user_input": user_input,
                "chat_buffer": merged_buffer,
                # 兼容旧字段（目前部分节点仍会读 user_profile/memories）
                "user_profile": db_data.get("user_inferred_profile") or {},
                "memories": db_data.get("conversation_summary") or "",
            }

        # fallback: MockMemory（仅当前进程内存，不落盘）
        profile = memory_service.get_profile(user_id)
        memories = memory_service.get_memories(user_id, limit=10)
        return {
            "user_profile": profile,
            "memories": memories,
            "user_input": user_input,
            "chat_buffer": chat_buffer,
        }

    return loader_node

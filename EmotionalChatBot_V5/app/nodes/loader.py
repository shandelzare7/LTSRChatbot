"""记忆加载节点：从 Memory Service 拉取 user_profile 与 memories。"""
from typing import TYPE_CHECKING, Any, Callable

from app.state import AgentState

if TYPE_CHECKING:
    from app.services.memory.base import MemoryBase


def create_loader_node(memory_service: "MemoryBase") -> Callable[[AgentState], dict]:
    def loader_node(state: AgentState) -> dict:
        user_id = state.get("user_id") or "default_user"
        profile = memory_service.get_profile(user_id)
        memories = memory_service.get_memories(user_id, limit=10)
        # 统一补齐 user_input / chat_buffer，便于 Detection/Reasoner 等节点使用
        messages = state.get("messages", []) or []
        last = messages[-1] if messages else None
        user_input = getattr(last, "content", "") if last else state.get("user_input", "")
        chat_buffer = state.get("chat_buffer") or messages
        return {
            "user_profile": profile,
            "memories": memories,
            "user_input": user_input,
            "chat_buffer": chat_buffer,
        }

    return loader_node

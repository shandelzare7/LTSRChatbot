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
        return {
            "user_profile": profile,
            "memories": memories,
        }

    return loader_node

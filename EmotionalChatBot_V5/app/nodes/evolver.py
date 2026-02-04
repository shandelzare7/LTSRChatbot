"""关系演化节点：更新关系 DB（Memory Service 写入）。"""
from typing import TYPE_CHECKING, Callable

from app.state import AgentState

if TYPE_CHECKING:
    from app.services.memory.base import MemoryBase


def create_evolver_node(memory_service: "MemoryBase") -> Callable[[AgentState], dict]:
    def evolver_node(state: AgentState) -> dict:
        user_id = state.get("user_id") or "default_user"
        segments = state.get("final_segments", [])
        final_text = " ".join(segments) if segments else state.get("draft_response", "")
        if final_text:
            memory_service.append_memory(
                user_id,
                f"Bot 回复: {final_text[:200]}",
                meta={"source": "evolver"},
            )
        return {}

    return evolver_node

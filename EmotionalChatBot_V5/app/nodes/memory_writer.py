"""Memory Writer：将 Bot 的最终回复写入 Memory Service。"""

from typing import TYPE_CHECKING, Callable

from app.state import AgentState

if TYPE_CHECKING:
    from app.services.memory.base import MemoryBase


def create_memory_writer_node(memory_service: "MemoryBase") -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        user_id = state.get("user_id") or "default_user"
        segments = state.get("final_segments", [])
        final_text = " ".join(segments) if segments else state.get("final_response") or state.get("draft_response") or ""
        final_text = (final_text or "").strip()
        if final_text:
            memory_service.append_memory(
                user_id,
                f"Bot 回复: {final_text[:200]}",
                meta={"source": "memory_writer"},
            )
        return {}

    return node


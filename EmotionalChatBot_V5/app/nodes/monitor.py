"""心理监测节点：调用 Engine（LLM 侧写）更新 current_mode。"""
from typing import TYPE_CHECKING, Callable

from app.state import AgentState

if TYPE_CHECKING:
    from app.core.engine import PsychoEngine


def create_monitor_node(engine: "PsychoEngine") -> Callable[[AgentState], dict]:
    def monitor_node(state: AgentState) -> dict:
        messages = state.get("messages", [])
        if not messages:
            return {}
        last = messages[-1]
        user_msg = getattr(last, "content", str(last))
        current_mode = state.get("current_mode")
        context_data = {
            "current_mode_id": current_mode.id if current_mode else "normal_mode",
            "user_profile": state.get("user_profile", {}),
        }
        new_mode = engine.detect_mode(user_msg, context_data)
        return {"current_mode": new_mode}

    return monitor_node

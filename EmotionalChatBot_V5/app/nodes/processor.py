"""拆句与延迟节点：受 Mode.split_strategy 与 typing_speed_multiplier 控制。"""
import re
from typing import Callable

from app.state import AgentState


def create_processor_node() -> Callable[[AgentState], dict]:
    def processor_node(state: AgentState) -> dict:
        mode = state.get("current_mode")
        draft = state.get("draft_response", "")
        strategy = getattr(mode, "split_strategy", "normal") if mode else "normal"
        multiplier = float(getattr(mode, "typing_speed_multiplier", 1.0) or 1.0)

        if strategy == "fragmented" and draft:
            # 碎碎念：按标点或短句拆
            segments = re.split(r"(?<=[。！？.!?])\s*", draft)
            segments = [s.strip() for s in segments if s.strip()]
            if not segments:
                segments = [draft]
        else:
            segments = [draft] if draft else []

        base_delay = 0.05
        final_delay = base_delay * (1.0 / multiplier) if multiplier > 0 else base_delay
        return {"final_segments": segments, "final_delay": final_delay}

    return processor_node

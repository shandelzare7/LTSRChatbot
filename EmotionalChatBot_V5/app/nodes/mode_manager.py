# app/nodes/mode_manager.py
from __future__ import annotations
from typing import Any, Callable, Dict

from utils.tracing import trace_if_enabled
from app.state import AgentState


def _maxv(d: Dict[str, Any]) -> float:
    vals = [v for v in d.values() if isinstance(v, (int, float))]
    return float(max(vals)) if vals else 0.0


def create_mode_manager_node(modes: list[Any]) -> Callable[[AgentState], dict]:
    modes_by_id = {getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None): m for m in modes}
    # 去掉 None key
    modes_by_id = {k: v for k, v in modes_by_id.items() if k}

    @trace_if_enabled(
        name="Perception/ModeManager",
        run_type="chain",
        tags=["node", "perception", "mode"],
        metadata={"state_outputs": ["mode_id", "current_mode"]},
    )
    def node(state: AgentState) -> dict:
        det = state.get("detection_signals") or {}
        comp = det.get("composite") or {}
        trace = det.get("trace") or {}
        stage_ctx = det.get("stage_ctx") or {}

        conflict_eff = float(comp.get("conflict_eff", 0.0) or 0.0)
        provocation  = float(comp.get("provocation", 0.0) or 0.0)
        pressure     = float(comp.get("pressure", 0.0) or 0.0)
        goodwill     = float(comp.get("goodwill", 0.0) or 0.0)

        sarcasm   = float(trace.get("sarcasm", 0.0) or 0.0)
        contempt  = float(trace.get("contempt", 0.0) or 0.0)
        loweffort = float(trace.get("low_effort", 0.0) or 0.0)
        toxicity  = float(trace.get("toxicity", 0.0) or 0.0)
        confusion = float(trace.get("confusion", 0.0) or 0.0)

        stage_violation = _maxv(stage_ctx)  # 你做的"阶段越界"最大值

        mood = state.get("mood_state") or {}
        busyness = float(mood.get("busyness", 0.0) or 0.0)

        # ---- 规则：你可以按口味微调阈值 ----
        # 注意：无论检测到什么模式（包括 mute_mode），都仅返回 mode，不进行特殊处理
        # 后续节点（如 reasoner、lats_search）会根据 mode 配置来处理行为
        
        # 1) mute_mode（冷处理不回）
        if conflict_eff >= 0.75 or provocation >= 0.85 or pressure >= 0.80:
            mode_id = "mute_mode"
        # 2) cold_mode（冷淡：嗯啊哦/一句话/不投入）
        elif (
            max(sarcasm, contempt, loweffort) >= 0.60 and goodwill < 0.45
        ) or (
            toxicity >= 0.55 and goodwill < 0.50
        ) or (
            stage_violation >= 0.70 and goodwill < 0.60
        ) or (
            busyness >= 0.80 and goodwill < 0.50
        ) or (
            confusion >= 0.70  # 对方表达太乱，先冷淡+澄清
        ):
            mode_id = "cold_mode"
        # 3) 默认 normal_mode
        else:
            mode_id = "normal_mode"

        current_mode = modes_by_id.get(mode_id) or modes_by_id.get("normal_mode")
        print(f"[Mode Manager] 确定 mode: {mode_id} (conflict_eff={conflict_eff:.2f}, provocation={provocation:.2f}, goodwill={goodwill:.2f})")
        
        # 仅返回 mode，不进行特殊处理
        return {
            "mode_id": mode_id,
            "current_mode": current_mode,
        }

    return node

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from utils.tracing import trace_if_enabled


def _as_float(d: Dict[str, Any], k: str, default: float = 0.0) -> float:
    try:
        return float(d.get(k, default))
    except Exception:
        return default


def _as_int(d: Dict[str, Any], k: str, default: int = 0) -> int:
    try:
        return int(d.get(k, default))
    except Exception:
        return default


def _topic_history_count(spt: Dict[str, Any]) -> int:
    th = spt.get("topic_history")
    if isinstance(th, set):
        return len(th)
    if isinstance(th, (list, tuple)):
        return len(th)
    return 0


def check_transition(
    stage: str,
    scores: Dict[str, Any],
    spt: Dict[str, Any],
    *,
    deltas_applied: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Knapp stage transition logic (hierarchy):
    1) Global crashes FIRST
    2) Growth (Coming Together) SECOND
    3) Decay (Coming Apart) THIRD
    4) Special jumps (event-driven) applied on top (highest priority after crashes)
    """
    stage = str(stage or "initiating")
    deltas_applied = deltas_applied or {}

    # scores
    trust = _as_float(scores, "trust", 0.0)
    closeness = _as_float(scores, "closeness", 0.0)
    liking = _as_float(scores, "liking", 0.0)
    respect = _as_float(scores, "respect", 0.0)
    warmth = _as_float(scores, "warmth", 0.0)
    power = _as_float(scores, "power", 50.0)

    # spt
    depth = _as_int(spt, "current_depth_level", 1)
    max_depth = _as_int(spt, "max_depth_reached", 1)
    breadth = _topic_history_count(spt)

    # 0) Special "Jump" Logic (Event Driven)
    # Crash: if trust_delta <= -30 (single turn), Jump to Terminating.
    # Breakup: if liking_delta <= -25 (single turn), Jump to Differentiating.
    # (Note: in current engine deltas are usually small; this is kept for completeness.)
    if _as_float(deltas_applied, "trust", 0.0) <= -30:
        return "terminating"
    if _as_float(deltas_applied, "liking", 0.0) <= -25:
        return "differentiating"

    # 1) Global Safety Net (Checks for immediate failure)
    if trust <= -20 or closeness <= -10:
        return "terminating"

    # 2) Phase 1: Coming Together (Growth)
    if stage == "initiating":
        if closeness >= 10 or liking >= 10:
            return "experimenting"
        if liking < 0:
            return "terminating"  # Early exit: Bad first impression

    elif stage == "experimenting":
        if closeness >= 40 and trust >= 30 and depth >= 2:
            return "intensifying"
        if liking < 10 and breadth > 3:
            return "avoiding"  # Early exit: Boring connection

    elif stage == "intensifying":
        power_gap = abs(power - 50) * 2
        if closeness >= 70 and trust >= 60 and depth >= 3:
            if power_gap <= 40:
                return "integrating"

    elif stage == "integrating":
        if closeness >= 90 and trust >= 90 and depth == 4:
            if respect >= 60:
                return "bonding"

    # 3) Phase 2: Coming Apart (Decay)
    if stage in ["bonding", "integrating"]:
        if closeness > 60 and (respect < 40 or liking < 40):
            return "differentiating"

    elif stage == "differentiating":
        # spec: trust < 50 OR spt_depth decreasing trend
        depth_decreasing = depth < max_depth
        if trust < 50 or depth_decreasing:
            return "circumscribing"

    elif stage == "circumscribing":
        if warmth < 30 and depth <= 1:
            return "stagnating"

    elif stage == "stagnating":
        if closeness < 20:
            return "avoiding"

    elif stage == "avoiding":
        if closeness <= 0:
            return "terminating"

    return stage


def create_stage_manager_node() -> Callable[[Dict[str, Any]], dict]:
    @trace_if_enabled(
        name="Relationship/StageManager",
        run_type="chain",
        tags=["node", "relationship", "knapp", "stage_manager"],
        metadata={"state_outputs": ["current_stage"]},
    )
    def node(state: Dict[str, Any]) -> dict:
        scores = state.get("relationship_state") or {}
        spt = state.get("spt_state") or {}
        stage = state.get("current_stage") or "initiating"
        deltas_applied = state.get("relationship_deltas_applied") or {}

        new_stage = check_transition(
            str(stage),
            scores,
            spt,
            deltas_applied=deltas_applied,
        )

        # 记录阶段变化（可选）
        if new_stage != stage:
            return {"current_stage": new_stage, "stage_transition": {"from": stage, "to": new_stage}}
        return {"current_stage": new_stage}

    return node


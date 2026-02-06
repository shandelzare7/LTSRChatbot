"""
Relationship Updater (Assets, Facts)

职责：维护单调累积的 RelationshipAssets（历史资产），并产生本轮 asset_updates flags。
流水线位置：Processor -> Updater -> Evolver -> StageManager -> Responder
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Set

from app.state import AgentState
from utils.tracing import trace_if_enabled


def _ensure_assets(state: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(state)
    assets = dict(s.get("relationship_assets") or {})

    th = assets.get("topic_history")
    if isinstance(th, set):
        topic_history: Set[str] = th
    elif isinstance(th, (list, tuple)):
        topic_history = set([str(x) for x in th])
    else:
        topic_history = set()
    assets["topic_history"] = topic_history

    assets.setdefault("breadth_score", len(topic_history))
    assets.setdefault("max_spt_depth", 1)
    assets.setdefault("intellectual_capital", 0)
    s["relationship_assets"] = assets

    s.setdefault("asset_updates", {"is_new_topic": False, "is_spt_breakthrough": False, "is_intellectually_deep": False})
    return s


@trace_if_enabled(
    name="Relationship/Updater(Assets)",
    run_type="chain",
    tags=["node", "relationship", "assets", "updater"],
    metadata={"state_outputs": ["relationship_assets", "asset_updates"]},
)
def updater_node(state: AgentState) -> dict:
    safe = _ensure_assets(state)
    processor_output = safe.get("processor_output") or {}

    topic_category = str(processor_output.get("topic_category") or "general")
    try:
        spt_level = int(processor_output.get("spt_level") or 1)
    except Exception:
        spt_level = 1
    spt_level = max(1, min(4, spt_level))
    is_intellectually_deep = bool(processor_output.get("is_intellectually_deep") or False)

    assets: Dict[str, Any] = dict(safe.get("relationship_assets") or {})
    topic_history: Set[str] = assets.get("topic_history") if isinstance(assets.get("topic_history"), set) else set()

    old_breadth = len(topic_history)
    is_new_topic = topic_category not in topic_history
    if is_new_topic:
        topic_history.add(topic_category)

    new_breadth = len(topic_history)
    old_max_depth = int(assets.get("max_spt_depth") or 1)
    new_max_depth = max(old_max_depth, spt_level)
    is_spt_breakthrough = new_max_depth > old_max_depth

    intellectual_capital = int(assets.get("intellectual_capital") or 0)
    if is_intellectually_deep:
        intellectual_capital += 1

    assets.update(
        {
            "topic_history": topic_history,
            "breadth_score": new_breadth,
            "max_spt_depth": new_max_depth,
            "intellectual_capital": intellectual_capital,
        }
    )

    flags = {
        "is_new_topic": bool(is_new_topic),
        "is_spt_breakthrough": bool(is_spt_breakthrough),
        "is_intellectually_deep": bool(is_intellectually_deep),
    }

    return {"relationship_assets": assets, "asset_updates": flags}


def create_updater_node() -> Callable[[AgentState], dict]:
    return updater_node


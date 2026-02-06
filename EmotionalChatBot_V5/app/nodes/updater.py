"""updater.py

关系资产更新节点（Monotonic Assets Updater）

职责：
- 基于 processor 输出（topic_category / spt_level / is_intellectually_deep）
  更新跨回合累积的 RelationshipAssets
- 产出本回合 transient flags（asset_updates），供 evolver 节点做 bonus/触发
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from app.state import AgentState, AssetDelta, RelationshipAssets


def relationship_updater_node(state: AgentState) -> Dict[str, Any]:
    """
    Updates monotonic relationship assets based on Processor output.
    Calculates deltas (flags) for the Evolver node to use.
    """
    processor_output = state.get("processor_output", {}) or {}
    current_assets = state.get("relationship_assets")

    if not current_assets:
        current_assets = {
            "topic_history": [],
            "breadth_score": 0,
            "max_spt_depth": 1,
            "intellectual_capital": 0,
        }

    new_topic = str(processor_output.get("topic_category", "general") or "general")
    try:
        current_spt_level = int(processor_output.get("spt_level", 1) or 1)
    except Exception:
        current_spt_level = 1
    current_spt_level = max(1, min(4, current_spt_level))
    is_deep_turn = bool(processor_output.get("is_intellectually_deep", False))

    # --- Topic breadth ---
    topic_set = set([str(t) for t in (current_assets.get("topic_history") or [])])
    is_new_topic = False
    if new_topic and new_topic not in topic_set:
        topic_set.add(new_topic)
        is_new_topic = True

    updated_topic_history: List[str] = sorted(list(topic_set))
    updated_breadth_score = len(updated_topic_history)

    # --- SPT depth high-water mark ---
    try:
        old_max = int(current_assets.get("max_spt_depth", 1) or 1)
    except Exception:
        old_max = 1
    old_max = max(1, min(4, old_max))
    new_max = max(old_max, current_spt_level)
    is_spt_breakthrough = new_max > old_max

    # --- Intellectual capital ---
    try:
        current_capital = int(current_assets.get("intellectual_capital", 0) or 0)
    except Exception:
        current_capital = 0
    new_capital = current_capital + (1 if is_deep_turn else 0)

    updated_assets: RelationshipAssets = {
        "topic_history": updated_topic_history,
        "breadth_score": int(updated_breadth_score),
        "max_spt_depth": int(new_max),
        "intellectual_capital": int(new_capital),
    }

    asset_updates: AssetDelta = {
        "is_new_topic": bool(is_new_topic),
        "is_spt_breakthrough": bool(is_spt_breakthrough),
        "is_intellectually_deep": bool(is_deep_turn),
    }

    return {"relationship_assets": updated_assets, "asset_updates": asset_updates}


def create_updater_node() -> Callable[[AgentState], dict]:
    return relationship_updater_node


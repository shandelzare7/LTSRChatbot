"""
Evolver（Relationship Metrics / Feelings）

职责：更新 6 维关系分数（relationship_metrics），它是动态波动值。
输入：
- processor_output: 本轮基础变化量 base_deltas + 标签
- asset_updates: Updater 产生的 flags（本轮是否新话题/深聊/突破）

核心：
1) Base Change: base_deltas
2) Apply Asset Bonuses（硬编码）：
   - is_new_topic: +2 closeness, +2 liking
   - is_intellectually_deep: +3 respect
   - is_spt_breakthrough: +10 closeness, +10 trust
3) 可选阻尼：边际收益递减/背叛惩罚（保留，避免无穷增长）

输出：
- relationship_metrics: Dict[str,int]
并同步写回 relationship_state（float），供现有 styler/其他旧节点读取
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from app.state import AgentState
from utils.tracing import trace_if_enabled


REL_DIMS = ("closeness", "trust", "liking", "respect", "warmth", "power")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def calculate_damped_delta(current_score: float, raw_delta: int) -> float:
    # 正向变化：分数越高越难涨
    if raw_delta > 0:
        if current_score >= 90:
            return raw_delta * 0.1
        if current_score >= 60:
            return raw_delta * 0.5
        return raw_delta * 1.0
    # 负向变化：高分更痛（背叛惩罚）
    if raw_delta < 0:
        if current_score >= 80:
            return raw_delta * 1.5
        return raw_delta * 1.0
    return 0.0


def _ensure_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(state)
    metrics = dict(s.get("relationship_metrics") or {})
    metrics.setdefault("closeness", 0)
    metrics.setdefault("trust", 0)
    metrics.setdefault("liking", 0)
    metrics.setdefault("respect", 0)
    metrics.setdefault("warmth", 0)
    metrics.setdefault("power", 50)
    s["relationship_metrics"] = metrics
    s.setdefault("asset_updates", {"is_new_topic": False, "is_spt_breakthrough": False, "is_intellectually_deep": False})
    s.setdefault("processor_output", {"base_deltas": {k: 0 for k in REL_DIMS}})
    return s


@trace_if_enabled(
    name="Relationship/Evolver(Metrics)",
    run_type="chain",
    tags=["node", "relationship", "metrics", "evolver"],
    metadata={"state_outputs": ["relationship_metrics", "relationship_state"]},
)
def evolver_node(state: AgentState) -> dict:
    safe = _ensure_metrics(state)
    metrics: Dict[str, int] = dict(safe.get("relationship_metrics") or {})
    flags = safe.get("asset_updates") or {}
    processor_output = safe.get("processor_output") or {}
    base_deltas = dict(processor_output.get("base_deltas") or {})

    # Asset bonuses（硬编码）
    bonus = {k: 0 for k in REL_DIMS}
    if flags.get("is_new_topic"):
        bonus["closeness"] += 2
        bonus["liking"] += 2
    if flags.get("is_intellectually_deep"):
        bonus["respect"] += 3
    if flags.get("is_spt_breakthrough"):
        bonus["closeness"] += 10
        bonus["trust"] += 10

    applied: Dict[str, float] = {}
    for dim in REL_DIMS:
        cur = int(metrics.get(dim, 0 if dim != "power" else 50))
        raw = int(base_deltas.get(dim, 0)) + int(bonus.get(dim, 0))
        if raw == 0:
            applied[dim] = 0.0
            continue
        real = float(calculate_damped_delta(float(cur), int(raw)))

        if dim == "power":
            nxt = _clamp(cur + real, 0.0, 100.0)
        else:
            # 允许负值用于 Crash gates
            nxt = _clamp(cur + real, -100.0, 100.0)

        metrics[dim] = int(round(nxt))
        applied[dim] = round(real, 3)

    # 同步到旧字段 relationship_state（float）
    rel_state = {
        "closeness": float(metrics.get("closeness", 0)),
        "trust": float(metrics.get("trust", 0)),
        "liking": float(metrics.get("liking", 0)),
        "respect": float(metrics.get("respect", 0)),
        "warmth": float(metrics.get("warmth", 0)),
        "power": float(metrics.get("power", 50)),
    }

    return {
        "relationship_metrics": metrics,
        "relationship_state": rel_state,
        "relationship_deltas_applied": applied,
    }


def create_evolver_node() -> Callable[[AgentState], dict]:
    return evolver_node

from __future__ import annotations

"""
state_schema.py
补充 State 的结构化校验（供 stage_manager 使用）。

目标：
- 在不强依赖 app/state.py 的 TypedDict 完整性的情况下，
  用 Pydantic 对 stage_manager 所需字段做校验/补齐默认值。
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RelationshipStateModel(BaseModel):
    closeness: float = Field(50.0, ge=0.0, le=100.0)
    trust: float = Field(50.0, ge=0.0, le=100.0)
    liking: float = Field(50.0, ge=0.0, le=100.0)
    respect: float = Field(50.0, ge=0.0, le=100.0)
    warmth: float = Field(50.0, ge=0.0, le=100.0)
    power: float = Field(50.0, ge=0.0, le=100.0)


class SPTInfoModel(BaseModel):
    """
    Social Penetration Theory info for stage gating.
    - depth: 1..4
    - breadth: unique topic count
    - topic_list: raw history list
    - depth_trend: stable|increasing|decreasing
    - recent_signals: symbolic signals (self_disclosure, we_talk, contempt...)
    """

    depth: int = Field(1, ge=1, le=4)
    breadth: int = Field(0, ge=0)
    topic_list: List[str] = Field(default_factory=list)
    depth_trend: Literal["stable", "increasing", "decreasing"] = "stable"
    recent_signals: List[str] = Field(default_factory=list)


class StageManagerInput(BaseModel):
    current_stage: str = "initiating"
    relationship_state: RelationshipStateModel = Field(default_factory=RelationshipStateModel)
    # deltas: raw deltas dict; may be in [-3..3] or already points; stage_manager will normalize
    relationship_deltas: Dict[str, float] = Field(default_factory=dict)
    # optional more realistic: damped deltas applied (points)
    relationship_deltas_applied: Dict[str, float] = Field(default_factory=dict)
    # optional analysis payload
    latest_relationship_analysis: Dict[str, Any] = Field(default_factory=dict)
    # SPT info
    spt_info: Optional[SPTInfoModel] = None


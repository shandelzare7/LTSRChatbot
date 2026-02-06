from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class RelationshipDeltas(BaseModel):
    """6维关系的数值变化量 (-3 到 +3)"""

    closeness: int = Field(..., description="Range: -3 to 3", ge=-3, le=3)
    trust: int = Field(..., description="Range: -3 to 3", ge=-3, le=3)
    liking: int = Field(..., description="Range: -3 to 3", ge=-3, le=3)
    respect: int = Field(..., description="Range: -3 to 3", ge=-3, le=3)
    warmth: int = Field(..., description="Range: -3 to 3", ge=-3, le=3)
    power: int = Field(..., description="Range: -3 to 3", ge=-3, le=3)


class RelationshipAnalysis(BaseModel):
    """LLM 对关系的完整分析结果"""

    thought_process: str = Field(
        ..., description="Step-by-step reasoning linking user input to signals and context."
    )
    detected_signals: List[str] = Field(
        default_factory=list,
        description="Specific cues found in input (e.g., 'User shared a secret').",
    )
    topic_category: str = Field(
        "general",
        description="A coarse topic label for breadth tracking (e.g., work, family, love, health, hobbies).",
    )
    self_disclosure_depth_level: int = Field(
        1,
        ge=1,
        le=4,
        description="SPT depth level: 1=Public,2=Preferences,3=Private,4=Core",
    )
    is_intellectually_deep: bool = Field(
        False,
        description="Whether the message shows intellectually deep/reflective content (depth bonus).",
    )
    deltas: RelationshipDeltas = Field(..., description="The calculated score changes.")


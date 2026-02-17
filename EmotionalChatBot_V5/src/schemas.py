from __future__ import annotations

from typing import Any, Dict, List

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
    deltas: RelationshipDeltas = Field(..., description="The calculated score changes.")

    # --- User Profiling (optional, extracted from conversation) ---
    basic_info_updates: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted user basic info from conversation: name/age/gender/occupation/location. Only non-empty when info is found.",
    )
    new_inferred_entries: Dict[str, str] = Field(
        default_factory=dict,
        description="Other user profile observations: key=trait name, value=description.",
    )


from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# 符合 OpenAI 严格 JSON Schema：禁止额外属性，避免 Invalid schema 警告
_pydantic_extra_forbid = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Evolver (Relationship)
# ---------------------------------------------------------------------------


class RelationshipDeltas(BaseModel):
    model_config = _pydantic_extra_forbid

    """6维关系的数值变化量 (-3 到 +3)"""

    closeness: int = Field(..., description="Range: -3 to 3", ge=-3, le=3)
    trust: int = Field(..., description="Range: -3 to 3", ge=-3, le=3)
    liking: int = Field(..., description="Range: -3 to 3", ge=-3, le=3)
    respect: int = Field(..., description="Range: -3 to 3", ge=-3, le=3)
    attractiveness: int = Field(..., description="Range: -3 to 3", ge=-3, le=3)
    power: int = Field(..., description="Range: -3 to 3", ge=-3, le=3)


class RelationshipAnalysis(BaseModel):
    model_config = _pydantic_extra_forbid

    """LLM 对关系的完整分析结果"""

    thought_process: str = Field(
        ..., description="Step-by-step reasoning linking user input to signals and context."
    )
    detected_signals: List[str] = Field(
        default_factory=list,
        description="Specific cues found in input (e.g., 'User shared a secret').",
    )
    deltas: RelationshipDeltas = Field(..., description="The calculated score changes.")

    # --- Task completion (optional, judged by LLM from bot reply + tasks_for_lats) ---
    completed_task_ids: List[str] = Field(
        default_factory=list,
        description="Task ids from tasks_for_lats that were completed this turn (bot actually did the task in its reply).",
    )
    attempted_task_ids: List[str] = Field(
        default_factory=list,
        description="Task ids from tasks_for_lats that were attempted or in scope this turn.",
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class DetectionOutput(BaseModel):
    model_config = _pydantic_extra_forbid

    """Detection 节点 LLM 输出：0-10 整数 + subtext。"""

    hostility_level: int = Field(0, ge=0, le=10, description="0-10")
    engagement_level: int = Field(0, ge=0, le=10, description="0-10")
    topic_appeal: int = Field(0, ge=0, le=10, description="0-10")
    stage_pacing: Literal["正常", "过分亲密", "过分生疏"] = Field(
        "正常",
        description="关系节奏：正常=与当前阶段匹配；过分亲密=交浅言深；过分生疏=突然冷淡/回避。",
    )
    urgency: int = Field(0, ge=0, le=10, description="0-10")
    subtext: str = Field("", description="用户潜台词/意图简述")


# ---------------------------------------------------------------------------
# Inner Monologue
# ---------------------------------------------------------------------------


class InnerMonologueOutput(BaseModel):
    model_config = _pydantic_extra_forbid

    """Inner monologue 节点 LLM 输出。"""

    monologue: str = Field("", description="内心独白文本")
    selected_profile_keys: List[str] = Field(default_factory=list, description="从 profile 中选中的键名")


# ---------------------------------------------------------------------------
# Processor (humanize segments)
# ---------------------------------------------------------------------------


class ProcessorSegment(BaseModel):
    model_config = _pydantic_extra_forbid

    content: str = Field(..., description="单条气泡文本")
    delay: float = Field(0.5, ge=0, le=60, description="秒")
    action: Literal["typing", "idle"] = Field("typing")


class ProcessorOutput(BaseModel):
    model_config = _pydantic_extra_forbid

    """Processor 节点 LLM 输出：分段回复。"""

    segments: List[ProcessorSegment] = Field(..., description="多条气泡")
    is_macro_delay: bool = Field(False)
    macro_delay_seconds: float = Field(0.0, ge=0, le=60)


# ---------------------------------------------------------------------------
# Task Planner
# ---------------------------------------------------------------------------


class TaskPlannerOutput(BaseModel):
    model_config = _pydantic_extra_forbid

    """Task planner 节点 LLM 输出。"""

    word_budget: int = Field(60, ge=0, le=60)
    task_budget_max: int = Field(2, ge=0, le=2)
    top2_indices: List[int] = Field(default_factory=list, description="最多 2 个候选索引")
    random_index: Optional[int] = Field(None, description="额外随机一个候选索引")


# ---------------------------------------------------------------------------
# Reply Planner
# ---------------------------------------------------------------------------


class ReplyPlannerSingle(BaseModel):
    model_config = _pydantic_extra_forbid

    """k=1 时单条回复。"""

    reply: str = Field(..., description="一条完整回复文本")


class ReplyPlanCandidate(BaseModel):
    model_config = _pydantic_extra_forbid

    reply: str = Field(..., description="一条完整回复文本")


class ReplyPlannerCandidates(BaseModel):
    model_config = _pydantic_extra_forbid

    """k>1 时多候选。"""

    candidates: List[ReplyPlanCandidate] = Field(..., description="多条候选回复")


# ---------------------------------------------------------------------------
# Strategy Router
# ---------------------------------------------------------------------------


class StrategyRouterOutput(BaseModel):
    model_config = _pydantic_extra_forbid

    """策略路由 LLM 输出：命中策略 id 或 null。"""

    hit: Optional[str] = Field(None, description="命中的策略 id，未命中为 null")


# ---------------------------------------------------------------------------
# Memory Manager
# ---------------------------------------------------------------------------


class TranscriptMeta(BaseModel):
    model_config = _pydantic_extra_forbid

    entities: List[str] = Field(default_factory=list)
    topic: Optional[str] = None
    importance: float = Field(0.0, ge=0, le=1)
    short_context: Optional[str] = None


class MemoryNote(BaseModel):
    model_config = _pydantic_extra_forbid

    note_type: Literal["fact", "preference", "activity", "decision", "other"] = "other"
    content: str = Field(...)
    importance: float = Field(0.5, ge=0, le=1)


class BasicInfoFields(BaseModel):
    model_config = _pydantic_extra_forbid

    """basic_info 五字段，均为可选。"""

    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    location: Optional[str] = None


class BasicInfoConfidence(BaseModel):
    model_config = _pydantic_extra_forbid

    """basic_info 各字段置信度 0~1，符合 OpenAI 严格 schema（无 Dict）。"""

    name: Optional[float] = Field(None, ge=0, le=1)
    age: Optional[float] = Field(None, ge=0, le=1)
    gender: Optional[float] = Field(None, ge=0, le=1)
    occupation: Optional[float] = Field(None, ge=0, le=1)
    location: Optional[float] = Field(None, ge=0, le=1)


class InferredEntry(BaseModel):
    model_config = _pydantic_extra_forbid

    """单条 inferred 画像条目，符合 OpenAI 严格 schema。"""

    key: str = Field(..., description="画像键名")
    value: str = Field(..., description="画像值")


class MemoryManagerOutput(BaseModel):
    model_config = _pydantic_extra_forbid

    """Memory manager 节点 LLM 输出。"""

    new_summary: str = Field("", description="本轮更新后的对话摘要")
    transcript_meta: TranscriptMeta = Field(default_factory=TranscriptMeta)
    notes: List[MemoryNote] = Field(default_factory=list)
    basic_info_updates: BasicInfoFields = Field(default_factory=BasicInfoFields)
    basic_info_confidence: BasicInfoConfidence = Field(default_factory=BasicInfoConfidence)
    basic_info_evidence: BasicInfoFields = Field(default_factory=BasicInfoFields)
    new_inferred_entries: List[InferredEntry] = Field(default_factory=list, description="少而精的画像条目")
    new_topics: List[str] = Field(default_factory=list)
    # 仅当用户性别未知时，可根据本轮对话推断并填写（男/女/其他）；否则留空
    inferred_user_gender: Optional[str] = Field(default=None, description="If user gender is unknown, infer from conversation; otherwise leave empty.")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class DimensionScore(BaseModel):
    model_config = _pydantic_extra_forbid

    """单维度得分，符合 OpenAI 严格 schema（无 Dict）。"""

    dimension: str = Field(..., description="维度名")
    score: float = Field(0.0, ge=0, le=1)


class EvaluatorSoftScore(BaseModel):
    model_config = _pydantic_extra_forbid

    """Soft scorer 单条：overall_score + score_breakdown。"""

    overall_score: float = Field(0.0, ge=0, le=1)
    score_breakdown: List[DimensionScore] = Field(default_factory=list, description="各维度得分")


class EvaluatorSoftScoreItem(BaseModel):
    model_config = _pydantic_extra_forbid

    idx: int = Field(...)
    overall_score: float = Field(0.0, ge=0, le=1)
    score_breakdown: List[DimensionScore] = Field(default_factory=list)


class EvaluatorSoftScoreBatch(BaseModel):
    model_config = _pydantic_extra_forbid

    results: List[EvaluatorSoftScoreItem] = Field(default_factory=list)


class Gate1ResultItem(BaseModel):
    model_config = _pydantic_extra_forbid

    idx: int = Field(...)
    assistantiness_ok: bool = Field(...)
    identity_ok: bool = Field(...)
    immersion_ok: bool = Field(...)


class EvaluatorGate1Batch(BaseModel):
    model_config = _pydantic_extra_forbid

    results: List[Gate1ResultItem] = Field(default_factory=list)


class EvaluatorJudgeResult(BaseModel):
    model_config = _pydantic_extra_forbid

    """单条 judge：score + sub_scores（维度名不固定）。"""

    score: float = Field(0.0, ge=0, le=1)
    sub_scores: List[DimensionScore] = Field(default_factory=list)


class EvaluatorJudgeBatchItem(BaseModel):
    model_config = _pydantic_extra_forbid

    idx: int = Field(...)
    score: float = Field(0.0, ge=0, le=1)
    sub_scores: List[DimensionScore] = Field(default_factory=list)


class EvaluatorJudgeBatch(BaseModel):
    model_config = _pydantic_extra_forbid

    results: List[EvaluatorJudgeBatchItem] = Field(default_factory=list)


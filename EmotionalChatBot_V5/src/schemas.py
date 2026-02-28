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

    """Detection 节点 LLM 输出：4 维客观量（去掉 topic_appeal 和 subtext，这两个由 extract 吸收）。"""

    hostility_level: int = Field(0, ge=0, le=10, description="0-10")
    engagement_level: int = Field(0, ge=0, le=10, description="0-10")
    stage_pacing: Literal["正常", "过分亲密", "过分生疏"] = Field(
        "正常",
        description="关系节奏：正常=与当前阶段匹配；过分亲密=交浅言深；过分生疏=突然冷淡/回避。",
    )
    urgency: int = Field(0, ge=0, le=10, description="0-10")
    knowledge_gap: bool = Field(
        False,
        description="用户问事实/数据/现状/发生了什么，或提及专有名词/生僻词/不确定需查证的内容，或涉及当前/未来可能变化的事实（价格、人事、政策、赛事、天气等）需查证→true；仅纯情绪/观点/打招呼/闲聊→false。拿不准倾向true。",
    )
    search_keywords: str = Field(
        "",
        description="knowledge_gap=True 时从问题提炼 3–8 字关键词（中文），可含专有名词或时效词；否则留空。",
    )


# ---------------------------------------------------------------------------
# Inner Monologue
# ---------------------------------------------------------------------------


class InnerMonologueOutput(BaseModel):
    # 允许 LLM 多返回 reasoning/target_mode_id 等，仅取 monologue，避免 extra forbid 报错
    model_config = ConfigDict(extra="ignore")

    """Inner monologue 节点 LLM 输出：只输出纯文本独白（结构化提取由 extract 节点完成）。"""

    monologue: str = Field("", description="内心独白文本，600-1200字符，只写感受/态度/意愿，不做分类或选择")



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
    new_inferred_entries: List[InferredEntry] = Field(default_factory=list, description="用户画像条目：偏好/习惯/职业特征/个人情况等")
    new_topics: List[str] = Field(default_factory=list)
    # 仅当用户性别未知时，可根据本轮对话推断并填写（男/女/其他）；否则留空
    inferred_user_gender: Optional[str] = Field(default=None, description="If user gender is unknown, infer from conversation; otherwise leave empty.")
    # 仅在会话边界（新会话第一轮，有旧摘要）时填写：对上一个 session 的最大自我披露深度评估（1-5）；其他情况留 null
    spt_depth_last_session: Optional[int] = Field(default=None, ge=1, le=5, description="Max SPT depth of previous session (1-5); only fill at session boundary")
    # 仅在会话边界时填写：上一会话的精华摘要（500-1000字），比 new_summary 更侧重保留关键情感节点/承诺/事实；其他情况留 null
    session_summary: Optional[str] = Field(default=None, description="Rich session-level summary of previous session (500-1000 chars); only fill at session boundary")


# ---------------------------------------------------------------------------
# Monologue Extract（新架构：extract 节点）
# ---------------------------------------------------------------------------


class MonologueExtractOutput(BaseModel):
    # 允许 LLM 多返回 reasoning/target_mode_id 等，仅取约定字段，避免 extra forbid 报错
    model_config = ConfigDict(extra="ignore")

    """从内心独白中结构化提取信号，同时完成 profile_keys 选择和 move_ids 选择。"""

    emotion_tag: str = Field("", description="当前情绪标签，如 心疼/烦躁/期待/无聊/开心 等")
    attitude: str = Field("", description="对用户的态度倾向，如 主动配合/被动应付/想转移话题/好奇/享受 等")
    momentum_delta: float = Field(0.0, ge=-1.0, le=1.0, description="冲量变化量 -1.0~+1.0，正=想继续，负=想结束")
    topic_appeal: float = Field(5.0, ge=0.0, le=10.0, description="话题吸引力 0-10（替换旧 detection.topic_appeal）")
    subtext_guess: str = Field("", description="对用户潜台词的猜测，无则空字符串")
    selected_profile_keys: List[str] = Field(default_factory=list, description="当前最相关的用户画像键名，0-5个")
    selected_content_move_ids: List[int] = Field(
        default_factory=list,
        description="当轮选中的 content move id，2-4个，对应 content_moves.yaml 中 pure_content_transformations 的 id",
    )
    inferred_gender: Optional[str] = Field(
        None,
        description="从对话上下文推断的用户性别（男/女/其他）；性别已知时输出 null",
    )


# ---------------------------------------------------------------------------
# Judge（新架构：judge 节点）
# ---------------------------------------------------------------------------


class JudgeOutput(BaseModel):
    # 允许 LLM 多返回 reasoning 等，且缺 winner_index 时兜底 0，减少 fallback 噪音
    model_config = ConfigDict(extra="ignore")

    """Judge 节点 LLM 输出：从所有候选中选出最符合内心独白的那条。"""

    winner_index: int = Field(0, ge=0, description="generation_candidates 列表中最优候选的索引")
    justification: str = Field("", description="简短说明为什么选这条（对照独白的态度/情绪）")


# ---------------------------------------------------------------------------
# Safety（新架构：safety 节点）
# ---------------------------------------------------------------------------


class SafetyOutput(BaseModel):
    model_config = _pydantic_extra_forbid

    """Safety 节点 LLM 输出：是否触发安全策略。"""

    triggered: bool = Field(False, description="是否触发安全层（注入/脱角色/高危诉求）")
    strategy_id: Optional[str] = Field(None, description="命中的策略 id，未触发为 null")
    reason: str = Field("", description="触发原因简述")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
# (Gate1 / evaluate_candidate 已移除，主流程使用 evaluate_27_candidates_single_llm)


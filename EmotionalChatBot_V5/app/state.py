"""
【状态层】定义 LangGraph Agent 的全局状态。
支持高度拟人化的 AI 聊天机器人，包含大五人格、动态人设、6维关系模型和PAD情绪模型。
"""
from typing import TypedDict, List, Dict, Any, Optional, Union, Annotated, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def _merge_profile(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """合并并行节点写入的 _profile（拼接 nodes 列表），供 LTSR_PROFILE_STEPS 使用。"""
    a = left or {}
    b = right or {}
    nodes = list(a.get("nodes") or []) + list(b.get("nodes") or [])
    return {**a, **b, "nodes": nodes}

# 时间戳在 state 中通常为 ISO 字符串，便于 JSON 序列化
TimestampStr = str


# ==========================================
# 0. Knapp 关系阶段定义 (Relationship Stages)
# ==========================================

# Knapp 十阶段类型定义
# 阶段元数据请从 config/stages/*.yaml 文件加载，使用 utils.yaml_loader.load_stage_by_id()
KnappStage = Literal[
    "initiating",      # Stage 1: 起始
    "experimenting",   # Stage 2: 探索
    "intensifying",    # Stage 3: 加深
    "integrating",     # Stage 4: 融合
    "bonding",         # Stage 5: 承诺
    "differentiating", # Stage 6: 分化
    "circumscribing",  # Stage 7: 限缩
    "stagnating",      # Stage 8: 停滞
    "avoiding",        # Stage 9: 回避
    "terminating"      # Stage 10: 结束
]


# ==========================================
# 1. 机器人核心定义 (Identity & Soul)
# ==========================================

class BotBasicInfo(TypedDict):
    """机器人的硬性身份信息 (Static Facts)"""
    name: str
    gender: str
    age: int
    region: str
    occupation: str
    education: str
    native_language: str
    # 核心设定，如 "说话喜欢用倒装句"
    speaking_style: str


class BotBigFive(TypedDict):
    """机器人的大五人格基准 (用于计算性格底色) - Range: [-1.0, 1.0]"""
    openness: float          # O: 开放性 (脑洞 vs 现实)
    conscientiousness: float # C: 尽责性 (严谨 vs 随性)
    extraversion: float      # E: 外向性 (热情 vs 内向)
    agreeableness: float     # A: 宜人性 (配合 vs 毒舌)
    neuroticism: float       # N: 神经质 (情绪波动率)


class BotPersona(TypedDict, total=False):
    """
    机器人的动态人设 (Dynamic & Semi-structured)
    使用松散结构以便动态增删爱好、经历等
    """
    # 键值对属性 (e.g. {"fav_color": "Blue", "catchphrase": "Just kidding"})
    attributes: Dict[str, str]
    
    # 集合列表 (e.g. {"hobbies": ["Skiing", "Painting"], "skills": ["Python"]})
    collections: Dict[str, List[str]]
    
    # 背景故事片段 (e.g. {"origin": "Born in Mars...", "secret": "..."})
    lore: Dict[str, str]


# ==========================================
# 2. 用户侧写 (User Profiling)
# ==========================================

class UserBasicInfo(TypedDict):
    """用户的显性信息 (Explicit Facts)"""
    name: Optional[str]
    gender: Optional[str]
    age: Optional[int]
    location: Optional[str]
    occupation: Optional[str]


class UserInferredProfile(TypedDict, total=False):
    """
    AI 分析出的用户隐性侧写 (Inferred by Analyzer)，用于校准 Bot 的态度。
    无固定字段，为可扩展 JSON；下游将整块注入 prompt。
    """
    pass


# ==========================================
# 3. 心理与关系引擎 (Physics Engine)
# ==========================================

class RelationshipState(TypedDict):
    """
    6维核心关系属性 (The Essential 6) - Range: [0, 100]
    决定了 Bot 对 User 的'态度'
    """
    closeness: float  # 亲密 (陌生 -> 熟人)
    trust: float      # 信任 (防备 -> 依赖)
    liking: float     # 喜爱 (工作伙伴 -> 喜欢的伙伴)
    respect: float    # 尊重 (损友 -> 导师)
    warmth: float     # 暖意 (高冷 -> 热情)
    power: float      # 权力 (Bot处于弱势 -> Bot处于强势/支配)


class MoodState(TypedDict):
    """
    当前情绪状态 (Transient State)
    """
    # PAD 情绪模型 - Range: [-1.0, 1.0]
    pleasure: float   # P: 愉悦度
    arousal: float   # A: 唤醒度/激动度
    dominance: float # D: 掌控感
    
    # 繁忙度/资源限制 - Range: [0.0, 1.0]
    # > 0.8 时会强制缩短回复
    busyness: float


# ==========================================
# 3.4 任务 (Bot Task)
# ==========================================

class Task(TypedDict, total=False):
    """
    Bot 的单项任务：用于待办/提醒/跟进等。
    - id: 唯一标识（UUID 字符串）
    - task_type: 任务类型（如 remind / follow_up / ask / custom）
    - description: 自然语言描述
    - importance: 重要性（0-1 或 1-5，由业务约定）
    - created_at: 创建时间（ISO 字符串）
    - expires_at: 有效期/截止时间（ISO 字符串，可选）
    - last_attempt_at: 上次尝试时间（ISO 字符串，可选）
    - attempt_count: 尝试次数（非负整数）
    """
    id: str
    task_type: str
    description: str
    importance: float
    created_at: TimestampStr
    expires_at: Optional[TimestampStr]
    last_attempt_at: Optional[TimestampStr]
    attempt_count: int


# ==========================================
# 3.5 拟人化行为表现层 (Behavioral Layer)
# ==========================================

class ResponseSegment(TypedDict):
    """单个回复气泡的定义（给客户端执行的“剧本”）。"""
    content: str
    delay: float  # 发送该气泡前的等待时间（秒，相对于上一条气泡）
    action: Literal["typing", "idle"]  # typing=显示“正在输入…”，idle=长离线纯等待


class HumanizedOutput(TypedDict, total=False):
    """Processor 节点输出：包含分段与延迟的拟人化结果（含宏观长延迟门控）。"""
    total_latency_seconds: float  # 总响应耗时（秒）
    segments: List[ResponseSegment]
    is_macro_delay: bool  # 是否触发宏观长延迟（如睡眠/忙碌/策略性沉默）

    # --- Optional debug/backward-compat fields ---
    total_latency_simulated: float
    latency_breakdown: Dict[str, float]


# 兼容旧命名：TimelineSegment（老代码/文档可能引用）
TimelineSegment = ResponseSegment


# ==========================================
# 3.5.1 LATS / Choreography Planning Schemas
# ==========================================

ReplyMsgFunction = Literal[
    "empathy",      # 共情/安抚
    "stance",       # 站队/态度
    "answer",       # 直接回应/结论
    "explain",      # 解释原因/背景
    "advice",       # 给建议/行动
    "boundary",     # 边界/拒绝
    "question",     # 反问/追问
    "closing",      # 收束/邀约继续
]

PauseType = Literal["none", "thinking", "polite", "beat", "long"]
DelayBucket = Literal["instant", "short", "medium", "long", "offline"]


class ReplyPlanMessage(TypedDict, total=False):
    """场景化编排下的单条消息计划（不是最终 UI 消息，但应接近可落地文本）。"""
    id: str
    function: ReplyMsgFunction
    content: str
    key_points: List[str]
    target_length: int
    info_density: Literal["low", "medium", "high"]
    pause_after: PauseType
    delay_bucket: DelayBucket


class ReplyPlan(TypedDict, total=False):
    """ReplyPlanner 的输出：对话编排计划（A:意图 + B:节奏），供编译器生成可执行 ProcessorPlan。"""
    intent: str
    speech_act: str
    # 变体生成用：显式策略标签（用于强制候选多样性，而非只做同义改写）
    strategy_tag: Optional[str]
    stakes: Literal["low", "medium", "high"]
    first_message_role: ReplyMsgFunction
    pacing_strategy: str
    # 生成端硬结构：用于避免“先生成再惩罚”，让 planner 显式思考预算与覆盖分配
    messages_count: int
    messages: List[ReplyPlanMessage]
    # must_cover_points -> message.id 映射（用于对齐计划目标与消息分配）
    must_cover_map: Dict[str, str]
    justification: str  # 简短自我解释（为什么这样编排）
    # 任务结算（最小闭环）：由 LATS/ReplyPlanner 输出，供 evolver 在本轮结束时结算
    attempted_task_ids: List[str]
    completed_task_ids: List[str]


class ProcessorPlan(TypedDict, total=False):
    """可执行落地计划：最终 messages/delays/actions（必须可被 processor 执行器直接消费）。"""
    messages: List[str]
    delays: List[float]  # 相对上一条的等待秒数
    actions: List[Literal["typing", "idle"]]
    meta: Dict[str, Any]  # 例如拆分原因、来源范围、关键点分配映射等


class RequirementsChecklist(TypedDict, total=False):
    """本轮必须满足的硬约束/需求清单（LATS 的硬门槛与 must-have 主要来源）。"""
    must_have: List[str]
    forbidden: List[str]
    safety_notes: List[str]
    first_message_rule: str
    max_messages: int
    min_first_len: int
    max_message_len: int
    stage_pacing_notes: str
    # requirements_policy 相关字段
    must_have_policy: str  # "soft" | "none"
    must_have_min_coverage: float
    allow_short_reply: bool
    allow_empty_reply: bool
    # 明确的约束清单（来自 reasoner/style/stage）
    plan_goals: Dict[str, Any]  # {"must_cover_points": List[str], "avoid_points": List[str]}
    style_targets: Dict[str, float]  # 12维目标（verbal_length, social_distance, tone_temperature, etc.）
    stage_targets: Dict[str, Any]  # {"stage": str, "pacing_notes": List[str], "violation_sensitivity": float}
    # mode 行为策略（用于让 mode 不止约束条数/长度，而是进入可评估目标）
    mode_behavior_targets: List[str]


class EvalCheckFailure(TypedDict, total=False):
    id: str
    reason: str
    evidence: str


class SimReport(TypedDict, total=False):
    """evaluator 的结构化输出：可直接作为 reward 与调参诊断依据。"""
    found_solution: bool
    eval_score: float
    failed_checks: List[EvalCheckFailure]
    score_breakdown: Dict[str, float]
    improvement_notes: List[str]
    # LLM soft scorer 状态：ok / failed / skipped（便于诊断 llm_overall=None 的来源）
    llm_status: str
    # 可选：LLM soft scorer 的结构化证据/逐条对齐输出（便于调参、反思与可视化）
    llm_details: Dict[str, Any]


# ==========================================
# 3.6 关系资产层 (Relationship Assets)
# ==========================================

class RelationshipAssets(TypedDict):
    """
    关系资产（可序列化的长期积累数据）：
    - topic_history: 话题历史（用 List 存储，便于 JSON 序列化）
    - breadth_score: 话题广度（unique topic 数量）
    - max_spt_depth: 历史最高自我暴露深度
    """
    topic_history: List[str]
    breadth_score: int
    max_spt_depth: int


# ==========================================
# 3.7 SPT 信息 (Stage Manager 用)
# ==========================================

class SPTInfo(TypedDict, total=False):
    """社会穿透理论相关输入（用于阶段门控）。"""
    depth: int  # 1-4
    breadth: int  # 话题数量（unique）
    topic_list: List[str]
    depth_trend: str  # "stable" | "increasing" | "decreasing"
    recent_signals: List[str]


# ==========================================
# 4. 主状态定义 (Main Agent State)
# ==========================================

class AgentState(TypedDict, total=False):
    """
    Agent 主状态：支持高度拟人化的 AI 聊天机器人
    
    架构说明：
    - Identity: BotBasicInfo, BigFive, BotPersona (我是谁)
    - Perception: UserInferredProfile (我看你是谁)
    - Physics: RelationshipState, MoodState (我们的关系和我的心情)
    - Memory: chat_buffer, conversation_summary, retrieved_memories
    - Output: llm_instructions (12维输出驱动), final_response
    """

    # --- Input Context ---
    messages: Annotated[List[BaseMessage], add_messages]  # 对话消息列表（LangGraph 自动合并）
    user_input: str
    # 仅外部可见的用户文本（防止 internal prompt/debug 污染 user_input）
    external_user_text: Optional[str]
    current_time: str
    # 业务语义时间戳（用于 DB 写入 created_at 语义）
    user_received_at: Optional[str]
    ai_sent_at: Optional[str]
    user_id: str  # 用户ID，用于数据库查询
    bot_id: str   # Bot 的外部/固定ID（用于 DB/本地持久化定位同一条关系）

    # --- Persistence Identifiers (optional) ---
    # relationship_id: 当前「Bot 下用户」的主键（loader 从 DB 读取后填充，即 users.id）
    relationship_id: Optional[str]
    # 会话/线程标识：用于 Store A 以 session/thread/turn 组织原文（可由入口注入）
    session_id: Optional[str]
    thread_id: Optional[str]
    turn_index: Optional[int]
    
    # --- Static/Semi-Static Profiles (Loaded from DB) ---
    bot_basic_info: BotBasicInfo
    bot_big_five: BotBigFive
    bot_persona: BotPersona
    
    user_basic_info: UserBasicInfo
    user_inferred_profile: UserInferredProfile
    
    # --- Dynamic Core (Read/Write frequently) ---
    relationship_state: RelationshipState
    mood_state: MoodState
    # --- Relationship Assets (长期积累资产，可选) ---
    relationship_assets: Optional[RelationshipAssets]
    # --- SPT Info (可选，供 stage_manager 判定) ---
    spt_info: Optional[SPTInfo]
    # --- Stage Manager 输出（可选） ---
    stage_narrative: Optional[str]
    stage_transition: Optional[Dict[str, Any]]
    
    # --- Knapp Relationship Stage ---
    # 当前关系阶段（根据 Knapp 理论模型）
    current_stage: KnappStage

    # --- Bot 任务清单（读写数据库）---
    # 该 Bot 对应当前用户的完整任务列表，由 loader 从 DB 加载，可由节点更新后经 save_turn 写回
    bot_task_list: List[Task]
    # 当前会话要处理的任务子集，通常 0-3 条，仅内存使用不持久化
    current_session_tasks: List[Task]

    # --- TaskPlanner 输出（LATS 之前节点写入，供 LATS / reply_planner 使用）---
    # 本轮交给 LATS 的至多 3 条自然语言任务（带 id 便于回写完成）
    tasks_for_lats: List[Dict[str, Any]]  # [{"id": str, "description": str, "task_type": str?}, ...]
    # 本轮允许完成的任务数 0/1/2；0 时仍可带任务包，但倾向“隐式完成”
    task_budget_max: int
    # 回复字数上限（0-60）
    word_budget: int
    # 第三任务加权随机时的温度（可选）
    completion_temperature: Optional[float]

    # --- Task settlement (from LATS/ReplyPlanner; evolver consumes) ---
    attempted_task_ids: Optional[List[str]]
    completed_task_ids: Optional[List[str]]

    # --- Writer flags ---
    # Web 并发输入：用户消息可能已提前逐条落库，此时 save_turn 不应再写入 user_input（避免出现合并 user 行）
    skip_user_message_write: Optional[bool]
    
    # --- Memory System ---
    # 短期记忆窗口 (最近 10-20 条)
    chat_buffer: List[BaseMessage]
    # 长期记忆摘要
    conversation_summary: str
    # RAG 检索到的相关记忆 (事实 + 关键事件)
    retrieved_memories: List[str]
    # 检索是否成功（失败时下游 prompt 只用 summary，不拼旧 retrieved）
    retrieval_ok: Optional[bool]
    # 统一注入提示词的记忆块（chat_buffer + summary + retrieved 合并后的文本）
    memory_context: str
    
    # --- Analysis Artifacts (中间产物) ---
    # inner_monologue 节点输出：内心独白文本 + 选中的 inferred_profile 键名列表
    inner_monologue: Optional[str]
    selected_profile_keys: Optional[List[str]]
    response_strategy: Optional[str]
    # Analyzer 输出的意图
    user_intent: Optional[str]
    # Analyzer 输出的属性变化值 (Deltas)
    relationship_deltas: Optional[Dict[str, float]]
    # Relationship Engine：本轮 LLM 分析结果（包含 signals + raw deltas）
    latest_relationship_analysis: Optional[Dict[str, Any]]
    # Relationship Engine：本轮阻尼后实际应用的变化量（real change）
    relationship_deltas_applied: Optional[Dict[str, float]]
    
    # --- Detection（感知：scores/brief/stage_judge/immediate_tasks）---
    detection_signals: Optional[Dict[str, Any]]
    detection_scores: Optional[Dict[str, float]]   # friendly, hostile, overstep, low_effort, confusion
    detection_meta: Optional[Dict[str, int]]       # target_is_assistant, quoted_or_reported_speech
    detection_brief: Optional[Dict[str, Any]]      # gist, references, unknowns, subtext, understanding_confidence, reaction_seed
    detection_stage_judge: Optional[Dict[str, Any]]  # current_stage, implied_stage, delta, direction, evidence_spans
    detection_immediate_tasks: Optional[List[Dict[str, Any]]]  # 当轮任务，交给 planner 写入任务库
    # 紧急任务：Detection 产生的当轮必须执行的任务（直接注入 LATS，不参与打分）
    detection_urgent_tasks: Optional[List[Dict[str, Any]]]
    # 紧急任务：从 DB 加载的开发者/bot/user 级别紧急任务（直接注入 LATS，执行后从 DB 删除）
    db_urgent_tasks: Optional[List[Dict[str, Any]]]
    # 内部标记：本轮是否消费了 DB 紧急任务（供 save_turn 清除用）
    _urgent_tasks_consumed: Optional[bool]
    # 兼容/日志用（不再用于路由；路由改由 word_budget/no_reply）
    detection_result: Optional[str]
    detection_category: Optional[str]
    # 是否本轮不回复（由 task_planner 在 word_budget=0 时设置，graph 条件边短路）
    no_reply: Optional[bool]
    # 直觉思考：由 Inner Monologue 节点生成（原 detection 的“先想再分类”现移入 inner_monologue）
    intuition_thought: Optional[str]
    # 关系滤镜：由 Inner Monologue 生成，此刻对 TA 的主观关系感受（字符串，非 relationship_state 数值）
    relationship_filter: Optional[str]
    # 安全检测结果：由 SecurityCheck 节点生成，用于路由到安全响应节点
    # {"is_injection_attempt": bool, "is_ai_test": bool, "is_user_treating_as_assistant": bool, "needs_security_response": bool, "reasoning": str}
    security_check: Optional[Dict[str, Any]]
    
    # --- Mode Management ---
    # 当前模式 ID（由 mode_manager 节点确定）
    mode_id: Optional[str]
    # 当前模式配置对象（PsychoMode，包含 behavior_contract, lats_budget, requirements_policy 等）
    current_mode: Optional[Any]  # PsychoMode 类型，但避免循环导入

    # --- Profiling (devtools) ---
    # 节点级耗时与 LLM 调用增量（由 app/graph.py 的 profiling wrapper 写入；devtools 使用）
    # 并行节点（如 detection + inner_monologue）会同时写入，用 reducer 合并 nodes 列表
    _profile: Annotated[Optional[Dict[str, Any]], _merge_profile]
    
    # --- Output Drivers (The 12 Dimensions) ---
    # 这里的 Key 对应 12 维输出定义
    # Strategy 维度:
    #   - self_disclosure: 自我暴露程度
    #   - topic_adherence: 话题粘性
    #   - initiative: 主动性
    #   - advice_style: 建议风格
    #   - subjectivity: 主观性
    #   - memory_hook: 记忆钩子
    # Style 维度:
    #   - verbal_length: 语言长度
    #   - social_distance: 社交距离
    #   - tone_temperature: 语调温度
    #   - emotional_display: 情绪表达
    #   - wit_and_humor: 机智幽默
    #   - non_verbal_cues: 非语言 cues
    llm_instructions: Dict[str, Any]

    # --- LATS / Choreography Planning (NEW) ---
    # 从 reasoner/style/mode 编译出的硬约束/需求清单
    requirements: Optional[RequirementsChecklist]
    # 作为风格/节奏画像的统一口径（默认可与 llm_instructions 同步）
    style_profile: Optional[Dict[str, Any]]

    # 当前候选与其编排计划（用于搜索与调试）
    candidate_text: Optional[str]
    reply_plan: Optional[ReplyPlan]
    processor_plan: Optional[ProcessorPlan]
    sim_report: Optional[SimReport]

    # LATS 搜索树与统计（保持灵活，便于迭代）
    lats_tree: Optional[Dict[str, Any]]
    lats_best_id: Optional[str]
    lats_rollouts: Optional[int]
    lats_expand_k: Optional[int]
    # 是否启用 LLM soft scorer（用于 plan_alignment/style/stage/memory/persona/relationship 等“可优化维度”评审）
    lats_enable_llm_soft_scorer: Optional[bool]
    # LATS 两阶段评审参数（可选；必须进入 AgentState 否则会被 LangGraph 丢弃）
    lats_llm_soft_top_n: Optional[int]
    lats_llm_soft_max_concurrency: Optional[int]
    lats_assistant_check_top_n: Optional[int]
    # LATS 早退/停止相关阈值（必须进入 AgentState，否则 LangGraph 传播会丢字段）
    lats_early_exit_root_score: Optional[float]
    lats_early_exit_plan_alignment_min: Optional[float]
    lats_early_exit_assistantiness_max: Optional[float]
    lats_early_exit_mode_fit_min: Optional[float]
    # bot-to-bot/压测用：禁用早退（强制至少跑完 rollouts）
    lats_disable_early_exit: Optional[bool]
    # 可选：低风险回合跳过 LATS rollout 搜索（只用根计划）
    lats_skip_low_risk: Optional[bool]
    # P0：至少跑完 N 次 rollout 才允许 early-exit（root_plan 直接早退）。
    # - 默认 1：避免“树永远不长”
    # - 设为 0：允许 root_plan 直接早退（旧行为）
    lats_min_rollouts_before_early_exit: Optional[int]
    
    # --- Final Output ---
    final_response: str
    
    # --- Legacy/Compatibility Fields (可选，用于向后兼容) ---
    # 如果现有代码依赖这些字段，可以保留
    deep_reasoning_trace: Optional[Dict[str, Any]]  # 思考过程 (Reasoner)
    style_analysis: Optional[str]  # 风格分析 (Styler)
    draft_response: Optional[str]  # 初稿 (Generator)
    critique_feedback: Optional[str]  # 批评意见 (Critic)
    retry_count: Optional[int]  # 重试次数
    final_segments: Optional[List[str]]  # 最终分段
    final_delay: Optional[float]  # 最终延迟
    # Processor 开关：是否启用 LLM 做拆句与节奏（必须进 AgentState，否则 LangGraph 传播会丢字段）
    processor_use_llm: Optional[bool]

    # --- Behavioral Layer Output (Processor) ---
    # 更细粒度的“拟人化输出”，供客户端按 delay 播放打字/气泡
    humanized_output: Optional[HumanizedOutput]

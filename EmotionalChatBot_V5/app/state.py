"""
【状态层】定义 LangGraph Agent 的全局状态。
支持高度拟人化的 AI 聊天机器人，包含大五人格、动态人设、6维关系模型和PAD情绪模型。
"""
from typing import TypedDict, List, Dict, Any, Optional, Union, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


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
    nickname: Optional[str]
    gender: Optional[str]
    age_group: Optional[str]
    location: Optional[str]
    occupation: Optional[str]


class UserInferredProfile(TypedDict):
    """
    AI 分析出的用户隐性侧写 (Inferred by Analyzer)
    用于校准 Bot 的态度
    """
    # 沟通风格 (e.g. "casual, uses emojis, short")
    communication_style: str 
    # 表达欲基准 (low/medium/high) - 用于判断用户情绪权重
    expressiveness_baseline: str
    # 兴趣图谱 (用于主动发起话题)
    interests: List[str]
    # 雷区/禁忌 (用于 Guardrail)
    sensitive_topics: List[str]


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
    current_time: str
    user_id: str  # 用户ID，用于数据库查询
    
    # --- Static/Semi-Static Profiles (Loaded from DB) ---
    bot_basic_info: BotBasicInfo
    bot_big_five: BotBigFive
    bot_persona: BotPersona
    
    user_basic_info: UserBasicInfo
    user_inferred_profile: UserInferredProfile
    
    # --- Dynamic Core (Read/Write frequently) ---
    relationship_state: RelationshipState
    mood_state: MoodState
    
    # --- Memory System ---
    # 短期记忆窗口 (最近 10-20 条)
    chat_buffer: List[BaseMessage]
    # 长期记忆摘要
    conversation_summary: str
    # RAG 检索到的相关记忆 (事实 + 关键事件)
    retrieved_memories: List[str]
    
    # --- Analysis Artifacts (中间产物) ---
    # Analyzer 输出的意图
    user_intent: Optional[str]
    # Analyzer 输出的属性变化值 (Deltas)
    relationship_deltas: Optional[Dict[str, float]]
    
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

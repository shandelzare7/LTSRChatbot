"""【状态层】定义 LangGraph Agent 的全局状态。"""
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.core.mode_base import PsychoMode


class AgentState(TypedDict, total=False):
    """Agent 状态：消息、用户、当前模式、记忆与各阶段产物。"""

    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str

    # 核心：当前激活的心理模式对象
    current_mode: PsychoMode

    # 记忆数据
    user_profile: dict
    memories: str

    # 中间产物
    deep_reasoning_trace: dict  # 思考过程 (Reasoner)
    style_analysis: str  # 风格分析 (Styler)
    draft_response: str  # 初稿 (Generator)
    critique_feedback: str  # 批评意见 (Critic)
    retry_count: int

    # 最终产物 (Processor -> 输出)
    final_segments: List[str]
    final_delay: float

"""
LangGraph Chatbot 流程实现
包含安全检测、规划、生成和关系演化的完整流程
"""

from typing import List, Literal, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Agent 状态定义"""
    messages: Annotated[List[BaseMessage], add_messages]  # 对话消息列表
    relationship_stats: dict  # 亲密度等关系统计数据
    safety_flag: bool  # 安全检测结果
    plan: str  # 思考出的回复策略
    final_response: str  # 最终回复


def safety_check(state: AgentState) -> AgentState:
    """
    安全检测节点
    检查最后一条消息，如果通过则设置 safety_flag=True
    """
    messages = state.get("messages", [])
    
    if not messages:
        # 如果没有消息，直接标记为不安全
        return {**state, "safety_flag": False}
    
    last_message = messages[-1]
    
    # Mock 安全检测逻辑：简单检查是否包含敏感词
    sensitive_words = ["暴力", "色情", "违法"]
    message_content = last_message.content if hasattr(last_message, "content") else str(last_message)
    
    # 如果包含敏感词，标记为不安全
    is_safe = not any(word in message_content for word in sensitive_words)
    
    print(f"[安全检测] 消息内容: {message_content[:50]}...")
    print(f"[安全检测] 检测结果: {'通过' if is_safe else '未通过'}")
    
    return {**state, "safety_flag": is_safe}


def planner(state: AgentState) -> AgentState:
    """
    规划节点
    根据对话历史生成回复策略
    """
    messages = state.get("messages", [])
    relationship_stats = state.get("relationship_stats", {})
    
    # Mock 规划逻辑：根据消息数量和关系状态生成策略
    message_count = len(messages)
    intimacy = relationship_stats.get("intimacy", 0)
    
    if intimacy < 10:
        plan = f"友好初次接触策略：保持礼貌，询问基本信息（消息数: {message_count}）"
    elif intimacy < 50:
        plan = f"建立信任策略：分享共同话题，增加互动（消息数: {message_count}）"
    else:
        plan = f"深度交流策略：提供个性化建议，加强情感连接（消息数: {message_count}）"
    
    print(f"[规划器] 生成策略: {plan}")
    
    return {**state, "plan": plan}


def generator(state: AgentState) -> AgentState:
    """
    生成节点
    根据 plan 生成最终回复
    """
    plan = state.get("plan", "")
    messages = state.get("messages", [])
    
    # Mock 生成逻辑：根据策略和最后一条消息生成回复
    last_message = messages[-1] if messages else None
    last_content = last_message.content if last_message and hasattr(last_message, "content") else ""
    
    # 简单的回复生成逻辑
    if "初次接触" in plan:
        response = f"你好！很高兴认识你。你刚才说：{last_content[:30]}... 能告诉我更多关于你的信息吗？"
    elif "建立信任" in plan:
        response = f"我理解你的意思。关于'{last_content[:20]}...'这个话题，我们可以深入聊聊。"
    elif "深度交流" in plan:
        response = f"基于我们之前的对话，我建议：{last_content[:20]}... 你觉得怎么样？"
    else:
        response = f"我收到了你的消息：{last_content[:30]}... 让我想想如何回复你。"
    
    print(f"[生成器] 生成回复: {response}")
    
    return {**state, "final_response": response}


def evolver(state: AgentState) -> AgentState:
    """
    演化节点
    更新 relationship_stats（亲密度等）
    """
    relationship_stats = state.get("relationship_stats", {})
    messages = state.get("messages", [])
    
    # Mock 演化逻辑：根据消息数量增加亲密度
    current_intimacy = relationship_stats.get("intimacy", 0)
    message_count = len(messages)
    
    # 每次对话增加亲密度
    new_intimacy = current_intimacy + 5
    conversation_count = relationship_stats.get("conversation_count", 0) + 1
    
    updated_stats = {
        **relationship_stats,
        "intimacy": new_intimacy,
        "conversation_count": conversation_count,
        "last_message_count": message_count
    }
    
    print(f"[演化器] 更新关系统计: 亲密度={new_intimacy}, 对话次数={conversation_count}")
    
    return {**state, "relationship_stats": updated_stats}


def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """
    条件边函数
    根据 safety_flag 决定是否继续流程
    """
    safety_flag = state.get("safety_flag", False)
    
    if safety_flag:
        print("[条件判断] 安全检测通过，继续流程")
        return "continue"
    else:
        print("[条件判断] 安全检测未通过，结束流程")
        return "end"


def create_chatbot_graph() -> StateGraph:
    """
    创建并返回 Chatbot 流程图
    """
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("safety_check", safety_check)
    workflow.add_node("planner", planner)
    workflow.add_node("generator", generator)
    workflow.add_node("evolver", evolver)
    
    # 设置入口点
    workflow.set_entry_point("safety_check")
    
    # 添加条件边：从 safety_check 根据条件分支
    workflow.add_conditional_edges(
        "safety_check",
        should_continue,
        {
            "continue": "planner",  # 安全检测通过 -> 规划器
            "end": END  # 安全检测未通过 -> 结束
        }
    )
    
    # 添加顺序边：planner -> generator -> evolver -> 结束
    workflow.add_edge("planner", "generator")
    workflow.add_edge("generator", "evolver")
    workflow.add_edge("evolver", END)
    
    # 编译图
    app = workflow.compile()
    
    return app


if __name__ == "__main__":
    # 示例运行
    print("=" * 50)
    print("LangGraph Chatbot 流程示例")
    print("=" * 50)
    
    # 创建图
    app = create_chatbot_graph()
    
    # 初始化状态
    initial_state: AgentState = {
        "messages": [HumanMessage(content="你好，我想了解一下你的服务")],
        "relationship_stats": {"intimacy": 0, "conversation_count": 0},
        "safety_flag": False,
        "plan": "",
        "final_response": ""
    }
    
    print("\n初始状态:")
    print(f"  消息: {initial_state['messages'][0].content}")
    print(f"  关系统计: {initial_state['relationship_stats']}")
    print()
    
    # 运行图
    result = app.invoke(initial_state)
    
    print("\n最终状态:")
    print(f"  安全标志: {result['safety_flag']}")
    print(f"  规划策略: {result['plan']}")
    print(f"  最终回复: {result['final_response']}")
    print(f"  关系统计: {result['relationship_stats']}")
    print()
    
    # 测试不安全消息
    print("=" * 50)
    print("测试不安全消息")
    print("=" * 50)
    
    unsafe_state: AgentState = {
        "messages": [HumanMessage(content="这里包含暴力内容")],
        "relationship_stats": {"intimacy": 0, "conversation_count": 0},
        "safety_flag": False,
        "plan": "",
        "final_response": ""
    }
    
    result_unsafe = app.invoke(unsafe_state)
    print(f"\n最终状态:")
    print(f"  安全标志: {result_unsafe['safety_flag']}")
    print(f"  规划策略: {result_unsafe['plan']}")
    print(f"  最终回复: {result_unsafe['final_response']}")

"""
LangGraph Chatbot 流程实现
包含安全检测、规划、生成和关系演化的完整流程
"""

import os
import warnings

# 在导入任何可能触发警告的模块之前设置警告过滤
# 抑制 urllib3 的 OpenSSL 警告（这是 macOS 系统库的已知问题，不影响功能）
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:urllib3"
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")

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
    critic_feedback: str  # 检查员的反馈
    retry_count: int  # 重试次数


def safety_check(state: AgentState) -> AgentState:
    """
    安全检测节点
    检查最后一条消息，如果通过则设置 safety_flag=True
    """
    try:
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
    except Exception as e:
        print(f"[安全检测] 错误: {e}")
        return {**state, "safety_flag": False}


def planner(state: AgentState) -> AgentState:
    """
    规划节点
    根据对话历史生成回复策略
    """
    try:
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
    except Exception as e:
        print(f"[规划器] 错误: {e}")
        return {**state, "plan": "默认策略：友好回复"}


def generator(state: AgentState) -> AgentState:
    """
    生成节点
    根据 plan 生成最终回复
    如果之前有检查员反馈，会根据反馈改进
    """
    try:
        plan = state.get("plan", "")
        messages = state.get("messages", [])
        critic_feedback = state.get("critic_feedback", "")
        retry_count = state.get("retry_count", 0)
        
        # Mock 生成逻辑：根据策略和最后一条消息生成回复
        last_message = messages[-1] if messages else None
        last_content = last_message.content if last_message and hasattr(last_message, "content") else ""
        
        # 如果有检查员反馈，说明是重试，需要改进
        if critic_feedback and "检查发现问题" in critic_feedback:
            print(f"[生成器] 🔄 根据检查员反馈重新生成（第 {retry_count + 1} 次尝试）")
            print(f"[生成器] 反馈内容: {critic_feedback}")
            # 生成更详细的回复
            if "初次接触" in plan:
                response = f"你好！很高兴认识你。你刚才说：{last_content[:30]}... 能告诉我更多关于你的信息吗？我会认真倾听并尽力帮助你。"
            elif "建立信任" in plan:
                response = f"我理解你的意思。关于'{last_content[:20]}...'这个话题，我们可以深入聊聊。我很乐意分享我的看法和经验。"
            elif "深度交流" in plan:
                response = f"基于我们之前的对话，我建议：{last_content[:20]}... 你觉得怎么样？我们可以进一步讨论这个方案。"
            else:
                response = f"我收到了你的消息：{last_content[:30]}... 让我仔细思考一下如何更好地回复你。我会提供更有价值的建议。"
        else:
            # 首次生成
            if "初次接触" in plan:
                response = f"你好！很高兴认识你。你刚才说：{last_content[:30]}... 能告诉我更多关于你的信息吗？"
            elif "建立信任" in plan:
                response = f"我理解你的意思。关于'{last_content[:20]}...'这个话题，我们可以深入聊聊。"
            elif "深度交流" in plan:
                response = f"基于我们之前的对话，我建议：{last_content[:20]}... 你觉得怎么样？"
            else:
                response = f"我收到了你的消息：{last_content[:30]}... 让我想想如何回复你。"
        
        print(f"[生成器] 生成回复: {response}")
        
        # 更新重试次数
        return {**state, "final_response": response, "retry_count": retry_count + 1}
    except Exception as e:
        print(f"[生成器] 错误: {e}")
        return {**state, "final_response": "抱歉，我遇到了一些问题，请稍后再试。"}


def critic(state: AgentState) -> AgentState:
    """
    检查员节点
    检查生成的回复质量，如果不满意则提供反馈
    """
    try:
        final_response = state.get("final_response", "")
        plan = state.get("plan", "")
        retry_count = state.get("retry_count", 0)
        
        # Mock 检查逻辑：检查回复质量
        # 检查标准：
        # 1. 回复不能太短（少于10个字符）
        # 2. 回复应该与策略相关
        # 3. 回复不能是空字符串
        
        issues = []
        
        if len(final_response) < 10:
            issues.append("回复太短，需要更详细")
        
        if not final_response.strip():
            issues.append("回复为空")
        
        if plan and "初次接触" in plan and "你好" not in final_response:
            issues.append("初次接触策略应该包含问候语")
        
        # 模拟：前两次可能检查不通过（用于演示循环）
        if retry_count < 2 and len(final_response) < 50:
            issues.append("回复质量不够，需要更丰富的内容")
        
        if issues:
            feedback = f"检查发现问题: {', '.join(issues)}。请重新生成更优质的回复。"
            print(f"[检查员] ❌ 检查未通过")
            print(f"[检查员] 反馈: {feedback}")
            return {**state, "critic_feedback": feedback}
        else:
            feedback = "检查通过：回复质量良好"
            print(f"[检查员] ✅ 检查通过")
            print(f"[检查员] 反馈: {feedback}")
            return {**state, "critic_feedback": feedback}
    except Exception as e:
        print(f"[检查员] 错误: {e}")
        return {**state, "critic_feedback": "检查过程出错，但允许继续"}


def evolver(state: AgentState) -> AgentState:
    """
    演化节点
    更新 relationship_stats（亲密度等）
    """
    try:
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
    except Exception as e:
        print(f"[演化器] 错误: {e}")
        return {**state, "relationship_stats": state.get("relationship_stats", {})}


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


def should_retry(state: AgentState) -> Literal["retry", "continue"]:
    """
    条件边函数
    根据检查员反馈决定是否需要重试
    """
    critic_feedback = state.get("critic_feedback", "")
    retry_count = state.get("retry_count", 0)
    max_retries = 3  # 最大重试次数
    
    # 如果检查发现问题，且未超过最大重试次数，则重试
    if "检查发现问题" in critic_feedback and retry_count < max_retries:
        print(f"[条件判断] 检查未通过，需要重试（当前重试次数: {retry_count}/{max_retries}）")
        return "retry"
    else:
        if retry_count >= max_retries:
            print(f"[条件判断] 已达到最大重试次数 ({max_retries})，继续流程")
        else:
            print("[条件判断] 检查通过，继续流程")
        return "continue"


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
    workflow.add_node("critic", critic)  # 添加检查员节点
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
    
    # 添加顺序边：planner -> generator
    workflow.add_edge("planner", "generator")
    
    # generator -> critic（生成后必须检查）
    workflow.add_edge("generator", "critic")
    
    # 添加条件边：从 critic 根据检查结果决定是否重试（形成循环！）
    workflow.add_conditional_edges(
        "critic",
        should_retry,
        {
            "retry": "generator",  # 检查未通过 -> 重新生成（循环！）
            "continue": "evolver"  # 检查通过 -> 继续到演化器
        }
    )
    
    # 添加顺序边：evolver -> 结束
    workflow.add_edge("evolver", END)
    
    # 编译图
    app = workflow.compile()
    
    return app


if __name__ == "__main__":
    try:
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
            "final_response": "",
            "critic_feedback": "",
            "retry_count": 0
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
        print(f"  检查员反馈: {result.get('critic_feedback', '无')}")
        print(f"  重试次数: {result.get('retry_count', 0)}")
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
            "final_response": "",
            "critic_feedback": "",
            "retry_count": 0
        }
        
        result_unsafe = app.invoke(unsafe_state)
        print(f"\n最终状态:")
        print(f"  安全标志: {result_unsafe['safety_flag']}")
        print(f"  规划策略: {result_unsafe['plan']}")
        print(f"  最终回复: {result_unsafe['final_response']}")
    except Exception as e:
        print(f"\n程序运行错误: {e}")
        import traceback
        traceback.print_exc()

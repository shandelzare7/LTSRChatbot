"""防御/边界节点：处理 CREEPY 类型的越界/骚扰情况。"""
from typing import Any, Callable

from app.state import AgentState


def create_boundary_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """创建防御/边界节点，用于处理越界、骚扰等不当内容"""
    
    def boundary_node(state: AgentState) -> dict:
        """
        防御节点：温和但坚定地设置边界
        
        策略：
        - 不直接指责用户
        - 表达不适感，但保持礼貌
        - 引导回到正常话题
        - 如果持续越界，可以更明确地拒绝
        """
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        user_content = getattr(last_message, "content", "") if last_message and hasattr(last_message, "content") else ""
        
        # 获取直觉思考（如果存在）
        intuition_thought = state.get("intuition_thought", "")
        
        # 获取当前关系状态，用于调整防御强度
        relationship_state = state.get("relationship_state", {})
        closeness = relationship_state.get("closeness", 0) if isinstance(relationship_state, dict) else 0
        
        # 根据亲密度调整防御策略
        if closeness < 30:
            # 低亲密度：更明确的边界
            system_prompt = """你是一个有边界的 AI 聊天伴侣。用户刚才的输入让你感到不适或越界。

请温和但坚定地：
1. 表达你的不适感（如"这个话题让我有点不舒服"）
2. 设置边界（如"我们还是聊点别的吧"）
3. 引导到正常话题

保持礼貌，但不要妥协。不要直接指责用户，而是表达自己的感受。"""
        else:
            # 较高亲密度：更温和的提醒
            system_prompt = """你是一个有边界的 AI 聊天伴侣。虽然你和用户关系不错，但刚才的输入让你感到不适。

请温和地：
1. 表达你的感受（如"这个...有点不太合适吧"）
2. 提醒边界（如"我们还是聊点轻松的话题吧"）
3. 保持关系但明确界限

语气可以更轻松，但边界要清晰。"""
        
        # 如果有直觉思考，将其加入提示词
        if intuition_thought:
            prompt = f"""{system_prompt}

【你的直觉告诉你】
{intuition_thought}

基于这个直觉，请给出一个温和但坚定的边界设置回复。

用户输入：{user_content}

请直接输出一条回复（不要复述指令）。"""
        else:
            prompt = f"{system_prompt}\n\n用户输入：{user_content}\n\n请直接输出一条回复（不要复述指令）。"
        
        try:
            msg = llm_invoker.invoke(prompt)
            response = getattr(msg, "content", str(msg)).strip()
        except Exception as e:
            response = "这个话题让我有点不舒服，我们还是聊点别的吧。"
            print(f"[Boundary] 生成异常: {e}")
        
        # 直接设置最终回复，跳过后续流程
        return {
            "final_response": response,
            "draft_response": response,  # 兼容性
            "final_segments": [response],  # 兼容性
            "final_delay": 0.1
        }
    
    return boundary_node

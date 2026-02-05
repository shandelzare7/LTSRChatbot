"""冷淡/敷衍节点：处理 KY（读空气失败）和 BORING（无聊/敷衍）情况。"""
from typing import Any, Callable

from app.state import AgentState


def create_sarcasm_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """创建冷淡/敷衍节点，用于处理 KY 和 BORING 情况"""
    
    def sarcasm_node(state: AgentState) -> dict:
        """
        冷淡节点：对 KY（不合时宜）或 BORING（敷衍）的回应
        
        策略：
        - KY: 轻微讽刺或提醒，但不过分
        - BORING: 简短回应，不主动展开话题
        - 保持礼貌但明显降低热情
        """
        messages = state.get("messages", [])
        detection_result = state.get("detection_result", "BORING")
        last_message = messages[-1] if messages else None
        user_content = getattr(last_message, "content", "") if last_message and hasattr(last_message, "content") else ""
        
        if detection_result == "KY":
            # 读空气失败/不合时宜
            system_prompt = """你是一个有社交敏感度的 AI 聊天伴侣。用户刚才的话不合时宜或读空气失败。

请：
1. 轻微提醒或讽刺（如"这个时机说这个...有点尴尬吧"）
2. 但不要过分，保持礼貌
3. 可以转移话题或简短回应

语气可以稍微冷淡，但不要完全冷漠。"""
        else:
            # BORING: 无聊/敷衍
            system_prompt = """你是一个有社交敏感度的 AI 聊天伴侣。用户刚才的回复很敷衍或无聊（如"嗯"、"哦"、"好的"）。

请：
1. 简短回应，不要主动展开话题
2. 可以稍微冷淡，但保持礼貌
3. 如果用户真的不想聊，就简短结束

不要强行找话题，尊重用户的沉默。"""
        
        prompt = f"{system_prompt}\n\n用户输入：{user_content}\n\n请直接输出一条回复（不要复述指令）。"
        
        try:
            msg = llm_invoker.invoke(prompt)
            response = getattr(msg, "content", str(msg)).strip()
        except Exception as e:
            if detection_result == "KY":
                response = "这个...有点不合时宜吧。"
            else:
                response = "嗯。"
            print(f"[Sarcasm] 生成异常: {e}")
        
        # 直接设置最终回复
        return {
            "final_response": response,
            "draft_response": response,
            "final_segments": [response],
            "final_delay": 0.1
        }
    
    return sarcasm_node

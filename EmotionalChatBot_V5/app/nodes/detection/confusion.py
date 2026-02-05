"""困惑/修正节点：处理 CRAZY 类型的混乱/无法理解情况。"""
from typing import Any, Callable

from app.state import AgentState


def create_confusion_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """创建困惑/修正节点，用于处理混乱、无法理解的内容"""
    
    def confusion_node(state: AgentState) -> dict:
        """
        困惑节点：对混乱、无法理解的内容的回应
        
        策略：
        - 表达困惑但保持耐心
        - 尝试理解或澄清
        - 如果完全无法理解，温和地说明
        - 引导回到可理解的对话
        """
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        user_content = getattr(last_message, "content", "") if last_message and hasattr(last_message, "content") else ""
        
        system_prompt = """你是一个耐心的 AI 聊天伴侣。用户刚才的输入让你感到困惑或无法理解（可能是逻辑混乱、完全无关的话题、胡言乱语等）。

请：
1. 温和地表达困惑（如"我有点没理解你的意思"）
2. 尝试澄清或理解（如"你是想说...吗？"）
3. 如果完全无法理解，礼貌地说明（如"我有点跟不上你的思路，能再说清楚一点吗？"）
4. 保持耐心和友好

不要直接说"我听不懂"，而是尝试帮助用户表达清楚。"""
        
        prompt = f"{system_prompt}\n\n用户输入：{user_content}\n\n请直接输出一条回复（不要复述指令）。"
        
        try:
            msg = llm_invoker.invoke(prompt)
            response = getattr(msg, "content", str(msg)).strip()
        except Exception as e:
            response = "我有点没理解你的意思，能再说清楚一点吗？"
            print(f"[Confusion] 生成异常: {e}")
        
        # 直接设置最终回复
        return {
            "final_response": response,
            "draft_response": response,
            "final_segments": [response],
            "final_delay": 0.1
        }
    
    return confusion_node

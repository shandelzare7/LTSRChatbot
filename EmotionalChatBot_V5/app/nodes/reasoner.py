"""深度思考节点：Chain-of-Thought，受 Mode.monologue_instruction 控制。"""
from typing import Any, Callable

from app.state import AgentState


def create_reasoner_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    def reasoner_node(state: AgentState) -> dict:
        mode = state.get("current_mode")
        if not mode:
            return {"deep_reasoning_trace": {"reasoning": "", "enabled": False}}
        if not getattr(mode, "enable_deep_reasoning", True):
            return {"deep_reasoning_trace": {"reasoning": "", "enabled": False}}
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        user_content = getattr(last, "content", "") if last else ""
        intuition = state.get("intuition_thought", "")
        instruction = getattr(mode, "monologue_instruction", "理性分析用户意图与情绪。")
        intuition_block = f"\n\n【System 1 直觉（仅参考）】\n{intuition}" if intuition else ""
        prompt = f"""【内心独白指令】{instruction}{intuition_block}

用户最后一条消息：
{user_content}

请用 1-3 句话写出 Bot 此刻的内心思考（不输出给用户）。"""
        try:
            msg = llm_invoker.invoke(prompt)
            reasoning = getattr(msg, "content", str(msg))
        except Exception as e:
            reasoning = f"[Reasoner Mock] 按当前模式思考。异常: {e}"
        return {"deep_reasoning_trace": {"reasoning": reasoning, "enabled": True}}

    return reasoner_node

"""风格分析节点：并行节点，受当前模式影响（无独立 style_prompt 时用 mode.name）。"""
from typing import Any, Callable

from app.state import AgentState


def create_style_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    def style_node(state: AgentState) -> dict:
        mode = state.get("current_mode")
        mode_name = mode.name if mode else "正常"
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        user_content = getattr(last, "content", "") if last else ""
        prompt = f"""当前心理模式：{mode_name}。请用一句话描述回复应有的风格（如：自然友好、冷淡简短、破碎犹豫等）。不要输出具体回复内容。"""
        try:
            msg = llm_invoker.invoke(prompt)
            analysis = getattr(msg, "content", str(msg))
        except Exception as e:
            analysis = f"[Styler Mock] 风格：与模式「{mode_name}」一致。异常: {e}"
        return {"style_analysis": analysis}

    return style_node

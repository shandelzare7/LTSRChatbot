"""回复生成节点：初稿生成与 Refiner 重写，受 Mode.system_prompt_template 控制。"""
from typing import Any, Callable

from app.state import AgentState


def create_generator_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    def generator_node(state: AgentState) -> dict:
        mode = state.get("current_mode")
        system_prompt = (
            getattr(mode, "system_prompt_template", "你是一个陪伴型 Bot，自然回复。")
            if mode
            else "你是一个陪伴型 Bot，自然回复。"
        )
        messages = state.get("messages", [])
        reasoning = state.get("deep_reasoning_trace") or {}
        style = state.get("style_analysis", "")
        critique = state.get("critique_feedback", "")
        retry_count = state.get("retry_count", 0)
        is_refine = bool(critique and retry_count > 0)
        last = messages[-1] if messages else None
        user_content = getattr(last, "content", "") if last else ""
        memories = state.get("memories", "")

        user_block = f"用户消息：\n{user_content}\n\n近期记忆：\n{memories}"
        if is_refine:
            user_block += f"\n\n【质检未通过，请按以下意见修改】\n{critique}\n\n当前初稿（请重写）：\n{state.get('draft_response', '')}"
        else:
            if reasoning.get("reasoning"):
                user_block += f"\n\n内心思考（仅参考）：\n{reasoning.get('reasoning', '')}"
            if style:
                user_block += f"\n\n风格要求：{style}"

        full_prompt = f"{system_prompt}\n\n{user_block}\n\n请直接输出一条回复（不要复述指令）。"
        try:
            msg = llm_invoker.invoke(full_prompt)
            draft = getattr(msg, "content", str(msg)).strip()
        except Exception as e:
            draft = f"[Generator Mock] 已根据当前模式生成占位回复。异常: {e}"
        out = {"draft_response": draft}
        if is_refine:
            out["retry_count"] = state.get("retry_count", 0) + 1
        return out

    return generator_node

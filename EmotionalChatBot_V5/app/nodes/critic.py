"""循环质检节点：根据 Mode.critic_criteria 判断是否通过，并写入 critique_feedback。"""
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from app.state import AgentState


def create_critic_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    def critic_node(state: AgentState) -> dict:
        mode = state.get("current_mode")
        criteria = (
            getattr(mode, "critic_criteria", ["回复自然、得体"]) if mode else ["回复自然、得体"]
        )
        draft = state.get("draft_response", "")
        criteria_text = "\n".join([f"- {c}" for c in criteria])
        # 仅用 summary + retrieved 放入 system；chat_buffer 分条放正文
        summary = state.get("conversation_summary") or ""
        retrieved = state.get("retrieved_memories") or []
        system_memory_parts = []
        if summary:
            system_memory_parts.append("近期对话摘要：\n" + summary)
        if retrieved:
            system_memory_parts.append("相关记忆片段：\n" + "\n".join(retrieved))
        system_memory = "\n\n".join(system_memory_parts) if system_memory_parts else "（无）"
        system_content = f"""你是质检员。请判断以下回复是否满足以下标准。若全部满足，只回复「通过」；否则回复「不通过」并简要说明需要修改的点（1-2 句）。

标准：
{criteria_text}

（供一致性检查的记忆，仅摘要与检索片段）
{system_memory}
"""
        chat_buffer = state.get("chat_buffer") or []
        body_messages = list(chat_buffer[-20:])
        try:
            msg = llm_invoker.invoke([
                SystemMessage(content=system_content),
                *body_messages,
                HumanMessage(content=f"待检回复：\n{draft}"),
            ])
            feedback = getattr(msg, "content", str(msg)).strip()
        except Exception as e:
            feedback = "通过"
        passed = "通过" in feedback and "不通过" not in feedback[:5]
        print("[Critic] pass" if passed else "[Critic] retry")
        return {"critique_feedback": "" if passed else feedback}

    return critic_node


def check_critic_result(state: AgentState) -> str:
    """条件边：通过 -> processor，未通过 -> refiner；超过最大重试也放行。"""
    feedback = state.get("critique_feedback", "")
    if not feedback:
        return "pass"
    max_retries = 3
    retry = state.get("retry_count", 0)
    if retry >= max_retries:
        return "pass"
    return "retry"

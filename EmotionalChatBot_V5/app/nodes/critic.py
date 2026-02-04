"""循环质检节点：根据 Mode.critic_criteria 判断是否通过，并写入 critique_feedback。"""
from typing import Any, Callable

from app.state import AgentState


def create_critic_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    def critic_node(state: AgentState) -> dict:
        mode = state.get("current_mode")
        criteria = (
            getattr(mode, "critic_criteria", ["回复自然、得体"]) if mode else ["回复自然、得体"]
        )
        draft = state.get("draft_response", "")
        criteria_text = "\n".join([f"- {c}" for c in criteria])
        prompt = f"""你是质检员。请判断以下回复是否满足以下标准。若全部满足，只回复「通过」；否则回复「不通过」并简要说明需要修改的点（1-2 句）。

标准：
{criteria_text}

回复：
{draft}
"""
        try:
            msg = llm_invoker.invoke(prompt)
            feedback = getattr(msg, "content", str(msg)).strip()
        except Exception as e:
            feedback = "通过"
        passed = "通过" in feedback and "不通过" not in feedback[:5]
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

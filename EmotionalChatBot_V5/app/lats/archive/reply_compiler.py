from __future__ import annotations

from typing import Any, Dict, List

from app.state import ProcessorPlan, ReplyPlan
from utils.detailed_logging import log_computation


def compile_reply_plan_to_processor_plan(
    reply_plan: ReplyPlan,
    state: Dict[str, Any],
    *,
    max_messages: int = 5,
) -> ProcessorPlan:
    """
    将 ReplyPlan 编译成 ProcessorPlan，仅产出 messages。
    延迟由 processor 节点统一计算（纯 processor 计算），此处不产出 delays/actions。
    """
    messages_raw = reply_plan.get("messages") or []
    msgs: List[str] = []

    for m in messages_raw[:max_messages]:
        if isinstance(m, str):
            c = m.strip()
            if not c:
                continue
            msgs.append(c)
            continue
        if not isinstance(m, dict):
            continue
        c = str(m.get("content") or "").strip()
        if not c:
            continue
        msgs.append(c)

    if not msgs:
        text = (state.get("final_response") or state.get("draft_response") or "").strip()
        if text:
            msgs = [text]
        else:
            msgs = ["（刚才生成回复失败了。可能是模型服务不可用/余额不足。你可以稍后再试。）"]

    meta: Dict[str, Any] = {
        "source": "reply_plan_compiler",
        "reply_plan_justification": reply_plan.get("justification"),
        "reply_plan_messages_count": reply_plan.get("messages_count"),
        "reply_plan_must_cover_map": reply_plan.get("must_cover_map"),
        "reply_plan_first_message_role": reply_plan.get("first_message_role"),
    }

    log_computation(
        "ReplyCompiler",
        "编译完成（仅 messages，延迟由 processor 计算）",
        outputs={"processor_plan": {"messages_count": len(msgs), "meta": meta}},
    )

    return {"messages": msgs, "meta": meta}

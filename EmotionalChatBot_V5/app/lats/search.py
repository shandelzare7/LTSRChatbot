from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.lats.evaluator import evaluate_27_candidates_single_llm
from app.lats.reply_planner import plan_reply_27_via_content_moves
from app.state import ProcessorPlan, ReplyPlan, SimReport


def _compile_reply_plan_to_text_plan(reply_plan: ReplyPlan, *, max_messages: int) -> ProcessorPlan:
    """
    LATS no longer generates/uses delays/actions. For judging, only keep text messages[].
    Converts ReplyPlan (now with single "reply" field) to ProcessorPlan.
    The reply text will be split by processor later, so we put it as a single message here.
    """
    # 优先从 reply 字段获取完整文本（新格式）
    reply_text = (reply_plan or {}).get("reply")
    if isinstance(reply_text, str) and reply_text.strip():
        # 新格式：完整文本，processor 会负责分割
        return {"messages": [reply_text.strip()]}
    
    # 回退到旧格式：从 messages 数组获取（向后兼容）
    msgs_raw = (reply_plan or {}).get("messages") or []
    out: List[str] = []
    if isinstance(msgs_raw, list):
        for m in msgs_raw[: int(max_messages)]:
            if isinstance(m, str):
                t = m.strip()
            elif isinstance(m, dict):
                t = str(m.get("content") or "").strip()
            else:
                t = str(m).strip()
            if t:
                out.append(t)
    
    # 如果没有找到任何消息，返回空数组
    return {"messages": out if out else []}


def _repair_reply_via_llm(
    state: Dict[str, Any],
    original_reply: str,
    repair_instructions: str,
    llm_invoker: Any,
) -> Optional[str]:
    """LATS V3：用轻模型做最多一次定向修复。输入原候选+修复指令，输出一条新 reply。"""
    if not llm_invoker or not (repair_instructions or "").strip():
        return None
    from langchain_core.messages import HumanMessage, SystemMessage
    from app.lats.prompt_utils import get_chat_buffer_body_messages, safe_text

    system = "你是回复改写助手。根据「修复指令」对「原候选回复」做补丁式改写，只输出一条最终回复正文，不要解释、不要编号。"
    user = f"原候选回复：\n{safe_text(original_reply)}\n\n修复指令：\n{safe_text(repair_instructions)}\n\n请输出一条修复后的完整回复（仅正文）："
    body = get_chat_buffer_body_messages(state or {}, limit=30)
    try:
        resp = llm_invoker.invoke([SystemMessage(content=system), *body, HumanMessage(content=user)])
        content = (getattr(resp, "content", "") or "").strip()
        if content:
            return content
    except Exception:
        pass
    return None


def _lats_search_v3(
    state: Dict[str, Any],
    llm_gen: Any,
    llm_eval: Any,
    *,
    max_messages: int = 5,
) -> Tuple[Optional[ReplyPlan], Optional[ProcessorPlan], Optional[SimReport], Dict[str, Any]]:
    """LATS V3：27 并行生成 → 单模型评估 → 通过即返回 / 不通过则最多一次轻模型修复或 fallback。"""
    requirements = state.get("requirements") or {}
    requirements = requirements if isinstance(requirements, dict) else {}
    tree: Dict[str, Any] = {"version": "v3", "best_id": None}

    candidates_27 = plan_reply_27_via_content_moves(state, llm_gen)
    if not candidates_27:
        return None, None, None, {**tree, "error": "plan_reply_27_failed"}

    eval_result = evaluate_27_candidates_single_llm(state, candidates_27, requirements, llm_invoker=llm_eval)
    best_id = max(0, min(26, int(eval_result.get("best_id", 0))))
    accept = bool(eval_result.get("accept", False))
    fail_type = eval_result.get("fail_type")
    repair_instructions = eval_result.get("repair_instructions")
    fallback = eval_result.get("fallback")

    best_candidate = next((c for c in candidates_27 if int(c.get("id", -1)) == best_id), None)
    if not best_candidate:
        best_candidate = candidates_27[best_id] if best_id < len(candidates_27) else candidates_27[0]
    original_reply = (best_candidate or {}).get("reply") or ""

    tree["best_id"] = best_id
    tree["accept"] = accept

    if accept:
        reply_plan: ReplyPlan = {"reply": original_reply}
        proc = _compile_reply_plan_to_text_plan(reply_plan, max_messages=max_messages)
        sim_report: SimReport = {
            "found_solution": True,
            "eval_score": 1.0,
            "failed_checks": [],
            "score_breakdown": {"best_id": best_id, "accept": True},
            "llm_status": "ok",
            "llm_details": {},
        }
        return reply_plan, proc, sim_report, tree

    # 关闭 repair：未通过时直接用 fallback，不再多一次 LLM 调用
    final_reply: Optional[str] = None
    if (fallback or "").strip():
        final_reply = (fallback or "").strip()
        tree["used_fallback"] = True
    if final_reply is None:
        final_reply = original_reply or "（暂时无法给出合适回复，稍后再聊。）"
        tree["used_fallback"] = True

    reply_plan = {"reply": final_reply}
    proc = _compile_reply_plan_to_text_plan(reply_plan, max_messages=max_messages)
    sim_report = {
        "found_solution": False,
        "eval_score": 0.0,
        "failed_checks": [{"id": "single_eval_reject", "reason": fail_type or "accept=false"}],
        "score_breakdown": {"best_id": best_id, "accept": False, "fail_type": fail_type},
        "llm_status": "ok",
        "llm_details": {},
    }
    return reply_plan, proc, sim_report, tree


def lats_search_best_plan(
    state: Dict[str, Any],
    *,
    llm_gen: Any,
    llm_eval: Any,
    max_messages: int = 5,
) -> Tuple[Optional[ReplyPlan], Optional[ProcessorPlan], Optional[SimReport], Dict[str, Any]]:
    """
    LATS：9 并行生成（gpt-4o-mini）+ 单模型评估（gpt-4o）→ 通过即返回 / 不通过则最多一次轻模型修复或 fallback。
    """
    if not llm_gen or not llm_eval:
        return None, None, None, {"error": "llm_gen_and_llm_eval_required"}
    return _lats_search_v3(state, llm_gen, llm_eval, max_messages=max_messages)


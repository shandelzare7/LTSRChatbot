"""
Fast 节点：当策略 route_path 为 fast 时由 strategy_resolver 路由到此节点。
使用与 reply_planner 相同的提示词构建（含必须遵守规则，优先级从高到低：当前策略 > 时间与会话上下文 > 阶段意图与行为准则 > 写作要求；冲突时靠前的优先），
仅输出单条回复，使用 main 的 LLM（gpt-4o），不经过 LATS。
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from app.state import AgentState
from utils.tracing import trace_if_enabled

from app.lats.prompt_utils import build_style_profile
from app.lats.requirements import compile_requirements
from app.lats.reply_planner import plan_reply_via_llm
from app.lats.search import _compile_reply_plan_to_text_plan
from utils.external_text import strip_candidate_prefix


def create_fast_reply_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """
    Fast 回复节点：提示词与 reply_planner 一致（必须遵守顺序：当前策略 > 时间/意图 > 写作要求），单条输出，使用 llm_invoker（main 的 LLM，即 gpt-4o）。
    不经过 LATS 搜索与评分，直接生成一条回复；final_segments 由 processor 节点产出。
    """

    @trace_if_enabled(
        name="Response/Fast_Reply",
        run_type="chain",
        tags=["node", "fast_reply"],
        metadata={
            "state_outputs": [
                "requirements",
                "style_profile",
                "reply_plan",
                "sim_report",
                "final_response",
            ]
        },
    )
    def node(state: AgentState) -> Dict[str, Any]:
        requirements = compile_requirements(state)
        style_profile = build_style_profile(state)

        merged = dict(state)
        merged["requirements"] = requirements
        merged["style_profile"] = style_profile

        rp = plan_reply_via_llm(merged, llm_invoker, user_message_only=True)
        proc = _compile_reply_plan_to_text_plan(rp or {}, max_messages=1)
        final_text = " ".join([strip_candidate_prefix(str(x)) for x in (proc.get("messages") or [])]).strip()

        return {
            "requirements": requirements,
            "style_profile": style_profile,
            "reply_plan": rp,
            "sim_report": {
                "found_solution": True,
                "eval_score": 1.0,
                "failed_checks": [],
                "score_breakdown": {"fast_reply": 1.0},
            },
            "lats_tree": {"skipped": True, "reason": "fast_route"},
            "lats_best_id": "fast",
            "final_response": final_text,
        }

    return node

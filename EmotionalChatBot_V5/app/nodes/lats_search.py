from __future__ import annotations

from typing import Any, Callable, Dict

from app.state import AgentState
from utils.tracing import trace_if_enabled

from app.lats.prompt_utils import build_style_profile
from app.lats.requirements import compile_requirements
from app.lats.search import lats_search_best_plan
from app.lats.reply_planner import plan_reply_via_llm
from app.lats.reply_compiler import compile_reply_plan_to_processor_plan
from app.lats.evaluator import evaluate_candidate

import re


def create_lats_search_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """
    LATS Choreography Search Node
    - 输入：reasoner/style 已写入的策略/风格/关系参数 + 记忆
    - 输出：best ReplyPlan + 编译后的 ProcessorPlan（messages/delays/actions）
    - reward：基于 post-processor 形态（messages/delays）评审得分
    """

    @trace_if_enabled(
        name="Response/LATS_Search",
        run_type="chain",
        tags=["node", "lats", "search", "choreography"],
        metadata={
            "state_outputs": [
                "requirements",
                "style_profile",
                "reply_plan",
                "processor_plan",
                "sim_report",
                "lats_tree",
                "lats_best_id",
                "lats_rollouts",
                "final_response",
                "final_segments",
            ]
        },
    )
    def node(state: AgentState) -> Dict[str, Any]:
        # Mode 早停检查：如果 mode 禁用 LATS，直接返回空响应
        mode = state.get("current_mode") or {}
        budget = (mode.get("lats_budget") if isinstance(mode, dict) else getattr(mode, "lats_budget", None)) or {}
        enabled = budget.get("enabled", True) if isinstance(budget, dict) else (getattr(budget, "enabled", True) if budget else True)
        
        if not enabled:
            # mute：不跑任何搜索，不生成多消息
            # 返回空字符串作为最终输出
            print("[LATS] Mode 禁用 LATS，跳过搜索（mute_mode 早停）")
            return {
                "reply_plan": {"mode": "mute", "messages": []},
                "processor_plan": {"messages": [], "delays": [], "actions": []},
                "final_response": "",  # 空回复
                "sim_report": {
                    "eval_score": 1.0,
                    "found_solution": True,
                    "failed_checks": [],
                    "score_breakdown": {"mode_noop": 1.0},
                },
                "lats_tree": {"disabled": True},
            }
        
        # 1) compile requirements/style_profile
        requirements = compile_requirements(state)
        style_profile = build_style_profile(state)

        merged = dict(state)
        merged["requirements"] = requirements
        merged["style_profile"] = style_profile

        # 2.0) 允许“完全跳过 LATS”（只用 draft_response/fallback），用于压测/降载
        if (state.get("lats_rollouts") is not None and int(state.get("lats_rollouts") or 0) <= 0) and (
            state.get("lats_expand_k") is not None and int(state.get("lats_expand_k") or 0) <= 0
        ):
            print("[LATS] rollouts/expand_k=0：跳过搜索（直接用根计划一次生成）")
            rp = plan_reply_via_llm(
                merged,
                llm_invoker,
                max_messages=int(requirements.get("max_messages", 3) or 3),
            )
            proc = compile_reply_plan_to_processor_plan(rp, merged, max_messages=int(requirements.get("max_messages", 3) or 3))
            rep = evaluate_candidate(merged, rp, proc, requirements, llm_soft_scorer=None)
            final_segments = list(proc.get("messages") or [])
            final_text = " ".join([str(x) for x in final_segments]).strip()
            return {
                "requirements": requirements,
                "style_profile": style_profile,
                "reply_plan": rp,
                "processor_plan": proc,
                "sim_report": rep,
                "lats_tree": {"skipped": True, "reason": "rollouts_expand_k_zero"},
                "lats_best_id": "root",
                "final_response": final_text,
                "final_segments": final_segments,
            }

        # 2.1) 可选：低风险回合跳过 LATS rollout（节省 token、降低污染面）
        # 仅在显式开启时生效（默认不改变线上行为）。
        if bool(state.get("lats_skip_low_risk")):
            ext = str(state.get("external_user_text") or state.get("user_input") or "").strip()
            stage_id = str(state.get("current_stage") or "initiating")
            composite = (state.get("detection_signals") or {}).get("composite") or {}
            try:
                risk = max(float(composite.get("conflict_eff", 0.0) or 0.0), float(composite.get("pressure", 0.0) or 0.0))
            except Exception:
                risk = 0.0
            greeting_pat = re.compile(r"^\s*(hi|hello|hey|你好|您好|嗨|哈喽|早上好|中午好|晚上好|晚安).{0,24}$", re.IGNORECASE)
            is_greeting = bool(ext) and bool(greeting_pat.match(ext))
            if stage_id in ("initiating", "experimenting") and is_greeting and risk < 0.15:
                print("[LATS] low-risk 回合：跳过 rollout 搜索，仅用 ReplyPlanner 根计划")
                rp = plan_reply_via_llm(
                    merged,
                    llm_invoker,
                    max_messages=int(requirements.get("max_messages", 3) or 3),
                )
                proc = compile_reply_plan_to_processor_plan(rp, merged, max_messages=int(requirements.get("max_messages", 3) or 3))
                rep = evaluate_candidate(merged, rp, proc, requirements, llm_soft_scorer=None)
                final_segments = list(proc.get("messages") or [])
                final_text = " ".join([str(x) for x in final_segments]).strip()
                return {
                    "requirements": requirements,
                    "style_profile": style_profile,
                    "reply_plan": rp,
                    "processor_plan": proc,
                    "sim_report": rep,
                    "lats_tree": {"skipped": True},
                    "lats_best_id": "root",
                    "final_response": final_text,
                    "final_segments": final_segments,
                }

        # 2) LATS search best ReplyPlan
        # 规则：state 显式配置优先（便于压测/实验），否则回退到 mode.lats_budget，再回退默认值
        lats_budget = None
        if mode and hasattr(mode, "lats_budget"):
            lats_budget = mode.lats_budget

        rollouts_state = state.get("lats_rollouts")
        expand_k_state = state.get("lats_expand_k")

        if rollouts_state is not None:
            rollouts = int(rollouts_state)
        elif lats_budget and hasattr(lats_budget, "rollouts"):
            rollouts = int(lats_budget.rollouts)
        else:
            # P1：阶段感知预算（让早期阶段更像“会长树”的搜索，而不是 2-sample rerank）
            stage_id = str(state.get("current_stage") or "")
            if not stage_id and isinstance(requirements, dict):
                st = requirements.get("stage_targets") or {}
                if isinstance(st, dict) and st.get("stage"):
                    stage_id = str(st.get("stage"))
            stage_id = stage_id or "initiating"
            if stage_id in ("initiating", "experimenting"):
                rollouts = 8
            elif stage_id in ("intensifying", "integrating"):
                rollouts = 6
            else:
                rollouts = 2

        if expand_k_state is not None:
            expand_k = int(expand_k_state)
        elif lats_budget and hasattr(lats_budget, "expand_k"):
            expand_k = int(lats_budget.expand_k)
        else:
            stage_id = str(state.get("current_stage") or "")
            if not stage_id and isinstance(requirements, dict):
                st = requirements.get("stage_targets") or {}
                if isinstance(st, dict) and st.get("stage"):
                    stage_id = str(st.get("stage"))
            stage_id = stage_id or "initiating"
            if stage_id in ("initiating", "experimenting"):
                expand_k = 1
            else:
                expand_k = 2
        
        # P0：默认应启用 LLM soft scorer（至少用于 Top1 纠偏），避免“只靠启发式就过关”的短路。
        # 若用户显式设置为 False 才禁用。
        enable_llm_soft = bool(state.get("lats_enable_llm_soft_scorer", True))
        print(f"[LATS] 配置: rollouts={rollouts}, expand_k={expand_k}, llm_soft_scorer={'启用' if enable_llm_soft else '禁用'}")
        
        best_reply_plan, best_proc_plan, best_report, tree = lats_search_best_plan(
            merged,
            llm_invoker,
            llm_soft_scorer=(llm_invoker if enable_llm_soft else None),
            rollouts=rollouts,
            expand_k=expand_k,
            max_messages=int(requirements.get("max_messages", 5) or 5),
        )

        # 3) finalize outputs
        if not best_proc_plan:
            print("[LATS] ⚠ 未找到有效计划，使用 fallback")
            # 极端 fallback：不阻断主流程
            text = (state.get("draft_response") or state.get("final_response") or "").strip()
            best_proc_plan = {
                "messages": [text] if text else ["…"],
                "delays": [0.8],
                "actions": ["typing"],
                "meta": {"source": "lats_fallback"},
            }
        final_segments = list(best_proc_plan.get("messages") or [])
        final_text = " ".join([str(x) for x in final_segments]).strip()
        
        if best_report:
            final_score = best_report.get("eval_score", 0.0)
            final_found = best_report.get("found_solution", False)
            final_delays = best_proc_plan.get("delays", [])
            total_delay = sum(float(d) for d in final_delays) if final_delays else 0.0
            print(f"[LATS] 最终输出: {len(final_segments)}条消息, score={final_score:.4f}, found={final_found}, 总延迟={total_delay:.2f}秒")

        print("[LATS] done")
        return {
            "requirements": requirements,
            "style_profile": style_profile,
            "reply_plan": best_reply_plan,
            "processor_plan": best_proc_plan,
            "sim_report": best_report,
            "lats_tree": tree,
            "lats_best_id": tree.get("best_id") if isinstance(tree, dict) else None,
            # 用于观测/可调：本节点执行的 rollout 数
            "lats_rollouts": rollouts,
            "final_response": final_text,
            "final_segments": final_segments,
        }

    return node

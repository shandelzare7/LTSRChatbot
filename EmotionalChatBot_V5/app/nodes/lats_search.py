from __future__ import annotations

import time
from typing import Any, Callable, Dict

from app.state import AgentState
from utils.tracing import trace_if_enabled

from app.lats.prompt_utils import build_style_profile
from app.lats.requirements import compile_requirements
from app.lats.search import lats_search_best_plan
from app.lats.reply_planner import plan_reply_via_llm
from app.lats.evaluator import hard_gate

import re


def create_lats_search_node(llm_invoker: Any, *, llm_soft_scorer: Any = None) -> Callable[[AgentState], dict]:
    """
    LATS Choreography Search Node
    - 输入：reasoner/style 已写入的策略/风格/关系参数 + 记忆
    - 输出：best ReplyPlan + 文本 ProcessorPlan（仅 messages[]；延迟系统交由 processor 节点）
    - reward：基于候选 messages[] 文本评审得分（不评估 delays/actions）
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
                # 任务结算闭环（由 ReplyPlanner/LATS 输出，供 evolver 使用）
                "attempted_task_ids",
                "completed_task_ids",
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
                "processor_plan": {"messages": []},
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

        # -----------------------------
        # LATS tuning defaults (balanced, soft-scorer ON)
        # -----------------------------
        stage_id = str(state.get("current_stage") or "")
        if not stage_id and isinstance(requirements, dict):
            st = requirements.get("stage_targets") or {}
            if isinstance(st, dict) and st.get("stage"):
                stage_id = str(st.get("stage"))
        stage_id = stage_id or "initiating"

        # Keep soft scorer enabled by default; allow explicit disable only via state/env.
        enable_llm_soft = bool(state.get("lats_enable_llm_soft_scorer", True))

        # Limit how often soft scorer is invoked (still ON).
        merged.setdefault("lats_llm_soft_top_n", 1)
        merged.setdefault("lats_llm_soft_max_concurrency", 1)
        # Stage 1.5 assistant-check is an extra LLM call; default off (soft scorer already measures assistantiness).
        merged.setdefault("lats_assistant_check_top_n", 0)

        # LATS V2 defaults (planner 8 candidates, up to 2 regens; strict gate then 3-judge aggregate)
        merged.setdefault("lats_candidate_k", 8)
        merged.setdefault("lats_max_regens", 2)
        merged.setdefault("lats_gate_pass_rate_min", 0.65)
        merged.setdefault("lats_final_score_threshold", 0.68)
        merged.setdefault("lats_dim_w_relationship", 1.0 / 3.0)
        merged.setdefault("lats_dim_w_stage", 1.0 / 3.0)
        merged.setdefault("lats_dim_w_mood_busy", 1.0 / 3.0)

        # Early-exit guard: initiating/experimenting must explore at least 1 rollout.
        if "lats_min_rollouts_before_early_exit" not in merged:
            merged["lats_min_rollouts_before_early_exit"] = 1 if stage_id in ("initiating", "experimenting") else 0

        # Stricter early-exit gates in early stages to avoid "generic opener" winning too often.
        if stage_id in ("initiating", "experimenting"):
            merged.setdefault("lats_early_exit_root_score", 0.82)
            merged.setdefault("lats_early_exit_plan_alignment_min", 0.80)
            merged.setdefault("lats_early_exit_assistantiness_max", 0.18)
            merged.setdefault("lats_early_exit_mode_fit_min", 0.65)

        # 2.0) 允许“完全跳过 LATS”（只用 draft_response/fallback），用于压测/降载
        if (state.get("lats_rollouts") is not None and int(state.get("lats_rollouts") or 0) <= 0) and (
            state.get("lats_expand_k") is not None and int(state.get("lats_expand_k") or 0) <= 0
        ):
            print("[LATS] rollouts/expand_k=0：跳过搜索（直接用根计划一次生成）")
            rp = plan_reply_via_llm(
                merged,
                llm_invoker,
                max_messages=1,
            )
            proc = {"messages": list((rp or {}).get("messages") or [])}
            failures = hard_gate(proc, requirements)
            final_segments = list(proc.get("messages") or [])
            final_text = " ".join([str(x) for x in final_segments]).strip()
            return {
                "requirements": requirements,
                "style_profile": style_profile,
                "reply_plan": rp,
                "processor_plan": proc,
                "sim_report": {
                    "found_solution": len(failures) == 0,
                    "eval_score": 1.0 if len(failures) == 0 else 0.0,
                    "failed_checks": failures,
                    "score_breakdown": {"skip_lats": 1.0},
                },
                "lats_tree": {"skipped": True, "reason": "rollouts_expand_k_zero"},
                "lats_best_id": "root",
                "final_response": final_text,
                "final_segments": final_segments,
            }

        # 2.1) 可选：低风险回合跳过 LATS rollout（节省 token、降低污染面）
        # 仅在显式开启时生效（默认不改变线上行为）。
        if bool(state.get("lats_skip_low_risk")):
            ext = str(state.get("external_user_text") or state.get("user_input") or "").strip()
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
                    max_messages=1,
                )
                proc = {"messages": list((rp or {}).get("messages") or [])}
                failures = hard_gate(proc, requirements)
                final_segments = list(proc.get("messages") or [])
                final_text = " ".join([str(x) for x in final_segments]).strip()
                return {
                    "requirements": requirements,
                    "style_profile": style_profile,
                    "reply_plan": rp,
                    "processor_plan": proc,
                    "sim_report": {
                        "found_solution": len(failures) == 0,
                        "eval_score": 1.0 if len(failures) == 0 else 0.0,
                        "failed_checks": failures,
                        "score_breakdown": {"skip_lats_low_risk": 1.0},
                    },
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
            # Stage-aware balanced defaults
            if stage_id in ("initiating", "experimenting"):
                rollouts = 4
            elif stage_id in ("intensifying", "integrating"):
                rollouts = 2
            elif stage_id in ("differentiating", "circumscribing", "stagnating", "avoiding", "terminating"):
                rollouts = 3
            else:
                rollouts = 2

        if expand_k_state is not None:
            expand_k = int(expand_k_state)
        elif lats_budget and hasattr(lats_budget, "expand_k"):
            expand_k = int(lats_budget.expand_k)
        else:
            if stage_id in ("initiating", "experimenting"):
                expand_k = 2
            elif stage_id in ("intensifying", "integrating"):
                expand_k = 1
            elif stage_id in ("differentiating", "circumscribing", "stagnating", "avoiding", "terminating"):
                expand_k = 1
            else:
                expand_k = 1
        
        print(f"[LATS] 配置: rollouts={rollouts}, expand_k={expand_k}, llm_soft_scorer={'启用' if enable_llm_soft else '禁用'}")
        
        # ### 6.2 需要监控的参数 - LATS 搜索的平均耗时
        lats_start_time = time.perf_counter()
        
        best_reply_plan, best_proc_plan, best_report, tree = lats_search_best_plan(
            merged,
            llm_invoker,
            llm_soft_scorer=(llm_soft_scorer if enable_llm_soft else None),
            rollouts=rollouts,
            expand_k=expand_k,
            max_messages=int(requirements.get("max_messages", 5) or 5),
        )
        
        lats_duration_ms = (time.perf_counter() - lats_start_time) * 1000.0
        print(f"[MONITOR] lats_search_duration_ms={lats_duration_ms:.2f}")
        
        # ### 6.2 需要监控的参数 - 早退触发的频率
        if isinstance(tree, dict):
            early_exit = tree.get("early_exit", False)
            if early_exit:
                exit_reason = tree.get("early_exit_reason", "unknown")
                exit_at_rollout = tree.get("early_exit_at_rollout", 0)
                print(f"[MONITOR] lats_early_exit_triggered: reason={exit_reason}, at_rollout={exit_at_rollout}/{rollouts}")

        # 3) finalize outputs (LATS does NOT generate delays/actions)
        if not best_proc_plan:
            print("[LATS] ⚠ 未找到有效计划，使用 fallback")
            # 极端 fallback：不阻断主流程
            text = (state.get("draft_response") or state.get("final_response") or "").strip()
            if not text:
                # 避免输出“…”导致下一轮 user_input 变成占位符，整段对话崩塌。
                # 这里给出可读、可定位的降级提示（尤其是上游 402/限流/网络异常时）。
                text = "（当前模型服务不可用或余额不足，暂时无法生成回复。请管理员检查 API Key/余额后重试。）"
            best_proc_plan = {"messages": [text], "meta": {"source": "lats_fallback"}}
        final_segments = list(best_proc_plan.get("messages") or [])
        final_text = " ".join([str(x) for x in final_segments]).strip()
        
        if best_report:
            final_score = best_report.get("eval_score", 0.0)
            final_found = best_report.get("found_solution", False)
            print(f"[LATS] 最终输出: {len(final_segments)}条消息, score={final_score:.4f}, found={final_found}")

        print("[LATS] done")
        attempted_task_ids: list[str] = []
        completed_task_ids: list[str] = []
        try:
            if isinstance(best_reply_plan, dict):
                a = best_reply_plan.get("attempted_task_ids")
                c = best_reply_plan.get("completed_task_ids")
                if isinstance(a, list):
                    attempted_task_ids = [str(x) for x in a if str(x).strip()][:8]
                if isinstance(c, list):
                    completed_task_ids = [str(x) for x in c if str(x).strip()][:2]
        except Exception:
            pass

        # 防御：只允许从 tasks_for_lats 的 id 中回写；并按 task_budget_max 截断 completed
        try:
            valid_ids = set()
            req = requirements if isinstance(requirements, dict) else {}
            tfl = req.get("tasks_for_lats", []) if isinstance(req, dict) else []
            if isinstance(tfl, list):
                for t in tfl:
                    if isinstance(t, dict) and t.get("id"):
                        valid_ids.add(str(t.get("id")))
            if valid_ids:
                attempted_task_ids = [x for x in attempted_task_ids if x in valid_ids]
                completed_task_ids = [x for x in completed_task_ids if x in valid_ids]
            tb = int(req.get("task_budget_max", 2) or 2) if isinstance(req, dict) else 2
            tb = max(0, min(2, tb))
            completed_task_ids = completed_task_ids[:tb]
        except Exception:
            pass
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
            "attempted_task_ids": attempted_task_ids,
            "completed_task_ids": completed_task_ids,
        }

    return node

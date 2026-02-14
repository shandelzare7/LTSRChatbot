from __future__ import annotations

import math
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.state import ProcessorPlan, ReplyPlan, SimReport
from utils.llm_json import parse_json_from_llm

from app.lats.evaluator import evaluate_candidate, check_assistant_like_via_llm
from app.lats.reply_compiler import compile_reply_plan_to_processor_plan
from app.lats.reply_planner import plan_reply_via_llm
from app.lats.reflection import build_reflection_patch_via_llm
from app.lats.prompt_utils import (
    build_system_memory_block,
    build_style_profile,
    get_chat_buffer_body_messages,
    safe_text,
    summarize_state_for_planner,
)
from utils.detailed_logging import log_prompt_and_params, log_llm_response, log_computation


VARIANTS_SYSTEM = """你是 ReplyPlanner 的“多样化扩展器”(Expand)。
给定同一轮的约束与上下文，请生成多个不同的 ReplyPlan 候选。
差异必须体现在“对话编排/节奏组织/互动动作选择（答/反问/边界/轻松调侃等）”，而不是只换同义词。

必须保证：
- 第一条就可用（先回应/先态度或结论）
- 多条合起来满足 plan_goals/style_targets/stage_targets/mode budget
- 像同一个人在连续发消息（连贯、人味、不像助手）

请严格输出 JSON：
{
  "candidates": [ ReplyPlan, ReplyPlan, ... ]
}

每个 ReplyPlan 必须包含：
- strategy_tag（必须从下列枚举里选 1 个，用于强制多样性，且同一批 candidates 的 strategy_tag 尽量互不相同）：
  - "direct_answer"
  - "empathy_reflect"
  - "self_disclosure"
  - "light_tease"
  - "ask_back"
  - "co_create"
- messages_count（必须等于 messages 条数）
- must_cover_map（把 must_cover_points 逐条定位到 message id）

其余结构与 ReplyPlanner 输出一致（intent/speech_act/stakes/first_message_role/pacing_strategy/messages/justification）。""".strip()


def _plan_text(plan: Dict[str, Any]) -> str:
    try:
        msgs = plan.get("messages") or []
        if isinstance(msgs, list):
            return "\n".join([str(m.get("content") if isinstance(m, dict) else m) for m in msgs])
    except Exception:
        pass
    return str(plan)


def _infer_strategy_tag(plan: Dict[str, Any]) -> str:
    """最低配策略标签推断：用于 candidates 未输出 strategy_tag 或重复时做兜底。"""
    try:
        sa = str(plan.get("speech_act") or "").strip().lower()
    except Exception:
        sa = ""
    try:
        fmr = str(plan.get("first_message_role") or "").strip().lower()
    except Exception:
        fmr = ""
    text = _plan_text(plan)
    t = text.lower()
    q_cnt = t.count("?") + t.count("？")
    if "tease" in fmr or "light_tease" in fmr:
        return "light_tease"
    if q_cnt >= 1 or "question" in fmr:
        return "ask_back"
    if "empathy" in fmr or "empa" in fmr or "共情" in sa:
        return "empathy_reflect"
    if "advice" in sa or "建议" in sa:
        return "direct_answer"
    # 自我披露：出现“我最近/我其实/我一直/我有点…”
    if any(x in t for x in ["我最近", "我其实", "我一直", "我有点", "我也会", "我喜欢", "我常常"]):
        return "self_disclosure"
    return "direct_answer"


def _diversify_candidates(
    cands: List[Dict[str, Any]],
    *,
    k: int,
    base_tag: Optional[str] = None,
    sim_threshold: float = 0.88,
) -> List[Dict[str, Any]]:
    """
    P1：候选多样性增强（非温度）：strategy_tag 去重 + 文本相似度去重 + 贪心 MMR。
    """
    import difflib

    # 1) ensure strategy_tag
    for c in cands:
        tag = str(c.get("strategy_tag") or "").strip()
        if not tag:
            c["strategy_tag"] = _infer_strategy_tag(c)

    # 2) drop tag duplicates (prefer first), and avoid base_tag if possible
    out: List[Dict[str, Any]] = []
    used_tags: set[str] = set()
    for c in cands:
        tag = str(c.get("strategy_tag") or "").strip()
        if base_tag and tag == base_tag:
            continue
        if tag in used_tags:
            continue
        used_tags.add(tag)
        out.append(c)
        if len(out) >= max(2, min(k, 6)):
            break

    pool = out if out else list(cands)

    # 3) MMR-ish select by minimizing similarity
    selected: List[Dict[str, Any]] = []
    while pool and len(selected) < k:
        if not selected:
            selected.append(pool.pop(0))
            continue
        best_i = None
        best_score = -1e9
        for i, c in enumerate(pool):
            txt = _plan_text(c)
            max_sim = 0.0
            for s in selected:
                max_sim = max(max_sim, difflib.SequenceMatcher(None, txt, _plan_text(s)).ratio())
            score = 1.0 - max_sim
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        cand = pool.pop(best_i)
        # hard dedup by threshold
        too_similar = any(
            difflib.SequenceMatcher(None, _plan_text(cand), _plan_text(s)).ratio() >= sim_threshold
            for s in selected
        )
        if too_similar:
            continue
        selected.append(cand)

    return selected[:k]


def failures_to_actionable_hints(failed_checks: List[Dict[str, str]], max_hints: int = 2) -> str:
    """将 failed_checks 转换为 actionable guidance（避免重复无意义反馈）。"""
    if not failed_checks or not isinstance(failed_checks, list):
        return ""
    # 只取 TopN，避免 prompt 过长
    top_n = failed_checks[:max_hints]
    hints: List[str] = []
    for f in top_n:
        fid = str(f.get("id", "")).strip()
        reason = str(f.get("reason", "")).strip()
        evidence = str(f.get("evidence", "")).strip()
        
        # 转换为 actionable fix（基于常见失败模式）
        if fid == "first_too_short":
            hints.append("确保第一条消息至少 8 字，且必须包含对用户问题的直接回应或明确态度（不能只是铺垫/寒暄）")
        elif fid == "too_many_messages":
            hints.append(f"控制消息条数在 5 条以内，优先合并相关消息（当前: {reason}）")
        elif fid == "message_too_long":
            hints.append("单条消息长度控制在 200 字以内，过长内容应拆分成多条或精简")
        elif fid == "empty_message":
            hints.append("确保每条消息都有实际内容，不能为空")
        elif fid == "must_have_missing":
            hints.append(f"必须覆盖所有 must-have 要求（缺失: {evidence[:40]}...）")
        elif fid == "assistant_like_response":
            hints.append("禁止使用助手式回答（如'我在这里帮助你'、'提供信息'、'为您服务'等），回复应该像真人朋友聊天，自然、直接、有情感")
        elif fid == "empty":
            hints.append("确保至少生成 1 条有效消息")
        else:
            # 通用 fallback：直接转述 reason
            hints.append(f"{reason[:60]}")
    
    if not hints:
        return ""
    return "\n".join([f"- {h}" for h in hints])


def _detect_repeated_failures(
    tree: Dict[str, Any],
    recent_rollouts: int = 3,
) -> Dict[str, int]:
    """检测最近 rollouts 中重复出现的失败模式（用于触发全局 guidelines）。"""
    nodes = tree.get("nodes", {})
    if not nodes:
        return {}
    
    # 收集最近 expanded 节点的失败模式
    failure_counts: Dict[str, int] = {}
    expanded_nodes = [n for n in nodes.values() if n.get("expanded")]
    # 按某种顺序取最近 N 个（简化：取最后 N 个）
    recent_nodes = expanded_nodes[-recent_rollouts:] if len(expanded_nodes) > recent_rollouts else expanded_nodes
    
    for node in recent_nodes:
        rep = node.get("sim_report")
        if not isinstance(rep, dict):
            continue
        fails = rep.get("failed_checks", [])
        if not isinstance(fails, list):
            continue
        for f in fails:
            fid = str(f.get("id", "")).strip()
            if fid:
                failure_counts[fid] = failure_counts.get(fid, 0) + 1
    
    # 只返回出现 >= 2 次的（重复模式）
    return {k: v for k, v in failure_counts.items() if v >= 2}


def _build_reflection_patch(
    repeated_patterns: Dict[str, int],
    state: Dict[str, Any],
    llm_invoker: Any,
) -> Dict[str, Any]:
    """用 LLM 从重复失败模式中生成结构化 patch。"""
    if not repeated_patterns or llm_invoker is None:
        return {}
    
    # 转换为 List[Tuple[str, int]] 格式
    repeated_failures = [(fid, count) for fid, count in repeated_patterns.items()]
    
    # 调用新的结构化 patch 函数
    patch = build_reflection_patch_via_llm(state, llm_invoker, repeated_failures)
    
    return patch


def _ucb1(value_sum: float, visits: int, parent_visits: int, c: float = 1.2) -> float:
    if visits <= 0:
        return float("inf")
    exploit = value_sum / float(visits)
    explore = c * math.sqrt(math.log(max(1, parent_visits)) / float(visits))
    return exploit + explore


def generate_variants_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    *,
    base_plan: Optional[ReplyPlan] = None,
    base_sim_report: Optional[SimReport] = None,
    global_guidelines: Optional[str] = None,
    k: int = 4,
    max_messages: int = 5,
    force_strategy_tags: Optional[List[str]] = None,
) -> List[ReplyPlan]:
    if llm_invoker is None:
        print(f"  [扩展] ⚠ LLM invoker 不可用，跳过变体生成")
        return []

    system_memory = build_system_memory_block(state)
    style_profile = build_style_profile(state)
    requirements = state.get("requirements") or {}
    plan_goals = requirements.get("plan_goals") if isinstance(requirements, dict) else None
    style_targets = requirements.get("style_targets") if isinstance(requirements, dict) else None
    stage_targets = requirements.get("stage_targets") if isinstance(requirements, dict) else None
    snapshot = summarize_state_for_planner(state)
    user_input = safe_text(state.get("external_user_text") or state.get("user_input"))
    strategy = safe_text(state.get("response_strategy"))

    system_prompt = f"""{VARIANTS_SYSTEM}

## Memory (Summary + Retrieved)
{system_memory}

## State Snapshot
{snapshot}

## Style Profile (12D)
{safe_text(style_profile)}

## Requirements (Checklist)
{safe_text(requirements)}

## Hard Targets (Planner MUST obey)
- max_messages_each: {int(requirements.get("max_messages", max_messages) or max_messages)}
- plan_goals.must_cover_points: {safe_text((plan_goals or {}).get("must_cover_points", [])) if isinstance(plan_goals, dict) else "[]"}
- plan_goals.avoid_points: {safe_text((plan_goals or {}).get("avoid_points", [])) if isinstance(plan_goals, dict) else "[]"}
- style_targets(12D): {safe_text(style_targets) if isinstance(style_targets, dict) else "（无）"}
- stage_targets: {safe_text(stage_targets) if isinstance(stage_targets, dict) else "（无）"}
- mode_behavior_targets: {safe_text(requirements.get("mode_behavior_targets", [])) if isinstance(requirements, dict) else "[]"}

## Limits
- candidates: {int(k)}
- max_messages_each: {int(max_messages)}
""".strip()

    hint = f"参考父计划但不要只改同义词：\n{safe_text(base_plan)}" if base_plan else "（无父计划参考）"
    base_tag = None
    if isinstance(base_plan, dict) and base_plan:
        base_tag = str(base_plan.get("strategy_tag") or "").strip() or _infer_strategy_tag(base_plan)
    
    # 方案 C: 父节点失败点转 actionable hints
    actionable_hints = ""
    if base_sim_report:
        base_failures = base_sim_report.get("failed_checks", [])
        if base_failures:
            actionable_hints = failures_to_actionable_hints(base_failures, max_hints=2)
    
    # 方案 A: 全局 guidelines（如果提供）
    guidelines_block = ""
    if global_guidelines:
        guidelines_block = f"\n\n全局指导原则（基于最近搜索经验）：\n{global_guidelines}"
    
    avoid_block = f"\n\n避免事项（必须遵守）：\n{actionable_hints}" if actionable_hints else ""
    
    # P1：候选差异化约束（不是温度）：要求候选 strategy_tag 与 base_tag / 彼此不同
    diversify_block = ""
    if base_tag:
        diversify_block = f"""
差异化硬要求（必须遵守）：
- base_strategy_tag = "{base_tag}"
- 本批 candidates 的 strategy_tag 必须尽量两两不同，且尽量不要等于 base_strategy_tag
- 至少覆盖 2 类不同 strategy_tag（如果 candidates>=2）
""".strip()
    if isinstance(force_strategy_tags, list) and force_strategy_tags:
        tags_txt = ", ".join([str(x).strip() for x in force_strategy_tags if str(x).strip()][:6])
        if tags_txt:
            extra = f'\n- 本次必须优先覆盖这些 strategy_tag（尽量一一对应到不同候选）：{tags_txt}'
            diversify_block = (diversify_block + extra).strip() if diversify_block else extra.strip()

    # 为了能做去重/MMR，允许 LLM 先多产一些，再在代码里筛到 k 个
    raw_k = int(max(k, min(8, k * 2)))

    task = f"""用户输入：
{user_input}

导演策略：
{strategy}

父计划参考：
{hint}{avoid_block}{guidelines_block}

{diversify_block}

请生成 {int(raw_k)} 个编排差异明显的 ReplyPlan 候选。""".strip()

    body_messages = get_chat_buffer_body_messages(state, limit=20)
    
    # 记录扩展变体生成的提示词和参数
    log_prompt_and_params(
        "LATS Variants Generator",
        system_prompt=system_prompt,
        user_prompt=task,
        messages=body_messages,
        params={
            "user_input": user_input,
            "strategy": strategy,
            "base_plan": str(base_plan)[:200] + "..." if base_plan and len(str(base_plan)) > 200 else str(base_plan) if base_plan else None,
            "base_failures": base_sim_report.get("failed_checks", []) if base_sim_report else [],
            "actionable_hints": actionable_hints,
            "global_guidelines": global_guidelines,
            "k": k,
            "max_messages": max_messages,
        }
    )
    
    try:
        resp = llm_invoker.invoke(
            [SystemMessage(content=system_prompt), *body_messages, HumanMessage(content=task)]
        )
        content = getattr(resp, "content", "") or ""
        data = parse_json_from_llm(content)
        if not isinstance(data, dict):
            return []
        cands = data.get("candidates")
        if not isinstance(cands, list):
            return []
        out: List[ReplyPlan] = []
        for c in cands:
            if not (isinstance(c, dict) and isinstance(c.get("messages"), list) and c.get("messages")):
                continue
            # 规范化：messages_count 必须匹配
            try:
                c["messages_count"] = int(c.get("messages_count") or len(c["messages"]))
            except Exception:
                c["messages_count"] = len(c.get("messages") or [])
            if int(c.get("messages_count") or 0) != len(c.get("messages") or []):
                c["messages_count"] = len(c.get("messages") or [])

            # 如果存在 must_cover_points，则至少保证 must_cover_map 字段为 dict，便于下游提示缺失
            if isinstance(plan_goals, dict) and plan_goals.get("must_cover_points"):
                if not isinstance(c.get("must_cover_map"), dict):
                    c["must_cover_map"] = {}

            out.append(c)  # type: ignore[list-item]
        
        # 记录 LLM 响应
        log_llm_response("LATS Variants Generator", resp, parsed_result={"candidates_count": len(out), "candidates": out[:2]})
        
        # P1：去重 + 差异化筛选（避免同质候选导致“越聊越像固定模板人格”）
        diversified = _diversify_candidates([x for x in out if isinstance(x, dict)], k=int(k), base_tag=base_tag)
        return diversified[:k]  # type: ignore[return-value]
    except Exception as e:
        print(f"  [扩展] ❌ 异常: {e}")
        return []


def lats_search_best_plan(
    state: Dict[str, Any],
    llm_planner: Any,
    *,
    llm_soft_scorer: Any = None,
    rollouts: int = 6,
    expand_k: int = 4,
    max_messages: int = 5,
) -> Tuple[Optional[ReplyPlan], Optional[ProcessorPlan], Optional[SimReport], Dict[str, Any]]:
    """MCTS-like LATS search over ReplyPlan (planner output)."""
    requirements = state.get("requirements") or {}
    
    print(f"\n[LATS] ========== 开始搜索 (rollouts={rollouts}, expand_k={expand_k}, max_messages={max_messages}) ==========")
    user_input = safe_text(state.get("external_user_text") or state.get("user_input", ""))[:60]
    print(f"[LATS] 用户输入: {user_input}...")
    print(f"[LATS] 硬约束: max_messages={max_messages}, must_have={requirements.get('must_have', [])}")

    # 方案 A: 检测重复失败模式并生成结构化 patch（如果树中已有节点）
    tree: Dict[str, Any] = {"nodes": {}, "root_id": "root", "best_id": "root"}
    nodes = tree["nodes"]
    reflection_patch: Dict[str, Any] = {}
    global_guidelines: Optional[str] = None  # 向后兼容：转换为文本格式
    
    # 如果 state 中已有 lats_tree，检测重复模式
    existing_tree = state.get("lats_tree")
    # 0) 优先使用“带 TTL 的 active_patch”，避免每轮叠加新 patch 导致漂移
    active_patch = None
    if isinstance(existing_tree, dict):
        ap = existing_tree.get("active_patch")
        if isinstance(ap, dict):
            try:
                ttl_rem = int(ap.get("ttl_remaining") or 0)
            except Exception:
                ttl_rem = 0
            if ttl_rem > 0:
                active_patch = dict(ap)

    if isinstance(active_patch, dict) and active_patch:
        print(f"[LATS] 使用 active_patch (ttl_remaining={int(active_patch.get('ttl_remaining') or 0)})")
        from app.lats.requirements import apply_reflection_patch
        requirements = apply_reflection_patch(requirements, active_patch)
        # 消耗 TTL
        try:
            active_patch["ttl_remaining"] = max(0, int(active_patch.get("ttl_remaining") or 0) - 1)
        except Exception:
            active_patch["ttl_remaining"] = 0
        reflection_patch = active_patch
        tree["active_patch"] = active_patch
    elif isinstance(existing_tree, dict) and existing_tree.get("nodes"):
        # 1) 没有 active_patch 时才根据重复失败生成新的 patch
        repeated = _detect_repeated_failures(existing_tree, recent_rollouts=3)
        if repeated:
            print(f"[LATS] 检测到重复失败模式: {repeated}")
            reflection_patch = _build_reflection_patch(repeated, state, llm_planner)
            if reflection_patch:
                # 默认 TTL（最低配）：避免 patch 永久叠加
                if "ttl_turns" not in reflection_patch:
                    reflection_patch["ttl_turns"] = int(state.get("lats_patch_ttl_turns", 3) or 3)
                if "ttl_remaining" not in reflection_patch:
                    reflection_patch["ttl_remaining"] = int(reflection_patch.get("ttl_turns") or 3)

                print(f"[LATS] ✓ 结构化 patch: {reflection_patch}")
                # 检查 stop_now
                if reflection_patch.get("stop_now"):
                    print(f"[LATS] ⚠ Patch 要求停止扩展（stop_now=true）")
                # 应用 reflection_patch 到 requirements
                from app.lats.requirements import apply_reflection_patch
                requirements = apply_reflection_patch(requirements, reflection_patch)
                print(f"[LATS] ✓ 已应用 reflection_patch 到 requirements")

                # 消耗 1 次 TTL（本轮已应用）
                try:
                    reflection_patch["ttl_remaining"] = max(0, int(reflection_patch.get("ttl_remaining") or 0) - 1)
                except Exception:
                    reflection_patch["ttl_remaining"] = 0
                tree["active_patch"] = dict(reflection_patch)

                # 向后兼容：转换为文本格式的 global_guidelines
                from app.lats.reflection import build_global_guidelines_via_llm
                repeated_list = [(fid, count) for fid, count in repeated.items()]
                global_guidelines = build_global_guidelines_via_llm(state, llm_planner, repeated_list)
            else:
                print(f"[LATS] ⚠ 结构化 patch 生成失败")

    root_plan = plan_reply_via_llm(state, llm_planner, max_messages=max_messages, global_guidelines=global_guidelines)
    if not root_plan:
        print("[LATS] ❌ 根计划生成失败")
        return None, None, None, {"error": "root_plan_failed"}
    
    print(f"[LATS] ✓ 根计划已生成: intent={root_plan.get('intent', '')[:40]}..., messages={len(root_plan.get('messages', []))}条")

    def _eval(plan: ReplyPlan) -> Tuple[ProcessorPlan, SimReport]:
        # 使用修正后的 requirements（已应用 reflection_patch）
        proc = compile_reply_plan_to_processor_plan(plan, state, max_messages=max_messages)
        rep = evaluate_candidate(
            state,
            plan,
            proc,
            requirements,  # 使用修正后的 requirements
            llm_soft_scorer=llm_soft_scorer,
        )
        return proc, rep

    def _eval_fast(plan: ReplyPlan) -> Tuple[ProcessorPlan, SimReport]:
        """Fast eval without LLM soft scorer (hard_gate + heuristic only)."""
        # 使用修正后的 requirements（已应用 reflection_patch）
        proc = compile_reply_plan_to_processor_plan(plan, state, max_messages=max_messages)
        rep = evaluate_candidate(
            state,
            plan,
            proc,
            requirements,  # 使用修正后的 requirements
            llm_soft_scorer=None,
        )
        return proc, rep


    root_proc, root_rep = _eval(root_plan)
    root_score = float(root_rep.get("eval_score", 0.0) or 0.0)
    root_found = bool(root_rep.get("found_solution"))
    root_fails = root_rep.get("failed_checks", [])
    
    print(f"[LATS] [根节点] 评估完成:")
    print(f"  - 分数: {root_score:.4f}")
    print(f"  - 通过硬门槛: {root_found}")
    print(f"  - 失败检查: {len(root_fails)}项")
    if root_fails:
        for f in root_fails[:3]:
            print(f"    × {f.get('id', '')}: {f.get('reason', '')}")
    print(f"  - 最终消息数: {len(root_proc.get('messages', []))}")
    
    nodes["root"] = {
        "id": "root",
        "parent": None,
        "children": [],
        "visits": 1,
        "value_sum": root_score,
        "reply_plan": root_plan,
        "processor_plan": root_proc,
        "sim_report": root_rep,
        "expanded": False,
    }

    best_id = "root"
    best_score = root_score

    # P0: 至少跑完 N 次 rollout 才允许 root_plan 早退（否则树永远是“根节点 → 选一个回复”）
    try:
        min_rollouts_before_early_exit = int(state.get("lats_min_rollouts_before_early_exit", 1) or 1)
    except Exception:
        min_rollouts_before_early_exit = 1
    if min_rollouts_before_early_exit < 0:
        min_rollouts_before_early_exit = 0

    # 若预算 rollouts 太小，至少保证能跑到 min_rollouts（否则 early-exit 禁掉了但 rollouts=0 会啥也不做）
    try:
        if int(rollouts) < int(min_rollouts_before_early_exit):
            rollouts = int(min_rollouts_before_early_exit)
    except Exception:
        rollouts = int(min_rollouts_before_early_exit)

    # Early-exit: 仅当多条件同时满足才早退，避免“根计划一过线就不探索”
    # initiating 阶段默认阈值更高（更容易是“通用开场白”，需要多探索一层才能更像“这个人”）
    stage_id = str(state.get("current_stage") or "")
    if not stage_id and isinstance(requirements, dict):
        st = requirements.get("stage_targets") or {}
        if isinstance(st, dict) and st.get("stage"):
            stage_id = str(st.get("stage"))
    stage_id = stage_id or "initiating"

    default_early_exit_score = 0.80 if stage_id == "initiating" else 0.65
    default_plan_min = 0.75 if stage_id == "initiating" else 0.70
    default_assistant_max = 0.22 if stage_id == "initiating" else 0.25
    default_mode_min = 0.60 if stage_id == "initiating" else 0.55

    early_exit_score = float(state.get("lats_early_exit_root_score", default_early_exit_score) or default_early_exit_score)
    early_exit_plan_min = float(state.get("lats_early_exit_plan_alignment_min", default_plan_min) or default_plan_min)
    early_exit_assistant_max = float(state.get("lats_early_exit_assistantiness_max", default_assistant_max) or default_assistant_max)
    early_exit_mode_min = float(state.get("lats_early_exit_mode_fit_min", default_mode_min) or default_mode_min)

    bd_root = root_rep.get("score_breakdown", {}) if isinstance(root_rep, dict) else {}
    # P0：当 llm_soft_scorer 可用时，early-exit 必须“以 LLM gates 为准”。
    # 如果 breakdown 中缺少对应字段（例如 LLM 解析失败），就按保守失败处理，从而阻止 early-exit。
    can_use_llm_gates = (llm_soft_scorer is not None and isinstance(bd_root, dict))
    if can_use_llm_gates:
        llm_plan = float(bd_root.get("llm_plan_alignment", 0.0) or 0.0)
        llm_mode_fit = float(bd_root.get("llm_mode_behavior_fit", 0.0) or 0.0)
        # 缺失 assistantiness 时按保守失败（更像助手），避免误早退
        assistantiness = float(bd_root.get("assistantiness", bd_root.get("llm_assistantiness", 1.0)) or 1.0)
    else:
        llm_plan = 0.0
        llm_mode_fit = 0.0
        assistantiness = 1.0
    has_mode_gate = isinstance(bd_root, dict) and ("llm_mode_behavior_fit" in bd_root)
    disable_early_exit_flag = bool(state.get("lats_disable_early_exit"))
    allow_root_early_exit = (min_rollouts_before_early_exit <= 0)
    early_exit_ok = (
        root_found and
        best_score >= early_exit_score and
        (not disable_early_exit_flag) and
        allow_root_early_exit and
        (
            not can_use_llm_gates
            or (
                llm_plan >= early_exit_plan_min
                and assistantiness <= early_exit_assistant_max
                and (not has_mode_gate or llm_mode_fit >= early_exit_mode_min)
            )
        )
    )
    if early_exit_ok:
        if can_use_llm_gates:
            print(
                f"[LATS] ⚡ 早退: 根计划多条件满足 "
                f"(score={best_score:.4f}>= {early_exit_score:.2f}, plan={llm_plan:.2f}>= {early_exit_plan_min:.2f}, "
                f"assistant={assistantiness:.2f}<= {early_exit_assistant_max:.2f}, "
                f"mode_fit={llm_mode_fit:.2f}{'>=' + str(round(early_exit_mode_min,2)) if has_mode_gate else '(skip)'} )"
            )
        else:
            print(f"[LATS] ⚡ 早退: 根计划满足条件 (score={best_score:.4f} >= {early_exit_score:.2f}, found_solution=True)")
        tree["best_id"] = best_id
        return root_plan, root_proc, root_rep, tree
    else:
        # 让“你以为关了 early-exit 但其实没关上”的问题在日志中直接可见
        if root_found and best_score >= early_exit_score and (not allow_root_early_exit):
            print(
                f"[LATS] ⏭ 跳过 root 早退（min_rollouts_before_early_exit={min_rollouts_before_early_exit}，"
                f"disable_early_exit={disable_early_exit_flag}）"
            )
        elif root_found and best_score >= early_exit_score and disable_early_exit_flag:
            print(f"[LATS] ⏭ root 早退被禁用（lats_disable_early_exit=True）")
    
    print(f"[LATS] 开始 {rollouts} 轮 rollout 搜索...")

    for rollout_idx in range(int(rollouts)):
        print(f"\n[LATS] --- Rollout {rollout_idx + 1}/{rollouts} ---")
        # Selection: choose best UCB leaf
        current_id = "root"
        path = [current_id]
        selection_path_str = ["root"]
        ucb_details = []  # 在循环外初始化，避免未定义错误
        while True:
            node = nodes[current_id]
            children = node.get("children") or []
            if not children:
                break
            parent_visits = int(node.get("visits", 1) or 1)
            best_child = None
            best_ucb = -1e9
            ucb_details = []  # 每次循环重新初始化
            for cid in children:
                ch = nodes[cid]
                visits = int(ch.get("visits", 0) or 0)
                value_sum = float(ch.get("value_sum", 0.0))
                u = _ucb1(value_sum, visits, parent_visits)
                ucb_details.append((cid[:6], f"{u:.3f}", f"v={value_sum:.3f}", f"n={visits}"))
                if u > best_ucb:
                    best_ucb = u
                    best_child = cid
            
            # 记录 UCB 计算过程
            if rollout_idx == 0 or len(ucb_details) > 0:  # 只在第一次或有关键选择时记录
                log_computation(
                    "LATS Selection",
                    f"UCB计算 (Rollout {rollout_idx + 1}, Node {current_id[:6]})",
                    inputs={
                        "parent_visits": parent_visits,
                        "children_count": len(children),
                    },
                    intermediate_steps=[
                        {
                            "child_id": cid,
                            "visits": visits,
                            "value_sum": value_sum,
                            "ucb_score": u,
                            "exploit": value_sum / max(1, visits),
                            "explore": 1.2 * math.sqrt(math.log(max(1, parent_visits)) / max(1, visits)),
                        }
                        for cid, visits, value_sum, u in [
                            (cid[:6], int(nodes[cid].get("visits", 0) or 0), float(nodes[cid].get("value_sum", 0.0)), _ucb1(float(nodes[cid].get("value_sum", 0.0)), int(nodes[cid].get("visits", 0) or 0), parent_visits))
                            for cid in children[:5]  # 只记录前5个
                        ]
                    ],
                    outputs={
                        "selected_child": best_child[:6] if best_child else None,
                        "best_ucb": best_ucb,
                    },
                )
            if best_child is None:
                break
            current_id = best_child
            path.append(current_id)
            selection_path_str.append(current_id[:6])
            if not nodes[current_id].get("expanded"):
                break
        
        print(f"  [选择] 路径: {' -> '.join(selection_path_str)}")
        # P0：每个 rollout 都记录 path_length + path（避免只看 rollout0 误判树没长）
        log_computation(
            "LATS Rollout Path",
            f"SelectionPath (Rollout {rollout_idx + 1})",
            inputs={
                "rollout_idx": rollout_idx + 1,
                "path_length": len(path),
                "path": list(selection_path_str),
                "leaf_id": current_id[:6] if current_id != "root" else "root",
                "leaf_expanded": bool(nodes.get(current_id, {}).get("expanded")),
            },
        )
        if len(ucb_details) > 0:
            top_ucb = sorted(ucb_details, key=lambda x: float(x[1]), reverse=True)[:3]
            print(f"  [选择] UCB Top3: {', '.join([f'{cid}({ucb})' for cid, ucb, _, _ in top_ucb])}")

        leaf = nodes[current_id]
        leaf_id_short = current_id[:6] if current_id != "root" else "root"
        if not leaf.get("expanded"):
            # Expand
            print(f"  [扩展] 节点 {leaf_id_short}: 生成 {expand_k} 个变体候选...")
            base = leaf.get("reply_plan") if isinstance(leaf.get("reply_plan"), dict) else None
            base_intent = base.get("intent", "")[:30] if base else "无"
            print(f"  [扩展] 父计划意图: {base_intent}...")
            
            # 获取父节点的 sim_report（用于方案 C：父节点失败点反馈）
            base_sim_report = leaf.get("sim_report") if isinstance(leaf.get("sim_report"), dict) else None
            
            variants = generate_variants_via_llm(
                state,
                llm_planner,
                base_plan=base,
                base_sim_report=base_sim_report,
                global_guidelines=global_guidelines,
                k=expand_k,
                max_messages=max_messages,
            )
            # P1：行为签名覆盖约束（至少 3 类 strategy_tag）
            # 若候选同质化，LATS 只能在“同一种模板”里 rerank，长期会漂成固定腔调。
            try:
                min_tags = int(state.get("lats_min_strategy_tags", 3) or 3)
            except Exception:
                min_tags = 3
            min_tags = max(0, min(min_tags, int(expand_k)))
            if min_tags >= 3 and len(variants) >= 3:
                try:
                    present = []
                    for v in variants:
                        if isinstance(v, dict):
                            t = str(v.get("strategy_tag") or "").strip() or _infer_strategy_tag(v)
                            present.append(t)
                    uniq = []
                    for t in present:
                        if t and t not in uniq:
                            uniq.append(t)
                    if len(uniq) < min_tags:
                        desired = ["direct_answer", "ask_back", "self_disclosure", "empathy", "light_tease", "co_create"]
                        missing = [t for t in desired if t not in uniq][: max(0, min_tags - len(uniq))]
                        if missing:
                            print(f"  [扩展] ⚠ strategy_tag 覆盖不足({len(uniq)}/{min_tags})，二次补齐: {missing}")
                            more = generate_variants_via_llm(
                                state,
                                llm_planner,
                                base_plan=base,
                                base_sim_report=base_sim_report,
                                global_guidelines=global_guidelines,
                                k=expand_k,
                                max_messages=max_messages,
                                force_strategy_tags=missing,
                            )
                            # merge then diversify again
                            merged = [x for x in (variants + (more or [])) if isinstance(x, dict)]
                            variants = _diversify_candidates(merged, k=int(expand_k), base_tag=base_tag)[: int(expand_k)]
                except Exception:
                    pass
            print(f"  [扩展] ✓ 生成了 {len(variants)} 个变体")

            # ---- Two-stage evaluation ----
            # Stage 1) cheap ranking: hard_gate + heuristic only
            staged: List[Dict[str, Any]] = []
            for v in variants:
                proc_fast, rep_fast = _eval_fast(v)
                staged.append({"plan": v, "proc": proc_fast, "rep": rep_fast})

            # Rank: prefer hard_pass (found_solution) then higher score
            staged.sort(
                key=lambda x: (
                    1 if bool((x.get("rep") or {}).get("found_solution")) else 0,
                    float((x.get("rep") or {}).get("eval_score", 0.0) or 0.0),
                ),
                reverse=True,
            )

            # Stage 1.5) 轻量级 LLM classifier：对通过 hard_gate 的 top2-3 候选检查助手式回答
            assistant_check_top_n = int(state.get("lats_assistant_check_top_n", 3) or 3)
            assistant_check_top_n = max(0, min(assistant_check_top_n, len(staged)))
            
            # 只对通过 hard_gate 的候选进行检查
            hard_pass_candidates = [item for item in staged[:assistant_check_top_n] 
                                    if bool((item.get("rep") or {}).get("found_solution"))]
            
            if hard_pass_candidates and llm_soft_scorer is not None:
                print(f"  [助手检测] 对 {len(hard_pass_candidates)} 个通过硬门槛的候选进行 LLM 助手味检测...")
                for item in hard_pass_candidates:
                    proc = item.get("proc")
                    msgs = proc.get("messages", []) if proc else []
                    if msgs:
                        llm_check_result = check_assistant_like_via_llm(msgs, llm_soft_scorer)
                        if llm_check_result is not None:
                            is_assistant, confidence = llm_check_result
                            if is_assistant and confidence > 0.5:
                                # 标记为助手式回答，降低评分
                                rep = item.get("rep", {})
                                current_score = float(rep.get("eval_score", 0.0) or 0.0)
                                # 大幅惩罚：降低到原来的 20%
                                rep["eval_score"] = current_score * 0.2
                                rep["failed_checks"] = rep.get("failed_checks", []) + [{
                                    "id": "assistant_like_response_llm",
                                    "reason": f"LLM检测到助手式回答（confidence={confidence:.2f}），不符合拟人化要求",
                                    "evidence": "\n".join([str(m) for m in msgs])[:200],
                                }]
                                rep["found_solution"] = False  # 标记为未通过
                                print(f"    [助手检测] ⚠ 检测到助手式回答 (confidence={confidence:.2f})，已惩罚")
                            else:
                                print(f"    [助手检测] ✓ 非助手式回答 (confidence={1.0-confidence:.2f})")
                
                # 重新排序（因为评分可能已改变）
                staged.sort(
                    key=lambda x: (
                        1 if bool((x.get("rep") or {}).get("found_solution")) else 0,
                        float((x.get("rep") or {}).get("eval_score", 0.0) or 0.0),
                    ),
                    reverse=True,
                )

            enable_llm_soft = llm_soft_scorer is not None
            # P0：无论 found_solution 与否，至少对 Top1 跑一次 LLM soft scorer（避免启发式短路误判）
            # 允许用户显式设置为 0 来完全禁用（但默认不允许短路）
            raw_top_n = state.get("lats_llm_soft_top_n")
            try:
                top_n = int(raw_top_n) if raw_top_n is not None else 1
            except Exception:
                top_n = 1
            if enable_llm_soft and top_n <= 0:
                top_n = 1
            max_conc = int(state.get("lats_llm_soft_max_concurrency") or 2)
            top_n = max(0, min(top_n, len(staged)))

            print(f"  [评审] 两阶段评审: cheap评审={len(staged)}个, LLM精评={'启用' if enable_llm_soft else '禁用'}")
            if enable_llm_soft and top_n > 0:
                print(f"  [评审] 仅对 Top{top_n} 做 LLM soft scorer，并发={min(max_conc, top_n)}")

                def _eval_llm_only(item: Dict[str, Any]) -> Tuple[SimReport, str]:
                    plan = item["plan"]
                    proc = item["proc"]
                    rep = evaluate_candidate(
                        state,
                        plan,
                        proc,
                        requirements,
                        llm_soft_scorer=llm_soft_scorer,
                    )
                    return rep, safe_text(plan.get("intent", ""))[:30]

                # 并行只做 LLM soft scorer（返回 rep），树写入主线程串行
                with ThreadPoolExecutor(max_workers=min(max_conc, top_n)) as ex:
                    futures = {ex.submit(_eval_llm_only, staged[i]): i for i in range(top_n)}
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        try:
                            rep_llm, intent_hint = fut.result()
                            staged[idx]["rep"] = rep_llm
                            print(f"    [LLM精评] idx={idx+1}/{top_n}, intent={intent_hint}..., score={float(rep_llm.get('eval_score',0.0) or 0.0):.4f}")
                        except Exception as e:
                            print(f"    [LLM精评] ⚠ idx={idx+1} 失败: {str(e)[:80]}")

            # Stage 2) serialize tree insertion & best update (no locks needed)
            for item in staged:
                v = item["plan"]
                proc = item["proc"]
                rep = item["rep"]
                nid = str(uuid.uuid4())[:8]
                score = float(rep.get("eval_score", 0.0) or 0.0)
                found = bool(rep.get("found_solution"))
                fails = rep.get("failed_checks", [])
                msg_count = len(proc.get("messages", []))

                nodes[nid] = {
                    "id": nid,
                    "parent": current_id,
                    "children": [],
                    "visits": 1,
                    "value_sum": score,
                    "reply_plan": v,
                    "processor_plan": proc,
                    "sim_report": rep,
                    "expanded": False,
                }
                leaf.setdefault("children", []).append(nid)

                print(f"    [{nid[:6]}] score={score:.4f}, found={found}, msgs={msg_count}, fails={len(fails)}")
                if fails:
                    print(f"      × {fails[0].get('id', '')}: {fails[0].get('reason', '')[:50]}")

                if score > best_score and found:
                    best_score = score
                    best_id = nid
                    print(f"    ⭐ 新最佳候选: {nid[:6]} (score={best_score:.4f})")
            leaf["expanded"] = True
        else:
            print(f"  [跳过] 节点 {leaf_id_short} 已扩展")

        # Backprop: update path with best leaf score (value of current node)
        reward = float(nodes[current_id].get("sim_report", {}).get("eval_score", 0.0) or 0.0)
        print(f"  [回传] reward={reward:.4f}, 路径节点数={len(path)}")
        
        # 记录回传过程
        backprop_updates = []
        for pid in path:
            n = nodes[pid]
            old_visits = int(n.get("visits", 0) or 0)
            old_value = float(n.get("value_sum", 0.0) or 0.0)
            n["visits"] = old_visits + 1
            n["value_sum"] = old_value + reward
            backprop_updates.append({
                "node_id": pid[:6] if pid != "root" else "root",
                "old_visits": old_visits,
                "new_visits": old_visits + 1,
                "old_value_sum": old_value,
                "new_value_sum": old_value + reward,
            })
        
        # P0：每个 rollout 都记录回传 path_length（否则结构化日志总是 rollout0=1）
        log_computation(
            "LATS Backpropagation",
            f"回传更新 (Rollout {rollout_idx + 1})",
            inputs={"reward": reward, "path_length": len(path), "path": list(selection_path_str)},
            outputs={"updates": backprop_updates},
        )

        # Early-stop: 找到很高质量解就收工（同样使用多条件门槛）
        # 压测/探索模式下允许关闭早停（强制跑完 rollouts）
        if bool(state.get("lats_disable_early_exit")):
            continue
        global_stop_score = float(state.get("lats_early_stop_score", 0.85) or 0.85)
        if best_score >= global_stop_score:
            best_node = nodes.get(best_id) or {}
            rep_best = best_node.get("sim_report", {}) if isinstance(best_node, dict) else {}
            bd_best = rep_best.get("score_breakdown", {}) if isinstance(rep_best, dict) else {}
            llm_plan_b = float(bd_best.get("llm_plan_alignment", 0.0) or 0.0) if isinstance(bd_best, dict) else 0.0
            llm_mode_fit_b = float(bd_best.get("llm_mode_behavior_fit", 0.0) or 0.0) if isinstance(bd_best, dict) else 0.0
            assistant_b = float(bd_best.get("assistantiness", bd_best.get("llm_assistantiness", 0.0)) or 0.0) if isinstance(bd_best, dict) else 0.0
            found_b = bool(rep_best.get("found_solution")) if isinstance(rep_best, dict) else False

            can_use_llm_gates_b = (
                llm_soft_scorer is not None
                and isinstance(bd_best, dict)
                and ("llm_plan_alignment" in bd_best or "assistantiness" in bd_best or "llm_assistantiness" in bd_best)
            )
            has_mode_gate_b = isinstance(bd_best, dict) and ("llm_mode_behavior_fit" in bd_best)
            ok_b = (
                found_b and
                (
                    not can_use_llm_gates_b
                    or (
                        llm_plan_b >= early_exit_plan_min
                        and assistant_b <= early_exit_assistant_max
                        and (not has_mode_gate_b or llm_mode_fit_b >= early_exit_mode_min)
                    )
                )
            )
            if ok_b:
                print(f"[LATS] ⚡ 早停: 找到高质量解 (score={best_score:.4f} >= {global_stop_score:.2f})")
                break

    tree["best_id"] = best_id
    best_node = nodes.get(best_id) or nodes["root"]
    best_final_score = float(best_node.get("sim_report", {}).get("eval_score", 0.0) or 0.0)
    best_final_found = bool(best_node.get("sim_report", {}).get("found_solution"))
    best_final_msgs = len(best_node.get("processor_plan", {}).get("messages", []))
    
    print(f"\n[LATS] ========== 搜索完成 ==========")
    print(f"[LATS] 最佳节点: {best_id[:6] if best_id != 'root' else 'root'}")
    print(f"[LATS] 最终分数: {best_final_score:.4f}")
    print(f"[LATS] 通过硬门槛: {best_final_found}")
    print(f"[LATS] 最终消息数: {best_final_msgs}")
    
    # 显示最佳计划的关键信息
    best_plan = best_node.get("reply_plan", {})
    best_proc_plan_full = best_node.get("processor_plan", {})
    if best_plan:
        best_intent = best_plan.get("intent", "")[:50]
        best_pacing = best_plan.get("pacing_strategy", "")[:50]
        print(f"[LATS] 最佳计划意图: {best_intent}...")
        print(f"[LATS] 节奏策略: {best_pacing}...")
    
    # 显示延迟规划
    if best_proc_plan_full:
        delays = best_proc_plan_full.get("delays", [])
        actions = best_proc_plan_full.get("actions", [])
        msgs = best_proc_plan_full.get("messages", [])
        if delays and msgs:
            total_delay = sum(float(d) for d in delays)
            print(f"[LATS] 延迟规划:")
            print(f"  - 总延迟: {total_delay:.2f}秒")
            for i, (msg, delay, action) in enumerate(zip(msgs, delays, actions)):
                msg_preview = (str(msg) or "")[:30]
                delay_val = float(delay or 0.0)
                action_val = str(action or "typing")
                print(f"  [{i+1}] delay={delay_val:.2f}s, action={action_val}, \"{msg_preview}...\"")
    
    # 显示评估详情
    best_report_full = best_node.get("sim_report", {})
    if best_report_full:
        breakdown = best_report_full.get("score_breakdown", {})
        if breakdown:
            top_scores = sorted(breakdown.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            print(f"[LATS] 评分详情: {', '.join([f'{k}={v:.3f}' for k, v in top_scores])}")
        fails = best_report_full.get("failed_checks", [])
        if fails:
            print(f"[LATS] ⚠ 仍有 {len(fails)} 项失败检查")
    
    print(f"[LATS] 树节点总数: {len(nodes)}")
    # 统计树结构
    expanded_count = sum(1 for n in nodes.values() if n.get("expanded"))
    children_count = sum(len(n.get("children", [])) for n in nodes.values())
    print(f"[LATS] 树统计: 已扩展节点={expanded_count}, 总子节点数={children_count}")

    # P0：补充“调试不误导”的树/路径汇总（树深度不是未来回合 lookahead，只表示“变体的变体”深度）
    non_root = [n for k, n in nodes.items() if k != "root" and isinstance(n, dict)]
    revisit_nodes = sum(1 for n in non_root if int(n.get("visits", 0) or 0) > 1)
    revisit_rate = (revisit_nodes / max(1, len(non_root))) if non_root else 0.0

    def _depth(nid: str) -> int:
        d = 1  # include root
        cur = nid
        while cur and cur in nodes and cur != "root":
            d += 1
            cur = str((nodes.get(cur) or {}).get("parent") or "")
            if d > 32:
                break
        return d

    depths = [_depth(k) for k in nodes.keys() if k != "root"]
    max_depth = max(depths) if depths else 1
    avg_depth = (sum(depths) / max(1, len(depths))) if depths else 1.0

    print(f"[LATS] 路径统计(树深度估算): max_depth={max_depth}, avg_depth={avg_depth:.2f}, revisit_rate={revisit_rate:.2f}")
    log_computation(
        "LATS Summary",
        "TreeStats",
        inputs={"nodes": len(nodes), "expanded": expanded_count, "children": children_count},
        outputs={"max_depth": max_depth, "avg_depth": round(avg_depth, 3), "revisit_rate": round(revisit_rate, 3)},
    )
    
    # ==========================================
    # 对最终候选必跑一次 LLM scorer（用于反思与下一轮）
    # ==========================================
    final_best_plan = best_node.get("reply_plan")
    final_best_proc_plan = best_node.get("processor_plan")
    final_best_report = best_node.get("sim_report")
    
    if llm_soft_scorer and final_best_plan and final_best_proc_plan:
        print(f"[LATS] 🔍 对最终候选运行 LLM scorer（用于反思与下一轮）...")
        try:
            final_llm_report = evaluate_candidate(
                state,
                final_best_plan,
                final_best_proc_plan,
                requirements,
                llm_soft_scorer=llm_soft_scorer,
            )
            # 更新最终报告（保留原有信息，但用 LLM 评分覆盖）
            if final_llm_report:
                final_best_report = final_llm_report
                print(f"[LATS] ✅ LLM scorer 完成: score={final_llm_report.get('eval_score', 0.0):.4f}")
                breakdown = final_llm_report.get("score_breakdown", {})
                # 检查是否包含 plan_alignment、style_adherence、stage_fit
                if "plan_alignment" in breakdown or "llm_plan_alignment" in breakdown:
                    plan_align = breakdown.get("plan_alignment") or breakdown.get("llm_plan_alignment", 0.0)
                    print(f"[LATS]   - plan_alignment: {plan_align:.3f}")
                if "style_adherence" in breakdown or "llm_style_adherence" in breakdown:
                    style_adhere = breakdown.get("style_adherence") or breakdown.get("llm_style_adherence", 0.0)
                    print(f"[LATS]   - style_adherence: {style_adhere:.3f}")
                if "stage_fit" in breakdown or "llm_stage_fit" in breakdown:
                    stage_fit_val = breakdown.get("stage_fit") or breakdown.get("llm_stage_fit", 0.0)
                    print(f"[LATS]   - stage_fit: {stage_fit_val:.3f}")
        except Exception as e:
            print(f"[LATS] ⚠ 最终候选 LLM scorer 失败: {str(e)[:100]}")
            # 失败时继续使用原有报告
    
    return (
        final_best_plan,
        final_best_proc_plan,
        final_best_report,
        tree,
    )


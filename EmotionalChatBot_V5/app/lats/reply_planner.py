from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.state import ReplyPlan
from src.schemas import ReplyPlannerSingle, ReplyPlannerCandidates
from utils.external_text import strip_candidate_prefix
from utils.llm_json import parse_json_from_llm
from utils.detailed_logging import log_prompt_and_params, log_llm_response

from app.lats.prompt_utils import (
    build_style_profile,
    build_system_memory_block,
    get_chat_buffer_body_messages,
    get_chat_buffer_body_messages_with_time_slices,
    safe_text,
)
from app.services.llm import set_current_node, reset_current_node

# 限制回复计划生成的最大 token，避免输出过长被截断导致 LengthFinishReasonError。可通过 LTSR_REPLY_PLAN_MAX_TOKENS 覆盖（默认 8192）
def _reply_plan_max_tokens() -> int:
    try:
        v = (os.getenv("LTSR_REPLY_PLAN_MAX_TOKENS") or "").strip()
        if v:
            return max(1024, min(16384, int(v)))
    except Exception:
        pass
    return 8192


from utils.time_context import build_time_context_block, TIME_SLICE_BEHAVIOR_RULES

try:
    from utils.yaml_loader import load_stage_by_id
except Exception:
    load_stage_by_id = None


def _get_stage_prompts(state: Dict[str, Any]) -> tuple[str, str]:
    """从当前阶段配置取 distribution_prompt 与 intent_prompt（供 reply planner 注入 LLM）。"""
    distribution_prompt = ""
    intent_prompt = ""
    if load_stage_by_id is None:
        return distribution_prompt, intent_prompt
    stage_id = str(state.get("current_stage") or "experimenting").strip()
    try:
        stage_config = load_stage_by_id(stage_id)
        prompts = (stage_config or {}).get("prompts") or {}
        if isinstance(prompts, dict):
            distribution_prompt = (prompts.get("distribution_prompt") or "").strip()
            intent_prompt = (prompts.get("intent_prompt") or "").strip()
        # 兼容旧格式：act.system_prompt 整段作为 intent_prompt；旧 YAML 的 strategy_prompt 也接受
        if not intent_prompt and isinstance(prompts, dict):
            intent_prompt = (prompts.get("strategy_prompt") or "").strip()
        if not intent_prompt:
            act = (stage_config or {}).get("act") or {}
            sp = act.get("system_prompt") or ""
            if isinstance(sp, str) and sp.strip():
                intent_prompt = sp.strip()
    except Exception:
        pass
    return distribution_prompt, intent_prompt


def _get_current_strategy_prompt(state: Dict[str, Any]) -> str:
    """从 state 取当前策略（strategies.yaml 命中条）的 prompt。"""
    cur = state.get("current_strategy")
    if not isinstance(cur, dict):
        return ""
    prompt = cur.get("prompt")
    return (prompt or "").strip() if isinstance(prompt, str) else ""


def _select_user_profile(state: Dict[str, Any]) -> Any:
    """
    从 full_profile 中按 selected_profile_keys 选子集；若不满足条件则返回 full_profile 原样。
    不做“删字段”之外的任何改动（不改值、不重写结构）。
    """
    full_profile = state.get("user_inferred_profile") or state.get("user_profile") or {}
    selected_keys = state.get("selected_profile_keys") or []
    if selected_keys and isinstance(full_profile, dict):
        return {k: full_profile[k] for k in selected_keys if k in full_profile}
    return full_profile


def _sanitize_requirements_for_prompt(requirements: Any) -> Any:
    """
    要求：提示词里不出现 is_urgent / tasks_for_lats 的内部标记。
    做法：保留 requirements 其余字段原样，仅移除 tasks_for_lats（因为会带 is_urgent 等内部字段）。
    """
    if not isinstance(requirements, dict):
        return requirements
    out = dict(requirements)
    if "tasks_for_lats" in out:
        out["tasks_for_lats"] = "（已在“本轮必须完成”中明确列出；不在此重复内部任务列表）"
    return out


def _task_to_user_instruction(task: Any) -> str:
    """
    把内部 task dict 转成“确定结论”的自然语言指令（不暴露 is_urgent 等字段）。
    尽量稳健：优先使用 task 自带的可读字段；其次用 action/task_name 映射；再退化为简短描述。
    """
    if not isinstance(task, dict):
        t = safe_text(task).strip()
        return t if t else "完成系统指定任务"

    # 1) 若已有面向用户/助手的指令字段，优先用
    for key in ("user_facing_instruction", "instruction", "must_do", "directive", "goal", "description", "title"):
        val = task.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # 2) 常见 action / task_name 映射
    action = (task.get("action") or task.get("task_name") or task.get("task_type") or "").strip()
    action_l = str(action).lower()

    mapping = {
        "ask_user_name": "回复中必须明确询问对方的姓名或称呼",
        "ask_user_age": "回复中必须明确询问对方的年龄",
        "ask_user_gender": "回复中必须明确询问对方的性别",
        "ask_user_occupation": "回复中必须明确询问对方的职业/身份",
        "ask_user_location": "回复中必须明确询问对方所在城市/地区",
    }
    if action_l in mapping:
        return mapping[action_l]

    # 3) 尝试用字段拼一个简短可读描述
    #    尽量避免把整坨 dict 打出来（太工程/太长）
    bits: List[str] = []
    for k in ("id", "name", "field", "target_field", "question", "prompt"):
        v = task.get(k)
        if isinstance(v, str) and v.strip():
            # question/prompt 往往就是“要问的话”
            if k in ("question", "prompt"):
                return v.strip()
            bits.append(v.strip())

    if bits:
        return " / ".join(bits[:3])

    # 4) 兜底
    return "完成系统指定任务"


def _extract_required_tasks(requirements: Any) -> List[str]:
    """
    外部计算“本轮必须完成”的任务清单（不把 is_urgent 等内部字段带进 prompt）。
    """
    if not isinstance(requirements, dict):
        return []
    tasks = requirements.get("tasks_for_lats")
    if not isinstance(tasks, list):
        return []

    required: List[str] = []
    for t in tasks:
        if isinstance(t, dict):
            is_urgent = bool(t.get("is_urgent")) or (str(t.get("task_type") or "").lower() == "urgent")
            if is_urgent:
                required.append(_task_to_user_instruction(t))
        else:
            # 非 dict 的任务结构：无法判断 urgent 时不强行列为必做
            continue

    # 去重保序
    seen = set()
    out: List[str] = []
    for x in required:
        key = x.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


# 强调句：system 开头与 user 结尾各用一次，提醒严格遵守当前策略
STRICT_STRATEGY_REMINDER = "⚠️必须严格遵守【当前策略】的硬约束与意图，不得违背。"

# 必须遵守规则列表（仅模块名称）；下方顺序即冲突时的优先级（靠前的优先于靠后的）
MANDATORY_RULES_NAMES = """必须遵守以下部分（按下方顺序）。若规则之间冲突，则按本列表从上到下的优先顺序执行（靠前的优先于靠后的）：
- 当前策略
- 时间与会话上下文
- 阶段意图与行为准则
- 风格说明（style）
- 输出格式"""


def _build_system_prompt_b(
    *,
    bot_name: str,
    user_name: str,
    bot_basic_info: Dict[str, Any],
    bot_persona: Dict[str, Any],
    user_basic_info: Dict[str, Any],
    user_profile_selected: Any,
    system_memory: str,
    style_profile: Any,
    requirements: Any,
    required_tasks: List[str],
    k: int = 1,
    distribution_prompt: str = "",
    intent_prompt: str = "",
    strategies_prompt: str = "",
) -> Dict[str, str]:
    """
    返回 system prompt 各模块内容（不拼接），供 _invoke_planner_llm 按指定顺序组装。
    """
    header = f"你是 {bot_name}，正在和 {user_name} 对话。"

    background = f"""
可用背景信息（只用于生成，不要照抄给用户）：

- bot_basic_info：{safe_text(bot_basic_info)}
- bot_persona：{safe_text(bot_persona)}
- user_basic_info：{safe_text(user_basic_info)}
- user_profile（本轮选中字段）：{safe_text(user_profile_selected)}
""".strip()

    memory_block = f"【memory（摘要 + 检索）】\n{system_memory}".strip()

    style_block = f"【风格说明（style 节点 llm_instructions）】\n{safe_text(style_profile)}".strip()

    intent_block = f"【阶段意图与行为准则】\n{intent_prompt}".strip() if intent_prompt else ""
    strategy_block = f"【当前策略（本轮回调策略）】\n{strategies_prompt}".strip() if strategies_prompt else ""

    # 必须完成任务列表（紧急任务，放在策略后）
    required_tasks_block = ""
    if required_tasks:
        lines = [f"- {str(t).strip()}" for t in required_tasks if str(t).strip()]
        if lines:
            required_tasks_block = "【必须完成任务列表】\n" + "\n".join(lines)

    # 输出格式：写作要求内容 + 输出形式（JSON），不再单独设写作要求字段
    writing_content = f"""- 回复要自然、连贯、像真人说话；不要自称 AI/助手/模型/机器人。
- 避免客服模板句式或“出戏说明”（例如“作为一个模型/根据设定/我可以为你提供…”）。
- TIME_* 标记为元数据，不要复述；不要输出精确时间戳（除非用户明确问）。

{TIME_SLICE_BEHAVIOR_RULES}
- 不要频繁叫对方的名字。
- 可用话语留白或陈述收尾维持对话。
""".strip()

    if k <= 1:
        输出格式 = f"""
{writing_content}

（输出格式由系统约束，请直接给出一条完整回复文本。）
""".strip()
    else:
        输出格式 = f"""
{writing_content}

- 候选之间必须明显不同（语气/策略/互动方式不同），不能只是同义改写。
- 共输出 {int(k)} 条候选。（输出格式由系统约束。）
""".strip()

    return {
        "header": header,
        "background": background,
        "memory_block": memory_block,
        "style_block": style_block,
        "intent_block": intent_block,
        "strategy_block": strategy_block,
        "required_tasks_block": required_tasks_block,
        "输出格式": 输出格式,
    }


def _planner_sampling_for_round(gen_round: int) -> tuple[float, float]:
    """第 1 次 temperature=0.7, top_p=0.95；不合格则每次重试 +0.15 temp、-0.05 top_p。"""
    temperature = min(0.7 + gen_round * 0.15, 1.2)
    top_p = max(0.95 - gen_round * 0.05, 0.5)
    return (temperature, top_p)


def _parse_planner_response(data: Dict[str, Any], k: int) -> List[ReplyPlan]:
    """从 LLM 返回的 JSON 解析出 ReplyPlan 列表。k<=1 时期望 {"reply": "..."}，k>1 时期望 {"candidates": [...]}。"""
    out: List[ReplyPlan] = []

    def _one_plan(c: Any) -> str:
        raw = ""
        if isinstance(c, dict) and isinstance(c.get("reply"), str):
            raw = c.get("reply", "").strip()
        elif isinstance(c, dict) and isinstance(c.get("messages"), list) and c["messages"]:
            m0 = c["messages"][0]
            if isinstance(m0, str):
                raw = m0.strip()
            elif isinstance(m0, dict):
                raw = str(m0.get("content") or "").strip()
            else:
                raw = str(m0).strip()
        return strip_candidate_prefix(raw) if raw else ""

    if k <= 1:
        reply = (data.get("reply") or "").strip() if isinstance(data.get("reply"), str) else ""
        if not reply and isinstance(data.get("messages"), list) and data["messages"]:
            m0 = data["messages"][0]
            reply = _one_plan({"reply": m0} if isinstance(m0, str) else m0)
        else:
            reply = strip_candidate_prefix(reply) if reply else ""
        if reply:
            out.append({"reply": reply})  # type: ignore[typeddict-item]
    else:
        cands = data.get("candidates")
        if not isinstance(cands, list) and isinstance(data.get("reply"), str) and data.get("reply", "").strip():
            cands = [{"reply": data["reply"].strip()}]
        if isinstance(cands, list):
            for c in cands[: int(k)]:
                reply = _one_plan(c) if isinstance(c, dict) else ""
                if reply:
                    out.append({"reply": reply})  # type: ignore[list-item,typeddict-item]

    return out


def _invoke_planner_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    *,
    k: int = 1,
    extra_constraints_text: Optional[str] = None,
    global_guidelines: Optional[str] = None,
    gen_round: int = 0,
    user_message_only: bool = False,
) -> List[ReplyPlan]:
    """
    统一入口：同一套 prompt、一次 LLM 调用、统一解析。
    k=1 时要求输出单条 reply，返回长度 0 或 1；k>1 时要求输出 k 条 candidates，返回最多 k 条。
    """
    if llm_invoker is None:
        return []

    system_memory = build_system_memory_block(state)
    style_profile = build_style_profile(state)
    requirements = state.get("requirements") or {}
    required_tasks = _extract_required_tasks(requirements)

    bot_basic_info = state.get("bot_basic_info") or {}
    bot_persona = state.get("bot_persona") or {}
    user_basic_info = state.get("user_basic_info") or {}
    user_profile_selected = _select_user_profile(state)

    bot_name = safe_text((bot_basic_info or {}).get("name") or "Bot").strip() or "Bot"
    user_name = safe_text((user_basic_info or {}).get("name") or "不知道姓名的人").strip() or "不知道姓名的人"

    distribution_prompt, intent_prompt = _get_stage_prompts(state)
    strategies_prompt = _get_current_strategy_prompt(state)

    parts = _build_system_prompt_b(
        bot_name=bot_name,
        user_name=user_name,
        bot_basic_info=bot_basic_info,
        bot_persona=bot_persona,
        user_basic_info=user_basic_info,
        user_profile_selected=user_profile_selected,
        system_memory=system_memory,
        style_profile=style_profile,
        requirements=requirements,
        required_tasks=required_tasks,
        k=k,
        distribution_prompt=distribution_prompt,
        intent_prompt=intent_prompt,
        strategies_prompt=strategies_prompt,
    )

    time_context_block = build_time_context_block(state)
    user_input = safe_text(state.get("external_user_text") or state.get("user_input"))
    monologue = safe_text(state.get("inner_monologue"))
    monologue_block = f"【内心动机】（只当参考，不要照抄）：\n{monologue}" if monologue.strip() else ""
    extra_constraints_block = f"改进要点（基于上一轮数值评分摘要）：\n{extra_constraints_text}" if (extra_constraints_text and extra_constraints_text.strip()) else ""

    # 顺序：强调句(system 开头)，2 身份句，3 背景信息，…（fast_reply 与 LATS 共用）
    system_blocks: List[str] = []
    system_blocks.append(STRICT_STRATEGY_REMINDER)                  # system 开头：强调严格遵守 strategy
    system_blocks.append(parts["header"])                           # 2 身份句
    system_blocks.append(parts["background"])                       # 3 背景信息（不含 memory）
    if extra_constraints_block:
        system_blocks.append(extra_constraints_block)               # 3 改进要点（仅列表不为空时）
    system_blocks.append(MANDATORY_RULES_NAMES)                     # 必须遵守规则列表（仅模块名称）
    if parts["strategy_block"]:
        system_blocks.append(parts["strategy_block"])               # 7 当前策略（必须遵守中优先级最高）
    system_blocks.append(parts["style_block"])                      # 5 风格说明
    if monologue_block:
        system_blocks.append(monologue_block)                       # 11 内心动机（风格说明后，仅非空时）
    system_blocks.append(parts["memory_block"])                     # memory（内心动机后）
    if parts.get("required_tasks_block"):
        system_blocks.append(parts["required_tasks_block"])         # 必须完成任务列表（紧急任务，紧接策略后）
    system_blocks.append(time_context_block)                        # 1 时间与会话上下文
    if parts["intent_block"]:
        system_blocks.append(parts["intent_block"])                 # 6 阶段意图与行为准则
    system_blocks.append(parts["输出格式"])                          # 输出格式（含写作要求内容 + JSON 输出形式）

    system_prompt = "\n\n".join(system_blocks)
    print(f"[ReplyPlanner] system_len={len(system_prompt)} user_len(将用)={len(user_input)} required_tasks={len(required_tasks)}")

    # user 消息：user_message_only 时仅用户输入；否则为 内容分布靶标 + 说明 + 用户输入（输出格式已在 system）
    if user_message_only:
        last_user_content = user_input
    else:
        user_parts: List[str] = []
        if distribution_prompt:
            user_parts.append("【内容分布靶标（当前阶段）】\n" + distribution_prompt)
        user_parts.append("本轮用户输入在最后一条用户消息中，请根据其生成回复。")
        user_parts.append(user_input)
        last_user_content = "\n\n".join(p for p in user_parts if p.strip())
    # user 结尾：强调严格遵守 strategy（reply_planner / fast_reply 共用）
    last_user_content = (last_user_content or "").strip() + "\n\n" + STRICT_STRATEGY_REMINDER
    body_messages = get_chat_buffer_body_messages_with_time_slices(state, limit=20)

    log_name = "ReplyPlanGen" if k <= 1 else "ReplyPlanGen (Candidates)"
    log_prompt_and_params(
        log_name,
        system_prompt=system_prompt,
        user_prompt=last_user_content,
        messages=body_messages,
        params={
            "gen_round": gen_round,
            "k": int(k),
            "temperature": _planner_sampling_for_round(gen_round)[0],
            "top_p": _planner_sampling_for_round(gen_round)[1],
            "has_extra_constraints": bool(extra_constraints_text),
            "has_global_guidelines": bool(global_guidelines),
            "required_tasks": required_tasks,
        },
    )

    try:
        temperature, top_p = _planner_sampling_for_round(gen_round)
        messages = [SystemMessage(content=system_prompt), *body_messages, HumanMessage(content=last_user_content)]
        data = None
        resp = None

        if hasattr(llm_invoker, "with_structured_output"):
            schema = ReplyPlannerSingle if k <= 1 else ReplyPlannerCandidates
            # 1. 只套结构化输出，不使用 .bind()
            structured = llm_invoker.with_structured_output(schema)

            log_name_ctx = "ReplyPlanGen" if k <= 1 else "ReplyPlanGenCandidates"
            tok = set_current_node(log_name_ctx)
            try:
                # 2. 直接在 invoke 时把动态参数作为 kwargs 传进去；限制 max_tokens 避免输出被截断导致 LengthFinishReasonError
                obj = structured.invoke(
                    messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=_reply_plan_max_tokens(),
                )
                data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
            except Exception as e:
                reset_current_node(tok)
                # 输出被长度截断时无法解析，直接返回空列表由上层重试/兜底
                err_name, err_msg = type(e).__name__, str(e)
                if "LengthFinishReasonError" in err_name or "length limit" in err_msg.lower():
                    print(f"  [ReplyPlanner] ⚠ 输出达到长度上限被截断，无法解析: {err_name}", flush=True)
                    # 被截断的内容在异常的 completion 里，打出前 4000 字便于排查（可能被 langchain 包装，从 __cause__ 取）
                    try:
                        comp = getattr(e, "completion", None)
                        if comp is None and getattr(e, "__cause__", None) is not None:
                            comp = getattr(e.__cause__, "completion", None)
                        if comp is not None and getattr(comp, "choices", None):
                            c0 = comp.choices[0] if comp.choices else None
                            if c0 is not None and getattr(c0, "message", None):
                                raw = getattr(c0.message, "content", None) or ""
                                if isinstance(raw, str) and raw:
                                    cap = 4000
                                    snippet = raw[:cap] + ("..." if len(raw) > cap else "")
                                    print(f"  [ReplyPlanner] 被截断的输出（前{min(len(raw), cap)}字）:\n{snippet}", flush=True)
                    except Exception:
                        pass
                    return []
                print(f"  [ReplyPlanner] ⚠ structured_output 调用异常（不回退到 raw 解析）: {err_name}: {err_msg[:200]}", flush=True)
                raise
            reset_current_node(tok)
        else:
            # 仅当 LLM 不支持 with_structured_output 时才走 raw invoke，同样直接传 kwargs
            resp = llm_invoker.invoke(
                messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=_reply_plan_max_tokens(),
            )
            content = getattr(resp, "content", "") or ""
            data = parse_json_from_llm(content)
        if not isinstance(data, dict):
            content = getattr(resp, "content", "") or "" if resp is not None else ""
            preview = (content[:400] + "...") if len(content) > 400 else content
            print(f"  [ReplyPlanner] ⚠ JSON解析失败: {preview}", flush=True)
            clean = (content or "").strip()
            if clean and len(clean) < 4000:
                data = {"reply": clean} if k <= 1 else {"candidates": [{"reply": clean}]}
                print("  [ReplyPlanner] 已用原文作为单条 reply 兜底", flush=True)
            else:
                return []

        log_llm_response(log_name, resp, parsed_result=data if k <= 1 else {"candidates": len(data.get("candidates") or [])})

        plans = _parse_planner_response(data, k)
        if k <= 1:
            print("  [计划生成] ✓ reply=1条" if plans else "  [计划生成] ⚠ reply 为空")
        else:
            print(f"  [计划生成] ✓ candidates={len(plans)}条")
        return plans
    except Exception as e:
        import traceback
        err_type, err_msg = type(e).__name__, str(e)
        print(f"  [ReplyPlanner] ❌ 异常: {err_type}: {err_msg[:50]}")
        traceback.print_exc()
        return []


def plan_reply_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    *,
    global_guidelines: Optional[str] = None,
    gen_round: int = 0,
    user_message_only: bool = False,
) -> Optional[ReplyPlan]:
    """生成单条回复计划。与多候选共用同一套 prompt 与调用逻辑，仅 k=1。user_message_only=True 时最后一条 user 消息仅含用户输入（供 fast 节点）。"""
    plans = _invoke_planner_llm(
        state, llm_invoker, k=1,
        global_guidelines=global_guidelines,
        gen_round=gen_round,
        user_message_only=user_message_only,
    )
    return plans[0] if plans else None


def plan_reply_candidates_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    *,
    k: int = 10,
    extra_constraints_text: Optional[str] = None,
    global_guidelines: Optional[str] = None,
    gen_round: int = 0,
) -> List[ReplyPlan]:
    """生成 k 条回复候选（用于 LATS 选优）。与单条共用同一套 prompt，仅 k>1 且可带 extra_constraints_text。"""
    return _invoke_planner_llm(
        state, llm_invoker, k=int(k),
        extra_constraints_text=extra_constraints_text,
        global_guidelines=global_guidelines,
        gen_round=gen_round,
    )

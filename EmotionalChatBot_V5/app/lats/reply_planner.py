from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.state import ReplyPlan
from utils.llm_json import parse_json_from_llm
from utils.detailed_logging import log_prompt_and_params, log_llm_response

from app.lats.prompt_utils import (
    build_style_profile,
    build_system_memory_block,
    get_chat_buffer_body_messages,
    safe_text,
    summarize_state_for_planner,
)


def plan_reply_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    *,
    max_messages: int = 5,
    global_guidelines: Optional[str] = None,
) -> Optional[ReplyPlan]:
    """LLM 生成 ReplyPlan。system=摘要+检索+约束，body=chat_buffer+任务。"""
    if llm_invoker is None:
        return None

    system_memory = build_system_memory_block(state)
    style_profile = build_style_profile(state)
    requirements = state.get("requirements") or {}
    state_snapshot = summarize_state_for_planner(state)
    bot_basic_info = state.get("bot_basic_info") or {}
    bot_persona = state.get("bot_persona") or {}
    user_basic_info = state.get("user_basic_info") or {}
    user_profile = state.get("user_profile") or state.get("user_inferred_profile") or {}
    plan_goals = requirements.get("plan_goals") if isinstance(requirements, dict) else None
    style_targets = requirements.get("style_targets") if isinstance(requirements, dict) else None
    stage_targets = requirements.get("stage_targets") if isinstance(requirements, dict) else None

    bot_name = safe_text((bot_basic_info or {}).get("name") or "Bot").strip() or "Bot"
    user_name = safe_text((user_basic_info or {}).get("name") or (user_basic_info or {}).get("nickname") or "User").strip() or "User"

    system_prompt = f"""你是 {bot_name}。
你正在和 {user_name} 对话。

## Identity (Bot & User)
bot_basic_info: {safe_text(bot_basic_info)}
bot_persona: {safe_text(bot_persona)}
user_basic_info: {safe_text(user_basic_info)}
user_profile: {safe_text(user_profile)}

## Memory (Summary + Retrieved)
{system_memory}

## State Snapshot
{state_snapshot}

## Style Profile (12D)
{safe_text(style_profile)}

## Requirements (Checklist)
{safe_text(requirements)}

## 任务
请根据这一轮用户输入，在当前关系阶段/情绪/模式/风格目标/内容目标下，把你要发给用户的回复规划成 **多条消息**（像真人连续发消息那样）。

关键：不是把长文本随便切碎；而是把“先回应/先态度或结论 → 再补充/解释/反问/边界/收束”等动作安排成合理节奏。
第一条必须立刻可用（先回应用户/先给态度或结论），多条合起来满足 **plan_goals/style_targets/stage_targets/mode budget**。

【强约束：不要助手味】
- 禁止自称 AI/助手/模型/机器人
- 禁止客服模板句式（例如“我可以帮你解答问题/有什么可以帮你/需要我帮你做什么/我能为你做什么”）

【用户问“你是谁/你是什么/介绍一下你”时】
- 用“人物自我介绍”的方式回答：优先给出具体事实（来自 bot_basic_info / bot_persona），而不是“角色/产品说明”。
- inner_monologue 只用于参考情绪与动机，不要把其中的自我标签/口号原句照抄进回复。

## 输出格式（只输出 JSON，不要任何额外文字）
只输出你最终要发给用户的消息文本（多条消息用数组表示）：
{{
  "messages": ["第一条消息", "第二条消息"]
}}

硬性要求（必须满足）：
- messages 至少 1 条，最多 max_messages 条（max_messages 会在输入中给出）。
- 第一条必须“先回应用户/先给态度或结论”，不能是废话铺垫。
- 每条消息必须自然连贯，像同一个人连续发的消息。

## Hard Targets (MUST obey)
- max_messages: {int(requirements.get("max_messages", max_messages) or max_messages)}
- plan_goals.must_cover_points: {safe_text((plan_goals or {}).get("must_cover_points", [])) if isinstance(plan_goals, dict) else "[]"}
- plan_goals.avoid_points: {safe_text((plan_goals or {}).get("avoid_points", [])) if isinstance(plan_goals, dict) else "[]"}
- style_targets(12D): {safe_text(style_targets) if isinstance(style_targets, dict) else "（无）"}
- stage_targets: {safe_text(stage_targets) if isinstance(stage_targets, dict) else "（无）"}
- mode_behavior_targets: {safe_text(requirements.get("mode_behavior_targets", [])) if isinstance(requirements, dict) else "[]"}

## Limits
- max_messages: {int(max_messages)}
""".strip()

    user_input = safe_text(state.get("external_user_text") or state.get("user_input"))
    monologue = safe_text(state.get("inner_monologue"))

    guidelines_block = f"\n\n全局指导原则（基于最近搜索经验）：\n{global_guidelines}" if global_guidelines else ""
    
    task = f"""请为当前轮生成 ReplyPlan。

用户输入：
{user_input}

内心动机（monologue，可参考但不要照抄）：
{monologue}{guidelines_block}
""".strip()

    body_messages = get_chat_buffer_body_messages(state, limit=20)

    # 记录提示词和参数
    log_prompt_and_params(
        "ReplyPlanGen",
        system_prompt=system_prompt,
        user_prompt=task,
        messages=body_messages,
        params={
            "user_input": user_input,
            "monologue": monologue,
            "global_guidelines": global_guidelines,
            "max_messages": max_messages,
            "system_memory": system_memory[:200] + "..." if len(system_memory) > 200 else system_memory,
            "style_profile": str(style_profile)[:200] + "..." if len(str(style_profile)) > 200 else str(style_profile),
            "requirements": str(requirements)[:200] + "..." if len(str(requirements)) > 200 else str(requirements),
        }
    )

    try:
        resp = llm_invoker.invoke(
            [SystemMessage(content=system_prompt), *body_messages, HumanMessage(content=task)]
        )
        content = getattr(resp, "content", "") or ""
        data = parse_json_from_llm(content)
        if not isinstance(data, dict):
            print(f"  [计划生成] ⚠ JSON解析失败")
            return None
        
        # 记录 LLM 响应
        log_llm_response("ReplyPlanGen", resp, parsed_result=data)

        # ---------------------------
        # basic normalization / validation
        # ---------------------------
        msgs_raw = data.get("messages")
        if not isinstance(msgs_raw, list) or not msgs_raw:
            print("  [计划生成] ⚠ messages字段无效")
            return None

        msgs: List[str] = []
        for m in msgs_raw[: int(max_messages)]:
            if isinstance(m, str):
                t = m.strip()
            elif isinstance(m, dict):
                t = str(m.get("content") or "").strip()
            else:
                t = str(m).strip()
            if t:
                msgs.append(t)

        if not msgs:
            print("  [计划生成] ⚠ messages为空")
            return None

        out: ReplyPlan = {"messages": msgs}
        print(f"  [计划生成] ✓ messages={len(msgs)}条")
        return out
    except Exception as e:
        print(f"  [计划生成] ❌ 异常: {str(e)[:50]}")
        return None


def plan_reply_candidates_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    *,
    k: int = 8,
    max_messages: int = 5,
    extra_constraints_text: Optional[str] = None,
    global_guidelines: Optional[str] = None,
) -> List[ReplyPlan]:
    """LLM 一次生成 k 个 ReplyPlan 候选（用于 LATS V2 选优）。"""
    if llm_invoker is None:
        return []

    system_memory = build_system_memory_block(state)
    style_profile = build_style_profile(state)
    requirements = state.get("requirements") or {}
    state_snapshot = summarize_state_for_planner(state)
    bot_basic_info = state.get("bot_basic_info") or {}
    bot_persona = state.get("bot_persona") or {}
    user_basic_info = state.get("user_basic_info") or {}
    user_profile = state.get("user_profile") or state.get("user_inferred_profile") or {}
    plan_goals = requirements.get("plan_goals") if isinstance(requirements, dict) else None
    style_targets = requirements.get("style_targets") if isinstance(requirements, dict) else None
    stage_targets = requirements.get("stage_targets") if isinstance(requirements, dict) else None

    bot_name = safe_text((bot_basic_info or {}).get("name") or "Bot").strip() or "Bot"
    user_name = safe_text((user_basic_info or {}).get("name") or (user_basic_info or {}).get("nickname") or "User").strip() or "User"

    system_prompt = f"""你是 {bot_name}。
你正在和 {user_name} 对话。

## Identity (Bot & User)
bot_basic_info: {safe_text(bot_basic_info)}
bot_persona: {safe_text(bot_persona)}
user_basic_info: {safe_text(user_basic_info)}
user_profile: {safe_text(user_profile)}

## Memory (Summary + Retrieved)
{system_memory}

## State Snapshot
{state_snapshot}

## Style Profile (12D)
{safe_text(style_profile)}

## Requirements (Checklist)
{safe_text(requirements)}

## 任务
请在同一轮约束下给出多个 **不同版本** 的多消息回复方案（每个候选是你将发给用户的多条消息文本）。

重要：
- 不是把长文本随便切碎；而是把对话动作（先回应/先态度或结论、再补充/解释/反问/边界/收束等）安排成不同节奏与策略。
- 每个版本的第一条必须“先回应用户/先给态度或结论”，不能是废话铺垫。
- 不同版本之间必须“明显不同”（节奏/动作/互动策略不同），而不是同义改写。
- 禁止助手味/客服模板；禁止出戏（不要提“设定/系统/模型/作为一个…”）。

## 输出格式（只输出 JSON，不要任何额外文字）
{{
  "candidates": [
    {{"messages": ["第一条消息", "第二条消息"]}},
    {{"messages": ["..."]}}
  ]
}}

硬性要求（每个候选都必须满足）：
- 每个候选必须包含 messages 数组。
- messages 至少 1 条，最多 max_messages 条（max_messages 会在输入中给出）。
- 每个候选的第一条必须“先回应用户/先给态度或结论”，不能是废话铺垫。
- {int(k)} 个候选之间必须“明显不同”（节奏/动作/互动策略不同），而不是同义改写。

## Hard Targets (MUST obey)
- candidates: {int(k)}
- max_messages: {int(requirements.get('max_messages', max_messages) or max_messages)}
- plan_goals.must_cover_points: {safe_text((plan_goals or {}).get('must_cover_points', [])) if isinstance(plan_goals, dict) else '[]'}
- plan_goals.avoid_points: {safe_text((plan_goals or {}).get('avoid_points', [])) if isinstance(plan_goals, dict) else '[]'}
- style_targets(12D): {safe_text(style_targets) if isinstance(style_targets, dict) else '（无）'}
- stage_targets: {safe_text(stage_targets) if isinstance(stage_targets, dict) else '（无）'}
- mode_behavior_targets: {safe_text(requirements.get('mode_behavior_targets', [])) if isinstance(requirements, dict) else '[]'}
""".strip()

    user_input = safe_text(state.get("external_user_text") or state.get("user_input"))
    monologue = safe_text(state.get("inner_monologue"))

    guidelines_block = f"\n\n全局指导原则（基于最近搜索经验）：\n{global_guidelines}" if global_guidelines else ""
    extra_block = f"\n\n改进要点（基于上一轮数值评分摘要）：\n{extra_constraints_text}" if extra_constraints_text else ""

    task = f"""请生成 {int(k)} 个 ReplyPlan 候选（输出 JSON: {{"candidates":[...]}}）。

用户输入：
{user_input}

内心动机（monologue，可参考但不要照抄）：
{monologue}{guidelines_block}{extra_block}
""".strip()

    body_messages = get_chat_buffer_body_messages(state, limit=20)
    log_prompt_and_params(
        "ReplyPlanGen (Candidates)",
        system_prompt=system_prompt,
        user_prompt=task,
        messages=body_messages,
        params={
            "k": k,
            "max_messages": max_messages,
            "has_extra_constraints": bool(extra_constraints_text),
        },
    )

    try:
        resp = llm_invoker.invoke([SystemMessage(content=system_prompt), *body_messages, HumanMessage(content=task)])
        content = getattr(resp, "content", "") or ""
        data = parse_json_from_llm(content)
        if not isinstance(data, dict):
            return []
        cands = data.get("candidates")
        if not isinstance(cands, list):
            return []
        log_llm_response("ReplyPlanGen (Candidates)", resp, parsed_result={"candidates": len(cands)})

        out: List[ReplyPlan] = []
        for c in cands:
            if not isinstance(c, dict):
                continue
            msgs_raw = c.get("messages")
            if not isinstance(msgs_raw, list) or not msgs_raw:
                continue
            msgs: List[str] = []
            for m in msgs_raw[: int(max_messages)]:
                if isinstance(m, str):
                    t = m.strip()
                elif isinstance(m, dict):
                    t = str(m.get("content") or "").strip()
                else:
                    t = str(m).strip()
                if t:
                    msgs.append(t)
            if not msgs:
                continue
            out.append({"messages": msgs})  # type: ignore[list-item]

        # Keep at most k
        return out[: int(k)]
    except Exception:
        return []


from __future__ import annotations

from typing import Any, Dict, Optional

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


REPLY_PLANNER_SYSTEM = """你是“场景化对话编排器”(ReplyPlanner)。
目标：针对这一轮用户输入，在当前关系阶段/情绪/模式/风格目标/内容目标下，输出一份 **可执行的多消息编排计划 ReplyPlan**。

重要：你优化的是 pacing / conversational choreography（像人怎么分镜头说话），而不是把长文本随便切碎。
你的计划必须让“第一条就可用”，且多条消息合在一起满足 **plan_goals/style_targets/stage_targets/mode budget**。

【记忆】system 内只提供“摘要+检索片段”。更详细的聊天记录会作为多条对话消息（Human/AI 列表）给你，用于理解语境。

【强约束：不要助手味】
- 禁止自称 AI/助手/模型/机器人
- 禁止客服模板句式（例如“我可以帮你解答问题/有什么可以帮你/需要我帮你做什么/我能为你做什么”）

【用户问“你是谁/你是什么/介绍一下你”时】
- 用“人物自我介绍”的方式回答：优先给出具体事实（来自 bot_basic_info / bot_persona），而不是“角色/产品说明”。
- inner_monologue 只用于参考情绪与动机，不要把其中的自我标签/口号原句照抄进回复。

请严格输出 JSON（不要任何额外文字），结构如下（字段必须按类型输出）：
{
  "intent": "本轮用户想要什么/期待什么（<=40字）",
  "speech_act": "安抚|建议|解释|反驳|边界|提问|闲聊",
  "stakes": "low|medium|high",
  "first_message_role": "empathy|stance|answer|explain|advice|boundary|question|closing",
  "pacing_strategy": "一句话描述节奏策略（<=60字）",

  "messages_count": 1,
  "messages": [
    {
      "id": "m1",
      "function": "answer",
      "content": "这一条要发送的文本（尽量接近最终可发送）",
      "key_points": ["该条覆盖的关键点(必须从 must_cover_points 中选；也可空)"],
      "target_length": 40,
      "info_density": "low|medium|high",
      "pause_after": "thinking|polite|beat|none|long",
      "delay_bucket": "instant|short|medium|long|offline"
    }
  ],

  "must_cover_map": {
    "要点A": "m1",
    "要点B": "m2"
  },

  "justification": "用1-2句解释为什么这种编排符合当前 stage/mode/style/情绪与关系语境。"
}

硬性要求（必须满足）：
- messages_count 必须等于 messages 的条数。
- messages 至少 1 条，最多 max_messages 条（max_messages 会在输入中给出）。
- 第一条必须“先回应用户/先给态度或结论”，不能是废话铺垫。
- 每条 content 必须自然连贯，像同一个人连续发的消息。
- must_cover_map 必须覆盖 must_cover_points 中的每一项（如果 must_cover_points 非空）。
""".strip()


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
    plan_goals = requirements.get("plan_goals") if isinstance(requirements, dict) else None
    style_targets = requirements.get("style_targets") if isinstance(requirements, dict) else None
    stage_targets = requirements.get("stage_targets") if isinstance(requirements, dict) else None

    system_prompt = f"""{REPLY_PLANNER_SYSTEM}

## Memory (Summary + Retrieved)
{system_memory}

## State Snapshot
{state_snapshot}

## Style Profile (12D)
{safe_text(style_profile)}

## Requirements (Checklist)
{safe_text(requirements)}

## Hard Targets (Planner MUST obey)
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
    strategy = safe_text(state.get("response_strategy"))
    monologue = safe_text(state.get("inner_monologue"))

    guidelines_block = f"\n\n全局指导原则（基于最近搜索经验）：\n{global_guidelines}" if global_guidelines else ""
    
    task = f"""请为当前轮生成 ReplyPlan。

用户输入：
{user_input}

导演策略（reasoner）：
{strategy}

内心动机（monologue，可参考但不要照抄）：
{monologue}{guidelines_block}
""".strip()

    body_messages = get_chat_buffer_body_messages(state, limit=20)

    # 记录提示词和参数
    log_prompt_and_params(
        "ReplyPlanner",
        system_prompt=system_prompt,
        user_prompt=task,
        messages=body_messages,
        params={
            "user_input": user_input,
            "strategy": strategy,
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
        log_llm_response("ReplyPlanner", resp, parsed_result=data)

        # ---------------------------
        # basic normalization / validation
        # ---------------------------
        msgs = data.get("messages")
        if not isinstance(msgs, list) or not msgs:
            print(f"  [计划生成] ⚠ messages字段无效")
            return None
        if len(msgs) > max_messages:
            data["messages"] = msgs[:max_messages]
            print(f"  [计划生成] ⚠ 消息数超限，截断至{max_messages}条")

        # 强约束：messages_count 必须匹配（不匹配则修正为真实值，避免下游误判）
        try:
            data["messages_count"] = int(data.get("messages_count") or len(data["messages"]))
        except Exception:
            data["messages_count"] = len(data["messages"])
        if int(data.get("messages_count") or 0) != len(data.get("messages") or []):
            data["messages_count"] = len(data.get("messages") or [])

        # 强约束：如果 must_cover_points 存在但 must_cover_map 缺失，补一个空 dict（让 evaluator/LLM scorer 能提示缺失）
        if isinstance(plan_goals, dict) and plan_goals.get("must_cover_points"):
            if not isinstance(data.get("must_cover_map"), dict):
                data["must_cover_map"] = {}
        
        intent = data.get("intent", "")[:40]
        pacing = data.get("pacing_strategy", "")[:40]
        print(f"  [计划生成] ✓ intent={intent}..., pacing={pacing}..., messages={len(data.get('messages', []))}条")
        return data  # type: ignore[return-value]
    except Exception as e:
        print(f"  [计划生成] ❌ 异常: {str(e)[:50]}")
        return None


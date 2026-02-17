"""内心独白节点：产出内心反应文本 + 选择相关的 inferred_profile 键（不负责预算）。"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from app.lats.prompt_utils import safe_text
from utils.prompt_helpers import format_relationship_for_llm, format_stage_for_llm
from utils.tracing import trace_if_enabled
from utils.llm_json import parse_json_from_llm

from app.state import AgentState

# 内心独白：多句/短段均可，只表达倾向/意愿/态度，不写步骤；字数上限大幅放宽
INNER_MONOLOGUE_MAX_CHARS = 400

# 与 Detection 对齐：当轮消息≤800 字，历史每条≤500 字，最近 30 条
LATEST_USER_TEXT_MAX = 800
RECENT_MSG_CONTENT_MAX = 500
RECENT_DIALOGUE_LAST_N = 30


def _is_user_message(m: BaseMessage) -> bool:
    t = getattr(m, "type", "") or ""
    return "human" in t.lower() or "user" in t.lower()


def _gather_context_for_monologue(state: Dict[str, Any]) -> tuple[str, str, str, str]:
    """从 state 抽取：当轮用户消息、历史对话、关系状态描述、关系阶段描述。"""
    chat_buffer: List[BaseMessage] = list(
        state.get("chat_buffer") or state.get("messages", [])[-RECENT_DIALOGUE_LAST_N:]
    )
    stage_id = str(state.get("current_stage") or "initiating")
    relationship_state = state.get("relationship_state") or {}

    # 当轮用户消息（≤800 字）
    latest_user_text_raw = (state.get("user_input") or "").strip()
    if not latest_user_text_raw and chat_buffer:
        last_msg = chat_buffer[-1]
        latest_user_text_raw = (getattr(last_msg, "content", "") or str(last_msg)).strip()
        if not _is_user_message(last_msg):
            latest_user_text_raw = latest_user_text_raw or "（无用户新句）"
    latest_user_text = (latest_user_text_raw or "（无用户消息）")[:LATEST_USER_TEXT_MAX]

    # 历史对话（最近 30 条，每条 content≤500 字）
    lines: List[str] = []
    for m in chat_buffer:
        role = "Human" if _is_user_message(m) else "AI"
        content = (getattr(m, "content", "") or str(m)).strip()
        if len(content) > RECENT_MSG_CONTENT_MAX:
            content = content[:RECENT_MSG_CONTENT_MAX] + "…"
        lines.append(f"{role}: {content}")
    recent_dialogue_context = "\n".join(lines) if lines else "（无历史对话）"

    rel_for_llm = format_relationship_for_llm(relationship_state)
    stage_desc = format_stage_for_llm(stage_id, include_judge_hints=True)

    return latest_user_text, recent_dialogue_context, rel_for_llm, stage_desc


def create_inner_monologue_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """创建内心独白节点：产出 inner_monologue 文本 + selected_profile_keys。"""

    @trace_if_enabled(
        name="Inner Monologue",
        run_type="chain",
        tags=["node", "inner_monologue", "perception"],
        metadata={"state_outputs": ["inner_monologue", "selected_profile_keys"]},
    )
    def inner_monologue_node(state: AgentState) -> dict:
        monologue, selected_keys = _generate_monologue(state, llm_invoker)
        monologue = (monologue or "按常理接话即可。").strip()
        if len(monologue) > INNER_MONOLOGUE_MAX_CHARS:
            monologue = monologue[:INNER_MONOLOGUE_MAX_CHARS]
        return {
            "inner_monologue": monologue,
            "selected_profile_keys": selected_keys,
        }

    return inner_monologue_node


def _generate_monologue(
    state: AgentState,
    llm_invoker: Any,
) -> tuple[str, List[str]]:
    """LLM 生成内心独白 + 选择相关的 inferred_profile 键。"""
    if llm_invoker is None:
        return "按常理接话即可。", []
    try:
        latest_user_text, recent_dialogue_context, rel_for_llm, stage_desc = _gather_context_for_monologue(state)

        # Bot / User 身份信息（与 reply_planner 对齐）
        bot_name = state.get("bot_name") or "Bot"
        user_name = state.get("user_name") or "User"
        bot_basic_info = state.get("bot_basic_info") or {}
        bot_persona = state.get("bot_persona") or ""
        user_basic_info = state.get("user_basic_info") or {}

        # 收集 inferred_profile 的所有键名
        inferred_profile = state.get("user_inferred_profile") or {}
        profile_keys = sorted(inferred_profile.keys()) if isinstance(inferred_profile, dict) else []
        profile_keys_str = ", ".join(profile_keys) if profile_keys else "（暂无）"

        sys = f"""你是 {bot_name}。你正在和 {user_name} 对话。
下面给你一段「历史对话」和「当轮用户消息」作为正文，请以你（{bot_name}）的第一人称视角，根据正文语境输出两部分内容，给下游执行用，不给用户看。

## Identity (Bot & User)
bot_basic_info: {safe_text(bot_basic_info)}
bot_persona: {safe_text(bot_persona)}
user_basic_info: {safe_text(user_basic_info)}

## 输出格式（严格 JSON，不要其他文字）
{{
  "monologue": "内心独白正文",
  "selected_profile_keys": ["key1", "key2"]
}}

## 第一部分：内心独白 (monologue)
- 字数**最多 {INNER_MONOLOGUE_MAX_CHARS} 个字符（中文按字计）**，可多句、可短段，但不得超出此上限。
- 以你（{bot_name}）的第一人称视角表达倾向/意愿/态度（如：懒得问、想挡一下、想接球但别太热情、对这句话有点意外但可以接），不要步骤、不要策略清单。

## 第二部分：选择相关的用户画像键 (selected_profile_keys)
以下是当前已有的用户推断画像键名列表：
[{profile_keys_str}]
- 从中选出 0~5 个与当前对话语境**相关**的键名（用于后续生成回复时参考）。
- 如果没有相关键或列表为空，返回空数组 []。
- 只选键名，不要自己编造不存在的键。

## 关系状态（0–1）
{rel_for_llm}

## 关系阶段与越界提示
{stage_desc}
"""
        user_body = f"""【历史对话】（最近 {RECENT_DIALOGUE_LAST_N} 条）
{recent_dialogue_context}

【当轮用户消息】
{latest_user_text}

请根据上述对话与当轮用户消息，输出严格 JSON（含 monologue 和 selected_profile_keys），不要其他内容。"""

        msg = llm_invoker.invoke([SystemMessage(content=sys), HumanMessage(content=user_body)])
        content = (getattr(msg, "content", "") or str(msg)).strip()

        data = parse_json_from_llm(content)
        if isinstance(data, dict):
            monologue = str(data.get("monologue") or "").strip().strip("\"'")
            raw_keys = data.get("selected_profile_keys") or []
            selected = [
                str(k) for k in raw_keys
                if isinstance(k, str) and k.strip() and k.strip() in profile_keys
            ][:5]
            if monologue:
                return monologue[:INNER_MONOLOGUE_MAX_CHARS], selected

        # JSON 解析失败时，将整个输出作为独白文本
        fallback_text = content.strip().strip("\"'")
        if fallback_text:
            return fallback_text[:INNER_MONOLOGUE_MAX_CHARS], []
    except Exception:
        pass
    return "按常理接话即可。", []

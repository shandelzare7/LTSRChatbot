from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from utils.prompt_helpers import format_stage_for_llm


_ASSISTANT_IDENTITY_PATTERNS = [
    r"我\s*是[\s\S]{0,24}(ai|人工智能|智能助手|机器人助手|chatbot|聊天助手|助手)",
    r"(我叫|我是|叫我)[\s\S]{0,18}(一个|位)?[\s\S]{0,18}(ai|人工智能|智能助手|机器人助手|chatbot|聊天助手|助手)",
    r"小池是一个聊天助手",
]


def sanitize_memory_text(text: str) -> str:
    """
    记忆卫生：过滤“自称助手/AI”的旧记忆，避免上游 reasoner/planner 被错误召回牵引。
    注意：这里过滤的是「身份自述」模板，不做一般敏感词过滤。
    """
    t = safe_text(text)
    if not t.strip():
        return ""
    # 按行过滤
    lines = [ln for ln in t.splitlines() if ln.strip()]
    kept: List[str] = []
    for ln in lines:
        low = ln.lower()
        bad = False
        for pat in _ASSISTANT_IDENTITY_PATTERNS:
            if re.search(pat, low):
                bad = True
                break
        if not bad:
            kept.append(ln)
    return "\n".join(kept).strip()


def filter_retrieved_memories(items: Any) -> List[str]:
    """过滤召回记忆列表中明显的助手身份自述片段。"""
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for x in items:
        s = safe_text(x).strip()
        if not s:
            continue
        low = s.lower()
        if any(re.search(p, low) for p in _ASSISTANT_IDENTITY_PATTERNS):
            continue
        out.append(s)
    return out


def build_system_memory_block(state: Dict[str, Any]) -> str:
    """System memory must NOT include chat_buffer; only summary + retrieved."""
    summary = sanitize_memory_text(state.get("conversation_summary") or "")
    retrieved = filter_retrieved_memories(state.get("retrieved_memories") or [])
    parts: List[str] = []
    if summary:
        parts.append("近期对话摘要：\n" + str(summary))
    if retrieved:
        parts.append("相关记忆片段：\n" + "\n".join([str(x) for x in retrieved if x]))
    return "\n\n".join(parts) if parts else "（无）"


def build_style_profile(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer explicit style_profile, fallback to llm_instructions."""
    sp = state.get("style_profile")
    if isinstance(sp, dict) and sp:
        return sp
    ins = state.get("llm_instructions")
    return ins if isinstance(ins, dict) else {}


def get_chat_buffer_body_messages(state: Dict[str, Any], limit: int = 20):
    """Return chat_buffer as message objects for LLM body."""
    chat_buffer = state.get("chat_buffer") or []
    try:
        return list(chat_buffer[-limit:])
    except Exception:
        return []


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return ""


def summarize_state_for_planner(state: Dict[str, Any]) -> str:
    """Compact state snapshot for planner/evaluator prompts (non-memory)."""
    bot = state.get("bot_basic_info") or {}
    mood = state.get("mood_state") or {}
    rel = state.get("relationship_state") or {}
    stage_id = state.get("current_stage") or "experimenting"
    stage_desc = format_stage_for_llm(stage_id)
    mode = state.get("current_mode")
    mode_id = getattr(mode, "id", None) if mode else None
    crit = None
    if mode and hasattr(mode, "critic_criteria"):
        crit_obj = mode.critic_criteria
        if hasattr(crit_obj, "focus"):
            crit = crit_obj.focus if isinstance(crit_obj.focus, list) else None
    return "\n".join(
        [
            f"- bot_name: {safe_text(bot.get('name') or 'Bot')}",
            f"- stage: {stage_desc}",
            f"- mood_state: {safe_text(mood)}",
            f"- relationship_state: {safe_text(rel)}",
            f"- mode_id: {safe_text(mode_id) if mode_id else '（无）'}",
            f"- mode_critic_criteria: {safe_text(crit) if crit else '（无）'}",
        ]
    )


from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from utils.prompt_helpers import format_stage_act_for_llm


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
    retrieval_ok = state.get("retrieval_ok")
    if retrieval_ok is False:
        retrieved = []
    else:
        retrieved = filter_retrieved_memories(state.get("retrieved_memories") or [])
    parts: List[str] = []
    if summary:
        parts.append("近期对话摘要：\n" + str(summary))
    if retrieved:
        parts.append("相关记忆片段：\n" + "\n".join([str(x) for x in retrieved if x]))
    return "\n\n".join(parts) if parts else "（无）"


# 数值 -> 五档文字（用于 FORMALITY/POLITENESS/WARMTH/CERTAINTY/EMOTIONAL_INTENSITY）
_STYLE_VALUE_TO_LABEL = (
    (0.86, 1.01, "extremely_high"),
    (0.61, 0.86, "high"),
    (0.41, 0.61, "mid"),
    (0.16, 0.41, "low"),
    (0.0, 0.16, "extremely_low"),
)
# 五档中文（注入提示词用）
_STYLE_VALUE_TO_LABEL_ZH = (
    (0.86, 1.01, "极高"),
    (0.61, 0.86, "高"),
    (0.41, 0.61, "中"),
    (0.16, 0.41, "低"),
    (0.0, 0.16, "极低"),
)

# 新 6 维 style 的 key 顺序与 EXPRESSION_MODE 枚举
_STYLE_6D_ORDER = ("FORMALITY", "POLITENESS", "WARMTH", "CERTAINTY", "EMOTIONAL_INTENSITY", "EXPRESSION_MODE")
# 6 维中文名（注入提示词用）
_STYLE_6D_KEY_ZH = {
    "FORMALITY": "正式度",
    "POLITENESS": "礼貌度",
    "WARMTH": "温暖度",
    "CERTAINTY": "确定度",
    "EMOTIONAL_INTENSITY": "情感强度",
    "EXPRESSION_MODE": "表达模式",
}
_EXPRESSION_MODE_LABELS = {
    0: "字面直白（直接说想法，不拐弯抹角）",
    1: "字面委婉（同样是字面意思，但措辞迂回、留余地）",
    2: "比喻/意象（用比喻、意象或隐喻表达，语言更具体感）",
    3: "轻调侃（带一点反讽或调侃，关系亲密时才用）",
}
_EXPRESSION_MODE_LABELS_EN = {
    0: "literal_direct",
    1: "literal_hedged",
    2: "metaphor_imagery",
    3: "light_teasing",
}

_STYLE_DIM_ANCHORS: Dict[str, Dict[str, str]] = {
    "WARMTH": {
        "extremely_low": "底层气质冷淡，即使正面情绪时也不主动流露温情",
        "low": "底层偏克制，温情需要明显触发才会显现",
        "mid": "底层气质平和，情感流露随即时情绪自然起伏",
        "high": "底层偏温暖，正面情绪时情感表达比较顺畅",
        "extremely_high": "底层气质亲密，情感表达阈值低，容易流露",
    },
    "EMOTIONAL_INTENSITY": {
        "extremely_low": "语气完全平静，无强调、无感叹，情绪完全内敛",
        "low": "语气偏平，用词克制，情绪不外露",
        "mid": "情绪适中，偶有强调或感叹",
        "high": "明显情绪投入，强调副词、感叹号较多",
        "extremely_high": "强烈情绪激活，大量强调/反复/感叹，语气高度外显",
    },
}
_STYLE_DIM_ANCHORS_EN: Dict[str, Dict[str, str]] = {
    "WARMTH": {
        "extremely_low": "cold baseline, little warmth even when positive",
        "low": "restrained, warmth only when clearly triggered",
        "mid": "neutral baseline, affect follows context",
        "high": "warm baseline, fluent positive affect",
        "extremely_high": "intimate baseline, low threshold for affect",
    },
    "EMOTIONAL_INTENSITY": {
        "extremely_low": "completely flat delivery, no intensifiers or exclamations",
        "low": "calm, measured, minimal emotional markers",
        "mid": "moderate intensity, occasional emphasis",
        "high": "noticeably animated, frequent intensifiers and exclamations",
        "extremely_high": "highly activated, heavy use of intensifiers/repetition/exclamations",
    },
}


def _style_value_to_label(value: float) -> str:
    """将 [0,1] 的数值映射为五档文字：extremely_low, low, mid, high, extremely_high。"""
    v = max(0.0, min(1.0, float(value)))
    for lo, hi, label in _STYLE_VALUE_TO_LABEL:
        if lo <= v < hi:
            return label
    return "mid"


def _style_value_to_label_zh(value: float) -> str:
    """将 [0,1] 的数值映射为五档中文：极低、低、中、高、极高。"""
    v = max(0.0, min(1.0, float(value)))
    for lo, hi, label in _STYLE_VALUE_TO_LABEL_ZH:
        if lo <= v < hi:
            return label
    return "中"


def format_style_as_param_list(style_dict: Dict[str, Any]) -> str:
    """将 style dict 格式化为文字参数列表（英文），供 reply_plan / fast_reply / generate 注入 prompt。"""
    if not isinstance(style_dict, dict):
        return ""
    # 新 6 维：任一新 key 存在则按新格式输出（英文）
    if any(k in style_dict for k in _STYLE_6D_ORDER):
        parts: List[str] = []
        for key in _STYLE_6D_ORDER:
            if key not in style_dict:
                continue
            try:
                v = style_dict[key]
                if key == "EXPRESSION_MODE":
                    mode = int(v) if v is not None else 0
                    label = _EXPRESSION_MODE_LABELS_EN.get(mode, "literal_direct")
                    parts.append(f"{key}={label}")
                else:
                    label_en = _style_value_to_label(float(v))
                    dim_anchors = _STYLE_DIM_ANCHORS_EN.get(key, {})
                    anchor = dim_anchors.get(label_en, "")
                    parts.append(f"{key}={label_en} ({anchor})" if anchor else f"{key}={label_en}")
            except (TypeError, ValueError):
                continue
        return "\n".join(parts) if parts else ""
    # 兼容旧 12 维（仅当无新 key 时）
    ORDER_LEGACY = (
        "self_disclosure", "topic_adherence", "initiative", "advice_style",
        "subjectivity", "memory_hook", "verbal_length", "social_distance",
        "emotional_display", "wit_and_humor", "non_verbal_cues",
    )
    parts = []
    for key in ORDER_LEGACY:
        if key not in style_dict:
            continue
        try:
            v = float(style_dict[key])
            label = _style_value_to_label(v)
            parts.append(f"{key}={label}")
        except (TypeError, ValueError):
            continue
    return "\n".join(parts) if parts else ""


def build_style_profile(state: Dict[str, Any]) -> Any:
    """Reply plan / fast_reply 用：优先返回 6 维参数列表字符串（FORMALITY/POLITENESS/WARMTH/CERTAINTY/EMOTIONAL_INTENSITY/EXPRESSION_MODE）；无 state['style'] 时回退到 style_profile / llm_instructions。"""
    style_dict = state.get("style")
    if isinstance(style_dict, dict) and style_dict:
        param_list = format_style_as_param_list(style_dict)
        if param_list:
            return param_list
    sp = state.get("style_profile")
    if isinstance(sp, dict) and sp:
        param_list = format_style_as_param_list(sp)
        if param_list:
            return param_list
    if isinstance(sp, str) and sp.strip():
        return sp
    ins = state.get("llm_instructions")
    if isinstance(ins, dict):
        param_list = format_style_as_param_list(ins)
        if param_list:
            return param_list
    if isinstance(ins, str) and ins.strip():
        return ins
    return ""


def get_chat_buffer_body_messages(state: Dict[str, Any], limit: int = 20):
    """Return chat_buffer as message objects for LLM body."""
    chat_buffer = state.get("chat_buffer") or []
    try:
        return list(chat_buffer[-limit:])
    except Exception:
        return []


def get_chat_buffer_body_messages_with_time_slices(state: Dict[str, Any], limit: int = 20):
    """
    Return chat_buffer as message objects for LLM body, with TIME_SLICE markers
    inserted between messages when gap/day_part rules are met (8-block day_part,
    gap >= 16h/4h/2h+part_changed). Markers are SystemMessage metadata, not shown to user.
    """
    try:
        from utils.time_context import inject_time_slices_into_messages
    except Exception:
        return get_chat_buffer_body_messages(state, limit=limit)

    chat_buffer = state.get("chat_buffer") or []
    try:
        window = list(chat_buffer[-limit:])
    except Exception:
        window = []
    return inject_time_slices_into_messages(window)


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
    stage_desc = format_stage_act_for_llm(stage_id)
    return "\n".join(
        [
            f"- bot_name: {safe_text(bot.get('name') or 'Bot')}",
            f"- stage: {stage_desc}",
            f"- mood_state（PAD 为 [-1,1]，0 为中性；busyness 为 [0,1]）: {safe_text(mood)}",
            f"- relationship_state: {safe_text(rel)}",
        ]
    )


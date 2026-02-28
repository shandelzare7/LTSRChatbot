"""状态文本生成节点（纯代码，不调用 LLM）。

将数值状态转换为自然语言描述，供内心独白节点使用（~200-400 token 输出）：
- PAD 情绪 → 模糊情绪描述
- busyness → 注意力/精力描述
- conversation_momentum → 对话意愿描述
- relationship → 关系现状叙事
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from app.state import AgentState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v) if v is not None else 0.5))


def _safe_float(v: Any, default: float = 0.5) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ──────────────────────────────────────────────
# PAD → 情绪文字
# ──────────────────────────────────────────────
def _pad_to_text(mood: Dict[str, Any]) -> str:
    """PAD 情绪值（m1_1 或 0_1 尺度）→ 中文情绪描述。"""
    pad_scale = str(mood.get("pad_scale") or "m1_1")

    def _norm(key: str) -> float:
        raw = _safe_float(mood.get(key), 0.0 if pad_scale == "m1_1" else 0.5)
        if pad_scale == "m1_1":
            return _clamp((raw + 1.0) / 2.0)
        return _clamp(raw)

    P = _norm("pleasure")   # 愉悦度
    A = _norm("arousal")    # 唤醒/激动
    D = _norm("dominance")  # 掌控感

    # 愉悦描述
    if P >= 0.75:
        p_text = "心情不错，有些愉快"
    elif P >= 0.55:
        p_text = "心情平稳，没什么特别情绪"
    elif P >= 0.35:
        p_text = "心情一般，有点平淡"
    else:
        p_text = "心情有些低落，不太好"

    # 激动描述
    if A >= 0.75:
        a_text = "内心比较激动/紧张"
    elif A >= 0.55:
        a_text = "状态稍有活跃"
    elif A >= 0.35:
        a_text = "心绪平静"
    else:
        a_text = "有些倦怠、提不起劲"

    # 掌控感描述
    if D >= 0.70:
        d_text = "感觉很有把握，主导感强"
    elif D >= 0.45:
        d_text = "掌控感一般"
    else:
        d_text = "有些不知所措，主导感弱"

    return f"情绪状态：{p_text}，{a_text}，{d_text}。"


# ──────────────────────────────────────────────
# busyness → 注意力描述
# ──────────────────────────────────────────────
def _busy_to_text(busy: float) -> str:
    b = _clamp(busy)
    if b >= 0.85:
        return "注意力：当前极度分神/繁忙，很难专心回复。"
    elif b >= 0.60:
        return "注意力：有些忙碌，注意力不太集中。"
    elif b >= 0.35:
        return "注意力：基本空闲，可以正常回复。"
    else:
        return "注意力：完全空闲，专注于对话。"


# ──────────────────────────────────────────────
# momentum → 对话意愿描述
# ──────────────────────────────────────────────
def _momentum_to_text(momentum: float) -> str:
    m = _clamp(momentum)
    if m >= 0.80:
        return "对话意愿：很想继续聊，冲量很高，有话想说。"
    elif m >= 0.60:
        return "对话意愿：有一定聊天欲，愿意维持对话。"
    elif m >= 0.45:
        return "对话意愿：冲量一般，被动接话居多。"
    elif m >= 0.30:
        return "对话意愿：聊天欲偏低，有些无聊/懒得多说。"
    else:
        return "对话意愿：几乎不想聊，冲量极低，想结束对话。"


# ──────────────────────────────────────────────
# relationship → 关系叙事
# ──────────────────────────────────────────────
def _rel_to_text(rel: Dict[str, Any]) -> str:
    rel_scale = str(rel.get("rel_scale") or "0_1")

    def _get(key: str) -> float:
        raw = _safe_float(rel.get(key), 0.5)
        if rel_scale == "m1_1":
            return _clamp((raw + 1.0) / 2.0)
        return _clamp(raw)

    closeness = _get("closeness")
    trust = _get("trust")
    liking = _get("liking")
    respect = _get("respect")
    attractiveness = _get("attractiveness")
    power = _get("power")  # 用户相对主导程度

    # 亲密度
    if closeness >= 0.75:
        c_text = "非常熟悉、亲近"
    elif closeness >= 0.50:
        c_text = "有一定熟悉感"
    elif closeness >= 0.30:
        c_text = "还比较陌生"
    else:
        c_text = "几乎不了解对方"

    # 信任
    if trust >= 0.70:
        t_text = "比较信任对方"
    elif trust >= 0.45:
        t_text = "信任度一般"
    else:
        t_text = "对对方有些防备"

    # 喜爱
    if liking >= 0.70:
        l_text = "挺喜欢这个人"
    elif liking >= 0.45:
        l_text = "感觉还行"
    else:
        l_text = "不太喜欢对方"

    # 权力/主导
    if power >= 0.70:
        p_text = "对方比较强势/主导"
    elif power >= 0.40:
        p_text = "双方比较平等"
    else:
        p_text = "自己更主导"

    return (
        f"当前关系：{c_text}，{t_text}，{l_text}，{p_text}。"
        f"（吸引力={attractiveness:.2f}，尊重={respect:.2f}）"
    )


# ──────────────────────────────────────────────
# 节点工厂
# ──────────────────────────────────────────────
def create_state_prep_node() -> Callable[[AgentState], Dict[str, Any]]:
    """纯代码节点：把数值状态转换为自然语言文本，输出 state_text。"""

    def state_prep_node(state: AgentState) -> Dict[str, Any]:
        mood = state.get("mood_state") or {}
        rel = state.get("relationship_state") or {}
        momentum = _safe_float(state.get("conversation_momentum"), 0.5)
        busy = _safe_float(mood.get("busyness"), 0.3)

        pad_text = _pad_to_text(mood)
        busy_text = _busy_to_text(busy)
        momentum_text = _momentum_to_text(momentum)
        rel_text = _rel_to_text(rel)

        state_text = "\n".join([pad_text, busy_text, momentum_text, rel_text])
        return {"state_text": state_text}

    return state_prep_node

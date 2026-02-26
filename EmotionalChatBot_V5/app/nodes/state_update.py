"""状态更新节点（纯代码，不调用 LLM）。

根据 extract 节点产出的独白信号更新：
- conversation_momentum（独白 + 规则混合）
- mood_state.pleasure / arousal / dominance（情绪 lerp：向当前情绪目标靠近）

设计原则：
- PAD 用 lerp 而非 delta：情绪是当下状态，可以因一句话大幅变化
- ALPHA=0.60：每轮向情绪目标靠近 60%，1 轮基本到位，保留少量惯性防止 extract 分类抖动
- 和旧 evolver 的关系演化不冲突：evolver 在生成后做完整关系更新，这里只做即时情绪更新
- momentum 仍用 delta 方式：冲量是累积量，不应一轮内大幅跳变
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from app.state import AgentState
from utils.yaml_loader import load_momentum_formula_config

# 加载公式常量
_FORMULA_CFG = load_momentum_formula_config()
MOMENTUM_FLOOR: float = float(_FORMULA_CFG.get("momentum_floor", 0.4))
MOMENTUM_CEILING: float = 1.0

# extract.momentum_delta 对 conversation_momentum 的最大影响幅度（防止一轮内大幅波动）
MOMENTUM_DELTA_SCALE: float = 0.12   # delta in [-1,1] → 最多影响 ±0.12
MOMENTUM_DELTA_MAX_ABS: float = 0.10  # 每轮最大步长

# lerp 权重：每轮向情绪目标靠近的比例
# 0.60 → 1 轮到 60%，2 轮到 84%，3 轮到 94%
PAD_LERP_ALPHA: float = 0.60

# emotion_tag → PAD 目标值（均在 [-1, 1] 空间，基于 PAD 情绪心理学模型）
# P (pleasure)：正 = 愉悦，负 = 痛苦
# A (arousal)：正 = 激动，负 = 平静
# D (dominance)：正 = 主导，负 = 服从
_EMOTION_PAD_TARGET: Dict[str, Dict[str, float]] = {
    # ── 正向情绪 ──────────────────────────────────────────────────
    "开心":    {"pleasure": +0.70, "arousal": +0.40, "dominance": +0.10},
    "愉快":    {"pleasure": +0.55, "arousal": +0.25, "dominance": +0.05},
    "兴奋":    {"pleasure": +0.65, "arousal": +0.75, "dominance": +0.20},
    "期待":    {"pleasure": +0.40, "arousal": +0.55, "dominance": +0.15},
    "好奇":    {"pleasure": +0.25, "arousal": +0.45, "dominance": +0.05},
    "享受":    {"pleasure": +0.70, "arousal": +0.15, "dominance": +0.10},
    # ── 中性情绪 ──────────────────────────────────────────────────
    "平静":    {"pleasure": +0.20, "arousal": -0.20, "dominance": +0.05},
    "无聊":    {"pleasure": -0.25, "arousal": -0.35, "dominance": -0.15},
    "纠结":    {"pleasure": -0.20, "arousal": +0.25, "dominance": -0.30},
    "心疼":    {"pleasure": -0.15, "arousal": +0.20, "dominance": -0.10},
    # ── 轻度负向 ──────────────────────────────────────────────────
    "难过":    {"pleasure": -0.55, "arousal": -0.15, "dominance": -0.25},
    "难受":    {"pleasure": -0.50, "arousal": -0.10, "dominance": -0.20},
    "低落":    {"pleasure": -0.60, "arousal": -0.30, "dominance": -0.30},
    "委屈":    {"pleasure": -0.50, "arousal": +0.15, "dominance": -0.40},
    "尴尬":    {"pleasure": -0.35, "arousal": +0.35, "dominance": -0.30},
    "害羞":    {"pleasure": +0.10, "arousal": +0.40, "dominance": -0.45},
    # ── 中度负向 ──────────────────────────────────────────────────
    "烦躁":    {"pleasure": -0.40, "arousal": +0.55, "dominance": -0.10},
    "不耐烦":  {"pleasure": -0.45, "arousal": +0.50, "dominance": -0.05},
    "警惕":    {"pleasure": -0.20, "arousal": +0.45, "dominance": +0.30},
    # ── 强烈负向 ──────────────────────────────────────────────────
    "愤怒":    {"pleasure": -0.70, "arousal": +0.80, "dominance": +0.25},
    "排斥":    {"pleasure": -0.55, "arousal": +0.30, "dominance": +0.20},
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _update_momentum(current: float, momentum_delta: float) -> float:
    """
    用 extract 的 momentum_delta 对 conversation_momentum 做小步长更新。
    公式：new = current + clamp(delta * SCALE, -max_abs, +max_abs)
    然后 clamp 到 [FLOOR, CEILING]。
    """
    step = _clamp(momentum_delta * MOMENTUM_DELTA_SCALE, -MOMENTUM_DELTA_MAX_ABS, MOMENTUM_DELTA_MAX_ABS)
    new_val = _clamp(current + step, MOMENTUM_FLOOR, MOMENTUM_CEILING)
    return new_val


def _update_pad_from_emotion_tag(mood: Dict[str, Any], emotion_tag: str) -> Dict[str, Any]:
    """
    用 lerp 将 PAD 向 emotion_tag 的目标值靠近。
    公式：new = current + ALPHA * (target - current)
    只修改 pleasure / arousal / dominance，其他字段（busyness, pad_scale 等）不变。
    """
    target = _EMOTION_PAD_TARGET.get(emotion_tag, {})
    if not target:
        return {}

    pad_scale = str(mood.get("pad_scale") or "m1_1")
    result: Dict[str, Any] = {}

    for key in ("pleasure", "arousal", "dominance"):
        raw = mood.get(key)
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue

        t = target.get(key, 0.0)  # 目标值，[-1, 1] 空间

        if pad_scale == "m1_1":
            # 当前值和目标值都在 [-1, 1]，直接 lerp
            v_new = _clamp(v + PAD_LERP_ALPHA * (t - v), -1.0, 1.0)
        else:
            # 当前值在 [0, 1]，目标值转换到 [0, 1] 再 lerp
            t_01 = (t + 1.0) / 2.0
            v_new = _clamp(v + PAD_LERP_ALPHA * (t_01 - v), 0.0, 1.0)
        result[key] = v_new

    return result


def create_state_update_node() -> Callable[[AgentState], Dict[str, Any]]:
    """纯代码节点：根据 extract 信号更新 conversation_momentum 和 mood_state（PAD）。"""

    def state_update_node(state: AgentState) -> Dict[str, Any]:
        extract = state.get("monologue_extract") or {}
        mood = dict(state.get("mood_state") or {})
        current_momentum = _safe_float(state.get("conversation_momentum"), 0.5)

        # 1. 更新 momentum（仍用 delta 方式，冲量是累积量）
        momentum_delta = _safe_float(extract.get("momentum_delta"), 0.0)
        new_momentum = _update_momentum(current_momentum, momentum_delta)

        # 2. 更新 PAD（lerp 向情绪目标靠近）
        emotion_tag = str(extract.get("emotion_tag") or "").strip()
        pad_updates = _update_pad_from_emotion_tag(mood, emotion_tag) if emotion_tag else {}
        if pad_updates:
            mood.update(pad_updates)

        # 3. 镜像压力修正：低宜人性 bot 对过度镜像产生不安/警惕 → dominance 下降
        detection = state.get("detection") or {}
        mirroring_raw = detection.get("mirroring_score", 5)
        try:
            mirroring = _clamp(float(mirroring_raw) / 10.0, 0.0, 1.0)
        except (TypeError, ValueError):
            mirroring = 0.5

        A = _safe_float((state.get("bot_big_five") or {}).get("agreeableness", 0.5), 0.5)

        if mirroring > 0.70 and A < 0.40:
            # 镜像越强、宜人性越低，dominance 下降越多（心理学：低A者被过度讨好时感到不安而非愉悦）
            d_key = "dominance"
            d_val = mood.get(d_key)
            if d_val is not None:
                try:
                    d_current = float(d_val)
                    # 修正量最大约 -0.09（mirroring=1.0, A=0.0 时）
                    correction = -(mirroring - 0.70) * (0.40 - A) * 1.5
                    pad_scale = str(mood.get("pad_scale") or "m1_1")
                    if pad_scale == "m1_1":
                        mood[d_key] = _clamp(d_current + correction, -1.0, 1.0)
                    else:
                        mood[d_key] = _clamp(d_current + correction * 0.5, 0.0, 1.0)
                except (TypeError, ValueError):
                    pass

        out: Dict[str, Any] = {
            "conversation_momentum": new_momentum,
            "mood_state": mood,
        }

        # 4. 性别推断写入：若 extract 推断出性别且 user_basic_info.gender 仍为空，写回
        inferred_gender = str(extract.get("inferred_gender") or "").strip()
        if inferred_gender in ("男", "女", "其他"):
            user_basic_info = dict(state.get("user_basic_info") or {})
            if not str(user_basic_info.get("gender") or "").strip():
                user_basic_info["gender"] = inferred_gender
                out["user_basic_info"] = user_basic_info

        return out

    return state_update_node

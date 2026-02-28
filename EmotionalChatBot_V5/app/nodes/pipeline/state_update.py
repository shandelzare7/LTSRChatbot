"""状态更新节点（纯代码，不调用 LLM）。

根据 extract 节点产出的独白信号更新：
- conversation_momentum（全参数确定性公式，非 LLM 驱动）
- mood_state.pleasure / arousal / dominance（情绪 lerp：向当前情绪目标靠近）

momentum 公式（来自 config/momentum_formula.yaml）：
  E_turn(0-10) = E_user*w_e + T_bot*w_t + R_base*w_r - H_user*penalty
  Score_raw = EMA(M_prev*100, E_turn*10*Multiplier_arousal, α) - Penalty_fatigue
  Ceiling = 100*(1-busyness*0.5)
  M_next = clamp(min(Score_raw, Ceiling)/100, floor=0.4, ceil=1.0)

设计原则：
- PAD 用 lerp 而非 delta：情绪是当下状态，可以因一句话大幅变化
- ALPHA=0.60：每轮向情绪目标靠近 60%，1 轮基本到位，保留少量惯性防止 extract 分类抖动
- 和旧 evolver 的关系演化不冲突：evolver 在生成后做完整关系更新，这里只做即时情绪更新
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from app.state import AgentState
from utils.yaml_loader import load_momentum_formula_config

# 加载公式常量
_FORMULA_CFG = load_momentum_formula_config()
MOMENTUM_FLOOR: float = float(_FORMULA_CFG.get("momentum_floor", 0.4))
MOMENTUM_CEILING: float = 1.0

# 全参数公式常量（从 YAML 读取）
_EMA_ALPHA: float = float(_FORMULA_CFG.get("ema_alpha", 0.15))
_HOSTILITY_PENALTY_COEF: float = float(_FORMULA_CFG.get("hostility_penalty_coef", 0.75))
_W_E_USER: float = float(_FORMULA_CFG.get("e_turn_e_user_weight", 0.275))
_W_T_BOT: float = float(_FORMULA_CFG.get("e_turn_t_bot_weight", 0.375))
_W_R_BASE: float = float(_FORMULA_CFG.get("e_turn_r_base_weight", 0.3))
_AR_CFG: dict = _FORMULA_CFG.get("arousal") or {}
_AR_COEF: float = min(0.3, max(0.0, float(_AR_CFG.get("multiplier_coef", 0.3))))
_FATIGUE_CFG: dict = _FORMULA_CFG.get("fatigue") or {}
_FATIGUE_START_TURN: int = int(_FATIGUE_CFG.get("start_turn", 20))
_FATIGUE_PER_TURN: float = float(_FATIGUE_CFG.get("per_turn", 1.5))

# R_base 权重：0.8*attractiveness + 0.1*liking + 0.1*closeness（0-1 空间，再映射到 0-10）
_R_BASE_WEIGHTS = (0.8, 0.1, 0.1)
_R_BASE_NEUTRAL = 5.0  # 关系维度缺失时的中性值

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


def _f010(v: Any, default: float = 5.0) -> float:
    """Clamp to [0, 10]."""
    try:
        return max(0.0, min(10.0, float(v)))
    except (TypeError, ValueError):
        return default


def _f01(v: Any, default: float = 0.0) -> float:
    """Clamp to [0, 1]."""
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def _compute_momentum(state: Any) -> float:
    """
    全参数确定性动量公式（不依赖 LLM）：

    Step 1: E_turn(0-10) = E_user*w_e + T_bot*w_t + R_base*w_r - H_user*penalty
    Step 2: Multiplier_arousal = 1.0 + (arousal * coef)，高唤醒 → 更愿聊
    Step 3: Score_raw = EMA(M_prev*100, E_turn*10*Multiplier, α) - Penalty_fatigue
    Step 4: Ceiling = 100*(1 - busyness*0.5)，M_next = clamp(min/ceil, 0~100)/100
    """
    detection = state.get("detection") or {}
    mood = state.get("mood_state") or {}
    rel = state.get("relationship_state") or {}
    extract = state.get("monologue_extract") or {}

    m_prev = _f01(state.get("conversation_momentum"), 0.5)

    # E_user：用户投入度 0-10
    e_user = _f010(detection.get("engagement_level"), 5.0)
    # T_bot：话题吸引力来自 extract（比 detection 更及时，反映角色主观感受）
    t_bot = _f010(extract.get("topic_appeal"), 5.0)
    # H_user：敌意 0-10
    h_user = _f010(detection.get("hostility_level"), 0.0)

    # R_base：关系基值（0.8*attractiveness + 0.1*liking + 0.1*closeness，映射到 0-10）
    att = _f01(rel.get("attractiveness", rel.get("warmth", 0.5)))
    lik = _f01(rel.get("liking", 0.5))
    clo = _f01(rel.get("closeness", 0.5))
    r_base_01 = _R_BASE_WEIGHTS[0] * att + _R_BASE_WEIGHTS[1] * lik + _R_BASE_WEIGHTS[2] * clo
    r_base = _f010(r_base_01 * 10.0, _R_BASE_NEUTRAL)

    e_turn = (e_user * _W_E_USER) + (t_bot * _W_T_BOT) + (r_base * _W_R_BASE) - (h_user * _HOSTILITY_PENALTY_COEF)
    e_turn = max(0.0, min(10.0, e_turn))

    # Arousal multiplier（默认 [-1,1] 输入，0 为中性）
    arousal_raw = _safe_float(mood.get("arousal"), 0.0)
    arousal = max(-1.0, min(1.0, arousal_raw))
    multiplier_arousal = max(0.2, 1.0 + arousal * _AR_COEF)

    # Fatigue penalty（0~100 量纲）
    turn_count = int(state.get("turn_count_in_session") or 0)
    penalty_fatigue = max(0.0, (turn_count - _FATIGUE_START_TURN) * _FATIGUE_PER_TURN)

    # EMA（量纲统一到 0~100）
    e_turn_100 = e_turn * 10.0 * multiplier_arousal
    score_raw = (m_prev * 100.0) * (1.0 - _EMA_ALPHA) + e_turn_100 * _EMA_ALPHA - penalty_fatigue

    # Busyness 天花板
    busyness = _f01(mood.get("busyness"), 0.0)
    ceiling = 100.0 * (1.0 - busyness * 0.5)
    score_final = max(0.0, min(score_raw, ceiling))

    m_next = score_final / 100.0
    return _clamp(m_next, MOMENTUM_FLOOR, MOMENTUM_CEILING)


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

        # 1. 更新 momentum（全参数确定性公式，不依赖 LLM momentum_delta）
        new_momentum = _compute_momentum(state)

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

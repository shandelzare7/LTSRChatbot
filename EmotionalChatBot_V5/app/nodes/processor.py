"""
拟人化行为表现层（Processor Node）

按论文方案实现：
1) 延迟计算模型 (Latency Calculation Model)
   T_total = T_read(L_in) + T_cog(P, M, S) + T_type(L_out, P, M)

2) 分段概率模型 (Segmentation Probability Model)
   在标点处以 sigmoid 概率决定切分

输出：
- final_segments: List[str]（兼容旧客户端）
- final_delay: float（兼容旧客户端：第一段前的等待）
- humanized_output: { total_latency_simulated, segments:[{content,delay,action}], latency_breakdown }
"""

from __future__ import annotations

import math
import random
import re
from typing import Any, Callable, Dict, List, Tuple

from app.state import AgentState


# Knapp 阶段的时间策略因子（Chronemics）
STAGE_DELAY_FACTORS: Dict[str, float] = {
    "initiating": 1.2,      # 礼貌的距离感
    "experimenting": 1.0,   # 正常节奏
    "intensifying": 0.6,    # 渴望交流 (Low Latency)
    "integrating": 0.8,     # 舒适
    "bonding": 0.9,         # 稳定
    "differentiating": 1.1, # 开始独立
    "circumscribing": 1.3,  # 减少交流
    "stagnating": 2.5,      # 显著拖延 (High Latency)
    "avoiding": 3.0,        # 极度拖延
    "terminating": 2.0,     # 结束前的犹豫
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _get_big5(state: AgentState) -> Dict[str, float]:
    big5 = state.get("bot_big_five") or {}
    # Big Five in this repo is [-1,1]
    def f(k: str, default: float = 0.0) -> float:
        try:
            return float(big5.get(k, default))
        except Exception:
            return default

    return {
        "openness": f("openness"),
        "conscientiousness": f("conscientiousness"),
        "extraversion": f("extraversion"),
        "agreeableness": f("agreeableness"),
        "neuroticism": f("neuroticism"),
    }


def _get_mood(state: AgentState) -> Dict[str, float]:
    mood = state.get("mood_state") or {}
    def f(k: str, default: float = 0.0) -> float:
        try:
            return float(mood.get(k, default))
        except Exception:
            return default

    return {
        "pleasure": f("pleasure"),
        "arousal": f("arousal"),
        "dominance": f("dominance"),
        "busyness": _clamp(f("busyness", 0.0), 0.0, 1.0),
    }


def _get_rel(state: AgentState) -> Dict[str, float]:
    rel = state.get("relationship_state") or {}
    def f(k: str, default: float = 0.0) -> float:
        try:
            return float(rel.get(k, default))
        except Exception:
            return default

    return {
        "closeness": _clamp(f("closeness", 0.0), 0.0, 100.0),
        "trust": _clamp(f("trust", 0.0), 0.0, 100.0),
    }


def calculate_human_dynamics(state: AgentState) -> Dict[str, float]:
    """
    计算拟人化动态参数（性格、情绪、关系阶段共同作用）。
    """
    big5 = _get_big5(state)
    mood = _get_mood(state)
    rel = _get_rel(state)
    stage = state.get("current_stage", "experimenting")

    # Personality coefficients
    p_speed = 1.0 - (big5["extraversion"] * 0.2)  # E ↑ -> faster
    p_caution = 1.0 + (big5["conscientiousness"] * 0.3)  # C ↑ -> slower (think more)
    p_noise_var = 0.1 + (max(0.0, big5["neuroticism"]) * 0.4)  # N ↑ -> more jitter

    # State modifiers
    m_arousal_boost = 1.0 - (mood["arousal"] * 0.3)  # A ↑ -> faster
    m_busyness_drag = 1.0 + (mood["busyness"] * 1.5)  # busy -> slower

    # Relationship modifiers
    r_intimacy_frag = rel["closeness"] / 100.0
    r_stage_factor = float(STAGE_DELAY_FACTORS.get(str(stage), 1.0))

    speed_factor = p_speed * p_caution * m_arousal_boost * m_busyness_drag * r_stage_factor

    fragmentation_tendency = (
        (big5["extraversion"] * 0.5)
        + (r_intimacy_frag * 0.5)
        + (mood["arousal"] * 0.3)
        - (big5["conscientiousness"] * 0.3)
    )

    return {
        "speed_factor": _clamp(speed_factor, 0.2, 5.0),
        "noise_level": _clamp(p_noise_var, 0.05, 0.8),
        "fragmentation_tendency": fragmentation_tendency,
        "stage_factor": r_stage_factor,
    }


def _compute_latency(
    *,
    state: AgentState,
    final_text: str,
    dynamics: Dict[str, float],
) -> Tuple[float, float, float]:
    """
    延迟模型：
    - T_read = alpha * len(user_input) + C_base
    - T_cog  = beta * (1+busyness) * StageFactor * (1 + Neuroticism * rand_eps)
    - T_type = len(ai_text) / (TypingSpeed * (1 + Extraversion*0.2 + Arousal*0.3))
    """
    big5 = _get_big5(state)
    mood = _get_mood(state)

    user_len = len(state.get("user_input") or "")
    out_len = len(final_text or "")

    # --- T_read
    alpha = 0.05
    c_base = 0.5
    t_read = c_base + alpha * user_len

    # --- T_cog
    beta = 0.8
    eps = random.uniform(-1.0, 1.0)  # random(ε)
    t_cog = beta * (1.0 + mood["busyness"]) * float(dynamics["stage_factor"]) * (
        1.0 + (max(0.0, big5["neuroticism"]) * 0.35 * eps)
    )
    # 使用输出长度作为 cognitive load 的代理（让长回复更“想得久”）
    t_cog *= 1.0 + (out_len * 0.002)
    t_cog = _clamp(t_cog, 0.1, 20.0)

    # --- T_type
    typing_speed = 5.0  # chars/sec baseline
    speed_boost = 1.0 + (big5["extraversion"] * 0.2) + (mood["arousal"] * 0.3)
    speed_boost = max(0.2, speed_boost)
    t_type = out_len / (typing_speed * speed_boost) if out_len > 0 else 0.0
    t_type = _clamp(t_type, 0.0, 60.0)

    return t_read, t_cog, t_type


def _segment_text(
    *,
    state: AgentState,
    text: str,
    dynamics: Dict[str, float],
) -> List[str]:
    """
    分段概率模型：
    P(cut_i)=sigmoid(w1*E + w2*Closeness + w3*Arousal - w4*C)
    只在标点处评估切分。
    """
    if not text:
        return []

    big5 = _get_big5(state)
    mood = _get_mood(state)
    rel = _get_rel(state)

    # weights (可在论文中声明为超参)
    w1, w2, w3, w4 = 1.0, 2.0, 1.2, 1.0
    closeness_norm = rel["closeness"] / 100.0

    base_logit = (
        w1 * big5["extraversion"]
        + w2 * closeness_norm
        + w3 * mood["arousal"]
        - w4 * big5["conscientiousness"]
    )
    base_p = _sigmoid(base_logit)

    # fallback threshold (fragmentation tendency)
    split_threshold = 20 - (_clamp(dynamics["fragmentation_tendency"], -1.0, 2.0) * 15)
    split_threshold = _clamp(split_threshold, 5.0, 30.0)

    parts = re.split(r"([。！？.!?\n]+)", text)
    bubbles: List[str] = []
    buf = ""

    for part in parts:
        if not part:
            continue
        buf += part

        is_punc = bool(re.fullmatch(r"[。！？.!?\n]+", part))
        if not is_punc:
            continue

        # 强制换行切
        if "\n" in part:
            bubbles.append(buf.strip())
            buf = ""
            continue

        # 概率切分 + 长度阈值兜底
        p_cut = _clamp(base_p + random.uniform(-0.08, 0.08), 0.05, 0.95)
        if len(buf) >= split_threshold and random.random() < p_cut:
            bubbles.append(buf.strip())
            buf = ""

    if buf.strip():
        bubbles.append(buf.strip())

    return bubbles


def create_processor_node() -> Callable[[AgentState], dict]:
    def processor_node(state: AgentState) -> dict:
        # 取最终文本：优先 final_response（异常分支），否则 draft_response（正常分支）
        final_text = (state.get("final_response") or state.get("draft_response") or "").strip()
        dynamics = calculate_human_dynamics(state)

        # Latency model
        t_read, t_cog, t_type = _compute_latency(state=state, final_text=final_text, dynamics=dynamics)

        # Segmentation model
        bubbles = _segment_text(state=state, text=final_text, dynamics=dynamics)

        # Timeline: delay-before-each-bubble
        initial_delay = t_read + t_cog

        # typing speed depends on speed_factor (more delay => slower)
        base_typing_speed = 5.0  # chars/sec
        speed_factor = float(dynamics["speed_factor"])
        typing_speed_char_per_sec = max(1.0, base_typing_speed / speed_factor)

        timeline_segments: List[Dict[str, Any]] = []
        delays: List[float] = []
        next_delay = initial_delay
        for bub in bubbles:
            t_this_type = (len(bub) / typing_speed_char_per_sec) if bub else 0.0
            t_this_type *= random.uniform(0.9, 1.1)
            timeline_segments.append({"content": bub, "delay": round(next_delay, 2), "action": "typing"})
            delays.append(next_delay)
            # 下一段：上一段“打字时间”作为等待（模拟正在输入下一条）
            next_delay = _clamp(t_this_type, 0.05, 30.0)

        total_latency_simulated = float(initial_delay + sum(_clamp(len(b) / typing_speed_char_per_sec, 0.0, 60.0) for b in bubbles))

        out = {
            # 兼容旧字段
            "final_segments": bubbles,
            "final_delay": round(initial_delay, 2),
            # 新字段：给客户端更细粒度的时间线
            "humanized_output": {
                "total_latency_simulated": round(total_latency_simulated, 2),
                "segments": timeline_segments,
                "latency_breakdown": {
                    "t_read": round(t_read, 3),
                    "t_cog": round(t_cog, 3),
                    "t_type": round(t_type, 3),
                    "t_total": round(t_read + t_cog + t_type, 3),
                },
            },
        }
        return out

    return processor_node


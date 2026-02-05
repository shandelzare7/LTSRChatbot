"""
拟人化行为表现层（Processor Node）

按论文方案实现：
1) 延迟计算模型 (Latency Calculation Model)
   T_total = T_read(L_in) + T_cog(P, M, S) + T_type(L_out, P, M)

2) 分段概率模型 (Segmentation Probability Model)
   在标点处进行 TCU 式切分（由 fragmentation_tendency 决定阈值）

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

    Theoretical anchors (for paper):
    - Walther (1992) SIP Theory: time is a key relational cue in CMC
    - Big Five in CMC:
      - Extraversion -> faster response & more fragmented texting
      - Conscientiousness -> more deliberation, less fragmentation
      - Neuroticism -> hesitation/noise in timing
    - Knapp relationship stages -> chronemics strategy factor
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

    # fragmentation_tendency: higher -> easier to cut (speech-like bursts)
    fragmentation_tendency = (big5["extraversion"] * 0.5) + (r_intimacy_frag * 0.5) + (
        mood["arousal"] * 0.3
    )

    return {
        "speed_factor": _clamp(speed_factor, 0.2, 5.0),
        "noise_level": _clamp(p_noise_var, 0.05, 0.8),
        "fragmentation_tendency": fragmentation_tendency,
        "stage_factor": r_stage_factor,
    }


def _segment_text(
    *,
    state: AgentState,
    text: str,
    dynamics: Dict[str, float],
) -> List[str]:
    """
    TCU-based segmentation logic (Baron 2008 / Sacks et al. 1974 inspired):
    - Split by punctuation into Turn-Constructional-Units (TCUs)
    - Use a dynamic threshold controlled by fragmentation_tendency
    """
    if not text:
        return []

    # threshold 越低越容易切分
    split_threshold = 20 - (dynamics["fragmentation_tendency"] * 15)
    split_threshold = int(_clamp(float(split_threshold), 5.0, 30.0))

    segments = re.split(r"([。！？\n]+)", text)
    bubbles: List[str] = []
    current_buf = ""

    for seg in segments:
        if not seg:
            continue
        current_buf += seg
        is_punctuation = re.match(r"[。！？\n]+", seg)
        if not is_punctuation:
            continue
        # 如果是标点，且当前缓冲区长度超过阈值，或者强制分段符(换行)，则切分
        if len(current_buf) > split_threshold or "\n" in seg:
            bubbles.append(current_buf.strip())
            current_buf = ""

    if current_buf.strip():
        bubbles.append(current_buf.strip())

    return bubbles


def create_processor_node() -> Callable[[AgentState], dict]:
    def processor_node(state: AgentState) -> dict:
        # 取最终文本：优先 final_response（异常分支），否则 draft_response（正常分支）
        final_text = (state.get("final_response") or state.get("draft_response") or "").strip()
        dynamics = calculate_human_dynamics(state)

        # --- A. 延迟计算 (Latency Model) ---
        user_input_len = len(state.get("user_input") or "")

        # 1) Reading Time (Base Physiology)
        # 假设平均阅读速度 ~20字/秒 + 0.5s 反应基准
        t_read = 0.5 + (user_input_len * 0.05)

        # 2) Thinking/Processing Time (Cognitive Load proxy)
        # 使用 final_response 长度作为认知负荷代理（Hick's Law 变体）
        cognitive_load = len(final_text) * 0.02
        t_cog = (1.0 + cognitive_load) * float(dynamics["speed_factor"])

        # Neuroticism hesitation noise (Gaussian)
        noise = random.gauss(1.0, float(dynamics["noise_level"]))
        t_cog *= _clamp(noise, 0.5, 2.0)

        # 3) Typing latency（按段计算更真实，这里先给总体估计值）
        typing_speed_char_per_sec = 5.0 / float(dynamics["speed_factor"])  # 基准打字速度
        t_type_total_est = (len(final_text) / typing_speed_char_per_sec) if final_text else 0.0

        # --- B. 分段算法 (TCU-based Segmentation) ---
        bubbles = _segment_text(state=state, text=final_text, dynamics=dynamics)

        # --- C. 构建最终序列 (Scheduling) ---
        accumulated_delay = t_read + t_cog  # 初始延迟（读+想）
        timeline_segments: List[Dict[str, Any]] = []
        for bub in bubbles:
            t_type = (len(bub) / typing_speed_char_per_sec) if bub else 0.0
            t_type *= random.uniform(0.9, 1.1)
            timeline_segments.append(
                {"content": bub, "delay": round(accumulated_delay, 2), "action": "typing"}
            )
            # 下一句的延迟是基于上一句发完之后的（模拟正在打下一句）
            accumulated_delay = _clamp(t_type, 0.05, 30.0)

        # total_latency_simulated: 按参考代码定义为 delay 字段求和（用于论文指标）
        total_latency_simulated = float(sum(s["delay"] for s in timeline_segments)) if timeline_segments else float(t_read + t_cog)

        out = {
            # 兼容旧字段
            "final_segments": bubbles,
            "final_delay": round(t_read + t_cog, 2),
            # 新字段：给客户端更细粒度的时间线
            "humanized_output": {
                "total_latency_simulated": round(total_latency_simulated, 2),
                "segments": timeline_segments,
                "latency_breakdown": {
                    "t_read": round(t_read, 3),
                    "t_cog": round(t_cog, 3),
                    "t_type": round(t_type_total_est, 3),
                    "t_total": round(t_read + t_cog + t_type_total_est, 3),
                },
            },
        }
        return out

    return processor_node


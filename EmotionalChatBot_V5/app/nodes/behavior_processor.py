"""
拟人化行为表现层（Behavior Processor Node）

原 `app/nodes/processor.py` 的拆句/延迟实现搬迁到这里，
避免与“关系流水线 Processor（语义处理器）”职责冲突。
"""

from __future__ import annotations

import random
import re
from typing import Any, Callable, Dict, List

from app.state import AgentState


# Knapp 阶段的时间策略因子（Chronemics）
STAGE_DELAY_FACTORS: Dict[str, float] = {
    "initiating": 1.2,
    "experimenting": 1.0,
    "intensifying": 0.6,
    "integrating": 0.8,
    "bonding": 0.9,
    "differentiating": 1.1,
    "circumscribing": 1.3,
    "stagnating": 2.5,
    "avoiding": 3.0,
    "terminating": 2.0,
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _get_big5(state: AgentState) -> Dict[str, float]:
    big5 = state.get("bot_big_five") or {}

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
    big5 = _get_big5(state)
    mood = _get_mood(state)
    rel = _get_rel(state)
    stage = state.get("current_stage", "experimenting")

    # Personality coefficients
    p_speed = 1.0 - (big5["extraversion"] * 0.2)  # E ↑ -> faster
    p_caution = 1.0 + (big5["conscientiousness"] * 0.3)  # C ↑ -> slower
    p_noise_var = 0.1 + (max(0.0, big5["neuroticism"]) * 0.4)  # N ↑ -> jitter

    # State modifiers
    m_arousal_boost = 1.0 - (mood["arousal"] * 0.3)
    m_busyness_drag = 1.0 + (mood["busyness"] * 1.5)

    # Relationship modifiers
    r_intimacy_frag = rel["closeness"] / 100.0
    r_stage_factor = float(STAGE_DELAY_FACTORS.get(str(stage), 1.0))

    speed_factor = p_speed * p_caution * m_arousal_boost * m_busyness_drag * r_stage_factor
    fragmentation_tendency = (big5["extraversion"] * 0.5) + (r_intimacy_frag * 0.5) + (
        mood["arousal"] * 0.3
    )

    return {
        "speed_factor": _clamp(speed_factor, 0.2, 5.0),
        "noise_level": _clamp(p_noise_var, 0.05, 0.8),
        "fragmentation_tendency": fragmentation_tendency,
        "stage_factor": r_stage_factor,
    }


def _segment_text(*, text: str, dynamics: Dict[str, float]) -> List[str]:
    if not text:
        return []
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
        if len(current_buf) > split_threshold or "\n" in seg:
            bubbles.append(current_buf.strip())
            current_buf = ""

    if current_buf.strip():
        bubbles.append(current_buf.strip())
    return bubbles


def create_behavior_processor_node() -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        final_text = (state.get("final_response") or state.get("draft_response") or "").strip()
        dynamics = calculate_human_dynamics(state)

        user_input_len = len(state.get("user_input") or "")
        t_read = 0.5 + (user_input_len * 0.05)
        cognitive_load = len(final_text) * 0.02
        t_cog = (1.0 + cognitive_load) * float(dynamics["speed_factor"])

        noise = random.gauss(1.0, float(dynamics["noise_level"]))
        t_cog *= _clamp(noise, 0.5, 2.0)

        typing_speed_char_per_sec = 5.0 / float(dynamics["speed_factor"])
        t_type_total_est = (len(final_text) / typing_speed_char_per_sec) if final_text else 0.0

        bubbles = _segment_text(text=final_text, dynamics=dynamics)

        accumulated_delay = t_read + t_cog
        timeline_segments: List[Dict[str, Any]] = []
        for bub in bubbles:
            t_type = (len(bub) / typing_speed_char_per_sec) if bub else 0.0
            t_type *= random.uniform(0.9, 1.1)
            timeline_segments.append(
                {"content": bub, "delay": round(accumulated_delay, 2), "action": "typing"}
            )
            accumulated_delay = _clamp(t_type, 0.05, 30.0)

        total_latency_simulated = (
            float(sum(s["delay"] for s in timeline_segments)) if timeline_segments else float(t_read + t_cog)
        )

        return {
            "final_segments": bubbles,
            "final_delay": round(t_read + t_cog, 2),
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

    return node


"""
processor.py
拟人化行为表现层 (Anthropomorphic Behavioral Layer)

功能：
1) 宏观异步门控 (Macro Gating): 基于昼夜节律、忙碌度和关系策略计算长延迟（睡眠/忙碌/策略性沉默）。
2) 微观交互节奏 (Micro Dynamics): 基于 SIP 理论计算阅读、思考与打字时间（Big Five + Mood + Knapp）。
3) 动态分段 (Segmentation): 基于话轮构建单元 (TCUs) 将文本拆分为多个气泡。

理论锚点：
- Chronemics (Time as Nonverbal Cue)
- Social Information Processing Theory (Walther, 1992)
- Big Five Personality Traits in CMC
- Knapp's Relational Interaction Model

输出（兼容 + 新结构）：
- humanized_output: HumanizedOutput
- final_segments: List[str]（兼容旧客户端/下游 memory writer）
- final_delay: float（兼容旧客户端：第一段出现前等待时间）
"""

from __future__ import annotations

import random
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

from app.state import AgentState, HumanizedOutput, ResponseSegment

# ==========================================
# 配置常量 (Configuration)
# ==========================================

# 机器人作息表 (Circadian Rhythms)
BOT_SCHEDULE: Dict[str, int] = {
    "sleep_start": 23,  # 23:00 入睡
    "sleep_end": 7,  # 07:00 起床
    "work_start": 9,  # 09:00 工作开始（预留）
    "work_end": 18,  # 18:00 工作结束（预留）
}

# Knapp 阶段的时间策略因子 (基于 Chronemics & EVT)
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

# 基础生理参数
AVG_READING_SPEED = 0.05  # 秒/字符（约 20字/秒）
BASE_TYPING_SPEED = 5.0  # 字符/秒
MIN_BUBBLE_LENGTH = 2  # 防止切太碎


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class HumanizationProcessor:
    def __init__(self, state: AgentState):
        self.state = state
        self.big5 = state.get(
            "bot_big_five",
            {"extraversion": 0.0, "conscientiousness": 0.0, "neuroticism": 0.0},
        )
        self.mood = state.get(
            "mood_state", {"arousal": 0.0, "busyness": 0.0, "pleasure": 0.0}
        )
        self.rel = state.get("relationship_state", {"closeness": 50.0, "power": 50.0})
        self.stage = str(state.get("current_stage", "experimenting"))
        self.current_time_str = str(state.get("current_time") or "")

    # ------------------------------------------
    # Micro dynamics
    # ------------------------------------------
    def _calculate_dynamics_modifiers(self) -> Dict[str, float]:
        """
        返回:
        - speed_factor: 速度倍率（>1 更慢）
        - fragmentation_tendency: 0..1（越大越爱碎片）
        - noise_level: 0..（神经质犹豫抖动）
        """
        e = float(self.big5.get("extraversion", 0.0) or 0.0)
        c = float(self.big5.get("conscientiousness", 0.0) or 0.0)
        n = float(self.big5.get("neuroticism", 0.0) or 0.0)

        ar = float(self.mood.get("arousal", 0.0) or 0.0)
        busy = float(self.mood.get("busyness", 0.0) or 0.0)
        pleasure = float(self.mood.get("pleasure", 0.0) or 0.0)

        closeness = float(self.rel.get("closeness", 50.0) or 50.0)

        # 1) Personality coefficients
        p_speed = 1.0 - (e * 0.2)
        p_caution = 1.0 + (c * 0.3)
        p_noise = 0.1 + (max(0.0, n) * 0.4)

        # 2) State modifiers
        m_arousal_boost = 1.0 - (ar * 0.3)
        m_busyness_drag = 1.0 + (busy * 1.5)

        # 3) Relational modifiers
        stage_factor = float(STAGE_DELAY_FACTORS.get(self.stage, 1.0))
        total_speed_factor = p_speed * p_caution * m_arousal_boost * m_busyness_drag * stage_factor

        # 分段倾向：外向 + 亲密 + 兴奋
        frag_tendency = (e * 0.4) + ((closeness / 100.0) * 0.4) + (ar * 0.2)

        return {
            "speed_factor": _clamp(float(total_speed_factor), 0.2, 5.0),
            "fragmentation_tendency": _clamp(float(frag_tendency), 0.0, 1.0),
            "noise_level": _clamp(float(p_noise), 0.05, 0.8),
            # 额外返回供宏观策略使用的 mood（不强依赖）
            "pleasure": _clamp(float(pleasure), -1.0, 1.0),
            "stage_factor": stage_factor,
        }

    # ------------------------------------------
    # Macro gating
    # ------------------------------------------
    def _calculate_macro_delay(self, dyn: Dict[str, float]) -> Tuple[float, str]:
        """
        计算宏观不可用时间（睡眠、忙碌、策略性沉默）
        返回: (延迟秒数, 原因类型)
        """
        try:
            now = datetime.fromisoformat(self.current_time_str)
        except Exception:
            now = datetime.now()

        current_hour = int(now.hour)

        # A) 睡眠检测
        s_start = int(BOT_SCHEDULE["sleep_start"])
        s_end = int(BOT_SCHEDULE["sleep_end"])

        is_sleeping = False
        if s_start > s_end:  # 跨夜
            if current_hour >= s_start or current_hour < s_end:
                is_sleeping = True
        else:
            if s_start <= current_hour < s_end:
                is_sleeping = True

        if is_sleeping:
            if current_hour >= s_start:
                target_hour = s_end + 24
            else:
                target_hour = s_end
            hours_left = target_hour - current_hour
            wake_up_jitter = random.uniform(900.0, 2700.0)  # 15~45min
            total_sleep_delay = (
                (hours_left * 3600.0)
                - (now.minute * 60.0)
                - float(now.second)
                + wake_up_jitter
            )
            return max(0.0, total_sleep_delay), "sleep"

        # B) 关系策略：Strategic Silence / Ghosting
        ghosting_prob = 0.0
        if self.stage in ("avoiding", "terminating"):
            ghosting_prob = 0.8
        elif self.stage == "stagnating":
            ghosting_prob = 0.5

        # 心情不好更容易放大
        if float(dyn.get("pleasure", 0.0)) < -0.3:
            ghosting_prob += 0.3

        ghosting_prob = _clamp(ghosting_prob, 0.0, 0.95)
        if random.random() < ghosting_prob:
            return random.uniform(7200.0, 43200.0), "ghosting"  # 2~12小时

        # C) 忙碌
        busyness = float(self.mood.get("busyness", 0.0) or 0.0)
        if busyness > 0.85 and random.random() < 0.7:
            return random.uniform(1800.0, 14400.0), "busy"  # 30min~4h

        return 0.0, "online"

    # ------------------------------------------
    # Segmentation
    # ------------------------------------------
    def _segment_text(self, text: str, tendency: float) -> List[str]:
        """
        基于 TCU 的分段
        tendency: 0(保守长句) -> 1(激进短句)
        """
        clean_text = (text or "").replace("**", "").strip()
        if not clean_text:
            return []

        raw_parts = re.split(r"([。！？\n]+)", clean_text)
        bubbles: List[str] = []
        current_buf = ""

        # High tendency -> threshold ~5; Low -> ~45
        split_threshold = 45.0 - (float(tendency) * 40.0)
        split_threshold = _clamp(split_threshold, 5.0, 60.0)

        for part in raw_parts:
            if not part:
                continue
            current_buf += part

            is_punct = re.match(r"[。！？\n]+", part)
            if not is_punct:
                continue

            if "\n" in part:
                bubbles.append(current_buf.strip())
                current_buf = ""
            elif len(current_buf) >= split_threshold:
                bubbles.append(current_buf.strip())
                current_buf = ""

        if current_buf.strip():
            bubbles.append(current_buf.strip())

        filtered = [b for b in bubbles if len(b) >= MIN_BUBBLE_LENGTH]
        return filtered if filtered else bubbles

    # ------------------------------------------
    # Main
    # ------------------------------------------
    def process(self) -> HumanizedOutput:
        user_input = str(self.state.get("user_input") or "")
        final_response = (
            str(self.state.get("final_response") or self.state.get("draft_response") or "")
        ).strip()

        dyn = self._calculate_dynamics_modifiers()
        macro_delay, macro_reason = self._calculate_macro_delay(dyn)
        is_macro = macro_delay > 0.0

        # 读/想
        t_read = 0.5 + (len(user_input) * AVG_READING_SPEED)
        cognitive_load = len(final_response) * 0.02
        t_think = (1.0 + cognitive_load) * float(dyn["speed_factor"])
        noise = random.gauss(1.0, float(dyn["noise_level"]))
        t_think *= _clamp(float(noise), 0.5, 2.0)

        # 分段
        segments_text = self._segment_text(
            final_response, float(dyn["fragmentation_tendency"])
        )

        # 打字速度：与旧实现保持一致（speed_factor 越大越慢）
        typing_speed = BASE_TYPING_SPEED / float(dyn["speed_factor"])
        typing_speed = _clamp(float(typing_speed), 0.5, 30.0)

        segments: List[ResponseSegment] = []
        base_delay = float(macro_delay + t_read + t_think)

        for i, seg_text in enumerate(segments_text):
            motor_noise = random.uniform(0.9, 1.1)
            t_type = (len(seg_text) / typing_speed) * motor_noise if seg_text else 0.0
            t_type = _clamp(float(t_type), 0.05, 30.0)

            if i == 0:
                # 第一段：delay = 宏观等待 + 读 + 想 + 打字
                action: Any = "idle" if macro_delay > 300.0 else "typing"
                segments.append(
                    {"content": seg_text, "delay": round(base_delay + t_type, 2), "action": action}
                )
            else:
                # 后续段：相对上一气泡的等待（基本等于打字时间）
                segments.append({"content": seg_text, "delay": round(t_type, 2), "action": "typing"})

        total_latency = float(sum(float(s["delay"]) for s in segments)) if segments else float(base_delay)

        # 输出：严格字段 + 可选 debug
        out: HumanizedOutput = {
            "total_latency_seconds": round(total_latency, 2),
            "segments": segments,
            "is_macro_delay": bool(is_macro),
            # debug/兼容
            "total_latency_simulated": round(total_latency, 2),
            "latency_breakdown": {
                "macro_delay": round(float(macro_delay), 3),
                "t_read": round(float(t_read), 3),
                "t_think": round(float(t_think), 3),
                # 便于本地排查：online=0，其余=1
                "macro_reason": 0.0 if macro_reason == "online" else 1.0,
            },
        }
        return out


def humanize_response_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node: 将原始回复转换为拟人化的气泡序列
    """
    processor = HumanizationProcessor(state)
    result = processor.process()

    segs = result.get("segments") or []
    bubbles = [s.get("content", "") for s in segs]
    first_delay = float(segs[0].get("delay", 0.0)) if segs else 0.0

    return {
        "humanized_output": result,
        # 兼容旧输出
        "final_segments": bubbles,
        "final_delay": round(first_delay, 2),
    }


def create_processor_node() -> Callable[[AgentState], dict]:
    return humanize_response_node


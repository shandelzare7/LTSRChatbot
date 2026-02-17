"""
processor.py
拟人化行为表现层 (Anthropomorphic Behavioral Layer)

功能：
1) 宏观异步门控 (Macro Gating): 可由 LLM 决定或规则回退（睡眠/忙碌/策略性沉默）。
2) 微观交互节奏 (Micro Dynamics): 由 LLM 根据上下文、记忆、state 输出拆句与每条延迟。
3) 动态分段 (Segmentation): LLM 将回复拆成多条气泡并指定 delay/action。

记忆与上下文：与 reasoner/generator 一致——system 仅放 summary + retrieved；chat_buffer 分条放正文。
"""

from __future__ import annotations

import random
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.state import AgentState, HumanizedOutput, ResponseSegment
from utils.llm_json import parse_json_from_llm
from utils.tracing import trace_if_enabled

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
MIN_SEGMENT_DELAY_SECONDS = 1.2  # 第二条及以后最小间隔，避免「秒出」


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _clip01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _pad_to_01(v: Any, *, default: float = 0.5) -> float:
    """
    PAD values are commonly in -1..1, but sometimes already 0..1.
    Normalize to 0..1 (missing -> default).
    """
    if v is None:
        return _clip01(default)
    try:
        x = float(v)
    except Exception:
        return _clip01(default)
    # Heuristic: if outside 0..1, treat as -1..1.
    if x < 0.0 or x > 1.0:
        return _clip01((x + 1.0) / 2.0)
    return _clip01(x)


def _extreme_deadzone(x01: float, *, low: float = 0.2, high: float = 0.8) -> Tuple[float, float]:
    """
    Only extremes matter; mid-range has no effect.
    Returns (low_ext_strength, high_ext_strength), each 0..1.
    """
    x = _clip01(x01)
    lo = _clip01(low)
    hi = _clip01(high)
    if hi <= lo:
        lo, hi = 0.2, 0.8
    if x <= lo:
        return (0.0 if lo <= 0.0 else (lo - x) / lo), 0.0
    if x >= hi:
        return 0.0, (0.0 if hi >= 1.0 else (x - hi) / (1.0 - hi))
    return 0.0, 0.0


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
        self.rel = state.get("relationship_state", {"closeness": 0.5, "power": 0.5})
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
        # Big5 / relationship are expected to be 0..1; PAD mood can be -1..1 or 0..1.
        e = _clip01(self.big5.get("extraversion", 0.0) or 0.0)
        c = _clip01(self.big5.get("conscientiousness", 0.0) or 0.0)
        n = _clip01(self.big5.get("neuroticism", 0.0) or 0.0)

        # PAD: (-1..1) -> (0..1). If values are already 0..1, keep them.
        # Important: many pipelines use 0.0/0.0/0.0 to mean "neutral/unknown",
        # so we follow style.py behavior: when all are 0, treat as default 0.5.
        P_raw = float(self.mood.get("pleasure", 0.0) or 0.0)
        A_raw = float(self.mood.get("arousal", 0.0) or 0.0)
        D_raw = float(self.mood.get("dominance", 0.0) or 0.0)

        if (P_raw < 0.0 or P_raw > 1.0) or (A_raw < 0.0 or A_raw > 1.0) or (D_raw < 0.0 or D_raw > 1.0):
            pleasure01 = _clip01((P_raw + 1.0) / 2.0)
            arousal01 = _clip01((A_raw + 1.0) / 2.0)
            dominance01 = _clip01((D_raw + 1.0) / 2.0)
        else:
            pleasure01 = _clip01(P_raw)
            arousal01 = _clip01(A_raw)
            dominance01 = _clip01(D_raw)

        if pleasure01 == 0.0 and arousal01 == 0.0 and dominance01 == 0.0:
            pleasure01 = arousal01 = dominance01 = 0.5

        pleasure_raw = P_raw
        busy = _clip01(self.mood.get("busyness", 0.0) or 0.0)

        # relationship (6D, 0..1), default 0.5
        closeness = _clip01(self.rel.get("closeness", 0.5) or 0.5)
        trust = _clip01(self.rel.get("trust", 0.5) or 0.5)
        liking = _clip01(self.rel.get("liking", 0.5) or 0.5)
        respect = _clip01(self.rel.get("respect", 0.5) or 0.5)
        warmth = _clip01(self.rel.get("warmth", 0.5) or 0.5)
        power = _clip01(self.rel.get("power", 0.5) or 0.5)

        # 1) Personality coefficients
        p_speed = 1.0 - (e * 0.2)
        p_caution = 1.0 + (c * 0.3)
        p_noise = 0.1 + (max(0.0, n) * 0.4)

        # 2) State modifiers
        m_arousal_boost = 1.0 - (float(arousal01) * 0.3)
        m_busyness_drag = 1.0 + (busy * 1.5)

        # 3) Relational modifiers
        stage_factor = float(STAGE_DELAY_FACTORS.get(self.stage, 1.0))
        total_speed_factor = p_speed * p_caution * m_arousal_boost * m_busyness_drag * stage_factor

        # -------------------------------
        # Fragmentation tendency (0..1)
        #
        # - extraversion: EXTREMELY strong, linear (dominant)
        # - conscientiousness: higher => less fragmentation
        # - neuroticism: higher => more fragmentation
        # - mood & relationship: multi-dimensional; only extremes affect (deadzone in the middle)
        # -------------------------------
        frag = 0.05
        # Big5 (linear, extraversion dominates)
        frag += 1.10 * float(e)
        frag += -0.60 * float(c)
        frag += 0.55 * float(n)

        # Mood (extremes only)
        ar_lo, ar_hi = _extreme_deadzone(float(arousal01))
        p_lo, p_hi = _extreme_deadzone(float(pleasure01))
        d_lo, d_hi = _extreme_deadzone(float(dominance01))
        _, busy_hi = _extreme_deadzone(float(busy))

        frag += 0.25 * ar_hi + (-0.15) * ar_lo        # high arousal => more; very low => less
        frag += 0.15 * p_lo + (-0.05) * p_hi          # very low pleasure => more; very high => slightly less
        frag += 0.10 * d_lo + (-0.05) * d_hi          # very low dominance => more; high => less
        frag += (-0.20) * busy_hi                     # very busy => fewer segments

        # Relationship (6D, extremes only)
        clo_lo, clo_hi = _extreme_deadzone(float(closeness))
        tru_lo, tru_hi = _extreme_deadzone(float(trust))
        lik_lo, lik_hi = _extreme_deadzone(float(liking))
        res_lo, res_hi = _extreme_deadzone(float(respect))
        war_lo, war_hi = _extreme_deadzone(float(warmth))
        pow_lo, pow_hi = _extreme_deadzone(float(power))

        frag += 0.22 * clo_hi + (-0.12) * clo_lo
        frag += 0.12 * tru_hi + (-0.06) * tru_lo
        frag += 0.14 * lik_hi + (-0.07) * lik_lo
        frag += 0.14 * war_hi + (-0.07) * war_lo
        frag += (-0.10) * res_hi + 0.05 * res_lo      # high respect => fewer segments
        frag += 0.08 * pow_lo + (-0.08) * pow_hi      # high power => fewer segments

        frag_tendency = _clip01(frag)

        return {
            "speed_factor": _clamp(float(total_speed_factor), 0.2, 5.0),
            "fragmentation_tendency": _clamp(float(frag_tendency), 0.0, 1.0),
            "noise_level": _clamp(float(p_noise), 0.05, 0.8),
            # 额外返回供宏观策略使用的 mood（不强依赖）
            "pleasure": _clamp(float(pleasure_raw), -1.0, 1.0),
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
            ghosting_prob = 0.65
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
            return random.uniform(1800.0, 7200.0), "busy"  # 30min~2h

        return 0.0, "online"

    def compute_delays_for_messages(self, msgs: List[str]) -> Tuple[List[float], List[str]]:
        """
        对已给定的消息列表（如 LATS 的 processor_plan.messages）纯由 processor 计算每条延迟。
        与 process() 使用相同的读/想/打字节奏与阶段因子，不重新分段。
        """
        if not msgs:
            return [], []
        dyn = self._calculate_dynamics_modifiers()
        macro_delay, macro_reason = self._calculate_macro_delay(dyn)
        user_input = str(self.state.get("user_input") or "")
        full_len = sum(len(m) for m in msgs)
        t_read = 0.5 + (len(user_input) * AVG_READING_SPEED)
        cognitive_load = full_len * 0.02
        t_think = (1.0 + cognitive_load) * float(dyn["speed_factor"])
        noise = random.gauss(1.0, float(dyn["noise_level"]))
        t_think *= _clamp(float(noise), 0.5, 2.0)
        typing_speed = BASE_TYPING_SPEED / float(dyn["speed_factor"])
        typing_speed = _clamp(float(typing_speed), 0.5, 30.0)
        base_delay = float(macro_delay + t_read + t_think)
        delays: List[float] = []
        actions: List[str] = []
        for i, seg_text in enumerate(msgs):
            motor_noise = random.uniform(0.9, 1.1)
            t_type = (len(seg_text) / typing_speed) * motor_noise if seg_text else 0.0
            t_type = _clamp(float(t_type), MIN_SEGMENT_DELAY_SECONDS, 30.0) if i > 0 else _clamp(float(t_type), 0.05, 30.0)
            act = "idle" if macro_delay > 300.0 and i == 0 else "typing"
            actions.append(act)
            if i == 0:
                delays.append(round(base_delay + t_type, 2))
            else:
                delays.append(round(t_type, 2))
        return delays, actions

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

        # High tendency -> threshold ~8; Low -> ~45
        split_threshold = 45.0 - (float(tendency) * 40.0)
        split_threshold = _clamp(split_threshold, 8.0, 60.0)

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
        
        # ### 6.2 需要监控的参数 - fragmentation_tendency 的实际分布
        frag_tendency = float(dyn.get("fragmentation_tendency", 0.0))
        print(f"[MONITOR] fragmentation_tendency={frag_tendency:.3f} (reached_1.0={frag_tendency >= 0.99})")
        
        # ### 6.2 需要监控的参数 - 宏观延迟触发的频率
        if is_macro:
            delay_hours = macro_delay / 3600.0
            print(f"[MONITOR] macro_delay_triggered: reason={macro_reason}, delay={delay_hours:.2f}h ({macro_delay:.1f}s)")

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
            t_type = _clamp(float(t_type), MIN_SEGMENT_DELAY_SECONDS if i > 0 else 0.05, 30.0)

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


def _build_processor_system_prompt(state: Dict[str, Any]) -> str:
    """System 仅含 summary + retrieved + 角色与 state 上下文（不含 chat_buffer）。"""
    summary = state.get("conversation_summary") or ""
    retrieved = state.get("retrieved_memories") or []
    memory_parts = []
    if summary:
        memory_parts.append("近期对话摘要：\n" + summary)
    if retrieved:
        memory_parts.append("相关记忆片段：\n" + "\n".join(retrieved))
    system_memory = "\n\n".join(memory_parts) if memory_parts else "（无）"

    bot = state.get("bot_basic_info") or {}
    mood = state.get("mood_state") or {}
    stage = state.get("current_stage") or "experimenting"
    rel = state.get("relationship_state") or {}
    current_time = state.get("current_time") or ""
    # 作息提示（供 LLM 决定是否加长延迟）
    schedule = f"作息: {BOT_SCHEDULE.get('sleep_start', 23)}:00 入睡, {BOT_SCHEDULE.get('sleep_end', 7)}:00 起床"

    return f"""# Role
你是语感优秀、常识经验丰富的语言学专家。凭借你对自然语言节奏和真人聊天习惯的深刻理解，根据对话上下文、记忆和当前状态，将「待拆句的回复」拆成多条气泡，并为每条指定发送前的等待时间（delay，秒）和动作（action）。

# Memory (Summary + Retrieved)
{system_memory}

# Current State（供你决定节奏与是否长延迟）
- Bot: {bot.get('name', 'Bot')}，当前情绪 PAD: {mood}
- 关系阶段: {stage}，亲密/信任等: {rel}
- 当前时间: {current_time or '未提供'}
- {schedule}

# Output Format (STRICT JSON ONLY)
必须返回一个 JSON 对象，且仅此对象，不要其他文字：
{{
  "segments": [
    {{ "content": "第一句或第一段。", "delay": 1.2, "action": "typing" }},
    {{ "content": "第二句。", "delay": 0.8, "action": "typing" }}
  ],
  "is_macro_delay": false,
  "macro_delay_seconds": 0
}}

规则：
- content: 从待拆句回复中切出的完整子句/短语，不要截断语义。
- delay: 该条气泡发送前相对上一条的等待秒数（≥0）。第一条的 delay 可包含「读用户消息+思考+首段打字」的合成时间（建议 0.5～3.0）。
- action: 仅 "typing" 或 "idle"。若你判断当前应长时间不回复（如睡觉、忙碌、冷处理），可设 is_macro_delay=true、macro_delay_seconds>0，且第一条 segment 的 action 可为 "idle"。
- 至少返回 1 条 segment；若回复很短可只 1 条。
"""


def _humanize_via_llm(state: AgentState, llm_invoker: Any) -> HumanizedOutput | None:
    """用 LLM 做拆句与延迟；失败返回 None，由调用方回退规则。"""
    user_input = str(state.get("user_input") or "")
    final_response = (
        str(state.get("final_response") or state.get("draft_response") or "")
    ).strip()
    if not final_response:
        return None

    system_prompt = _build_processor_system_prompt(state)
    chat_buffer = state.get("chat_buffer") or []
    body_messages = list(chat_buffer[-20:])
    task_content = f"""请将以下回复拆成多条气泡，并为每条指定 delay（秒）和 action（typing 或 idle）。

用户刚说的：
{user_input}

待拆句的回复：
{final_response}
"""

    try:
        response = llm_invoker.invoke([
            SystemMessage(content=system_prompt),
            *body_messages,
            HumanMessage(content=task_content),
        ])
        content = getattr(response, "content", "") or ""
        data = parse_json_from_llm(content)
        if not isinstance(data, dict):
            return None
        segments_raw = data.get("segments")
        if not segments_raw or not isinstance(segments_raw, list):
            return None
        segments: List[ResponseSegment] = []
        for item in segments_raw:
            if not isinstance(item, dict):
                continue
            c = item.get("content")
            if c is None:
                continue
            d = item.get("delay")
            try:
                delay_val = float(d) if d is not None else 0.5
            except (TypeError, ValueError):
                delay_val = 0.5
            delay_val = max(0.0, min(60.0, delay_val))
            action = item.get("action")
            if action not in ("typing", "idle"):
                action = "typing"
            segments.append({"content": str(c).strip(), "delay": round(delay_val, 2), "action": action})
        if not segments:
            return None
        is_macro = bool(data.get("is_macro_delay", False))
        macro_sec = float(data.get("macro_delay_seconds", 0) or 0)
        total_latency = sum(s["delay"] for s in segments)
        return {
            "total_latency_seconds": round(total_latency, 2),
            "segments": segments,
            "is_macro_delay": is_macro,
            "total_latency_simulated": round(total_latency, 2),
            "latency_breakdown": {
                "macro_delay": round(macro_sec, 3),
                "t_read": 0.0,
                "t_think": 0.0,
                "macro_reason": 1.0 if is_macro else 0.0,
            },
        }
    except Exception:
        return None


def _humanize_from_processor_plan(state: AgentState) -> HumanizedOutput | None:
    """
    执行 LATS/编译器产出的 ProcessorPlan。
    ReplyPlan 已去掉延迟参数，仅含 messages；延迟由本 processor 纯计算。
    若 plan 带 delays/actions 且长度匹配则沿用，否则用 HumanizationProcessor 计算延迟。
    """
    plan = state.get("processor_plan") or {}
    if not isinstance(plan, dict):
        return None
    msgs = plan.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    # 归一化为字符串列表
    texts: List[str] = []
    for m in msgs:
        if isinstance(m, str):
            t = (m or "").strip()
        elif isinstance(m, dict):
            t = str(m.get("content") or "").strip()
        else:
            t = str(m or "").strip()
        if t:
            texts.append(t)
    if not texts:
        return None

    # 延迟/节奏统一由 processor 计算；不沿用 plan 内可能残留的 delays/actions（避免上游组件覆盖节奏）。
    proc = HumanizationProcessor(state)
    delays, actions = proc.compute_delays_for_messages(texts)

    segments: List[ResponseSegment] = []
    for i, (text, d, a) in enumerate(zip(texts, delays, actions)):
        try:
            delay_val = float(d)
        except Exception:
            delay_val = 0.5
        delay_val = max(0.0, min(86400.0, delay_val))
        action = a if a in ("typing", "idle") else "typing"
        segments.append({"content": text, "delay": round(delay_val, 2), "action": action})

    if not segments:
        return None

    total_latency = float(sum(float(s["delay"]) for s in segments))
    meta = plan.get("meta") if isinstance(plan.get("meta"), dict) else {}
    is_macro = bool(meta.get("macro") or meta.get("is_macro_delay") or False)
    macro_sec = float(meta.get("macro_delay_seconds", 0.0) or 0.0)
    return {
        "total_latency_seconds": round(total_latency, 2),
        "segments": segments,
        "is_macro_delay": bool(is_macro),
        "total_latency_simulated": round(total_latency, 2),
        "latency_breakdown": {
            "macro_delay": round(float(macro_sec), 3),
            "t_read": 0.0,
            "t_think": 0.0,
            "macro_reason": 1.0 if is_macro else 0.0,
        },
    }


def humanize_response_node(state: AgentState, llm_invoker: Any = None) -> Dict[str, Any]:
    """
    LangGraph Node: 将原始回复转换为拟人化的气泡序列。
    当启用 LLM 时优先用 LLM（更语义化的拆句与节奏），失败再回退到 plan/规则。
    """
    use_llm = str(state.get("processor_use_llm") or "").strip().lower() in ("1", "true", "yes", "on")
    result: HumanizedOutput | None = None

    # 1) Prefer LLM when enabled.
    if llm_invoker and use_llm:
        result = _humanize_via_llm(state, llm_invoker)

    # 2) Fall back to executable plan (when available).
    if not result:
        result = _humanize_from_processor_plan(state)

    # 3) Final fallback: deterministic rules.
    if not result:
        processor = HumanizationProcessor(state)
        result = processor.process()

    segs = result.get("segments") or []
    bubbles = [s.get("content", "") for s in segs]
    first_delay = float(segs[0].get("delay", 0.0)) if segs else 0.0
    total_latency = float(result.get("total_latency_seconds", 0.0) or 0.0)
    
    print("[Processor] done")
    print(f"[Processor] 延迟规划:")
    print(f"  - 消息数: {len(segs)}")
    print(f"  - 首条延迟: {first_delay:.2f}秒")
    print(f"  - 总延迟: {total_latency:.2f}秒")
    for i, seg in enumerate(segs):
        content_preview = (seg.get("content", "") or "")[:40]
        delay_val = float(seg.get("delay", 0.0) or 0.0)
        action_val = seg.get("action", "typing")
        print(f"  [{i+1}] delay={delay_val:.2f}s, action={action_val}, content=\"{content_preview}...\"")

    return {
        "humanized_output": result,
        "final_segments": bubbles,
        "final_delay": round(first_delay, 2),
    }


def create_processor_node(llm_invoker: Any = None) -> Callable[[AgentState], dict]:
    """创建 processor 节点；传入 llm 时使用 LLM 拆句+延迟，否则仅用规则。"""
    @trace_if_enabled(
        name="Response/Processor",
        run_type="chain",
        tags=["node", "processor", "humanize"],
        metadata={"state_outputs": ["humanized_output", "final_segments", "final_delay"]},
    )
    def node(state: AgentState) -> dict:
        return humanize_response_node(state, llm_invoker)
    return node


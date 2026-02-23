"""
processor.py
拟人化行为表现层 (Anthropomorphic Behavioral Layer)

功能：
1) 宏观异步门控 (Macro Gating): 可由 LLM 决定或规则回退（睡眠/忙碌/策略性沉默）。
2) 微观交互节奏 (Micro Dynamics): 提取大五人格、PAD情绪计算出速度倍率与碎片化倾向。
3) 动态分段与改写 (Segmentation + Punct Normalization via LLM):
   - 完全交由 LLM 进行分段，LLM 将严格参考底层算出的“碎片化倾向 (fragmentation_tendency)”，按语义情感自然切分
   - 去除明显 AI 出戏符号（：～——() 等），保留问号等关键情绪符号，但不丢失任何信息

记忆与上下文：system 仅放 summary + retrieved；chat_buffer 分条放正文。
"""

from __future__ import annotations

import random
import re
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.state import AgentState, HumanizedOutput, ResponseSegment
from src.schemas import ProcessorOutput
from utils.llm_json import parse_json_from_llm
from utils.tracing import trace_if_enabled

# ==========================================
# 配置常量 (Configuration)
# ==========================================

BOT_SCHEDULE: Dict[str, int] = {
    "sleep_start": 23,
    "sleep_end": 7,
    "work_start": 9,
    "work_end": 18,
}

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

AVG_READING_SPEED = 0.05
BASE_TYPING_SPEED = 1.8
MIN_BUBBLE_LENGTH = 2
MIN_SEGMENT_DELAY_SECONDS = 1.2


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _clip01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0.0: return 0.0
    if v > 1.0: return 1.0
    return v

def _pad_to_01(v: Any, *, default: float = 0.5) -> float:
    if v is None: return _clip01(default)
    try:
        x = float(v)
    except Exception:
        return _clip01(default)
    if x < 0.0 or x > 1.0:
        return _clip01((x + 1.0) / 2.0)
    return _clip01(x)

def _extreme_deadzone(x01: float, *, low: float = 0.2, high: float = 0.8) -> Tuple[float, float]:
    x = _clip01(x01)
    lo = _clip01(low)
    hi = _clip01(high)
    if hi <= lo: lo, hi = 0.2, 0.8
    if x <= lo: return (0.0 if lo <= 0.0 else (lo - x) / lo), 0.0
    if x >= hi: return 0.0, (0.0 if hi >= 1.0 else (x - hi) / (1.0 - hi))
    return 0.0, 0.0

def _estimate_formality(state: Dict[str, Any]) -> float:
    rel = state.get("relationship_state") or {}
    stage = str(state.get("current_stage") or "experimenting")
    closeness = _clip01(rel.get("closeness", 0.5) or 0.5)
    respect = _clip01(rel.get("respect", 0.5) or 0.5)

    stage_formality_map = {
        "initiating": 0.75, "experimenting": 0.60, "intensifying": 0.40,
        "integrating": 0.45, "bonding": 0.35, "differentiating": 0.60,
        "circumscribing": 0.70, "stagnating": 0.80, "avoiding": 0.85,
        "terminating": 0.85,
    }
    stage_formality = float(stage_formality_map.get(stage, 0.55))
    f = 0.45 * stage_formality + 0.40 * (1.0 - float(closeness)) + 0.15 * float(respect)
    return _clamp(float(f), 0.0, 1.0)

def _estimate_end_punct_ratio(formality01: float) -> float:
    return _clamp(0.08 + 0.22 * float(formality01), 0.05, 0.35)


class HumanizationProcessor:
    def __init__(self, state: AgentState):
        self.state = state
        self.big5 = state.get("bot_big_five", {"extraversion": 0.0, "conscientiousness": 0.0, "neuroticism": 0.0})
        self.mood = state.get("mood_state", {"arousal": 0.0, "busyness": 0.0, "pleasure": 0.0})
        self.rel = state.get("relationship_state", {"closeness": 0.5, "power": 0.5})
        self.stage = str(state.get("current_stage", "experimenting"))
        self.current_time_str = str(state.get("current_time") or "")

    def calculate_dynamics_modifiers(self) -> Dict[str, float]:
        """计算底层动态参数供 LLM 和后续逻辑使用"""
        e = _clip01(self.big5.get("extraversion", 0.0) or 0.0)
        c = _clip01(self.big5.get("conscientiousness", 0.0) or 0.0)
        n = _clip01(self.big5.get("neuroticism", 0.0) or 0.0)

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

        busy = _clip01(self.mood.get("busyness", 0.0) or 0.0)

        closeness = _clip01(self.rel.get("closeness", 0.5) or 0.5)
        trust = _clip01(self.rel.get("trust", 0.5) or 0.5)
        liking = _clip01(self.rel.get("liking", 0.5) or 0.5)
        respect = _clip01(self.rel.get("respect", 0.5) or 0.5)
        power = _clip01(self.rel.get("power", 0.5) or 0.5)

        p_speed = 1.0 - (e * 0.2)
        p_caution = 1.0 + (c * 0.3)
        p_noise = 0.1 + (max(0.0, n) * 0.4)

        m_arousal_boost = 1.0 - (float(arousal01) * 0.3)
        m_busyness_drag = 1.0 + (busy * 1.5)

        stage_factor = float(STAGE_DELAY_FACTORS.get(self.stage, 1.0))
        total_speed_factor = p_speed * p_caution * m_arousal_boost * m_busyness_drag * stage_factor

        frag = 0.05
        frag += 1.10 * float(e)
        frag += -0.60 * float(c)
        frag += 0.55 * float(n)

        ar_lo, ar_hi = _extreme_deadzone(float(arousal01))
        p_lo, p_hi = _extreme_deadzone(float(pleasure01))
        d_lo, d_hi = _extreme_deadzone(float(dominance01))
        _, busy_hi = _extreme_deadzone(float(busy))

        frag += 0.25 * ar_hi + (-0.15) * ar_lo
        frag += 0.15 * p_lo + (-0.05) * p_hi
        frag += 0.10 * d_lo + (-0.05) * d_hi
        frag += (-0.20) * busy_hi

        clo_lo, clo_hi = _extreme_deadzone(float(closeness))
        tru_lo, tru_hi = _extreme_deadzone(float(trust))
        lik_lo, lik_hi = _extreme_deadzone(float(liking))
        res_lo, res_hi = _extreme_deadzone(float(respect))
        pow_lo, pow_hi = _extreme_deadzone(float(power))

        frag += 0.22 * clo_hi + (-0.12) * clo_lo
        frag += 0.12 * tru_hi + (-0.06) * tru_lo
        frag += 0.14 * lik_hi + (-0.07) * lik_lo
        frag += (-0.10) * res_hi + 0.05 * res_lo
        frag += 0.08 * pow_lo + (-0.08) * pow_hi

        frag_tendency = _clip01(frag)
        momentum = _clip01(float(self.state.get("conversation_momentum", 1.0) or 1.0))
        frag_tendency *= momentum

        return {
            "speed_factor": _clamp(float(total_speed_factor), 0.2, 5.0),
            "fragmentation_tendency": _clamp(float(frag_tendency), 0.0, 1.0),
            "noise_level": _clamp(float(p_noise), 0.05, 0.8),
            "pleasure": _clamp(float(P_raw), -1.0, 1.0),
            "stage_factor": stage_factor,
        }

    def calculate_macro_delay(self, dyn: Dict[str, float]) -> Tuple[float, str]:
        try:
            now = datetime.fromisoformat(self.current_time_str)
        except Exception:
            now = datetime.now()

        current_hour = int(now.hour)
        s_start = int(BOT_SCHEDULE["sleep_start"])
        s_end = int(BOT_SCHEDULE["sleep_end"])

        is_sleeping = False
        if s_start > s_end:
            if current_hour >= s_start or current_hour < s_end:
                is_sleeping = True
        else:
            if s_start <= current_hour < s_end:
                is_sleeping = True

        if is_sleeping:
            target_hour = s_end + 24 if current_hour >= s_start else s_end
            hours_left = target_hour - current_hour
            wake_up_jitter = random.uniform(900.0, 2700.0)
            total_sleep_delay = ((hours_left * 3600.0) - (now.minute * 60.0) - float(now.second) + wake_up_jitter)
            return max(0.0, total_sleep_delay), "sleep"

        ghosting_prob = 0.0
        if self.stage in ("avoiding", "terminating"):
            ghosting_prob = 0.65
        elif self.stage == "stagnating":
            ghosting_prob = 0.5

        if float(dyn.get("pleasure", 0.0)) < -0.3:
            ghosting_prob += 0.3

        ghosting_prob = _clamp(ghosting_prob, 0.0, 0.95)
        if random.random() < ghosting_prob:
            return random.uniform(7200.0, 43200.0), "ghosting"

        busyness = float(self.mood.get("busyness", 0.0) or 0.0)
        if busyness > 0.85 and random.random() < 0.7:
            return random.uniform(1800.0, 7200.0), "busy"

        return 0.0, "online"

    def process_fallback(self, dyn: Dict[str, float]) -> HumanizedOutput:
        """
        极简兜底算法：当 LLM 彻底失败时触发。
        不再使用正则尝试切分气泡，直接将 final_response 作为一条发出，保证流程不中断。
        """
        user_input = str(self.state.get("user_input") or "")
        final_response = (str(self.state.get("final_response") or self.state.get("draft_response") or "")).strip()

        macro_delay, macro_reason = self.calculate_macro_delay(dyn)
        is_macro = macro_delay > 0.0

        if is_macro:
            delay_hours = macro_delay / 3600.0
            print(f"[MONITOR-Fallback] macro_delay: reason={macro_reason}, delay={delay_hours:.2f}h")

        t_read = 0.5 + (len(user_input) * AVG_READING_SPEED)
        cognitive_load = len(final_response) * 0.02
        t_think = (1.0 + cognitive_load) * float(dyn["speed_factor"])
        noise = random.gauss(1.0, float(dyn["noise_level"]))
        t_think *= _clamp(float(noise), 0.5, 2.0)

        typing_speed = BASE_TYPING_SPEED / float(dyn["speed_factor"])
        typing_speed = _clamp(float(typing_speed), 0.5, 30.0)

        base_delay = float(macro_delay + t_read + t_think)
        motor_noise = random.uniform(0.9, 1.1)
        t_type = (len(final_response) / typing_speed) * motor_noise if final_response else 0.0
        t_type = _clamp(float(t_type), 0.05, 60.0)

        action: Any = "idle" if macro_delay > 300.0 else "typing"
        segments: List[ResponseSegment] = [
            {"content": final_response, "delay": round(base_delay + t_type, 2), "action": action}
        ]

        total_latency = float(segments[0]["delay"])

        out: HumanizedOutput = {
            "total_latency_seconds": round(total_latency, 2),
            "segments": segments,
            "is_macro_delay": bool(is_macro),
            "total_latency_simulated": round(total_latency, 2),
            "latency_breakdown": {
                "macro_delay": round(float(macro_delay), 3),
                "t_read": round(float(t_read), 3),
                "t_think": round(float(t_think), 3),
                "macro_reason": 0.0 if macro_reason == "online" else 1.0,
            },
        }
        return out


def _build_processor_system_prompt(state: Dict[str, Any], dyn: Dict[str, float]) -> str:
    summary = state.get("conversation_summary") or ""
    retrieved = state.get("retrieved_memories") or []
    memory_parts = []
    if summary: memory_parts.append("近期对话摘要：\n" + summary)
    if retrieved: memory_parts.append("相关记忆片段：\n" + "\n".join(retrieved))
    system_memory = "\n\n".join(memory_parts) if memory_parts else "（无）"

    bot = state.get("bot_basic_info") or {}
    mood = state.get("mood_state") or {}
    stage = state.get("current_stage") or "experimenting"
    rel = state.get("relationship_state") or {}
    current_time = state.get("current_time") or ""
    schedule = f"作息: {BOT_SCHEDULE.get('sleep_start', 23)}:00 入睡, {BOT_SCHEDULE.get('sleep_end', 7)}:00 起床"

    formality = _estimate_formality(state)
    end_punct_ratio = _estimate_end_punct_ratio(formality)
    
    tendency = float(dyn.get("fragmentation_tendency", 0.0))
    avoid_symbols = "： : ～ ~ —— — ( ) （ ）"

    return f"""# Role
你是语感优秀、常识经验丰富的语言学专家 + 资深聊天写作编辑。
你要做两件事（同时完成）：
1) 将「待发送的回复内容」按当前的“碎片化倾向”、语义情绪和对话节奏，自然地拆成一条或多条聊天气泡
2) 在不丢失任何信息的前提下，把文字改成更像真人聊天的标点/符号习惯（保留问号，去除句尾的句号和逗号，避免出戏符号）

# Memory (Summary + Retrieved)
{system_memory}

# Current State
- Bot: {bot.get('name', 'Bot')}，当前情绪 PAD（[-1,1]，0 为中性；busyness [0,1]）: {mood}
- 关系阶段: {stage}，亲密/信任等: {rel}
- 当前时间: {current_time or '未提供'}
- {schedule}

# Style Targets（你必须遵守）
- 目标正式度 formality(0~1): {formality:.2f}
- 目标“句末标点比例” end_punct_ratio(0~1): {end_punct_ratio:.2f} (单条气泡句尾的逗号和句号大概率要删除【直接留空】，但问号「？」必须保留)
- 目标碎片化倾向 fragmentation_tendency(0~1): {tendency:.2f} (0表示倾向于一次性发大长文，1表示倾向于像机关枪一样连发短句)
- 需要尽量避免的出戏符号（用改写避免使用，非机械删除）：{avoid_symbols}

# Few-Shot Examples (切分与改写参考)
【示例 1 - 倾向较低，倾向合在一起发】
原文：我今天去看了电影《流浪地球2》，特效真的很棒，你要一起去二刷吗？
输出：
[
  {{"content": "我今天去看了电影流浪地球2，特效真的很棒，你要一起去二刷吗？", "delay": 4.0, "action": "typing"}}
]

【示例 2 - 倾向较高，碎片化连发，自然语义切分，句末无句号/逗号，保留问号】
原文：哎呀，今天真是累死我了（老板又让我加班）！而且路上还堵车……你想吃点什么吗？我给你点外卖。
输出：
[
  {{"content": "哎呀今天真是累死我了", "delay": 2.8, "action": "typing"}},
  {{"content": "老板又让我加班", "delay": 3.2, "action": "typing"}},
  {{"content": "而且路上还堵车", "delay": 2.8, "action": "typing"}},
  {{"content": "你想吃点什么吗？", "delay": 3.2, "action": "typing"}},
  {{"content": "我给你点外卖", "delay": 2.8, "action": "typing"}}
]

# Hard Constraints（最重要）
- 不允许丢失任何事实、数字、专有名词、条件等信息
- 如果文本里出现 URL、文件路径、代码块（```...```），必须原样保留
- 切分策略：根据“碎片化倾向”、语义情感和节奏来自然拆分。倾向高就拆得多，倾向低就合在一起发，不要被具体字数限制，以语义连贯自然为准。
- 标点处理核心规则：正常对话里，问号（？）必须保留！消息单句句尾的逗号（，）和句号（。）大概率直接去掉（直接留空结束）。
- 严禁在气泡末尾强行添加“...”或“……”作为停顿，除非原文里本来就有！
- 不要用括号插入补充说明（拆成新气泡或用“另外”引出），不要用破折号做插入语。
"""

def _humanize_via_llm(state: AgentState, llm_invoker: Any, dyn: Dict[str, float]) -> HumanizedOutput | None:
    user_input = str(state.get("user_input") or "").strip()
    final_response = str(state.get("final_response") or state.get("draft_response") or "").strip()
    if not final_response:
        return None

    system_prompt = _build_processor_system_prompt(state, dyn)
    chat_buffer = state.get("chat_buffer") or []
    body_messages = list(chat_buffer[-20:])

    task_content = f"""你将收到一段“待发送的回复内容（原文）”。
请在不丢失任何信息的前提下：
- 改写成更像真人聊天的标点/符号习惯
- 根据系统提示中的碎片化倾向，将其保留为长段落或拆分成多条气泡，并给每条气泡 delay 与 action

用户刚说的：
{user_input}

待发送的回复内容（原文）：
{final_response}
"""

    messages = [
        SystemMessage(content=system_prompt),
        *body_messages,
        HumanMessage(content=task_content),
    ]

    def _data_to_result(data: Dict[str, Any]):
        if not data:
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
            text = str(c).strip()
            if not text:
                continue
            d = item.get("delay")
            try:
                delay_val = float(d) if d is not None else 2.5
            except (TypeError, ValueError):
                delay_val = 2.5
            delay_val = max(0.5, min(60.0, delay_val))
            action = item.get("action")
            if action not in ("typing", "idle"):
                action = "typing"
            segments.append({"content": text, "delay": round(delay_val, 2), "action": action})
        if not segments:
            return None
        is_macro = bool(data.get("is_macro_delay", False))
        try:
            macro_sec = float(data.get("macro_delay_seconds", 0) or 0)
        except Exception:
            macro_sec = 0.0
        total_latency = sum(float(s["delay"]) for s in segments)
        return {
            "total_latency_seconds": round(total_latency, 2),
            "segments": segments,
            "is_macro_delay": is_macro,
            "total_latency_simulated": round(total_latency, 2),
            "latency_breakdown": {
                "macro_delay": round(float(macro_sec), 3),
                "t_read": 0.0,
                "t_think": 0.0,
                "macro_reason": 1.0 if is_macro else 0.0,
            },
        }

    try:
        data = None
        if hasattr(llm_invoker, "with_structured_output"):
            try:
                structured = llm_invoker.with_structured_output(ProcessorOutput)
                obj = structured.invoke(messages)
                data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
            except Exception:
                data = None
        if data is None:
            response = llm_invoker.invoke(messages)
            content = getattr(response, "content", "") or ""
            data = parse_json_from_llm(content)
        if isinstance(data, dict):
            return _data_to_result(data)
    except Exception as e:
        print(f"[Processor] LLM parsing failed: {e}")
    return None


def humanize_response_node(state: AgentState, llm_invoker: Any = None) -> Dict[str, Any]:
    processor = HumanizationProcessor(state)
    dyn = processor.calculate_dynamics_modifiers()
    
    use_llm = str(state.get("processor_use_llm") or "").strip().lower() in ("1", "true", "yes", "on")
    result: HumanizedOutput | None = None

    if llm_invoker and use_llm:
        result = _humanize_via_llm(state, llm_invoker, dyn)

    if not result:
        # LLM 解析失败或未开启时，触发极简兜底算法（不再切分）
        result = processor.process_fallback(dyn)

    segs = result.get("segments") or []
    bubbles = [s.get("content", "") for s in segs]
    first_delay = float(segs[0].get("delay", 0.0)) if segs else 0.0
    total_latency = float(result.get("total_latency_seconds", 0.0) or 0.0)

    frag_tendency = float(dyn.get("fragmentation_tendency", 0.0))
    # 分割信息同时打 stdout（进日志）和 stderr（控制台可见，因图内常将 stdout 重定向到 log）
    def _console(s: str) -> None:
        print(s)
        try:
            sys.stderr.write(s + "\n")
            sys.stderr.flush()
        except Exception:
            pass
    _console(f"[Processor] 核心动态: 碎片化倾向 tendency={frag_tendency:.2f}")
    _console(f"[Processor] 延迟规划: 消息数 {len(segs)}, 首条延迟 {first_delay:.2f}s, 总延迟 {total_latency:.2f}s")
    for i, seg in enumerate(segs):
        full = (seg.get("content", "") or "")
        content_preview = (full[:40] + "...") if len(full) > 40 else full
        _console(f"  [{i+1}] delay={float(seg.get('delay', 0.0)):.2f}s, content=\"{content_preview}\"")

    return {
        "humanized_output": result,
        "final_segments": bubbles,
        "final_delay": round(first_delay, 2),
    }

def create_processor_node(llm_invoker: Any = None) -> Callable[[AgentState], dict]:
    @trace_if_enabled(
        name="Response/Processor",
        run_type="chain",
        tags=["node", "processor", "humanize"],
        metadata={"state_outputs": ["humanized_output", "final_segments", "final_delay"]},
    )
    def node(state: AgentState) -> dict:
        return humanize_response_node(state, llm_invoker)
    return node
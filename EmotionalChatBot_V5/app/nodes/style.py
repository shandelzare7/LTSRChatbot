"""Style 节点：根据关系、情绪、信号和阶段语境计算 12 维风格参数和 2 个门控变量（纯计算，无 LLM）。"""
from __future__ import annotations

from typing import Any, Callable, Dict

from utils.tracing import trace_if_enabled
from app.state import AgentState


def _clip01(x: float) -> float:
    """将值限制在 [0.0, 1.0] 范围内。"""
    return max(0.0, min(1.0, x))


def _avg2(a: float, b: float) -> float:
    """计算两个数的平均值。"""
    return (a + b) / 2.0


def _safe_get(d: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    """安全地从嵌套字典中获取值。"""
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, {})
        else:
            return default
    if isinstance(current, (int, float)):
        return float(current)
    return default


# Knapp stage baseline（宏观先验，不替代关系）
STAGE_PROFILE: Dict[int, Dict[str, float]] = {
    1: {"invest": 0.15, "ctx": 0.10},
    2: {"invest": 0.25, "ctx": 0.20},
    3: {"invest": 0.45, "ctx": 0.40},
    4: {"invest": 0.60, "ctx": 0.55},
    5: {"invest": 0.75, "ctx": 0.70},
    6: {"invest": 0.68, "ctx": 0.65},
    7: {"invest": 0.55, "ctx": 0.55},
    8: {"invest": 0.40, "ctx": 0.45},
    9: {"invest": 0.25, "ctx": 0.30},
    10: {"invest": 0.10, "ctx": 0.15},
}


def _get_stage_index(stage: Any) -> int:
    """从 stage 字符串或数字获取 stage_index (1-10)。"""
    if isinstance(stage, int):
        return max(1, min(10, stage))
    if isinstance(stage, str):
        # 尝试映射常见阶段名到索引
        stage_lower = stage.lower()
        stage_map = {
            "initiating": 1,
            "experimenting": 2,
            "intensifying": 3,
            "integrating": 4,
            "bonding": 5,
            "differentiating": 6,
            "circumscribing": 7,
            "stagnating": 8,
            "avoiding": 9,
            "terminating": 10,
        }
        if stage_lower in stage_map:
            return stage_map[stage_lower]
        # 尝试解析数字
        try:
            idx = int(stage)
            return max(1, min(10, idx))
        except:
            pass
    return 1  # 缺省


def create_style_node(llm_invoker: Any = None) -> Callable[[AgentState], dict]:
    """
    创建 Style 节点：纯计算，根据关系、情绪、信号和阶段语境计算风格参数。
    注意：llm_invoker 参数保留以兼容 graph.py，但不会被使用。
    """

    @trace_if_enabled(
        name="Style",
        run_type="chain",
        tags=["node", "style", "computation"],
        metadata={"state_outputs": ["style", "llm_instructions"]},
    )
    def style_node(state: AgentState) -> dict:
        """
        计算 12 维风格参数和 2 个门控变量。
        
        输入来源：
        1. 6维关系（relationship_state）
        2. PAD 情绪（mood_state）
        3. busy（mood_state.busyness）
        4. detection_signals（composite, trace, instant_eff, stage_ctx）
        5. current_stage（Knapp stage）
        """
        
        # A. 输入提取
        relationship_state = state.get("relationship_state") or {}
        mood_state = state.get("mood_state") or {}
        detection_signals = state.get("detection_signals") or {}
        current_stage = state.get("current_stage") or "initiating"
        
        # 1) 6维关系（系统内部统一为 0-1）
        closeness = _clip01(float(relationship_state.get("closeness", 0.5) or 0.5))
        trust = _clip01(float(relationship_state.get("trust", 0.5) or 0.5))
        liking = _clip01(float(relationship_state.get("liking", 0.5) or 0.5))
        respect = _clip01(float(relationship_state.get("respect", 0.5) or 0.5))
        warmth = _clip01(float(relationship_state.get("warmth", 0.5) or 0.5))
        power = _clip01(float(relationship_state.get("power", 0.5) or 0.5))
        
        # 2) PAD（-1..1 转为 0..1，缺省 0.5）
        P_raw = mood_state.get("pleasure", 0.0) or 0.0
        A_raw = mood_state.get("arousal", 0.0) or 0.0
        D_raw = mood_state.get("dominance", 0.0) or 0.0
        
        # 将 -1..1 映射到 0..1
        P = _clip01((P_raw + 1.0) / 2.0) if P_raw < 0 or P_raw > 1 else _clip01(P_raw)
        A = _clip01((A_raw + 1.0) / 2.0) if A_raw < 0 or A_raw > 1 else _clip01(A_raw)
        D = _clip01((D_raw + 1.0) / 2.0) if D_raw < 0 or D_raw > 1 else _clip01(D_raw)
        
        if P == 0.0 and A == 0.0 and D == 0.0:
            P = A = D = 0.5  # 缺省
        
        # 3) busy（缺省 0）
        busy = _clip01(mood_state.get("busyness", 0.0) or 0.0)
        
        # 4) detection_signals
        composite = detection_signals.get("composite") or {}
        trace = detection_signals.get("trace") or {}
        instant_eff = detection_signals.get("instant_eff") or {}
        stage_ctx = detection_signals.get("stage_ctx") or {}
        
        pos = _safe_get(composite, "goodwill", default=0.0)
        neg = _safe_get(composite, "conflict_eff", default=0.0)
        prov = _safe_get(composite, "provocation", default=0.0)
        press = _safe_get(composite, "pressure", default=0.0)
        uncert = _safe_get(trace, "confusion", default=0.0)
        if uncert == 0.0:
            uncert = _safe_get(instant_eff, "confusion", default=0.0)
        
        # 5) stage_ctx（缺的当 0）
        too_close_too_fast = _safe_get(stage_ctx, "too_close_too_fast", default=0.0)
        too_distant_too_cold = _safe_get(stage_ctx, "too_distant_too_cold", default=0.0)
        betrayal_violation = _safe_get(stage_ctx, "betrayal_violation", default=0.0)
        over_caring = _safe_get(stage_ctx, "over_caring", default=0.0)
        dependency_bid = _safe_get(stage_ctx, "dependency_bid", default=0.0)
        possessiveness_jealousy = _safe_get(stage_ctx, "possessiveness_jealousy", default=0.0)
        power_move = _safe_get(stage_ctx, "power_move", default=0.0)
        stonewalling_intent = _safe_get(stage_ctx, "stonewalling_intent", default=0.0)
        
        # 6) Knapp stage（缺省 1）
        stage_index = _get_stage_index(current_stage)
        
        # C. 派生量计算
        # 1) 关系底色轴
        Aff = _clip01(0.55 * liking + 0.25 * warmth + 0.20 * closeness)  # 亲和
        Saf = _clip01(0.50 * trust + 0.35 * respect + 0.15 * closeness)  # 安全感
        PowC = power - 0.50  # -0.5..+0.5
        
        # 2) Knapp stage baseline
        stage_profile = STAGE_PROFILE.get(stage_index, STAGE_PROFILE[1])
        invest = stage_profile["invest"]
        ctx = stage_profile["ctx"]
        break_n = 0.0 if stage_index <= 5 else (stage_index - 5) / 5.0  # 6..10 -> 0.2..1.0
        
        # 3) stage_ctx 合成
        BoundaryNeed = _clip01(
            0.45 * betrayal_violation +
            0.35 * power_move +
            0.25 * stonewalling_intent +
            0.20 * too_distant_too_cold +
            0.20 * possessiveness_jealousy +
            0.15 * over_caring
        )
        
        Unease = _clip01(
            0.35 * too_close_too_fast +
            0.25 * dependency_bid +
            0.25 * over_caring +
            0.20 * possessiveness_jealousy +
            0.15 * power_move
        )
        
        # D. 12维 style 公式
        # 1) self_disclosure（自我暴露）
        self_disclosure = _clip01(
            0.10
            + 0.55 * _avg2(trust, closeness)
            - 0.25 * A
            + 0.15 * pos
            + 0.10 * invest
        )
        
        # 2) topic_adherence（话题粘性）
        topic_adherence = _clip01(
            0.20
            + 0.70 * respect
            - 0.25 * uncert
            - 0.15 * prov
            + 0.08 * (1 - ctx)  # 默契越低越守规矩
        )
        
        # 3) initiative（主动权）
        initiative = _clip01(
            0.15
            + 0.45 * power
            + 0.35 * liking
            - 0.35 * busy
            + 0.10 * neg
            + 0.08 * invest
        )
        
        # 4) advice_style（建议风格）
        advice_style = _clip01(
            0.10
            + 0.45 * power
            + 0.35 * liking
            + 0.20 * BoundaryNeed
            - 0.20 * busy
            + 0.06 * invest
        )
        
        # 5) subjectivity（主观性/立场强）
        subjectivity = _clip01(
            0.35
            + 0.55 * power
            - 0.45 * respect
            + 0.30 * D
            + 0.25 * BoundaryNeed
            + 0.10 * break_n
        )
        
        # 6) memory_hook（记忆回扣）
        memory_hook = _clip01(
            0.05
            + 0.80 * closeness
            + 0.10 * pos
            - 0.25 * busy
            + 0.15 * ctx
        )
        
        # 7) verbal_length（篇幅）
        verbal_length = _clip01(
            0.20
            + 0.45 * _avg2(warmth, closeness)
            - 0.55 * busy
            - 0.20 * neg
            - 0.15 * BoundaryNeed
            + 0.10 * invest
            - 0.20 * break_n
        )
        
        # 8) social_distance（社交距离）- 需要先计算，因为 emotional_display 依赖它
        social_distance = _clip01(
            0.40
            + 0.40 * power
            + 0.25 * respect
            - 0.55 * closeness
            + 0.20 * neg
            + 0.25 * BoundaryNeed
            - 0.20 * ctx
            + 0.30 * break_n
        )
        
        # 9) tone_temperature（情感温度）
        tone_temperature = _clip01(
            0.20
            + 0.45 * _avg2(warmth, liking)
            + 0.25 * P
            - 0.25 * neg
            - 0.20 * Unease
            + 0.10 * invest
            - 0.25 * break_n
        )
        
        # 10) emotional_display（情绪显露）- 依赖 social_distance，所以放在后面
        emotional_display = _clip01(
            0.10
            + 0.45 * _avg2(trust, closeness)
            + 0.35 * A
            - 0.25 * social_distance
            + 0.08 * invest
            - 0.12 * break_n
        )
        
        # 11) wit_and_humor（幽默机趣）
        wit_and_humor = _clip01(
            0.05
            + 0.45 * _avg2(liking, closeness)
            + 0.20 * pos
            - 0.30 * BoundaryNeed
            - 0.20 * neg
            + 0.15 * ctx
            - 0.25 * break_n
        )
        
        # 12) non_verbal_cues（动作/表情包倾向）
        non_verbal_cues = _clip01(
            0.05
            + 0.65 * closeness
            + 0.10 * pos
            - 0.45 * busy
            - 0.20 * social_distance
            + 0.10 * ctx
            - 0.15 * break_n
        )
        
        # E. 两个门控变量
        # coldness_gate（冷淡/敷衍/嗯啊哦/不回）
        coldness_gate = _clip01(
            0.10
            + 0.45 * stonewalling_intent
            + 0.25 * too_distant_too_cold
            + 0.25 * neg
            + 0.25 * busy
            - 0.20 * _avg2(closeness, warmth)
            + 0.35 * break_n
            - 0.10 * ctx
        )
        
        # boundary_gate（设边界/强硬/回怼）
        boundary_gate = _clip01(
            0.10
            + 0.45 * betrayal_violation
            + 0.25 * power_move
            + 0.20 * press
            + 0.15 * prov
            + 0.10 * D
            - 0.20 * Saf
            + 0.20 * break_n
            + 0.10 * invest
        )
        
        # F. 输出写回 state
        # 应用 mode.style_bias（如果有）
        mode = state.get("current_mode")
        if mode and hasattr(mode, "style_bias"):
            bias = mode.style_bias
            if hasattr(bias, "verbal_length") and bias.verbal_length is not None:
                verbal_length = _clip01(0.7 * verbal_length + 0.3 * bias.verbal_length)
            if hasattr(bias, "tone_temperature") and bias.tone_temperature is not None:
                tone_temperature = _clip01(0.7 * tone_temperature + 0.3 * bias.tone_temperature)
            if hasattr(bias, "social_distance") and bias.social_distance is not None:
                social_distance = _clip01(0.7 * social_distance + 0.3 * bias.social_distance)
            if hasattr(bias, "advice_style") and bias.advice_style is not None:
                advice_style = _clip01(0.7 * advice_style + 0.3 * bias.advice_style)
            if hasattr(bias, "wit_and_humor") and bias.wit_and_humor is not None:
                wit_and_humor = _clip01(0.7 * wit_and_humor + 0.3 * bias.wit_and_humor)
            if hasattr(bias, "emotional_display") and bias.emotional_display is not None:
                emotional_display = _clip01(0.7 * emotional_display + 0.3 * bias.emotional_display)
        
        style_output = {
            # 12维
            "self_disclosure": self_disclosure,
            "topic_adherence": topic_adherence,
            "initiative": initiative,
            "advice_style": advice_style,
            "subjectivity": subjectivity,
            "memory_hook": memory_hook,
            "verbal_length": verbal_length,
            "social_distance": social_distance,
            "tone_temperature": tone_temperature,
            "emotional_display": emotional_display,
            "wit_and_humor": wit_and_humor,
            "non_verbal_cues": non_verbal_cues,
            # 两个门控
            "coldness_gate": coldness_gate,
            "boundary_gate": boundary_gate,
            # 可选：调试派生量
            "derived": {
                "Aff": Aff,
                "Saf": Saf,
                "BoundaryNeed": BoundaryNeed,
                "Unease": Unease,
                "invest": invest,
                "ctx": ctx,
                "break_n": break_n,
            },
        }
        
        # 兼容旧字段名（llm_instructions）
        llm_instructions = {
            "self_disclosure": self_disclosure,
            "topic_adherence": topic_adherence,
            "initiative": initiative,
            "advice_style": advice_style,
            "subjectivity": subjectivity,
            "memory_hook": memory_hook,
            "verbal_length": verbal_length,
            "social_distance": social_distance,
            "tone_temperature": tone_temperature,
            "emotional_display": emotional_display,
            "wit_and_humor": wit_and_humor,
            "non_verbal_cues": non_verbal_cues,
        }
        
        print(
            f"[Style] 12D computed: disclosure={self_disclosure:.2f}, "
            f"length={verbal_length:.2f}, temp={tone_temperature:.2f}, "
            f"gates: cold={coldness_gate:.2f}, boundary={boundary_gate:.2f}"
        )
        
        return {
            "style": style_output,
            "llm_instructions": llm_instructions,  # 兼容旧代码
            "style_analysis": f"12D computed (cold_gate={coldness_gate:.2f}, boundary_gate={boundary_gate:.2f})",
        }
    
    return style_node

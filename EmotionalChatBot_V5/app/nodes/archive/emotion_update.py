"""情绪更新节点：根据 detection_signals、relationship_state、stage_ctx 和当前 mood_state 更新 PAD 情绪。"""
from __future__ import annotations

from typing import Any, Callable, Dict

from utils.tracing import trace_if_enabled
from app.state import AgentState


def _clip01(x: float) -> float:
    """将值限制在 [0.0, 1.0] 范围内。"""
    return max(0.0, min(1.0, x))


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


def create_emotion_update_node() -> Callable[[AgentState], dict]:
    """创建情绪更新节点：根据信号、关系和阶段语境更新 PAD 情绪。"""

    @trace_if_enabled(
        name="Emotion Update",
        run_type="chain",
        tags=["node", "emotion", "mood"],
        metadata={"state_outputs": ["mood_state"]},
    )
    def emotion_update_node(state: AgentState) -> dict:
        """
        根据 detection_signals、relationship_state、stage_ctx 更新 PAD 情绪。
        
        逻辑：
        1. 从 detection_signals 提取 composite、trace、instant_eff
        2. 从 relationship_state 提取 6 维关系（转换为 0-1）
        3. 从 detection_signals.stage_ctx 提取阶段语境信号
        4. 计算亲和度、安全感、权力倾向
        5. 计算阶段语境量（BoundaryNeed、Unease）
        6. 计算 P/A/D 目标值
        7. 用系数回归更新当前 PAD
        """
        # 1) 输入提取
        detection_signals = state.get("detection_signals") or {}
        relationship_state = state.get("relationship_state") or {}
        mood_state = state.get("mood_state") or {}
        
        # 从 detection_signals 提取
        composite = detection_signals.get("composite") or {}
        trace = detection_signals.get("trace") or {}
        instant_eff = detection_signals.get("instant_eff") or {}
        stage_ctx = detection_signals.get("stage_ctx") or {}
        
        pos = _safe_get(composite, "goodwill", default=0.0)
        neg = _safe_get(composite, "conflict_eff", default=0.0)
        pressure = _safe_get(composite, "pressure", default=0.0)
        provocation = _safe_get(composite, "provocation", default=0.0)
        uncert = _safe_get(trace, "confusion", default=0.0)
        if uncert == 0.0:
            uncert = _safe_get(instant_eff, "confusion", default=0.0)
        
        # 从 relationship_state 提取 6 维（系统内部统一为 0-1）
        Clo = _clip01(float(relationship_state.get("closeness", 0.5) or 0.5))
        Tru = _clip01(float(relationship_state.get("trust", 0.5) or 0.5))
        Lik = _clip01(float(relationship_state.get("liking", 0.5) or 0.5))
        Res = _clip01(float(relationship_state.get("respect", 0.5) or 0.5))
        War = _clip01(float(relationship_state.get("warmth", 0.5) or 0.5))
        Pow = _clip01(float(relationship_state.get("power", 0.5) or 0.5))
        
        # 从 stage_ctx 提取（0-1）
        too_close_too_fast = _safe_get(stage_ctx, "too_close_too_fast", default=0.0)
        too_distant_too_cold = _safe_get(stage_ctx, "too_distant_too_cold", default=0.0)
        betrayal_violation = _safe_get(stage_ctx, "betrayal_violation", default=0.0)
        over_caring = _safe_get(stage_ctx, "over_caring", default=0.0)
        dependency_bid = _safe_get(stage_ctx, "dependency_bid", default=0.0)
        possessiveness_jealousy = _safe_get(stage_ctx, "possessiveness_jealousy", default=0.0)
        power_move = _safe_get(stage_ctx, "power_move", default=0.0)
        stonewalling_intent = _safe_get(stage_ctx, "stonewalling_intent", default=0.0)
        
        # 可选：busy（从 mood_state 或 stage_ctx 取）
        busy = mood_state.get("busyness", 0.0) or 0.0
        if busy == 0.0:
            busy = _safe_get(stage_ctx, "busy", default=0.0)
        busy = _clip01(busy)
        
        # 2) 6 维关系压成两个"底色轴"
        # 亲和（更容易开心、更宽容）
        Aff = 0.55 * Lik + 0.25 * War + 0.20 * Clo
        
        # 安全感（不紧绷、能修复、不会随时爆）
        Saf = 0.50 * Tru + 0.35 * Res + 0.15 * Clo
        
        # 权力倾向（更主导/更压制）
        PowC = Pow - 0.50  # -0.5..+0.5，但后续会 clip01，所以这里保持原值
        
        # 3) stage_ctx 组合成两个"阶段语境量"
        # BoundaryNeed（需要立场/边界/强硬）→ 主要喂给 D
        BoundaryNeed = (
            0.45 * betrayal_violation +
            0.35 * power_move +
            0.25 * stonewalling_intent +
            0.20 * too_distant_too_cold +
            0.20 * possessiveness_jealousy +
            0.15 * over_caring
        )
        BoundaryNeed = _clip01(BoundaryNeed)
        
        # Unease（不适/尴尬/紧绷）→ 主要喂给 A
        Unease = (
            0.35 * too_close_too_fast +
            0.25 * dependency_bid +
            0.25 * over_caring +
            0.20 * possessiveness_jealousy +
            0.15 * power_move
        )
        Unease = _clip01(Unease)
        
        # 4) 计算每个维度的 target
        # 4.1 P_target（愉悦）
        P_target = _clip01(
            0.45
            + 0.35 * Aff
            + 0.25 * pos * (0.6 + 0.8 * Aff)
            - 0.45 * neg * (0.6 + 0.8 * (1 - Lik))
            - 0.15 * BoundaryNeed
            - 0.10 * Unease
            - 0.10 * busy
        )
        
        # 4.2 A_target（唤醒/紧绷）
        A_target = _clip01(
            0.35
            + 0.45 * (1 - Saf)
            + 0.20 * pressure
            + 0.15 * provocation
            + 0.15 * uncert
            + 0.20 * Unease
            + 0.10 * neg
            - 0.10 * pos
            + 0.10 * busy
        )
        
        # 4.3 D_target（支配/强硬）
        # PowC 是 -0.5..+0.5，需要映射到 0-1 范围用于计算
        PowC_normalized = _clip01(PowC + 0.5)  # 将 -0.5..+0.5 映射到 0..1
        D_target = _clip01(
            0.50
            + 0.60 * PowC_normalized
            + 0.35 * BoundaryNeed
            + 0.20 * neg
            + 0.10 * provocation
            - 0.20 * pos * (0.6 + 0.4 * Saf)
            - 0.10 * Aff
        )
        
        # 5) 更新方式：系数回归
        # 当前 PAD（从 mood_state 取，MoodState 定义是 -1..1 范围，映射到 0..1 进行计算）
        P_curr_raw = mood_state.get("pleasure", 0.0) or 0.0
        A_curr_raw = mood_state.get("arousal", 0.0) or 0.0
        D_curr_raw = mood_state.get("dominance", 0.0) or 0.0
        
        # 将 -1..1 范围映射到 0..1（用于计算）
        def _pad_to_01(x: float) -> float:
            """将 PAD 值从 -1..1 映射到 0..1。"""
            return _clip01((x + 1.0) / 2.0)
        
        def _01_to_pad(x: float) -> float:
            """将 0..1 值映射回 -1..1 范围。"""
            return max(-1.0, min(1.0, x * 2.0 - 1.0))
        
        P_curr = _pad_to_01(P_curr_raw)
        A_curr = _pad_to_01(A_curr_raw)
        D_curr = _pad_to_01(D_curr_raw)
        
        # 基础系数
        βP = 0.18
        βA = 0.12
        βD = 0.15
        
        # 可选：冲突越强，反应越快（拟人化"当场上头"）
        k = _clip01(0.6 + 0.6 * neg + 0.3 * BoundaryNeed)
        βP *= k
        βA *= k
        βD *= k
        
        # 更新 PAD（在 0..1 范围内计算）
        P_new_01 = _clip01(P_curr + βP * (P_target - P_curr))
        A_new_01 = _clip01(A_curr + βA * (A_target - A_curr))
        D_new_01 = _clip01(D_curr + βD * (D_target - D_curr))
        
        # 映射回 -1..1 范围（MoodState 期望的范围）
        P_new = _01_to_pad(P_new_01)
        A_new = _01_to_pad(A_new_01)
        D_new = _01_to_pad(D_new_01)
        
        # 更新 busyness（可选：根据压力/混乱等调整）
        busy_new = _clip01(busy + 0.1 * (pressure + uncert - pos * 0.5))
        
        # 返回更新后的 mood_state（PAD 在 -1..1 范围）
        new_mood_state = {
            "pleasure": P_new,
            "arousal": A_new,
            "dominance": D_new,
            "busyness": busy_new,
        }
        
        print(
            f"[Emotion Update] P: {P_curr_raw:.2f}→{P_new:.2f} (target_01: {P_target:.2f}), "
            f"A: {A_curr_raw:.2f}→{A_new:.2f} (target_01: {A_target:.2f}), "
            f"D: {D_curr_raw:.2f}→{D_new:.2f} (target_01: {D_target:.2f}), "
            f"Aff={Aff:.2f}, Saf={Saf:.2f}, BoundaryNeed={BoundaryNeed:.2f}, Unease={Unease:.2f}"
        )
        
        return {
            "mood_state": new_mood_state,
        }
    
    return emotion_update_node

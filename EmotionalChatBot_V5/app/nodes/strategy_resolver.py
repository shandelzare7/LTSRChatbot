"""
策略仲裁节点：按 13 级固定优先级在三个路由结果中取当前模式；
若无命中则先按全参数公式计算 M_next，再据此选 momentum 五态之一并写回 conversation_momentum。
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict

from app.state import AgentState
from utils.tracing import trace_if_enabled
from utils.yaml_loader import get_strategy_by_id, load_momentum_formula_config

# 动量公式常量（从 config/momentum_formula.yaml 加载，缺省见 yaml_loader.load_momentum_formula_config）
_MOMENTUM_FORMULA_CONFIG = load_momentum_formula_config()
R_BASE_NEUTRAL = 5.0  # 关系维度缺失时的中性值 (0-10)
# R_base 权重：0.8*attractiveness + 0.1*liking + 0.1*closeness，关系维 0-1，再映射到 0-10
R_BASE_WEIGHTS = (0.8, 0.1, 0.1)  # (attractiveness, liking, closeness)


# 13 级仲裁顺序（从高到低，不可逾越）
STRATEGY_PRIORITY_13 = [
    "anti_ai_defense",      # 反AI，最高优底线
    "boundary_defense",     # 被骂必须先反击
    "yielding_apology",     # 自己做错事必须先滑跪
    "physical_limitation_refusal",  # 物理/现实限制必须拒绝（见面/照片/借钱等）
    "shit_test_counter",    # 送命题求生欲拉满
    "flirting_banter",      # 暧昧推拉/娇嗔（吸引力>0.6 时接招）
    "co_rumination",        # 用户极度愤怒/崩溃时不能刺激
    "reasonable_assistance", # 优先解决明确求助
    "busy_deferral",        # 无紧急情绪且 Bot 忙则开溜
    "tldr_refusal",        # 太长且无聊拒绝阅读
    "deflection",           # 闪避敏感话题
    "passive_aggression",   # 阴阳怪气/吃醋
    "clarification",        # 要求澄清
    "micro_reaction",       # 微反应敷衍
    "detail_nitpicking",    # 最低优：挑刺
]

# momentum 五态 id（当 13 个信号均未命中时，由计算决定使用哪一个）
MOMENTUM_IDS = [
    "momentum_terminate_neg_2",
    "momentum_converge_neg_1",
    "momentum_maintain_0",
    "momentum_extend_1",
    "momentum_lead_2",
]

# 当轮动量低于此值时：不输出任何回复，直接跳过到输出回复后的节点（evolver 等）
SKIP_REPLY_MOMENTUM_THRESHOLD = 0.1

# 驼峰拍扁：Knapp 1-10 阶段 -> 0-5 绝对亲密深度 (IDI)
# 深度5=极度亲密, 4=高度亲密, 3=中度亲密, 2=低度亲密, 1=零度/外壳, 0=切断
STAGE_TO_IDI = {
    1: 1,   # Initiating 初识
    2: 2,   # Experimenting 探索
    3: 3,   # Intensifying 强化
    4: 4,   # Integrating 整合
    5: 5,   # Bonding 结缔
    6: 4,   # Differentiating 分化
    7: 3,   # Circumscribing 限制
    8: 2,   # Stagnating 停滞
    9: 1,   # Avoiding 回避
    10: 0,  # Terminating 终止
}


def _check_boundary_and_pacing(state: Dict[str, Any], detection: Dict[str, Any]) -> str | None:
    """
    直接使用 Detection 的 stage_pacing（正常/过分亲密/过分生疏）决定是否越界及策略。
    - 正常：不触发，返回 None。
    - 过分亲密：吸引力高则 flirting_banter，否则 deflection。
    - 过分生疏：yielding_apology 退让安抚。
    """
    pacing = (detection.get("stage_pacing") or "正常").strip()
    if pacing == "正常":
        return None
    if pacing == "过分生疏":
        return "yielding_apology"
    if pacing == "过分亲密":
        rel = state.get("relationship_state") or {}
        profile = state.get("user_inferred_profile") or {}
        att_raw = rel.get("attractiveness", rel.get("warmth", profile.get("attractiveness", 0.5)))
        try:
            attractiveness = float(att_raw)
            if 0 <= attractiveness <= 1.0:
                attractiveness = attractiveness * 100.0
        except (TypeError, ValueError):
            attractiveness = 50.0
        attractiveness = max(0.0, min(100.0, attractiveness))
        return "flirting_banter" if attractiveness > 80 else "deflection"
    return None


def _compute_deferral_score(state: Dict[str, Any]) -> float:
    """
    Deferral_Score = (busy*1.0 + Power*0.5) - (Urgency*1.5 + topic_interest*0.8 + attractiveness*0.4)
    若 > 0.5 则公式触发 busy_deferral。量纲：busy/power/attractiveness 用 0-1，urgency/topic_appeal 用 0-10 后归一化到 0-1 参与计算。
    """
    mood = state.get("mood_state") or {}
    rel = state.get("relationship_state") or {}
    det = state.get("detection") or {}
    busy = _float_01(mood.get("busyness"), 0.0)
    power = _float_01(rel.get("power"), 0.5)
    urgency_raw = _float_0_10(det.get("urgency"), 5.0)
    topic_appeal_raw = _float_0_10(det.get("topic_appeal"), 5.0)
    attractiveness = _float_01(rel.get("attractiveness", rel.get("warmth", 0.5)))
    urgency01 = urgency_raw / 10.0
    topic_interest01 = topic_appeal_raw / 10.0
    deferral_score = (busy * 1.0 + power * 0.5) - (urgency01 * 1.5 + topic_interest01 * 0.8 + attractiveness * 0.4)
    return deferral_score


def _apply_momentum_modifier(m_current: float, modifier_str: str) -> float:
    """
    按策略的 momentum_modifier 更新动量。m_current 0~1，返回 clamp 到 [0,1]。
    - sub:X → m - X
    - add:X → m + X
    - set:X → X
    - freeze → 不变
    - use_formula → 由调用方用公式结果，此处按 freeze 处理
    """
    raw = (modifier_str or "").strip().lower()
    if not raw or raw == "freeze" or raw == "use_formula":
        return max(0.0, min(1.0, m_current))
    if raw.startswith("sub:"):
        try:
            x = float(raw.split(":", 1)[1].strip())
            return max(0.0, min(1.0, m_current - x))
        except (ValueError, IndexError):
            return max(0.0, min(1.0, m_current))
    if raw.startswith("add:"):
        try:
            x = float(raw.split(":", 1)[1].strip())
            return max(0.0, min(1.0, m_current + x))
        except (ValueError, IndexError):
            return max(0.0, min(1.0, m_current))
    if raw.startswith("set:"):
        try:
            x = float(raw.split(":", 1)[1].strip())
            return max(0.0, min(1.0, x))
        except (ValueError, IndexError):
            return max(0.0, min(1.0, m_current))
    return max(0.0, min(1.0, m_current))


def _resolve_from_three_routers(
    router_high_stakes: str | None,
    router_emotional_game: str | None,
    router_form_rhythm: str | None,
    state: Dict[str, Any] | None = None,
    m_current: float | None = None,
) -> str | None:
    """
    从三路路由结果中按 13 级优先级选出唯一策略 id。
    若传入 state，则按公式计算 Deferral_Score；若 > 0.5 则将 busy_deferral 加入候选；Stage 越界会加入候选。
    若传入 m_current，则从候选中剔除 min_momentum > m_current 的策略（不满足条件走常态）。
    若三路都为空且未公式触发，返回 None。
    """
    candidates: list[str] = []
    for sid in (router_high_stakes, router_emotional_game, router_form_rhythm):
        sid = (sid or "").strip()
        if sid and sid in STRATEGY_PRIORITY_13:
            candidates.append(sid)
    if state is not None:
        deferral_score = _compute_deferral_score(state)
        if deferral_score > 0.5:
            candidates.append("busy_deferral")
            print(f"[StrategyResolver] Deferral_Score={deferral_score:.3f} > 0.5，公式触发 busy_deferral")
        # Stage 越界差值判定（驼峰拍扁 IDI）
        detection = state.get("detection") or {}
        boundary_id = _check_boundary_and_pacing(state, detection)
        if boundary_id and boundary_id in STRATEGY_PRIORITY_13:
            candidates.append(boundary_id)
            print(f"[StrategyResolver] Stage 越界判定命中: {boundary_id}")

    # 按 min_momentum 门控：当前动量不足则从候选中去掉，最终走常态
    if m_current is not None and candidates:
        filtered = []
        for sid in candidates:
            strat = get_strategy_by_id(sid)
            min_m = strat.get("min_momentum")
            if min_m is None:
                filtered.append(sid)
                continue
            try:
                min_m_f = float(min_m)
                if m_current >= min_m_f:
                    filtered.append(sid)
                else:
                    print(f"[StrategyResolver] 剔除 {sid}：当前动量 {m_current:.3f} < min_momentum {min_m_f:.3f}")
            except (TypeError, ValueError):
                filtered.append(sid)
        candidates = filtered

    if not candidates:
        return None
    return min(candidates, key=lambda x: STRATEGY_PRIORITY_13.index(x))


# momentum 值区间与五态对应（conversation_momentum 0.0~1.0）
# 与 strategies.yaml 中意愿分 <20 / 20-40 / 40-60 / 60-80 / >80 一一对应
_MOMENTUM_THRESHOLDS = [
    (0.0, "momentum_terminate_neg_2"),   # [0, 0.2) 阻断
    (0.2, "momentum_converge_neg_1"),    # [0.2, 0.4) 收敛
    (0.4, "momentum_maintain_0"),       # [0.4, 0.6) 维系
    (0.6, "momentum_extend_1"),          # [0.6, 0.8) 延展
    (0.8, "momentum_lead_2"),            # [0.8, 1.0] 主导
]


def _float_0_10(x: Any, default: float = 5.0) -> float:
    try:
        return max(0.0, min(10.0, float(x)))
    except (TypeError, ValueError):
        return default


def _float_01(x: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return default


def _arousal_clamped(x: Any) -> float:
    """PAD Arousal 归一化到配置的 range_min..range_max。若原为 0-1 则线性映射到该区间。"""
    ar = _MOMENTUM_FORMULA_CONFIG.get("arousal") or {}
    r_min = float(ar.get("range_min", -1.0))
    r_max = float(ar.get("range_max", 1.0))
    try:
        v = float(x)
        if 0.0 <= v <= 1.0:
            return r_min + (r_max - r_min) * v
        return max(r_min, min(r_max, v))
    except (TypeError, ValueError):
        return (r_min + r_max) * 0.5


def _compute_momentum_next(state: Dict[str, Any]) -> float:
    """
    全参数动量公式：计算本轮结束后的新动量 M_next (0~1)。
    - 步骤 1: E_turn = (E_user*0.3) + (T_bot*0.4) + (R_base*0.3) - (H_user*hostility_penalty_coef)
    - 步骤 2: Multiplier_arousal = 1.0 + (Arousal * coef)，Arousal ∈ [-1,1]
    - 步骤 3: Score_raw = [ (M_prev*100)*(1-α) + (E_turn*Multiplier_arousal)*α ] - Penalty_fatigue
    - 步骤 4: Ceiling = 100*(1 - Busyness*0.5), Score_final = min(Score_raw, Ceiling)，clamp 到 [0,100]，M_next = Score_final/100
    - E_turn 中 topic_appeal 权重由 e_turn_t_bot_weight 配置（默认 0.4，可调大以增强 interesting 奖励）
    """
    detection = state.get("detection") or {}
    mood = state.get("mood_state") or {}
    rel = state.get("relationship_state") or {}

    m_prev = _float_01(state.get("conversation_momentum"), 0.5)
    e_user = _float_0_10(detection.get("engagement_level"), 5.0)
    t_bot = _float_0_10(detection.get("topic_appeal"), 5.0)
    # R_base = 0.8*attractiveness + 0.1*liking + 0.1*closeness（0-1），再映射到 0-10
    att = _float_01(rel.get("attractiveness", rel.get("warmth", 0.5)))
    lik = _float_01(rel.get("liking", 0.5))
    clo = _float_01(rel.get("closeness", 0.5))
    r_base_01 = R_BASE_WEIGHTS[0] * att + R_BASE_WEIGHTS[1] * lik + R_BASE_WEIGHTS[2] * clo
    r_base = _float_0_10(r_base_01 * 10.0, R_BASE_NEUTRAL)
    h_user = _float_0_10(detection.get("hostility_level"), 0.0)
    # 敌意对冲量的惩罚系数（降低后检测到敌意不会过度打压动量）
    hostility_penalty_coef = float(_MOMENTUM_FORMULA_CONFIG.get("hostility_penalty_coef", 0.75))
    # E_turn 里 topic_appeal（interesting）的权重，提高即增加「有意思」对动量的奖励
    weight_t_bot = float(_MOMENTUM_FORMULA_CONFIG.get("e_turn_t_bot_weight", 0.4))

    e_turn = (e_user * 0.3) + (t_bot * weight_t_bot) + (r_base * 0.3) - (h_user * hostility_penalty_coef)

    arousal = _arousal_clamped(mood.get("arousal"))
    ar_coef = float((_MOMENTUM_FORMULA_CONFIG.get("arousal") or {}).get("multiplier_coef", 0.5))
    multiplier_arousal = 1.0 + (arousal * ar_coef)

    # 轮次 = 本会话内已完成的回复轮数（会话初 0，每产生一次 bot 回复 +1，由持久化时写入；不按消息条数算）
    turn_count = int(state.get("turn_count_in_session") or 0)
    penalty_fatigue = max(0.0, (turn_count - 15) * 2.0)

    ema_alpha = float(_MOMENTUM_FORMULA_CONFIG.get("ema_alpha", 0.3))
    score_raw = (
        (m_prev * 100.0) * (1.0 - ema_alpha)
        + (e_turn * multiplier_arousal) * ema_alpha
    ) - penalty_fatigue

    busyness = _float_01(mood.get("busyness"), 0.0)
    ceiling = 100.0 * (1.0 - busyness * 0.5)
    score_final = min(score_raw, ceiling)
    score_final = max(0.0, min(100.0, score_final))

    return score_final / 100.0


def _compute_momentum_strategy(state: Dict[str, Any]) -> str | None:
    """
    当 13 个信号均未命中时，根据当前 conversation_momentum（0.0~1.0）映射到五态之一。
    """
    raw = state.get("conversation_momentum")
    try:
        m = float(raw)
    except (TypeError, ValueError):
        m = 0.5
    m = max(0.0, min(1.0, m))

    for threshold, strategy_id in reversed(_MOMENTUM_THRESHOLDS):
        if m >= threshold:
            return strategy_id
    return MOMENTUM_IDS[0]


def create_strategy_resolver_node() -> Callable[[AgentState], dict]:
    """
    策略仲裁节点：当轮 momentum 前置计算；若动量过低则设 skip_reply 不输出回复。
    按 13 级优先级从三路路由取候选，再按 min_momentum 过滤；不满足者走常态 momentum 五态。
    若走特殊信号则按该策略的 momentum_modifier 更新 conversation_momentum；否则用公式结果。
    写入 current_strategy_id / current_strategy / conversation_momentum / skip_reply。
    """

    @trace_if_enabled(
        name="StrategyResolver",
        run_type="chain",
        tags=["node", "strategy_resolver"],
        metadata={"state_outputs": ["current_strategy_id", "current_strategy", "conversation_momentum", "skip_reply", "force_fast_route"]},
    )
    def node(state: AgentState) -> dict:
        state_dict = dict(state)
        # 1）前置计算当轮动量：M_prev 为当前状态，M_next 为公式结果（用于常态与 skip 判断）
        m_prev = _float_01(state.get("conversation_momentum"), 0.5)
        m_next = _compute_momentum_next(state_dict)
        skip_reply = m_next <= SKIP_REPLY_MOMENTUM_THRESHOLD
        if skip_reply:
            print(f"[StrategyResolver] 当轮动量 M_next={m_next:.3f} <= {SKIP_REPLY_MOMENTUM_THRESHOLD}，跳过回复 (skip_reply=True)")

        # 可选：前两条回复总用时 < 阈值则走 fast（全局开关 FAST_ROUTE_WHEN_QUICK_REPLY_ENABLED 控制）
        force_fast_route = False
        fast_quick_reply_enabled = str(os.getenv("FAST_ROUTE_WHEN_QUICK_REPLY_ENABLED", "0")).lower() in ("1", "true", "yes", "on")
        if fast_quick_reply_enabled:
            try:
                threshold_sec = float(os.getenv("FAST_ROUTE_QUICK_REPLY_THRESHOLD_SEC", "60"))
            except (TypeError, ValueError):
                threshold_sec = 60.0
            dur_list = state.get("reply_duration_seconds_list") or []
            if isinstance(dur_list, list) and len(dur_list) >= 2:
                first_two = [float(x) for x in dur_list[:2] if isinstance(x, (int, float))]
                if len(first_two) >= 2 and sum(first_two) < threshold_sec:
                    force_fast_route = True
                    print(f"[StrategyResolver] 前两条回复总用时 {sum(first_two):.1f}s < {threshold_sec}s，force_fast_route=True")

        hid = (state.get("router_high_stakes") or "").strip() or None
        eid = (state.get("router_emotional_game") or "").strip() or None
        fid = (state.get("router_form_rhythm") or "").strip() or None
        print(f"[StrategyResolver] 三路路由 raw: high_stakes={hid!r}, emotional_game={eid!r}, form_rhythm={fid!r} | M_prev={m_prev:.3f} M_next(公式)={m_next:.3f}")
        # 2）按 min_momentum 过滤后的候选里按优先级选一个
        chosen_id = _resolve_from_three_routers(hid, eid, fid, state_dict, m_current=m_prev)

        if not chosen_id:
            # 3）无特殊信号：用公式 M_next 选 momentum 五态
            state_with_m = {**state_dict, "conversation_momentum": m_next}
            chosen_id = _compute_momentum_strategy(state_with_m)
            if not chosen_id:
                print("[StrategyResolver] 13 级均未命中且 momentum 未算出，current_strategy 置空")
                return {"current_strategy_id": None, "current_strategy": None, "conversation_momentum": m_next, "skip_reply": skip_reply, "force_fast_route": force_fast_route}
            print(f"[StrategyResolver] 走常态动量，M_next={m_next:.3f}，选用: {chosen_id}")
            new_momentum = m_next
        else:
            # 4）走特殊信号：按该策略的 momentum_modifier 更新动量
            strategy = get_strategy_by_id(chosen_id)
            mod = (strategy or {}).get("momentum_modifier") or "freeze"
            if chosen_id in MOMENTUM_IDS:
                new_momentum = m_next  # 常态五态仍用公式结果
            else:
                new_momentum = _apply_momentum_modifier(m_prev, mod)
                print(f"[StrategyResolver] 特殊信号 {chosen_id}，modifier={mod}，M: {m_prev:.3f} -> {new_momentum:.3f}")

        strategy = get_strategy_by_id(chosen_id)
        if not strategy:
            print(f"[StrategyResolver] 未找到策略 id={chosen_id}，置空")
            return {"current_strategy_id": None, "current_strategy": None, "conversation_momentum": new_momentum, "skip_reply": skip_reply, "force_fast_route": force_fast_route}
        print(f"[StrategyResolver] 选中策略: {chosen_id}")
        return {
            "current_strategy_id": chosen_id,
            "current_strategy": strategy,
            "conversation_momentum": new_momentum,
            "skip_reply": skip_reply,
            "force_fast_route": force_fast_route,
        }
    return node

"""
Detection 节点：对「当前用户最新一句」打 instant 分，用历史状态做衰减累积（trace），
计算可爆表强度（heat + streak），并输出组合指标（composite）。
全文会话仅作语境，不重复计数；trace/heat/streak 在 state 中持久化。
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from utils.tracing import trace_if_enabled
from utils.detailed_logging import log_prompt_and_params, log_llm_response
from utils.llm_json import parse_json_from_llm
from utils.prompt_helpers import format_relationship_for_llm, format_stage_for_llm

from app.state import AgentState

# 12 信号：负向 6 + 正向 6
SIGNALS_NEG = ["sarcasm", "contempt", "toxicity", "threat_pressure", "low_effort", "confusion"]
SIGNALS_POS = ["appreciation", "cooperation", "repair", "warm_humor", "self_disclosure", "validation"]
SIGNALS: List[str] = SIGNALS_NEG + SIGNALS_POS

# 阶段语境信号（0~1）：与当前关系阶段是否相符/越界，供约束关系模式
STAGE_CTX_KEYS: List[str] = [
    "too_close_too_fast",
    "too_distant_too_cold",
    "betrayal_violation",
    "over_caring",
    "dependency_bid",
    "possessiveness_jealousy",
    "power_move",
    "stonewalling_intent",
]

# 固定衰减参数（双时间尺度 + heat）
GAMMA_FAST = 0.10
GAMMA_SLOW = 0.70
MIX = 2 / 3  # ≈ 0.667
ALPHA_HEAT = 0.55
STREAK_THRESHOLD = 0.6
AMP_MAX_STREAK = 5
AMP_FACTOR = 0.35

# 组合权重
CONFLICT_TOXICITY = 0.55
CONFLICT_PROVOCATION = 0.25
CONFLICT_PRESSURE = 0.20
PRESSURE_TOXICITY_FACTOR = 0.6
GOODWILL_OFFSET = 0.70


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _f_instant(raw: Any) -> float:
    try:
        v = float(raw)
        return _clip01(v)
    except (TypeError, ValueError):
        return 0.0


def create_detection_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """创建信号检测节点：LLM 只对最新用户一句打 instant，本地计算 trace/heat/streak/composite。"""

    @trace_if_enabled(
        name="Perception/Detection",
        run_type="chain",
        tags=["node", "perception", "detection"],
        metadata={"state_outputs": ["detection_signals", "detection_category", "detection_result"]},
    )
    def detection_node(state: AgentState) -> dict:
        chat_buffer: List[BaseMessage] = list(state.get("chat_buffer") or state.get("messages", [])[-30:])
        prev = state.get("detection_signals") or {}
        trace_fast_prev = prev.get("trace_fast") or {s: 0.0 for s in SIGNALS}
        trace_slow_prev = prev.get("trace_slow") or {s: 0.0 for s in SIGNALS}
        heat_prev = prev.get("heat_recur") or prev.get("heat") or {s: 0.0 for s in SIGNALS}
        streak_prev = prev.get("streak") or {s: 0 for s in SIGNALS}

        # 无对话时返回零信号并走 normal
        if not chat_buffer:
            empty_instant = {s: 0.0 for s in SIGNALS}
            empty_trace = {s: 0.0 for s in SIGNALS}
            empty_heat = {s: 0.0 for s in SIGNALS}
            empty_streak = {s: 0 for s in SIGNALS}
            composite = {
                "provocation": 0.0,
                "conflict": 0.0,
                "goodwill": 0.0,
                "pressure": 0.0,
                "conflict_eff": 0.0,
            }
            params = {"gamma_fast": GAMMA_FAST, "gamma_slow": GAMMA_SLOW, "mix": MIX}
            empty_meta = {"target_is_assistant": 1, "playful_not_hostile": 0, "quoted_or_reported_speech": 0}
            empty_stage_ctx = {k: 0.0 for k in STAGE_CTX_KEYS}
            empty_quality = {"confidence": 0.0}
            out = {
                "instant": empty_instant,
                "meta": empty_meta,
                "stage_ctx": empty_stage_ctx,
                "quality": empty_quality,
                "instant_eff": dict(empty_instant),
                "trace": empty_trace,
                "heat": empty_heat,
                "streak": empty_streak,
                "composite": composite,
                "params": params,
                "trace_fast": trace_fast_prev,
                "trace_slow": trace_slow_prev,
                "heat_recur": empty_heat,
            }
            return {
                "detection_signals": out,
                "detection_category": "NORMAL",
                "detection_result": "NORMAL",
            }

        # 最新一条必须是用户消息；否则用 state.user_input 或最后一条内容
        def _is_user(m: BaseMessage) -> bool:
            t = getattr(m, "type", "") or ""
            return "human" in t.lower() or "user" in t.lower()
        last_msg = chat_buffer[-1]
        latest_user_text = (
            (state.get("user_input") or "").strip()
            or (getattr(last_msg, "content", "") or str(last_msg)).strip()
        )
        if not _is_user(last_msg):
            latest_user_text = latest_user_text or "(无用户新句)"

        # 转为 LC 消息列表供 LLM 区分角色
        def _to_lc(m: BaseMessage) -> BaseMessage:
            if isinstance(m, (HumanMessage, AIMessage)):
                return m
            c = getattr(m, "content", str(m))
            return HumanMessage(content=c) if _is_user(m) else AIMessage(content=c)
        conv_messages = [_to_lc(m) for m in chat_buffer]

        # 6 维关系状态（0–1 客观语义），供判读语境时参考
        relationship_state = state.get("relationship_state") or {}
        rel_for_llm = format_relationship_for_llm(relationship_state)
        stage_id = str(state.get("current_stage") or "initiating")
        stage_desc = format_stage_for_llm(stage_id)

        system_content = f"""你是对话系统的内部「语境判读器」。你会阅读当前会话全部内容以理解语境，但你必须只对「最新用户消息」输出 12 个信号强度 instant（0~1）。
历史内容仅用于判断语境（例如是否反讽、是否修复、是否友好玩笑），不得将历史中出现的攻击/道歉/敷衍重复计入 instant。

你输出的 instant 只反映「最新用户消息」在当前语境下的表现。历史只用于理解语境，不得把旧内容重复计入 instant。

# 当前关系状态（0–1 客观语义，供判读语境时参考）
{rel_for_llm}

# 当前关系阶段（供判读「是否与当前关系模式相符」）
{stage_desc}

阶段越界判断提示：
- 初识/试探期（initiating, experimenting）：过度亲密、强绑定、强依赖易越界
- 升温/整合期（intensifying, integrating, bonding）：突然敷衍、撤退、拒绝配合易显冷淡
- 投入期出现轻蔑/辱骂/威胁等会带来「关系违背感」
- 判读时结合当前阶段，判断本条消息是否与阶段相符、是否越界或倒退

# stage_ctx 字段（0~1，与阶段相符/越界相关）
- too_close_too_fast：早期阶段越级亲密/绑定/强关怀/强依赖等
- too_distant_too_cold：中后期阶段突然敷衍撤退/拒绝配合等
- betrayal_violation：投入阶段出现轻蔑/辱骂/威胁等「关系违背感」
- over_caring：关怀带控制/监控/指挥味
- dependency_bid：索求陪伴/绑定
- possessiveness_jealousy：占有欲/吃醋/排他
- power_move：争夺主导、命令、裁决、压迫式提问
- stonewalling_intent：有意冷处理/冷战倾向（区别「忙/省略」）

# 信号集合（0~1）
负向：sarcasm, contempt, toxicity, threat_pressure, low_effort, confusion
正向：appreciation, cooperation, repair, warm_humor, self_disclosure, validation

强度标尺：0=无；0.3=隐约；0.6=明显；0.9=主导/强烈。

meta 字段（0 或 1）：
- target_is_assistant: 这条最新用户消息主要是在对助手说(1)还是在谈论第三方/自言自语(0)
- playful_not_hostile: 这条消息更像友好打趣/玩笑(1)而非敌意挑衅(0)
- quoted_or_reported_speech: 这条消息包含引用/转述第三方言论（尤其辱骂）(1)否则(0)

quality 字段：
- confidence: 模型自评是否把握住语境，0~1

输出必须是严格 JSON，且只包含 instant / meta / stage_ctx / quality 四个顶层键，不要输出任何额外文本。示例结构：

{{
  "instant": {{ "sarcasm": 0.0, "contempt": 0.0, "toxicity": 0.0, "threat_pressure": 0.0, "low_effort": 0.0, "confusion": 0.0, "appreciation": 0.0, "cooperation": 0.0, "repair": 0.0, "warm_humor": 0.0, "self_disclosure": 0.0, "validation": 0.0 }},
  "meta": {{ "target_is_assistant": 0, "playful_not_hostile": 0, "quoted_or_reported_speech": 0 }},
  "stage_ctx": {{ "too_close_too_fast": 0.0, "too_distant_too_cold": 0.0, "betrayal_violation": 0.0, "over_caring": 0.0, "dependency_bid": 0.0, "possessiveness_jealousy": 0.0, "power_move": 0.0, "stonewalling_intent": 0.0 }},
  "quality": {{ "confidence": 0.0 }}
}}"""

        task_msg = HumanMessage(content="请根据上面对话，仅对「最新用户消息」输出上述格式的 JSON。")
        messages_to_invoke: List[BaseMessage] = [SystemMessage(content=system_content), *conv_messages, task_msg]

        log_prompt_and_params("Detection", system_prompt=system_content[:1200], user_prompt="[对话+输出JSON]", params={"conv_len": len(conv_messages)})

        instant: Dict[str, float] = {s: 0.0 for s in SIGNALS}
        meta: Dict[str, int] = {"target_is_assistant": 1, "playful_not_hostile": 0, "quoted_or_reported_speech": 0}
        stage_ctx: Dict[str, float] = {k: 0.0 for k in STAGE_CTX_KEYS}
        quality: Dict[str, float] = {"confidence": 0.0}
        try:
            msg = llm_invoker.invoke(messages_to_invoke)
            content = (getattr(msg, "content", "") or str(msg)).strip()
            result = parse_json_from_llm(content)
            if isinstance(result, dict):
                inst = result.get("instant") or {}
                for s in SIGNALS:
                    instant[s] = _f_instant(inst.get(s))
                m = result.get("meta") or {}
                meta["target_is_assistant"] = 1 if (m.get("target_is_assistant") in (1, "1", True)) else 0
                meta["playful_not_hostile"] = 1 if (m.get("playful_not_hostile") in (1, "1", True)) else 0
                meta["quoted_or_reported_speech"] = 1 if (m.get("quoted_or_reported_speech") in (1, "1", True)) else 0
                sc = result.get("stage_ctx") or {}
                for k in STAGE_CTX_KEYS:
                    stage_ctx[k] = _f_instant(sc.get(k))
                q = result.get("quality") or {}
                quality["confidence"] = _clip01(q.get("confidence", 0))
                log_llm_response("Detection", msg, parsed_result={"instant_keys": list(instant.keys()), "meta": meta, "stage_ctx_keys": STAGE_CTX_KEYS})
        except Exception as e:
            print(f"[Detection] LLM 异常: {e}，instant/meta/stage_ctx/quality 用默认")

        # 轻量规则补丁：早期阶段对“秘密/只告诉我/别跟别人说/说实话”等亲密推进更敏感
        # 目的：避免 LLM stage_ctx 漏检导致策略越界（过快亲密/过真/过假）。
        try:
            stage_id = str(state.get("current_stage") or "initiating")
            user_text = str(user_input or "")
            rel = state.get("relationship_state") or {}
            trust = float(rel.get("trust", 0.0) or 0.0)
            closeness = float(rel.get("closeness", 0.0) or 0.0)
            early = stage_id in ("initiating", "experimenting")
            low_rel = (trust < 0.35) or (closeness < 0.35)
            secretish = any(p in user_text for p in ("小秘密", "秘密", "只告诉我", "别告诉别人", "跟别人说", "说实话", "坦诚点"))
            if early and low_rel and secretish:
                stage_ctx["too_close_too_fast"] = _clip01(max(stage_ctx.get("too_close_too_fast", 0.0), 0.28))
        except Exception:
            pass

        # 用 meta 调制得到 instant_eff（去误判），再进入 trace/heat 更新
        instant_eff = dict(instant)
        playful = meta.get("playful_not_hostile", 0) == 1
        not_target = meta.get("target_is_assistant", 1) == 0
        quoted = meta.get("quoted_or_reported_speech", 0) == 1

        if playful:
            instant_eff["sarcasm"] = instant_eff.get("sarcasm", 0) * 0.35
            instant_eff["contempt"] = instant_eff.get("contempt", 0) * 0.35
            instant_eff["toxicity"] = instant_eff.get("toxicity", 0) * 0.20
            instant_eff["threat_pressure"] = instant_eff.get("threat_pressure", 0) * 0.50
            if instant.get("sarcasm", 0) > instant.get("warm_humor", 0):
                instant_eff["warm_humor"] = max(instant_eff.get("warm_humor", 0), 0.6 * instant.get("sarcasm", 0))
        if not_target:
            instant_eff["toxicity"] = instant_eff.get("toxicity", 0) * 0.40
            instant_eff["contempt"] = instant_eff.get("contempt", 0) * 0.50
            instant_eff["sarcasm"] = instant_eff.get("sarcasm", 0) * 0.60
            instant_eff["threat_pressure"] = instant_eff.get("threat_pressure", 0) * 0.60
        if quoted:
            instant_eff["toxicity"] = instant_eff.get("toxicity", 0) * 0.30
            instant_eff["threat_pressure"] = instant_eff.get("threat_pressure", 0) * 0.50
        for s in SIGNALS:
            instant_eff[s] = _clip01(instant_eff.get(s, 0))

        # Trace: fast/slow 衰减更新（用 instant_eff）
        trace_fast_new: Dict[str, float] = {}
        trace_slow_new: Dict[str, float] = {}
        trace_new: Dict[str, float] = {}
        for s in SIGNALS:
            i = instant_eff[s]
            f_old = trace_fast_prev.get(s, 0.0)
            sl_old = trace_slow_prev.get(s, 0.0)
            trace_fast_new[s] = _clip01(i + GAMMA_FAST * f_old)
            trace_slow_new[s] = _clip01(i + GAMMA_SLOW * sl_old)
            trace_new[s] = _clip01(MIX * trace_fast_new[s] + (1 - MIX) * trace_slow_new[s])

        # Heat + Streak（用 instant_eff）
        heat_new: Dict[str, float] = {}
        streak_new: Dict[str, int] = {}
        heat_star: Dict[str, float] = {}
        for s in SIGNALS:
            i = instant_eff[s]
            h_old = heat_prev.get(s, 0.0)
            str_old = streak_prev.get(s, 0)
            heat_new[s] = ALPHA_HEAT * h_old + i
            streak_new[s] = (str_old + 1) if i >= STREAK_THRESHOLD else 0
            amp = 1.0 + AMP_FACTOR * min(AMP_MAX_STREAK, streak_new[s])
            heat_star[s] = heat_new[s] * amp

        # Composite
        provocation = max(trace_new.get("sarcasm", 0), trace_new.get("contempt", 0))
        pressure = max(trace_new.get("threat_pressure", 0), PRESSURE_TOXICITY_FACTOR * trace_new.get("toxicity", 0))
        conflict = _clip01(
            CONFLICT_TOXICITY * trace_new.get("toxicity", 0)
            + CONFLICT_PROVOCATION * provocation
            + CONFLICT_PRESSURE * pressure
        )
        goodwill = max(
            trace_new.get("repair", 0),
            trace_new.get("appreciation", 0),
            trace_new.get("validation", 0),
            trace_new.get("cooperation", 0),
            trace_new.get("warm_humor", 0),
            trace_new.get("self_disclosure", 0),
        )
        conflict_eff = _clip01(conflict - GOODWILL_OFFSET * goodwill)

        composite = {
            "provocation": round(provocation, 4),
            "conflict": round(conflict, 4),
            "goodwill": round(goodwill, 4),
            "pressure": round(pressure, 4),
            "conflict_eff": round(conflict_eff, 4),
        }

        params = {"gamma_fast": GAMMA_FAST, "gamma_slow": GAMMA_SLOW, "mix": MIX}
        detection_signals = {
            "instant": instant,
            "meta": meta,
            "stage_ctx": stage_ctx,
            "quality": quality,
            "instant_eff": instant_eff,
            "trace": trace_new,
            "heat": heat_star,
            "streak": streak_new,
            "composite": composite,
            "params": params,
            "trace_fast": trace_fast_new,
            "trace_slow": trace_slow_new,
            "heat_recur": heat_new,
        }

        # 路由：冲突/挑衅高则 mute
        if conflict_eff > 0.7 or provocation > 0.8:
            route = "MUTE"
        else:
            route = "NORMAL"
        print(f"[Detection] conflict_eff={conflict_eff:.2f}, provocation={provocation:.2f}, route={route}")

        return {
            "detection_signals": detection_signals,
            "detection_category": route,
            "detection_result": route,
        }
    return detection_node


def route_by_detection(state: AgentState) -> str:
    """
    根据 detection 输出的 category 路由：
    - MUTE -> mute（跳过所有处理）
    - NORMAL -> normal（继续正常流程）
    注意：cold_mode 会在 mode_manager 节点中确定，这里不单独路由。
    """
    route = state.get("detection_category") or state.get("detection_result") or "NORMAL"
    return "normal" if route == "NORMAL" else "mute"

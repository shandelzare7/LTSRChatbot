from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

from app.state import AgentState
from app.lats.prompt_utils import format_style_as_param_list

ExprMode = Literal[0, 1, 2, 3]
# 0=LITERAL_DIRECT, 1=LITERAL_INDIRECT, 2=FIGURATIVE, 3=IRONIC_LIGHT


# -----------------------------
# helpers
# -----------------------------
def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def centered(x: float) -> float:
    """Map 0..1 -> -0.5..+0.5."""
    return x - 0.5


def lin(
    base: float,
    terms: Dict[str, float],
    values: Dict[str, float],
    gain: float = 1.0,
) -> float:
    """
    base + sum(w_i * centered(v_i)), then apply gain to reduce saturation, clip to [0,1].
    gain < 1 => fewer "贴墙" values while keeping directions.
    """
    s = base
    for k, w in terms.items():
        s += w * centered(values.get(k, 0.5))
    s = 0.5 + gain * (s - 0.5)
    return clip01(s)


def contrast_gamma01(x: float, gamma: float = 1.25) -> float:
    """
    gamma>1: push values away from 0.5 (increase margin)
    gamma<1: pull values toward 0.5 (decrease margin)

    Smooth, symmetric around 0.5.
    """
    x = clip01(x)
    if gamma <= 0:
        return x
    if abs(gamma - 1.0) < 1e-9:
        return x
    if x <= 0.5:
        return 0.5 * (2.0 * x) ** gamma
    return 1.0 - 0.5 * (2.0 * (1.0 - x)) ** gamma


def _respect_mid01(respect01: float) -> float:
    """
    Mid-respect (~0.5) is safest for light play/irony; extremes are less safe.
    Returns 0..1.
    """
    return clip01(1.0 - abs(respect01 - 0.5) * 2.0)


def _det_jitter(momentum01: float, topic_appeal01: float) -> float:
    """
    Deterministic tiny jitter in about [-0.03, +0.03].
    Driven ONLY by momentum/topic_appeal for reproducibility.
    Used to avoid hard threshold lock-in and add slight "life".
    """
    return 0.03 * math.sin(2.0 * math.pi * (1.7 * momentum01 + 2.3 * topic_appeal01))


def _parse_01(x: Any, default: float = 0.5) -> float:
    if x is None:
        return default
    try:
        return clip01(float(x))
    except (TypeError, ValueError):
        return default


def _pad_to_01(
    x: Any,
    *,
    pad_scale: Literal["0_1", "m1_1"] = "m1_1",
    default: float = 0.5,
) -> float:
    """
    PAD 值映射到 [0, 1]。不再使用 auto，由上游通过 mood["pad_scale"] 或 state 传入尺度。
    - "m1_1": 输入为 [-1, 1]，0 为中性
    - "0_1":  输入为 [0, 1]，0.5 为中性
    缺省 m1_1（更安全）。
    """
    if x is None:
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default

    if pad_scale == "0_1":
        return clip01(v)
    # m1_1（默认）
    return clip01((v + 1.0) / 2.0)


# -----------------------------
# inputs
# -----------------------------
@dataclass
class Inputs:
    # Big Five (0..1)
    E: float
    A: float
    C: float
    O: float
    N: float

    # PAD (0..1)
    P: float
    Ar: float
    D: float

    # Busy (0..1)
    busy: float

    # New (0..1)
    momentum: float = 0.5
    topic_appeal: float = 0.5

    # Relationship (0..1)
    closeness: float = 0.5
    trust: float = 0.5
    liking: float = 0.5
    respect: float = 0.5
    attractiveness: float = 0.5
    power: float = 0.5

    # Evidence (optional)
    # 你现在没有来源 => 传 None
    # 未来接入后可传 0..1 来做 certainty cap
    evidence: Optional[float] = None


# -----------------------------
# core computation
# -----------------------------
def compute_style_keys(inp: Inputs) -> Dict[str, float | int]:
    # pack + clip
    evidence_opt = inp.evidence if inp.evidence is None else clip01(float(inp.evidence))

    v: Dict[str, float] = {
        "E": clip01(inp.E),
        "A": clip01(inp.A),
        "C": clip01(inp.C),
        "O": clip01(inp.O),
        "N": clip01(inp.N),
        "P": clip01(inp.P),
        "Ar": clip01(inp.Ar),
        "D": clip01(inp.D),
        "busy": clip01(inp.busy),
        "momentum": clip01(inp.momentum),
        "topic_appeal": clip01(inp.topic_appeal),
        "closeness": clip01(inp.closeness),
        "trust": clip01(inp.trust),
        "liking": clip01(inp.liking),
        "respect": clip01(inp.respect),
        "attractiveness": clip01(inp.attractiveness),
        "power": clip01(inp.power),
        # evidence 若 None，这里给 0.5 仅作为占位（真正是否 clamp 在下面单独判断）
        "evidence": 0.5 if evidence_opt is None else evidence_opt,
    }

    # tiny deterministic jitter (very small)
    j = _det_jitter(v["momentum"], v["topic_appeal"])

    # -----------------------------
    # derived latent signals (推算值)
    # -----------------------------
    familiarity = clip01(
        0.50
        + 0.45 * centered(v["closeness"])
        + 0.25 * centered(v["liking"])
        + 0.15 * centered(v["trust"])
        + 0.08 * centered(v["momentum"])
        + 0.05 * centered(v["topic_appeal"])
    )

    hierarchy = clip01(
        0.50
        + 0.45 * centered(v["power"])
        + 0.25 * centered(v["D"])
    )

    tension = clip01(
        0.50
        + 0.45 * centered(v["Ar"])
        - 0.45 * centered(v["P"])
        + 0.15 * centered(v["busy"])
        + 0.10 * centered(v["momentum"])
        - 0.06 * centered(v["topic_appeal"])
        + j * 0.3
    )

    safety_to_play = clip01(
        0.50
        + 0.35 * centered(v["closeness"])
        + 0.25 * centered(v["trust"])
        + 0.20 * centered(v["P"])
        - 0.35 * centered(tension)
        - 0.20 * centered(v["respect"])
        + 0.10 * centered(v["topic_appeal"])
        + 0.06 * centered(v["momentum"])
        + j * 0.4
    )

    v["familiarity"] = familiarity
    v["hierarchy"] = hierarchy
    v["tension"] = tension
    v["safety_to_play"] = safety_to_play

    # -----------------------------
    # 6 keys
    # -----------------------------

    # (a) FORMALITY
    FORMALITY = lin(
        base=0.50,
        terms={
            "C": +0.55,
            "respect": +0.225,  # 减半：原 +0.45
            "hierarchy": +0.35,
            "busy": +0.15,
            "familiarity": -0.40,
            "E": -0.15,
            "momentum": -0.10,
            "topic_appeal": -0.05,
        },
        values=v,
        gain=0.95,
    )

    # (b) POLITENESS (rebalanced: less "always polite", more context-sensitive)
    POLITENESS = lin(
        base=0.50,  # was 0.55; make neutral actually neutral
        terms={
            # deference / face-saving
            "respect": +0.275,  # 减半：原 +0.55
            # agreeableness should mostly show up in WARMTH; keep mild here
            "A": +0.25,  # was +0.60
            "N": +0.10,  # was +0.20
            # when it's safe & familiar => drop pleasantries
            "familiarity": -0.45,  # was -0.20
            "safety_to_play": -0.25,  # new
            # pressure => terse, less ceremonious
            "busy": -0.25,  # was -0.15
            "tension": -0.20,  # new
            # separate "user power" vs "bot dominance" (adjust sign if your semantics differ)
            "power": +0.20,  # new
            "D": -0.15,  # new
            # tiny dynamics
            "momentum": -0.03,
            "topic_appeal": +0.03,
        },
        values=v,
        gain=1.05,  # slightly >1 to increase spread (margin)
    )
    POLITENESS = clip01(POLITENESS + j * 0.5)

    # (c) WARMTH
    WARMTH = lin(
        base=0.50,
        terms={
            "A": +0.55,
            "E": +0.25,
            "P": +0.35,
            "liking": +0.30,
            "familiarity": +0.25,
            "topic_appeal": +0.18,
            "momentum": +0.10,
            "attractiveness": +0.08,
            "tension": -0.30,
            "hierarchy": -0.20,
            "busy": -0.25,
        },
        values=v,
        gain=0.95,
    )
    WARMTH = clip01(WARMTH + j * 0.5)

    # (d) CERTAINTY
    CERTAINTY = lin(
        base=0.50,
        terms={
            "C": +0.45,
            "D": +0.35,
            "trust": +0.30,
            "power": +0.20,
            "N": -0.45,
            "busy": +0.10,
            "momentum": +0.05,
        },
        values=v,
        gain=0.95,
    )

    # evidence clamp：你现在没有 evidence => 使用“温和默认 cap”，避免又回到 0.85 的锁死
    # - 有 evidence：按 evidence 调整 cap
    # - 无 evidence：cap=0.92（允许更“个性鲜明”的笃定，但仍留一点安全余量）
    if evidence_opt is None:
        certainty_cap = 0.92
    else:
        certainty_cap = clip01(0.70 + 0.30 * evidence_opt)
    CERTAINTY = min(CERTAINTY, certainty_cap)

    # (e) CHAT_MARKERS
    CHAT_MARKERS = lin(
        base=0.42,
        terms={
            "E": +0.35,
            "familiarity": +0.35,
            "P": +0.12,
            "Ar": +0.18,
            "momentum": +0.25,
            "topic_appeal": +0.20,
            "attractiveness": +0.10,
            "C": -0.35,
            "respect": -0.30,
            "hierarchy": -0.30,
            "busy": -0.45,
        },
        values=v,
        gain=0.90,
    )
    CHAT_MARKERS = clip01(CHAT_MARKERS + j)

    # weaken coupling: allow "formal but a bit chatty" & "casual but restrained"
    CHAT_MARKERS = clip01(CHAT_MARKERS - 0.15 * centered(FORMALITY))

    # (f) EXPRESSION_MODE
    indirectness = clip01(
        0.50
        + 0.35 * centered(1.0 - CERTAINTY)
        + 0.25 * centered(POLITENESS)
        + 0.20 * centered(v["respect"])
        + 0.10 * centered(1.0 - FORMALITY)
        - 0.20 * centered(v["busy"])
        + 0.15 * centered(v["familiarity"])
        + 0.12 * centered(v["momentum"])
        + 0.08 * centered(v["topic_appeal"])
        + j * 0.6
    )

    figurative_bias = clip01(
        0.50
        + 0.45 * centered(v["O"])
        + 0.15 * centered(v["attractiveness"])
        + 0.15 * centered(v["Ar"])
        + 0.18 * centered(v["topic_appeal"])
        + 0.12 * centered(v["momentum"])
        + 0.10 * centered(v["familiarity"])
        - 0.30 * centered(v["respect"])
        - 0.35 * centered(v["busy"])
        + j
    )

    # slightly softened thresholds to improve observability
    EXPRESSION_MODE: ExprMode
    if figurative_bias >= 0.60 and indirectness >= 0.47:
        EXPRESSION_MODE = 2
    elif indirectness >= 0.60:
        EXPRESSION_MODE = 1
    else:
        EXPRESSION_MODE = 0

    # -----------------------------
    # IRONIC_LIGHT via propensity (more observable but still safe)
    # -----------------------------
    respect_mid = _respect_mid01(v["respect"])
    irony_propensity = clip01(
        0.50
        + 0.50 * centered(v["safety_to_play"])
        + 0.25 * centered(figurative_bias)
        + 0.15 * centered(v["momentum"])
        + 0.15 * centered(v["topic_appeal"])
        + 0.10 * centered(respect_mid)
        - 0.25 * centered(v["busy"])
        + j * 0.6
    )

    if (
        irony_propensity >= 0.78
        and v["closeness"] >= 0.68
        and v["trust"] >= 0.62
        and v["P"] >= 0.58
        and v["tension"] <= 0.62
        and figurative_bias >= 0.70
    ):
        EXPRESSION_MODE = 3

    # -----------------------------
    # guardrails
    # -----------------------------
    if v["respect"] >= 0.80:
        if EXPRESSION_MODE == 3:
            EXPRESSION_MODE = 1
        CHAT_MARKERS = min(CHAT_MARKERS, 0.40)

    if v["trust"] <= 0.35 or v["closeness"] <= 0.35:
        if EXPRESSION_MODE in (2, 3):
            EXPRESSION_MODE = 1

    if v["busy"] >= 0.80:
        EXPRESSION_MODE = 0 if CERTAINTY >= 0.55 else 1
        CHAT_MARKERS = min(CHAT_MARKERS, 0.20)

    # -----------------------------
    # post shaping (increase "margin" for LLM observability)
    # -----------------------------
    FORMALITY = contrast_gamma01(FORMALITY, gamma=1.15)
    POLITENESS = contrast_gamma01(POLITENESS, gamma=1.30)
    WARMTH = contrast_gamma01(WARMTH, gamma=1.10)
    CHAT_MARKERS = contrast_gamma01(CHAT_MARKERS, gamma=1.20)

    # CERTAINTY is safety-sensitive: keep mild, and respect the cap
    CERTAINTY = min(contrast_gamma01(CERTAINTY, gamma=1.05), certainty_cap)

    return {
        "FORMALITY": float(FORMALITY),
        "POLITENESS": float(POLITENESS),
        "WARMTH": float(WARMTH),
        "CERTAINTY": float(CERTAINTY),
        "EXPRESSION_MODE": int(EXPRESSION_MODE),
        "CHAT_MARKERS": float(CHAT_MARKERS),
    }


# -----------------------------
# style node
# -----------------------------
def create_style_node(llm_invoker: Any = None) -> Callable[[AgentState], Dict[str, Any]]:
    """
    从 state 读取 Big Five、PAD、relationship_state、busy、momentum、topic_appeal
    计算 6 维 style，并输出 llm_instructions。
    """

    def style_node(state: AgentState) -> Dict[str, Any]:
        bf = state.get("bot_big_five") or {}
        mood = state.get("mood_state") or {}
        rel = state.get("relationship_state") or {}

        # Big Five (0..1)
        E = _parse_01(bf.get("extraversion"), 0.5)
        A = _parse_01(bf.get("agreeableness"), 0.5)
        C = _parse_01(bf.get("conscientiousness"), 0.5)
        O = _parse_01(bf.get("openness"), 0.5)
        N = _parse_01(bf.get("neuroticism"), 0.5)

        # Busy: unknown => 0.5 (neutral)
        busy = _parse_01(mood.get("busyness"), 0.5)

        # PAD
        pleasure_raw = mood.get("pleasure")
        arousal_raw = mood.get("arousal")
        dominance_raw = mood.get("dominance")

        # PAD：强制使用 mood["pad_scale"]，否则默认 m1_1（不再 auto 推断）
        pad_scale: Literal["0_1", "m1_1"] = "m1_1"
        if mood.get("pad_scale") in ("m1_1", "0_1"):
            pad_scale = mood["pad_scale"]

        P = _pad_to_01(pleasure_raw, pad_scale=pad_scale, default=0.5)
        Ar = _pad_to_01(arousal_raw, pad_scale=pad_scale, default=0.5)
        D = _pad_to_01(dominance_raw, pad_scale=pad_scale, default=0.5)

        # Relationship：强制使用 rel_scale，否则默认 0_1（避免 [−1,1] 的 0.2 被当成 [0,1] 的 0.2）
        rel_scale: Literal["0_1", "m1_1"] = "0_1"
        if rel.get("rel_scale") in ("m1_1", "0_1"):
            rel_scale = rel["rel_scale"]
        elif state.get("relationship_scale") in ("m1_1", "0_1"):
            rel_scale = state["relationship_scale"]

        def _get_rel(key: str, default: float = 0.5) -> float:
            rv = rel.get(key)
            if rv is None and key == "attractiveness":
                rv = rel.get("warmth", rel.get("liking"))
            if rv is None:
                return default
            try:
                f = float(rv)
            except (TypeError, ValueError):
                return default
            if rel_scale == "m1_1" and -1.0 <= f <= 1.0:
                return clip01((f + 1.0) / 2.0)
            return clip01(f)

        closeness = _get_rel("closeness", 0.5)
        trust = _get_rel("trust", 0.5)
        liking = _get_rel("liking", 0.5)
        respect = _get_rel("respect", 0.5)
        attractiveness = _get_rel("attractiveness", 0.5)
        power = _get_rel("power", 0.5)

        # momentum/topic_appeal（你稍后适配即可）
        momentum = _parse_01(mood.get("momentum", state.get("momentum")), 0.5)
        topic_appeal = _parse_01(mood.get("topic_appeal", state.get("topic_appeal")), 0.5)

        # evidence：你现在没有来源 => 传 None
        inp = Inputs(
            E=E,
            A=A,
            C=C,
            O=O,
            N=N,
            P=P,
            Ar=Ar,
            D=D,
            busy=busy,
            momentum=momentum,
            topic_appeal=topic_appeal,
            closeness=closeness,
            trust=trust,
            liking=liking,
            respect=respect,
            attractiveness=attractiveness,
            power=power,
            evidence=None,
        )

        style_dict = compute_style_keys(inp)
        llm_instructions = format_style_as_param_list(style_dict)
        return {"style": style_dict, "llm_instructions": llm_instructions}

    return style_node
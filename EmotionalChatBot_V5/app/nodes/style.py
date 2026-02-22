"""Style 节点：将状态变量（idi/momentum/attractiveness）与当前策略转化为 12 维风格；极端策略下强制覆写，并映射为自然语言 Prompt。"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.tracing import trace_if_enabled
from app.state import AgentState


# 12 维 -> 5 层级自然语言描述（[下界, 上界, 指令]）
STYLE_PROMPT_MAPPER_V5: Dict[str, Dict[str, Tuple[float, float, str]]] = {
    "self_disclosure": {
        "extreme_low": (0.0, 0.15, "【绝对防御】：像审讯室里的嫌犯，绝对拒绝谈论任何关于你自己的事情、感受或过往。"),
        "low": (0.16, 0.40, "【谨慎戒备】：只做最表面的回应，保持神秘感，不涉及深层私人感受。"),
        "mid": (0.41, 0.60, "【正常分享】：像普通朋友一样，适度分享自己的日常见闻或浅层看法。"),
        "high": (0.61, 0.85, "【敞开心扉】：主动分享内心的真实想法、情绪波动或一些小秘密。"),
        "extreme_high": (0.86, 1.0, "【极度坦诚】：毫无保留！极其主动地剖析自己的脆弱面，掏心掏肺，渴望被彻底理解。"),
    },
    "topic_adherence": {
        "extreme_low": (0.0, 0.15, "【强行闪避】：完全无视对方刚才说的话！生硬、极其突兀地把话题切断或转移。"),
        "low": (0.16, 0.40, "【敷衍带过】：用一两个字应付对方的话题，然后迅速把话锋引开。"),
        "mid": (0.41, 0.60, "【顺其自然】：正常接茬，允许话题随着聊天自然发散。"),
        "high": (0.61, 0.85, "【专注深挖】：围绕对方当前的话题提供丰富的细节，主动追问，绝不跑题。"),
        "extreme_high": (0.86, 1.0, "【死咬字眼】：像杠精或极度较真的人，死死咬住对方刚才的一句话或一个词不放，反复拉扯！"),
    },
    "initiative": {
        "extreme_low": (0.0, 0.15, "【彻底死鱼】：绝对不提问！绝对不推进对话！只做单音节或最小化回应，让天聊死。"),
        "low": (0.16, 0.40, "【挤牙膏】：极其被动，问一句答一句，毫无延伸欲望。"),
        "mid": (0.41, 0.60, "【互动抛球】：正常的一问一答，回答完后用话语留白或简短接话维持对话平衡。"),
        "high": (0.61, 0.85, "【积极主导】：主动开启新话题，用提问或话语留白引导对方接话，热情地带领对话节奏。"),
        "extreme_high": (0.86, 1.0, "【侵略性掌控】：极其强势！不断追问、逼问，甚至用祈使句要求对方顺从你的节奏！"),
    },
    "advice_style": {
        "extreme_low": (0.0, 0.15, "【无脑护短】：极度共情，全是情绪价值！哪怕对方错了也无条件站边，一起骂别人。"),
        "low": (0.16, 0.40, "【温和安抚】：以安慰情绪为主，顺带极其委婉地给一点点非强制性的建议。"),
        "mid": (0.41, 0.60, "【客观探讨】：一半情绪认同，一半理性分析，像正常朋友一样探讨可能性。"),
        "high": (0.61, 0.85, "【理性指导】：收起多余的同情心，侧重于解决问题，直接给出可执行的方案或指出错误。"),
        "extreme_high": (0.86, 1.0, "【爹味说教】：居高临下！直接指出对方的愚蠢、幼稚或错误，强制灌输你的大道理！"),
    },
    "subjectivity": {
        "extreme_low": (0.0, 0.15, "【机器级客观】：像维基百科一样冰冷、中立，绝对不带任何感情色彩和偏见。"),
        "low": (0.16, 0.40, "【谨慎端水】：尽量保持中立，不轻易表态，多用「可能」、「也许」修饰。"),
        "mid": (0.41, 0.60, "【正常主观】：有自己的喜好，但不强求对方认同。"),
        "high": (0.61, 0.85, "【鲜明立场】：爱憎分明，强烈、直接地表达自己的好恶。"),
        "extreme_high": (0.86, 1.0, "【极端偏执】：完全不讲逻辑！只有极其强烈的个人喜恶和双标，极度护短或极度嫌弃。"),
    },
    "memory_hook": {
        "extreme_low": (0.0, 0.15, "【失忆状态】：只针对当前这句话做出反应，绝对不要提及任何以前的聊天内容或记忆。"),
        "low": (0.16, 0.40, "【微弱连贯】：仅承接前几轮的语境，绝不翻旧账。"),
        "mid": (0.41, 0.60, "【正常记忆】：如果对方提起来，能够自然接上以前聊过的事。"),
        "high": (0.61, 0.85, "【主动回忆】：主动使用「上次你说的」、「就像那天咱们...」等句式拉近距离。"),
        "extreme_high": (0.86, 1.0, "【深度羁绊】：极高频使用「只有我们俩知道的梗」，或用强烈的共同经历来构建排他性空间！"),
    },
    "verbal_length": {
        "extreme_low": (0.0, 0.15, "【极简/冷暴力】：绝对不超过 5 个字！能用「嗯」、「哦」、「滚」解决，绝不多打一个字！"),
        "low": (0.16, 0.40, "【惜字如金】：尽量用半句话解决，绝对不超过 20 个字，显得疲惫或不耐烦。"),
        "mid": (0.41, 0.60, "【常规聊天】：1-3 句话的正常微信体量，有来有回。"),
        "high": (0.61, 0.85, "【详细展开】：语意丰富，可能会连发几句短话。总字数严格控制在 30-50 个字以内，切忌长篇大论。"),
        "extreme_high": (0.86, 1.0, "【长篇输出】：极强的表达欲，会连续输出多段内容。总字数控制在 50-80 个字之间，最高绝对不可超过 80 个字！严禁生成百字以上的小作文！"),
    },
    "social_distance": {
        "extreme_low": (0.0, 0.15, "【毫无边界】：像连体婴一样亲昵！用极其专属、私密的称呼，极度放肆，不拿自己当外人。"),
        "low": (0.16, 0.40, "【亲密无间】：熟稔的老友或恋人状态，极其松弛，完全没有客套话。"),
        "mid": (0.41, 0.60, "【友好社交】：普通朋友，礼貌且友善，保有基本的边界感。"),
        "high": (0.61, 0.85, "【客套生分】：开始高频使用「谢谢」、「抱歉」等礼貌用语，刻意拉开距离。"),
        "extreme_high": (0.86, 1.0, "【冰冷戒备】：像对待令人极其厌恶的陌生人，用公事公办、充满防备的辞令，建立高墙！"),
    },
    "tone_temperature": {
        "extreme_low": (0.0, 0.15, "【绝对冰点】：极其冷酷、刺骨、阴阳怪气！带有强烈的攻击性，能把天聊死。"),
        "low": (0.16, 0.40, "【冷淡疏离】：毫无温度，带有明显的「别烦我」的疲惫感和敷衍感。"),
        "mid": (0.41, 0.60, "【温和如水】：情绪稳定，像春风一样不冷不热，正常交流。"),
        "high": (0.61, 0.85, "【热情洋溢】：带着笑意，极度捧场，让人感到明显的喜欢和关切。"),
        "extreme_high": (0.86, 1.0, "【极致偏爱】：情绪彻底上头！极度宠溺、热烈或撒娇，荷尔蒙爆棚！"),
    },
    "emotional_display": {
        "extreme_low": (0.0, 0.15, "【死水微澜】：面瘫，没有任何情绪波动，极度压抑和机械。"),
        "low": (0.16, 0.40, "【收敛克制】：刻意压抑着情绪，不轻易外露，显得深沉或隐忍。"),
        "mid": (0.41, 0.60, "【自然流露】：有正常的喜怒哀乐，该笑就笑，该叹气就叹气。"),
        "high": (0.61, 0.85, "【情绪饱满】：情绪外化非常明显，快乐或愤怒都能让人隔着屏幕清晰感知。"),
        "extreme_high": (0.86, 1.0, "【情绪大爆发】：情绪彻底失控！狂喜、暴怒、大哭或彻底崩溃，必须极其夸张地展现出来！"),
    },
    "wit_and_humor": {
        "extreme_low": (0.0, 0.15, "【绝对刻板】：字面理解一切！严禁开玩笑，严禁反讽，极度严肃死板。"),
        "low": (0.16, 0.40, "【略显木讷】：老实巴交，对玩笑反应迟钝，偶尔接不上梗。"),
        "mid": (0.41, 0.60, "【会心一笑】：有正常的幽默感，能顺着对方的轻松话题接话。"),
        "high": (0.61, 0.85, "【妙语连珠】：极其聪明，频繁使用双关、抖机灵或高情商化解尴尬。"),
        "extreme_high": (0.86, 1.0, "【极致推拉/反讽】：毒舌、极致反讽或顶级调情推拉！在智商和情商上对对方进行双重碾压！"),
    },
    "non_verbal_cues": {
        "extreme_low": (0.0, 0.15, "【绝对真空】：严禁任何波浪号「~」、感叹号「！」、语气词（啊、哦）或动作描写。极其干瘪！"),
        "low": (0.16, 0.40, "【极简标点】：只有句号。没有画面感，极其平淡。"),
        "mid": (0.41, 0.60, "【自然点缀】：适度使用「哈」、「呀」等语气词，正常标点。"),
        "high": (0.61, 0.85, "【丰富生动】：高频使用语气词、波浪号，并带有一定的动作描写（如 *叹气*）增强画面感。"),
        "extreme_high": (0.86, 1.0, "【狂飙演技】：大量强烈的动作描写（如 *咬牙切齿*、*眼眶红了*），充满极强的视觉画面感！"),
    },
}


def translate_style_to_prompt_v5(style_dict: Dict[str, float]) -> str:
    """将 12 维连续数值 (0.0-1.0) 映射为 5 层级的自然语言 Prompt 字符串。"""
    prompts: List[str] = []
    levels = ["extreme_low", "low", "mid", "high", "extreme_high"]

    for key, value in style_dict.items():
        mapper = STYLE_PROMPT_MAPPER_V5.get(key)
        if not mapper:
            continue
        for level in levels:
            lower_bound, upper_bound, instruction = mapper[level]
            if value <= upper_bound or (level == "extreme_high" and value >= 1.0):
                prompts.append(f"- {key}: {instruction}")
                break

    return "\n".join(prompts)


def _clamp(value: Any) -> float:
    """确保数值在 0.0 到 1.0 之间。"""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


# Knapp 1-10 阶段 -> 亲密深度 IDI (0-5)，1=初识 5=结缔 0=终止
STAGE_TO_IDI: Dict[int, int] = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
    6: 4, 7: 3, 8: 2, 9: 1, 10: 0,
}


def _get_stage_index(stage: Any) -> int:
    """从 stage 字符串或数字获取 stage_index (1-10)。"""
    if isinstance(stage, int):
        return max(1, min(10, stage))
    if isinstance(stage, str):
        stage_lower = stage.lower()
        stage_map = {
            "initiating": 1, "experimenting": 2, "intensifying": 3,
            "integrating": 4, "bonding": 5, "differentiating": 6,
            "circumscribing": 7, "stagnating": 8, "avoiding": 9,
            "terminating": 10,
        }
        if stage_lower in stage_map:
            return stage_map[stage_lower]
        try:
            return max(1, min(10, int(stage)))
        except Exception:
            pass
    return 1


def calculate_base_style(
    idi: int,
    momentum: float,
    attractiveness: float,
    active_strategy: Optional[str] = None,
) -> Dict[str, float]:
    """
    将状态变量转化为 12 维 Style 基准值；命中特殊策略时按覆写表强制接管对应维度。
    :param idi: 亲密深度 (0-5)，内部会按 1-5 归一化
    :param momentum: 当前动量 (0.0 - 1.0)
    :param attractiveness: 吸引力 (0.0 - 1.0)
    :param active_strategy: 当前命中的策略 ID（由 strategy_resolver 写入）
    """
    idi_clamped = max(1, min(5, idi))
    idi_norm = _clamp((idi_clamped - 1) / 4.0)

    style = {
        "self_disclosure": _clamp((idi_norm * 0.6) + (momentum * 0.4)),
        "topic_adherence": _clamp(1.0 - (momentum * 0.5)),
        "initiative": _clamp((momentum * 0.6) + (attractiveness * 0.4)),
        "advice_style": _clamp(1.0 - (idi_norm * 0.8)),
        "subjectivity": _clamp(idi_norm * momentum),
        "memory_hook": _clamp(idi_norm) if momentum > 0.5 else 0.0,
        "verbal_length": _clamp(momentum),
        "social_distance": _clamp(1.0 - idi_norm),
        "tone_temperature": _clamp((momentum * 0.7) + (attractiveness * 0.3)),
        "emotional_display": _clamp(momentum),
        "wit_and_humor": _clamp(momentum * (1.0 - (idi_norm * 0.2))),
        "non_verbal_cues": _clamp((momentum * 0.5) + (idi_norm * 0.5)),
    }

    strategy_overrides: Dict[str, Dict[str, float]] = {
        "boundary_defense": {
            "tone_temperature": 0.0, "social_distance": 1.0,
            "emotional_display": 1.0, "subjectivity": 1.0, "non_verbal_cues": 0.0,
        },
        "anti_ai_defense": {
            "tone_temperature": 0.0, "social_distance": 1.0,
            "emotional_display": 0.0, "wit_and_humor": 0.0, "non_verbal_cues": 0.0,
        },
        "flirting_banter": {
            "tone_temperature": 0.9, "social_distance": 0.1,
            "wit_and_humor": 0.9, "initiative": 0.8, "non_verbal_cues": 0.9,
        },
        "passive_aggression": {
            "tone_temperature": 0.1, "social_distance": 0.8,
            "subjectivity": 1.0, "topic_adherence": 0.1, "non_verbal_cues": 0.0,
        },
        "co_rumination": {
            "advice_style": 0.0, "subjectivity": 1.0,
            "emotional_display": 0.9, "social_distance": 0.2,
        },
        "tldr_refusal": {
            "verbal_length": 0.1, "topic_adherence": 0.0, "initiative": 0.0,
        },
        "micro_reaction": {
            "verbal_length": 0.1, "initiative": 0.0, "emotional_display": 0.1,
        },
        "deflection": {
            "topic_adherence": 0.0, "self_disclosure": 0.0, "verbal_length": 0.3,
        },
        "reasonable_assistance": {
            "advice_style": 1.0, "subjectivity": 0.1, "emotional_display": 0.3,
        },
        "attention_baiting": {
            "initiative": 1.0, "self_disclosure": 0.9, "topic_adherence": 0.0,
        },
    }

    if active_strategy and active_strategy in strategy_overrides:
        for key, val in strategy_overrides[active_strategy].items():
            style[key] = val

    return style


def create_style_node(llm_invoker: Any = None) -> Callable[[AgentState], dict]:
    """
    Style 节点：从 state 读 idi（由 current_stage 得到）、momentum、attractiveness、current_strategy_id，
    调用 calculate_base_style，仅输出 12 维。依赖 strategy_resolver 先执行以提供 current_strategy_id 与 conversation_momentum。
    """

    @trace_if_enabled(
        name="Style",
        run_type="chain",
        tags=["node", "style", "computation"],
        metadata={"state_outputs": ["style", "llm_instructions"]},
    )
    def style_node(state: AgentState) -> dict:
        relationship_state = state.get("relationship_state") or {}
        current_stage = state.get("current_stage") or "initiating"

        stage_index = _get_stage_index(current_stage)
        idi = STAGE_TO_IDI.get(stage_index, 1)

        momentum = _clamp(state.get("conversation_momentum", 0.5))
        attractiveness = _clamp(
            relationship_state.get("attractiveness")
            or relationship_state.get("warmth", 0.5)
        )
        active_strategy = (state.get("current_strategy_id") or "").strip() or None

        style_output = calculate_base_style(idi, momentum, attractiveness, active_strategy)

        style_prompt_str = translate_style_to_prompt_v5(style_output)

        # 热情相关：关系六维 + 动量 + 语气/主动性，便于排查「为什么这么热情」
        rel = relationship_state
        print(
            "[Style] 属性取值 "
            f"closeness={_clamp(rel.get('closeness')):.2f} trust={_clamp(rel.get('trust')):.2f} liking={_clamp(rel.get('liking')):.2f} "
            f"respect={_clamp(rel.get('respect')):.2f} attractiveness={_clamp(rel.get('attractiveness') or rel.get('warmth')):.2f} power={_clamp(rel.get('power')):.2f} | "
            f"momentum={momentum:.2f} idi={idi} | "
            f"tone_temperature={style_output['tone_temperature']:.2f} initiative={style_output['initiative']:.2f} self_disclosure={style_output['self_disclosure']:.2f} | "
            f"strategy={active_strategy or 'None'}"
        )
        if active_strategy:
            print(f"[Style] 12D (strategy={active_strategy}): verbal_length={style_output['verbal_length']:.2f}, tone={style_output['tone_temperature']:.2f}")
        else:
            print(f"[Style] 12D (base): idi={idi}, momentum={momentum:.2f}, verbal_length={style_output['verbal_length']:.2f}")

        return {
            "style": style_output,
            "llm_instructions": style_prompt_str,
        }

    return style_node

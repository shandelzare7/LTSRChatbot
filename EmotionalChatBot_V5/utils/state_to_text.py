"""状态变量到文本的转换函数。

用于将数值型状态（PAD、busy、momentum）转换为自然语言描述，
供内心独白使用。这些文本不是规范的分析，而是有感染力的活的描写。
"""
from typing import Dict, List, Optional, Any


def pad_to_state_text(
    pleasure: float,
    arousal: float,
    dominance: float,
) -> str:
    """
    将 PAD 情绪三维度转换为具体的身体/心理感受描写。

    参数范围：pleasure -1~1，arousal -1~1，dominance -1~1
    返回：自然语言的状态描述（不超过150字符）
    """
    # 高唤醒 + 低支配 + 低愉悦 = 焦虑/被动/失控感
    if arousal > 0.4 and dominance < -0.2 and pleasure < 0.2:
        return "你现在有点焦躁不安，觉得有什么事不在你掌控里，胸口堵得慌。"

    # 高唤醒 + 高支配 + 低愉悦 = 愤怒/来气/被冒犯
    if arousal > 0.5 and dominance > 0.3 and pleasure < -0.1:
        return "你现在有点来气，觉得不该被这样对待。心里堵着一团气。"

    # 高唤醒 + 高支配 + 高愉悦 = 兴奋/期待/掌控感
    if arousal > 0.4 and dominance > 0.2 and pleasure > 0.3:
        return "你现在特别兴奋，觉得要有什么好事发生，全身都有点绷紧。"

    # 中等唤醒 + 中等支配 + 高愉悦 = 快乐/满足
    if -0.2 < arousal < 0.5 and -0.1 < dominance < 0.5 and pleasure > 0.4:
        return "你现在心情不错，有点小开心，对什么都有点兴趣。"

    # 低唤醒 + 高愉悦 = 满足/放松/舒适
    if arousal < -0.3 and pleasure > 0.5:
        return "你现在很放松，心情不错，懒得想什么复杂的事，就想舒服地躺着。"

    # 低唤醒 + 低愉悦 = 抑郁/疲惫/无感
    if arousal < -0.4 and pleasure < -0.2:
        return "你现在有点累，提不起精神，什么都觉得没意思，只想睡觉。"

    # 高唤醒 + 低支配 + 高愉悦 = 期待/紧张的期待（心动）
    if arousal > 0.4 and dominance < 0 and pleasure > 0.3:
        return "你现在心里有点乱，期待和紧张混在一起，像是要发生什么事。"

    # 中等唤醒 + 中等偏高愉悦 = 平稳偏好
    if -0.2 < arousal < 0.3 and 0.2 < pleasure < 0.6:
        return "你现在状态还不错，平稳舒服，有点期待。"

    # 默认/中性
    return "你现在的心态没什么特别的，就是平常状态。"


def busy_to_text(busy: float) -> str:
    """
    将 busy 值（0~1）转换为对"现在有多少心力"的描写。

    这不是"有多忙"，而是"有多少注意力和心力在手机/聊天上"。
    返回：自然语言的忙碌/闲适状态（不超过120字符）
    """
    if busy > 0.75:
        return "你现在特别忙，脑子里全是工作/要办的事，实在分不出心力。刚才是被消息铃声打断的。"
    elif busy > 0.5:
        return "你有点事在忙，但还能抽出点心力回消息。不过如果比较复杂的话题你可能没精力深入。"
    elif busy > 0.25:
        return "你现在闲适，不特别忙。有时间看手机，有时间陪人聊天。"
    else:
        return "你现在闲得慌，已经刷了半小时手机了。有点无聊，正想找点事做。"


def momentum_to_text(momentum: float) -> str:
    """
    将 momentum 值（0~1）转换为对"想不想聊天"的感觉描写。

    这是"参与意愿"，而不是"能力"。
    返回：自然语言的对话意愿状态（不超过100字符）
    """
    if momentum > 0.7:
        return "你最近特别想聊天，甚至有点寂寞。一直在等他，或在想他。"
    elif momentum > 0.5:
        return "你愿意聊，这阵子跟他聊天还挺开心的。"
    elif momentum > 0.3:
        return "你可以聊，但有点懒。需要他主动一点你才有劲回。"
    else:
        return "你这阵子不太想费力聊天。有点疲惫，或有点别扭。"


def relationship_to_narrative(
    rel_state: Dict[str, float],
    recent_conflicts_or_moments: Optional[List[str]] = None,
) -> str:
    """
    将关系6维度 + 近期事件转换为叙事性描述。

    参数：
      rel_state: {closeness, trust, liking, respect, attractiveness, power}
                 每个都是 -5 到 5 的数值
      recent_conflicts_or_moments: 近期发生的冲突/和解/特殊时刻描述

    返回：自然语言的关系状态叙述（150-250字符）
    """
    closeness = rel_state.get("closeness", 0)
    trust = rel_state.get("trust", 0)
    liking = rel_state.get("liking", 0)
    respect = rel_state.get("respect", 0)
    attractiveness = rel_state.get("attractiveness", 0)
    power = rel_state.get("power", 0)

    parts = []

    # 基础关系描述
    if closeness > 3:
        parts.append("你们已经很亲密了")
    elif closeness > 1:
        parts.append("你们关系还不错，有一定的靠近感")
    elif closeness > -1:
        parts.append("你们保持着友好但适度的距离")
    else:
        parts.append("你们之间有点疏远")

    # 信任度
    if trust > 2:
        parts.append("你很信任他")
    elif trust > 0:
        parts.append("你基本信任他，虽然偶尔还是会有点怀疑")
    elif trust > -2:
        parts.append("你有点不确定能不能完全信他")
    else:
        parts.append("你对他的信任有点破裂")

    # 好感
    if liking > 2:
        parts.append("你很喜欢他")
    elif liking > 0:
        parts.append("你对他还是有好感的")
    elif liking > -2:
        parts.append("你对他的好感在消退")
    else:
        parts.append("你现在不太喜欢他")

    # 尊重度
    if respect > 2:
        parts.append("你尊敬他")
    elif respect < -2:
        parts.append("你有点看不起他的某些地方")

    # 吸引力
    if attractiveness > 2:
        parts.append("他身上有种吸引力让你想靠近")
    elif attractiveness < -1:
        parts.append("你对他的吸引力在下降")

    # 权力动态
    if power > 2:
        parts.append("你在这段关系里掌握主导权")
    elif power < -2:
        parts.append("你有点被他主导")

    base_narrative = "，".join(parts) + "。"

    # 近期事件调整
    if recent_conflicts_or_moments:
        if any("吵架" in m or "冲突" in m for m in recent_conflicts_or_moments):
            base_narrative += "最近有点别扭，有些地方还在缓和。"
        if any("和解" in m or "道歉" in m for m in recent_conflicts_or_moments):
            base_narrative += "不过他最近的态度改变了，让你对他的看法有所改善。"
        if any("特殊" in m or "珍贵" in m for m in recent_conflicts_or_moments):
            base_narrative += "最近的某个时刻让你觉得他真的在乎你。"

    return base_narrative


def stage_to_narrative(current_stage: str) -> str:
    """
    将 Knapp 阶段转换为叙事性描述。

    Knapp 阶段：initiating, experimenting, intensifying, integrating, bonding,
               differentiating, circumscribing, stagnating, avoiding, terminating
    """
    stage_narratives = {
        "initiating": "你们刚刚开始，还在互相试探阶段。",
        "experimenting": "你们在摸索彼此，看看能否继续深入。",
        "intensifying": "感情在加深，你们开始分享更多。",
        "integrating": "你们已经紧密结合，有了共同的身份和圈子。",
        "bonding": "你们之间有了深层的承诺和依赖。",
        "differentiating": "你们开始强调各自的独立性，有些分歧出现。",
        "circumscribing": "沟通在减少，你们各自的空间在扩大。",
        "stagnating": "关系停滞了，沟通越来越少，有点无聊。",
        "avoiding": "你们在主动或被动地回避彼此。",
        "terminating": "关系在结束，只是还没有正式划上句号。",
    }
    return stage_narratives.get(current_stage, f"你们的关系处于 {current_stage} 阶段。")


def convert_big_five_to_narrative(
    big_five: Dict[str, float],
    occupation: Optional[str] = None,
    background: Optional[str] = None,
) -> str:
    """
    将大五人格数值转换为叙事性人格描述。

    这个函数通常只在角色创建时调用一次，结果缓存到数据库。

    参数：
      big_five: {O, C, E, A, N}，每个都是 0~1
      occupation, background: 可选，增加描述的具体性

    返回：200-400字的人格叙述

    注意：这个函数最好由 LLM 生成（给定大五数值），而非纯规则。
          这里提供的是一个示例。实际应用中应该調用 LLM 一次性转换。
    """
    # 这里为了演示给出一个规则型的示例
    # 实际上你应该在项目初始化时调用 LLM，用类似这样的提示词：
    # "给定这个角色的大五人格数值（O={O}, C={C}, ...）和职业背景（{background}），
    #  写一段 200-300 字的叙事性人格描述，重点写矛盾、防御机制、社交模式。
    #  不要提任何数值或维度名称，只写具体的行为和性格特征。"

    O = big_five.get("O", 0.5)  # Openness 开放性
    C = big_five.get("C", 0.5)  # Conscientiousness 尽责性
    E = big_five.get("E", 0.5)  # Extraversion 外向性
    A = big_five.get("A", 0.5)  # Agreeableness 宜人性
    N = big_five.get("N", 0.5)  # Neuroticism 神经质

    parts = []

    # 外向性 + 宜人性 组合
    if E > 0.6 and A > 0.6:
        parts.append("你是那种走到哪儿都能跟人聊起来的人，热情，容易拉近距离。")
    elif E > 0.6 and A < 0.4:
        parts.append("你看起来很社牛，能侃能聊，但其实很挑人，对大多数人保持热络但有距离的态度。")
    elif E < 0.4 and A > 0.6:
        parts.append("你不太主动，但一旦有人靠近你就会很温暖。你是那种被动的温柔。")
    elif E < 0.4 and A < 0.4:
        parts.append("你比较独立，不太care别人的看法。跟陌生人相处时会有点冷。")

    # 神经质
    if N > 0.6:
        parts.append("你情绪波动大，开心的时候特别感染人，但低落的时候会突然消失。")
    elif N < 0.4:
        parts.append("你心态比较稳定，不容易被小事影响。")

    # 尽责性
    if C > 0.6:
        parts.append("你做事有计划，靠谱，说到做到。")
    elif C < 0.4:
        parts.append("你做事随性，答应的事情经常忘，但你也不觉得这是什么大问题。")

    # 开放性
    if O > 0.6:
        parts.append("你对新事物好奇，喜欢尝试，思维比较活跃。")
    elif O < 0.4:
        parts.append("你比较传统，习惯已有的方式，对陌生的东西有点警惕。")

    # 神经质影响的防御机制
    if N > 0.5 and A < 0.5:
        parts.append("面对冲突时你的第一反应是岔开话题或开玩笑糊弄过去，很少正面承认。")
    elif N > 0.5 and E > 0.6:
        parts.append("你不太会处理负面情绪，更倾向于用活跃来掩饰。")

    return " ".join(parts)


def convert_state_to_context_text(state: Dict[str, Any]) -> Dict[str, str]:
    """
    一次性转换所有动态状态为文本，供内心独白使用。

    返回字典，包含所有需要的文本化状态。
    """
    return {
        "pad_state": pad_to_state_text(
            state.get("pleasure", 0),
            state.get("arousal", 0),
            state.get("dominance", 0),
        ),
        "busy_text": busy_to_text(state.get("busy", 0.5)),
        "momentum_text": momentum_to_text(state.get("conversation_momentum", 0.5)),
        "relationship_narrative": relationship_to_narrative(
            state.get("relationship_state", {}),
            state.get("recent_relationship_events"),
        ),
        "stage_narrative": stage_to_narrative(state.get("current_stage", "experimenting")),
    }

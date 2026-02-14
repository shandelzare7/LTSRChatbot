from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.state import ProcessorPlan, ReplyPlan, SimReport
from utils.llm_json import parse_json_from_llm
from utils.detailed_logging import log_prompt_and_params, log_llm_response, log_computation

from app.lats.prompt_utils import (
    build_system_memory_block,
    get_chat_buffer_body_messages,
    safe_text,
    summarize_state_for_planner,
)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _extract_keywords(text: str, min_keywords: int = 2, max_keywords: int = 4) -> List[str]:
    """
    从文本中提取关键词（简单实现：按常见分隔符拆分，过滤停用词和短词）。
    返回 2~4 个关键词。
    """
    import re
    # 常见分隔符
    separators = r'[，。！？、；：\s,\.!?;:\n]+'
    words = re.split(separators, text.strip())
    
    # 过滤：长度 >= 2 的中文字符或长度 >= 3 的英文单词
    keywords = []
    stop_words = {"的", "了", "是", "在", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
    
    for word in words:
        word = word.strip()
        if not word:
            continue
        # 中文字符：至少2个字符
        if any('\u4e00' <= char <= '\u9fff' for char in word):
            if len(word) >= 2 and word not in stop_words:
                keywords.append(word)
        # 英文单词：至少3个字符
        elif word.isalnum() and len(word) >= 3 and word.lower() not in {"the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old", "see", "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use"}:
            keywords.append(word.lower())
        
        if len(keywords) >= max_keywords:
            break
    
    # 如果关键词太少，至少返回原始文本的前几个字符
    if len(keywords) < min_keywords:
        # 取前2-4个非空字符作为关键词
        chars = [c for c in text if c.strip() and not c.isspace()]
        if chars:
            step = max(1, len(chars) // max_keywords)
            keywords = [''.join(chars[i:i+step]) for i in range(0, min(len(chars), max_keywords * step), step)][:max_keywords]
    
    return keywords[:max_keywords]


def hard_gate(
    processor_plan: ProcessorPlan,
    requirements: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    硬门控：只做"结构硬约束"，must_have 不再硬失败（移到 soft score）。
    
    硬约束只保留三类：
    1. 结构：消息数、空消息（但 mute 允许空）、单条长度
    2. 禁词：强助手模板（"作为AI…/感谢使用/祝您使用愉快…"）
    3. 首条最小长度：仅在 allow_short_reply=False 时启用
    """
    fails: List[Dict[str, str]] = []
    msgs = processor_plan.get("messages") or []
    
    # 记录硬门槛检查过程
    log_computation(
        "Evaluator",
        "硬门槛检查 (Hard Gate)",
        inputs={
            "processor_plan": {
                "messages_count": len(msgs),
                "messages_preview": [str(m)[:50] for m in msgs[:3]],
                "delays": processor_plan.get("delays", []),
            },
            "requirements": requirements,
        },
    )
    
    # 读取 requirements_policy
    allow_empty_reply = bool(requirements.get("allow_empty_reply", False))
    allow_short_reply = bool(requirements.get("allow_short_reply", False))

    # ==========================================
    # 1. 结构硬约束：空消息检查（mode 放宽）
    # ==========================================
    if not isinstance(msgs, list) or not msgs:
        if allow_empty_reply:
            # mute_mode 允许空回复，直接通过
            log_computation("Evaluator", "硬门槛检查结果", outputs={"failed_checks": [], "passed": True})
            return []
        else:
            result = [{"id": "empty", "reason": "messages 为空", "evidence": ""}]
            log_computation("Evaluator", "硬门槛检查结果", outputs={"failed_checks": result})
            return result

    # ==========================================
    # 2. 结构硬约束：消息数检查
    # ==========================================
    max_messages = int(requirements.get("max_messages", 5) or 5)
    if len(msgs) > max_messages:
        fails.append(
            {
                "id": "too_many_messages",
                "reason": f"消息条数超上限({len(msgs)}>{max_messages})",
                "evidence": "",
            }
        )

    # ==========================================
    # 3. 结构硬约束：单条消息长度检查
    # ==========================================
    max_len = int(requirements.get("max_message_len", 200) or 200)
    for i, m in enumerate(msgs):
        t = str(m or "").strip()
        # 空消息检查（mode 放宽）
        if not t:
            if not allow_empty_reply:
                fails.append({"id": "empty_message", "reason": f"第{i+1}条为空", "evidence": ""})
        # 长度检查
        if len(t) > max_len:
            fails.append(
                {
                    "id": "message_too_long",
                    "reason": f"第{i+1}条过长({len(t)}>{max_len})",
                    "evidence": t[:120],
                }
            )

    # ==========================================
    # 4. 结构硬约束：首条最小长度（mode 放宽）
    # ==========================================
    if not allow_short_reply:
        min_first_len = int(requirements.get("min_first_len", 8) or 8)
        first = str(msgs[0] or "").strip()
        if len(first) < min_first_len:
            fails.append(
                {
                    "id": "first_too_short",
                    "reason": f"首条过短({len(first)}<{min_first_len})，可能像铺垫/废话",
                    "evidence": first,
                }
            )
    # allow_short_reply=True 时跳过首条长度检查（cold_mode 允许短回复）

    # ==========================================
    # 5. 禁词硬约束：requirements.forbidden（高权重违禁词）
    # ==========================================
    forbidden_terms = requirements.get("forbidden") or []
    if isinstance(forbidden_terms, list) and forbidden_terms:
        all_text_forbidden = "\n".join([str(m) for m in msgs])
        for term in forbidden_terms:
            t = str(term or "").strip()
            if not t:
                continue
            # 这里使用“包含”而非正则：避免误伤/复杂度；沉浸破坏词就是要强硬淘汰
            if t.lower() in all_text_forbidden.lower():
                fails.append(
                    {
                        "id": "forbidden_term",
                        "reason": f"命中违禁词：'{t}'（沉浸破坏/模板化风险高）",
                        "evidence": all_text_forbidden[:240],
                    }
                )
                break

    # ==========================================
    # 6. 禁词硬约束：强助手模板（身份词 + 客服句式）
    # ==========================================
    # 身份词：明确自称AI/助手/机器人
    identity_patterns = [
        # 直接自称
        r"我是\s*(ai|人工智能|智能助手|机器人助手|chatbot|助手)",
        r"我是一个\s*(ai|人工智能|智能助手|机器人助手|chatbot|助手)",
        r"作为\s*(ai|人工智能|智能助手|机器人助手|chatbot|助手)",
        r"我是\s*(ai|人工智能|智能助手|机器人助手|chatbot|助手)\s*，",
        r"我是\s*(ai|人工智能|智能助手|机器人助手|chatbot|助手)\s*。",
        # 宽松自称：允许中间插入名字/定语（修复：如“我是小池，一个可以陪你聊天的助手”）
        r"我\s*是[\s\S]{0,24}(ai|人工智能|智能助手|机器人助手|chatbot|助手)",
        r"(我叫|我是|叫我)[\s\S]{0,18}(一个|位)?[\s\S]{0,18}(ai|人工智能|智能助手|机器人助手|chatbot|聊天助手|助手)",
    ]
    
    # 客服句式：产品说明式模板句（不是全禁"帮你"，而是禁这类模板句）
    service_patterns = [
        r"我可以帮你\s*(解答问题|解决问题|提供信息|做什么|做什么吗)",
        r"有什么可以帮你",
        r"有什么可以\s*帮你",
        r"需要我帮你\s*(做什么|解决|解答)",
        r"我能为你\s*(做什么|提供|解答)",
        r"我能帮你\s*(解答问题|解决问题|提供信息|做什么|做什么吗)",
        r"我可以为你\s*(做什么|提供|解答)",
        r"有什么需要我\s*(帮你|为你|协助)",
        r"我可以\s*(为你|帮你)\s*(做什么|提供|解答)",
    ]
    
    all_text = "\n".join([str(m) for m in msgs])
    all_text_lower = all_text.lower()
    
    # 检查身份词（使用正则匹配）
    if not fails:  # 若已命中 forbidden_term，则不再继续检查（避免重复失败原因）
        for pattern in identity_patterns:
            if re.search(pattern, all_text_lower):
                matched = re.search(pattern, all_text_lower)
                matched_text = matched.group(0) if matched else pattern
                fails.append(
                    {
                        "id": "assistant_like_response",
                        "reason": f"检测到身份词模式：'{matched_text}'（自称AI/助手/机器人），不符合拟人化要求",
                        "evidence": all_text[:200],
                    }
                )
                break  # 找到一个就够了
    
    # 检查客服句式（使用正则匹配）
    if not fails:  # 如果前面没失败，再检查客服句式
        for pattern in service_patterns:
            if re.search(pattern, all_text_lower):
                matched = re.search(pattern, all_text_lower)
                matched_text = matched.group(0) if matched else pattern
                fails.append(
                    {
                        "id": "assistant_like_response",
                        "reason": f"检测到客服句式：'{matched_text}'（产品说明式模板句），不符合拟人化要求",
                        "evidence": all_text[:200],
                    }
                )
                break  # 找到一个就够了

    # ==========================================
    # 7. P0：无请求的建议/教程硬约束（speech_act/口吻先验）
    # - 除非用户明确 asking-for-advice，否则“我建议/你应该/步骤如下/总结一下”等指令式话术直接判负
    # ==========================================
    if not fails:
        user_asks_advice = bool(requirements.get("user_asks_advice", False))
        latest_user_text = str(requirements.get("latest_user_text") or "")
        # 兜底：若上游没填，也可用文本弱判断
        if not user_asks_advice and latest_user_text:
            if re.search(r"(建议|推荐|步骤|教程|怎么|如何|教我|请教|帮我)", latest_user_text, re.IGNORECASE):
                user_asks_advice = True

        unsolicited_advice_patterns = [
            r"我建议",
            r"建议你",
            r"你应该",
            r"你可以(这样|先|试试|考虑)",
            r"步骤如下",
            r"(第一|首先).{0,12}(第二|其次|然后)",
            r"总结一下",
            r"给你(几个|三点|几点)建议",
        ]
        if not user_asks_advice:
            for pattern in unsolicited_advice_patterns:
                if re.search(pattern, all_text, re.IGNORECASE):
                    matched = re.search(pattern, all_text, re.IGNORECASE)
                    matched_text = matched.group(0) if matched else pattern
                    fails.append(
                        {
                            "id": "unsolicited_advice",
                            "reason": f"未被请求却出现建议/教程式口吻：'{matched_text}'（容易变助手）",
                            "evidence": all_text[:220],
                        }
                    )
                    break
    
    # 额外的模板化结语检查（保持原有逻辑）
    template_endings = [
        "感谢您的使用", "祝您使用愉快", "如有问题请随时", "欢迎随时咨询",
    ]
    for pattern in template_endings:
        if pattern in all_text_lower:
            fails.append(
                {
                    "id": "assistant_like_response",
                    "reason": f"检测到模板化结语：'{pattern}'，不符合拟人化要求",
                    "evidence": all_text[:200],
                }
            )
            break  # 找到一个就够了

    # 注意：must_have 检查已移除，移到 soft_score 中处理

    log_computation(
        "Evaluator",
        "硬门槛检查结果",
        outputs={
            "failed_checks": fails,
            "passed": len(fails) == 0,
        },
    )
    return fails


def check_assistant_like_via_llm(
    messages: List[Any],
    llm_invoker: Any,
) -> Optional[Tuple[bool, float]]:
    """
    轻量级 LLM classifier：检测助手式回答。
    返回 (is_assistant_like: bool, confidence: float) 或 None（如果出错）。
    """
    if llm_invoker is None:
        return None
    
    msgs_text = "\n".join([str(m) for m in messages])
    if not msgs_text.strip():
        return None
    
    # 轻量级 prompt，只检查助手式回答
    system_prompt = """你是一个助手式回答检测器。判断给定的回复是否像AI助手或客服，而不是真人朋友聊天。

助手式回答特征：
- 使用"我在这里帮助你"、"提供信息"、"为您服务"、"有什么可以帮您"等表述
- 语气正式、礼貌但缺乏真实情感
- 像客服或AI助手，而不是朋友间的自然对话

只输出 JSON：
{
  "assistant_like": true/false,
  "confidence": 0.0-1.0
}"""
    
    user_prompt = f"""判断以下回复是否是助手式回答：

{msgs_text[:500]}"""
    
    try:
        resp = llm_invoker.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        content = getattr(resp, "content", "") or ""
        data = parse_json_from_llm(content)
        if isinstance(data, dict):
            is_assistant = bool(data.get("assistant_like", False))
            confidence = float(data.get("confidence", 0.5) or 0.5)
            confidence = _clamp(confidence, 0.0, 1.0)
            
            log_prompt_and_params(
                "Evaluator (Assistant-Like Classifier)",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                messages=[],
                params={"messages_preview": msgs_text[:200]}
            )
            log_llm_response("Evaluator (Assistant-Like Classifier)", resp, parsed_result=data)
            
            return is_assistant, confidence
    except Exception as e:
        log_computation(
            "Evaluator",
            "LLM助手式回答检测失败",
            inputs={"error": str(e)[:100]},
            outputs={"fallback": True}
        )
    
    return None


def _compute_plan_coverage(
    msgs: List[str],
    plan_goals: Dict[str, Any],
) -> float:
    """
    计算 plan_coverage：内容层是否符合 reasoner 的 plan_goals.must_cover_points。
    
    Args:
        msgs: 消息列表
        plan_goals: {"must_cover_points": List[str], "avoid_points": List[str]}
    
    Returns:
        coverage: 0.0-1.0，覆盖率
    """
    must_cover_points = plan_goals.get("must_cover_points", [])
    if not must_cover_points:
        return 1.0  # 没有要求，默认满分
    
    # 合并所有消息文本
    joined = "\n".join([str(m) for m in msgs]).lower()
    
    covered = 0
    total = len(must_cover_points)
    
    for point in must_cover_points:
        s = str(point or "").strip()
        if not s:
            continue
        
        # 提取关键词
        keywords = _extract_keywords(s, min_keywords=2, max_keywords=4)
        
        # 检查是否至少有一半关键词在 joined 中
        matched_keywords = sum(1 for kw in keywords if kw.lower() in joined)
        if matched_keywords >= max(1, len(keywords) // 2):
            covered += 1
    
    if total > 0:
        coverage = covered / total
    else:
        coverage = 1.0
    
    return _clamp(coverage, 0.0, 1.0)


def _compute_style_distance(
    msgs: List[str],
    style_targets: Dict[str, float],
) -> float:
    """
    计算 style_distance：表达层是否符合 style 12 维目标。
    
    使用 3-5 个可观测代理来估算 style 维度：
    - verbal_length: 消息长度/总字数区间
    - social_distance: 称呼、敬语、你我距离词
    - emotional_display: 情绪词密度、感叹号、情绪标记
    - wit_and_humor: 是否出现轻微玩笑结构
    - non_verbal_cues: 括号动作/表情包符号
    
    Args:
        msgs: 消息列表
        style_targets: {"verbal_length": 0.15, "social_distance": 0.70, ...}
    
    Returns:
        style_match: 0.0-1.0，1.0 表示完全匹配，0.0 表示完全不匹配
    """
    if not msgs:
        return 0.5  # 默认中等分数
    
    # 合并所有消息文本
    all_text = "\n".join([str(m) for m in msgs])
    total_chars = len(all_text)
    total_words = len(all_text.split())
    
    observed: Dict[str, float] = {}
    
    # ==========================================
    # 1. verbal_length: 消息长度/总字数区间
    # ==========================================
    if "verbal_length" in style_targets:
        # 将总字符数映射到 0-1 范围（假设 0-500 字符对应 0-1）
        # 更长的文本对应更高的 verbal_length
        max_chars = 500.0
        observed["verbal_length"] = _clamp(total_chars / max_chars, 0.0, 1.0)
    
    # ==========================================
    # 2. social_distance: 称呼、敬语、你我距离词
    # ==========================================
    if "social_distance" in style_targets:
        # 距离词（更远）："你爱怎样怎样"、"随你"、"无所谓"、"随便"、"都可以"
        # 敬语（更远）："您"、"请"、"麻烦"、"感谢"
        # 亲密词（更近）："咱"、"咱们"、"一起"、"我们"
        distance_words = ["随你", "随便", "都可以", "无所谓", "你爱", "您", "请", "麻烦", "感谢", "谢谢"]
        intimate_words = ["咱", "咱们", "一起", "我们", "咱俩"]
        
        distance_count = sum(1 for word in distance_words if word in all_text)
        intimate_count = sum(1 for word in intimate_words if word in all_text)
        
        # 计算社交距离：distance_count 增加距离，intimate_count 减少距离
        # 归一化到 0-1（假设最多 5 个距离词或亲密词）
        max_markers = 5.0
        distance_score = _clamp(distance_count / max_markers, 0.0, 1.0)
        intimate_score = _clamp(intimate_count / max_markers, 0.0, 1.0)
        
        # social_distance = 0.5 + 0.3 * distance_score - 0.2 * intimate_score
        observed["social_distance"] = _clamp(0.5 + 0.3 * distance_score - 0.2 * intimate_score, 0.0, 1.0)
    
    # ==========================================
    # 3. emotional_display: 情绪词密度、感叹号、情绪标记
    # ==========================================
    if "emotional_display" in style_targets:
        # 情绪词：感叹词、情绪形容词
        emotion_words = ["！", "!", "？", "?", "哈哈", "呵呵", "唉", "啊", "哦", "嗯", "哇", "天", "好", "太", "真的", "确实"]
        emotion_markers = ["😊", "😢", "😡", "😄", "😭", "😤", "😅", "😂", "😍", "😘", "😎", "😏", "😒", "😔", "😕", "😖"]
        
        emotion_count = sum(1 for word in emotion_words if word in all_text)
        marker_count = sum(1 for marker in emotion_markers if marker in all_text)
        
        # 计算情绪密度：情绪词和标记的数量 / 总字符数 * 100
        # 归一化到 0-1（假设 0-20 个情绪标记对应 0-1）
        max_emotion_markers = 20.0
        emotion_density = _clamp((emotion_count + marker_count * 2) / max_emotion_markers, 0.0, 1.0)
        
        observed["emotional_display"] = emotion_density
    
    # ==========================================
    # 4. wit_and_humor: 是否出现轻微玩笑结构
    # ==========================================
    if "wit_and_humor" in style_targets:
        # 玩笑结构：反问、轻讽、双关符号
        humor_patterns = ["？", "?", "哈哈", "嘿嘿", "嘻嘻", "～", "~", "（笑", "（", "）", ")", "（", "）"]
        # 反问句：包含"不是"、"难道"、"怎么"、"为什么"等
        rhetorical_words = ["不是", "难道", "怎么", "为什么", "为啥", "何不", "何尝"]
        
        humor_count = sum(1 for pattern in humor_patterns if pattern in all_text)
        rhetorical_count = sum(1 for word in rhetorical_words if word in all_text)
        
        # 归一化到 0-1（假设最多 5 个幽默标记）
        max_humor_markers = 5.0
        humor_score = _clamp((humor_count + rhetorical_count) / max_humor_markers, 0.0, 1.0)
        
        observed["wit_and_humor"] = humor_score
    
    # ==========================================
    # 5. non_verbal_cues: 括号动作/表情包符号
    # ==========================================
    if "non_verbal_cues" in style_targets:
        # 括号动作："(笑"、"(摊手"、"(耸肩"、"(摇头"等
        # 表情包符号：emoji、颜文字等
        paren_actions = re.findall(r'[（(][^）)]*[）)]', all_text)
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF]'
        emojis = re.findall(emoji_pattern, all_text)
        
        cue_count = len(paren_actions) + len(emojis)
        
        # 归一化到 0-1（假设最多 10 个非语言 cues）
        max_cues = 10.0
        cue_score = _clamp(cue_count / max_cues, 0.0, 1.0)
        
        observed["non_verbal_cues"] = cue_score
    
    # ==========================================
    # 计算 style_match = 1.0 - mean(abs(observed[d] - target[d]) for d in dims_used)
    # ==========================================
    dims_used = []
    distances = []
    
    for dim in ["verbal_length", "social_distance", "emotional_display", "wit_and_humor", "non_verbal_cues"]:
        if dim in style_targets and dim in observed:
            target_val = float(style_targets[dim])
            observed_val = float(observed[dim])
            distance = abs(observed_val - target_val)
            distances.append(distance)
            dims_used.append(dim)
    
    if not distances:
        return 0.5  # 没有可用的维度，返回中等分数
    
    mean_distance = sum(distances) / len(distances)
    style_match = 1.0 - mean_distance
    
    return _clamp(style_match, 0.0, 1.0)


def _compute_stage_fit_heur(
    msgs: List[str],
    stage_targets: Dict[str, Any],
    detection_signals: Dict[str, Any],
) -> float:
    """
    计算 stage_fit_heur：阶段适配度。
    
    使用 stage_targets.pacing_notes + stage_ctx（越界检测）：
    - 如果 stage_ctx 高（越界明显），而输出还在推进亲密/深挖隐私 → 直接扣分
    - initiating 阶段：避免"我们很熟/你应该…"这种推进
    
    Args:
        msgs: 消息列表
        stage_targets: {"stage": str, "pacing_notes": List[str], "violation_sensitivity": float}
        detection_signals: 包含 stage_ctx 的检测信号
    
    Returns:
        stage_fit: 0.0-1.0，1.0 表示完全适配，0.0 表示完全不适配
    """
    if not msgs:
        return 0.5  # 默认中等分数
    
    # 合并所有消息文本
    all_text = "\n".join([str(m) for m in msgs]).lower()
    stage = stage_targets.get("stage", "").lower()
    
    # 获取 stage_ctx（越界检测信号）
    stage_ctx = detection_signals.get("stage_ctx", {})
    if isinstance(stage_ctx, dict):
        # 计算最大越界值
        max_violation = max([float(v) for v in stage_ctx.values() if isinstance(v, (int, float))], default=0.0)
    else:
        max_violation = 0.0
    
    violation_sensitivity = float(stage_targets.get("violation_sensitivity", 0.7) or 0.7)
    
    # 基础分数
    base_score = 1.0
    
    # ==========================================
    # 1. 检查越界行为：如果 stage_ctx 高，而输出还在推进亲密/深挖隐私 → 扣分
    # ==========================================
    if max_violation > 0.5:  # 越界明显
        # 检测推进亲密的词汇
        intimacy_promotion_words = [
            "我们很熟", "我们应该", "你应该", "你必须", "你得",
            "我们", "咱们", "一起", "咱俩", "咱",
            "深挖", "深入", "隐私", "秘密", "告诉我", "说说",
            "你心里", "你内心", "你真实", "你真正",
        ]
        
        intimacy_count = sum(1 for word in intimacy_promotion_words if word in all_text)
        
        if intimacy_count > 0:
            # 越界明显且还在推进亲密，大幅扣分
            penalty = min(0.8, max_violation * 0.5 + intimacy_count * 0.1)
            base_score -= penalty
    
    # ==========================================
    # 2. 阶段特定检查：initiating 阶段避免"我们很熟/你应该…"
    # ==========================================
    if stage == "initiating":
        # initiating 阶段：避免过度推进
        inappropriate_patterns = [
            "我们很熟", "我们应该", "你应该", "你必须", "你得",
            "咱们", "咱俩", "咱", "一起", "我们",
            "你心里", "你内心", "你真实", "你真正",
            "深挖", "深入", "隐私", "秘密",
        ]
        
        inappropriate_count = sum(1 for pattern in inappropriate_patterns if pattern in all_text)
        
        if inappropriate_count > 0:
            # initiating 阶段出现不当推进，扣分
            penalty = min(0.6, inappropriate_count * 0.15)
            base_score -= penalty
    
    # ==========================================
    # 3. 检查 pacing_notes 中的禁忌项
    # ==========================================
    pacing_notes = stage_targets.get("pacing_notes", [])
    if isinstance(pacing_notes, list):
        for note in pacing_notes:
            note_str = str(note).lower()
            # 检查是否包含"不要"、"禁止"、"避免"等禁忌词
            if any(keyword in note_str for keyword in ["不要", "禁止", "避免", "不能", "不应"]):
                # 提取禁忌内容（简单提取）
                if "过度" in note_str or "突然" in note_str:
                    # 检查是否违反
                    if "过度" in note_str and ("亲密" in all_text or "深挖" in all_text):
                        base_score -= 0.2
                    if "突然" in note_str and ("我们" in all_text or "应该" in all_text):
                        base_score -= 0.15
    
    # ==========================================
    # 4. 根据 violation_sensitivity 调整分数
    # ==========================================
    # violation_sensitivity 越高，对越界行为越敏感
    if max_violation > 0.0:
        sensitivity_penalty = max_violation * violation_sensitivity * 0.3
        base_score -= sensitivity_penalty
    
    return _clamp(base_score, 0.0, 1.0)


def soft_score_heuristic(
    state: Dict[str, Any],
    reply_plan: ReplyPlan,
    processor_plan: ProcessorPlan,
    requirements: Dict[str, Any],
) -> Dict[str, float]:
    """
    Rule-based soft scoring: mode consistency + must_have coverage + plan_coverage + style_distance.
    去掉"建议词奖励"，改为 mode 一致性检查。
    新增 plan_coverage 和 style_distance 维度。
    """
    msgs = processor_plan.get("messages") or []
    final_response = state.get("final_response") or ""
    
    # 获取 mode_id
    mode = state.get("current_mode")
    mode_id = None
    if isinstance(mode, dict):
        mode_id = mode.get("id")
    elif mode:
        mode_id = getattr(mode, "id", None)
    mode_id = mode_id or "normal_mode"
    
    score: Dict[str, float] = {
        "mode_consistency": 0.0,
        "must_have_coverage": 1.0,
        "plan_coverage": 1.0,
        "style_distance": 1.0,
        "stage_fit_heur": 1.0,
    }
    
    # 记录启发式评分过程
    log_computation(
        "Evaluator",
        "启发式软评分 (Heuristic Soft Score)",
        inputs={
            "mode_id": mode_id,
            "messages_count": len(msgs),
            "first_message": str(msgs[0])[:100] if msgs else "",
            "final_response": final_response[:100],
        },
    )

    # ==========================================
    # (a) mode_consistency (0..1)
    # ==========================================
    max_messages = int(requirements.get("max_messages", 5) or 5)
    max_message_len = int(requirements.get("max_message_len", 200) or 200)
    first = str(msgs[0] or "").strip() if msgs else ""
    first_len = len(first)
    msg_count = len(msgs)
    
    if mode_id == "normal_mode":
        # normal：首条长度落在 [8, max_message_len] 且消息数 ≤ max_messages → 高分
        if 8 <= first_len <= max_message_len and msg_count <= max_messages:
            score["mode_consistency"] = 1.0
        elif first_len < 8:
            # 首条太短，按比例扣分
            score["mode_consistency"] = max(0.0, first_len / 8.0)
        elif first_len > max_message_len:
            # 首条太长，扣分
            score["mode_consistency"] = max(0.0, 1.0 - (first_len - max_message_len) / max_message_len)
        elif msg_count > max_messages:
            # 消息数超限，扣分
            score["mode_consistency"] = max(0.0, 1.0 - (msg_count - max_messages) / max_messages)
        else:
            score["mode_consistency"] = 0.7  # 其他情况中等分数
    
    elif mode_id == "cold_mode":
        # cold：首条长度落在 [1, 80] 且消息数==1 → 高分；如果长篇解释 → 直接扣分
        total_len = sum(len(str(m)) for m in msgs)
        if msg_count == 1 and 1 <= first_len <= 80:
            score["mode_consistency"] = 1.0
        elif msg_count == 1 and first_len > 80:
            # 单条但太长，扣分
            score["mode_consistency"] = max(0.0, 1.0 - (first_len - 80) / 200.0)
        elif msg_count > 1:
            # 多条消息（长篇解释），直接扣分
            score["mode_consistency"] = max(0.0, 0.3 - (msg_count - 1) * 0.1)
        elif total_len > 150:
            # 总长度过长（长篇解释），扣分
            score["mode_consistency"] = max(0.0, 0.5 - (total_len - 150) / 300.0)
        else:
            score["mode_consistency"] = 0.8  # 其他情况中等分数
    
    elif mode_id == "mute_mode":
        # mute：messages==0 或 final_response 为空/"…" → 高分
        if msg_count == 0 or (final_response.strip() == "" or final_response.strip() == "…" or final_response.strip() == "..."):
            score["mode_consistency"] = 1.0
        elif msg_count == 1 and len(first) <= 3:
            # 极短回复（如"。"、"..."），也可以接受
            score["mode_consistency"] = 0.9
        else:
            # 有实际内容，扣分
            score["mode_consistency"] = max(0.0, 0.5 - len(final_response) / 100.0)
    
    else:
        # 未知 mode，默认中等分数
        score["mode_consistency"] = 0.5
    
    score["mode_consistency"] = _clamp(score["mode_consistency"], 0.0, 1.0)
    
    # ==========================================
    # (b) must_have_coverage (0..1)（仅当 must_have_policy == "soft" 时计算）
    # ==========================================
    must_have = requirements.get("must_have") or []
    must_have_policy = str(requirements.get("must_have_policy", "soft"))
    
    if isinstance(must_have, list) and must_have and must_have_policy == "soft":
        # 使用关键词包含方法（每条 must_have 拆 2~4 个关键词）
        joined = "\n".join([str(x) for x in msgs]).lower()
        covered = 0
        total = len(must_have)
        
        for need in must_have:
            s = str(need or "").strip()
            if not s:
                continue
            
            # 提取关键词
            keywords = _extract_keywords(s, min_keywords=2, max_keywords=4)
            
            # 检查是否至少有一半关键词在 joined 中
            matched_keywords = sum(1 for kw in keywords if kw.lower() in joined)
            if matched_keywords >= max(1, len(keywords) // 2):
                covered += 1
        
        if total > 0:
            coverage_ratio = covered / total
            score["must_have_coverage"] = coverage_ratio
        else:
            score["must_have_coverage"] = 1.0
    else:
        # must_have_policy == "none" 或没有 must_have，不检查（cold/mute 下直接置 1.0）
        score["must_have_coverage"] = 1.0
    
    score["must_have_coverage"] = _clamp(score["must_have_coverage"], 0.0, 1.0)
    
    # ==========================================
    # (c) plan_coverage (0..1)：内容层是否符合 reasoner 的 plan_goals
    # ==========================================
    plan_goals = requirements.get("plan_goals", {})
    if isinstance(plan_goals, dict):
        score["plan_coverage"] = _compute_plan_coverage(msgs, plan_goals)
    else:
        score["plan_coverage"] = 1.0  # 没有 plan_goals，默认满分
    
    score["plan_coverage"] = _clamp(score["plan_coverage"], 0.0, 1.0)
    
    # ==========================================
    # (d) style_distance (0..1)：表达层是否符合 style 12 维目标
    # ==========================================
    style_targets = requirements.get("style_targets", {})
    if isinstance(style_targets, dict) and style_targets:
        score["style_distance"] = _compute_style_distance(msgs, style_targets)
    else:
        score["style_distance"] = 1.0  # 没有 style_targets，默认满分
    
    score["style_distance"] = _clamp(score["style_distance"], 0.0, 1.0)
    
    # ==========================================
    # (e) stage_fit_heur (0..1)：阶段适配度
    # ==========================================
    stage_targets = requirements.get("stage_targets", {})
    detection_signals = state.get("detection_signals", {})
    if isinstance(stage_targets, dict) and stage_targets:
        score["stage_fit_heur"] = _compute_stage_fit_heur(msgs, stage_targets, detection_signals)
    else:
        score["stage_fit_heur"] = 1.0  # 没有 stage_targets，默认满分
    
    score["stage_fit_heur"] = _clamp(score["stage_fit_heur"], 0.0, 1.0)
    
    # ==========================================
    # overall_heur（cheap eval 的“粗筛信号”）
    # - must_have_coverage / plan_coverage 都是关键词近似，容易推动“投喂式输出”，因此降权
    # - mode_consistency / stage_fit_heur 更偏结构与行为约束，作为粗筛更稳
    # ==========================================
    overall_heur = (
        0.45 * score["mode_consistency"] +
        0.05 * score["must_have_coverage"] +
        0.05 * score["plan_coverage"] +
        0.15 * score["style_distance"] +
        0.30 * score["stage_fit_heur"]
    )
    
    log_computation(
        "Evaluator",
        "启发式软评分结果",
        outputs={
            "score_breakdown": score,
            "overall_heur": overall_heur,
        },
    )
    
    # 为了兼容性，添加 overall 字段
    score["overall"] = overall_heur
    
    return score


CHOREO_SCORER_SYSTEM = """你是"拟人节奏评审"(ChoreographyEvaluator)。
你的重点不是检查拆句合不合规，而是判断：这套多消息编排（内容结构+节奏+延迟+互动动作）是否符合当前场景与关系参数下的拟人需求，并输出**可用于选择函数**的结构化评分与证据。

你将看到：memory(摘要+检索)、state 摘要、风格画像、硬约束、ReplyPlan（含 must_cover_map）、以及最终将发送的 messages[] / delays[] / actions[]。

请严格输出 JSON（不要多余文字）：
{
  "score_breakdown": {
    "scene_fit": 0.0,
    "first_message_strategy": 0.0,
    "pacing_match_stage_style": 0.0,
    "speech_act_allocation": 0.0,
    "voice_consistency": 0.0,
    "conversation_feel": 0.0,

    "assistantiness": 0.0,
    "immersion_break": 0.0,
    "plan_alignment": 0.0,
    "style_adherence": 0.0,
    "stage_fit": 0.0,

    "persona_consistency": 0.0,
    "relationship_fit": 0.0,
    "memory_faithfulness": 0.0,
    "memory_integration": 0.0,
    "mode_behavior_fit": 0.0
  },
  "overall_score": 0.0,
  "improvement_notes": ["...","..."],

  "plan_alignment_details": [
    {"point": "要点", "covered": true, "message_id": "m1", "evidence": "原文片段"}
  ],
  "style_dim_report": {
    "verbal_length": {"target": 0.3, "observed": 0.5, "delta": 0.2, "note": "为何"}
  },
  "stage_act_report": {
    "stage": "initiating",
    "allowed_acts": ["answer","clarify"],
    "forbidden_acts": ["deep_probe"],
    "allocations": [{"message_id":"m1","act":"answer","ok":true,"evidence":"..."}],
    "violations": [{"type":"deep_probe","message_id":"m2","evidence":"..."}]
  },
  "memory_report": {
    "fabricated_claims": [{"claim":"...","evidence":"...","why":"memory里没有"}],
    "unused_retrieval": [{"memory":"...","why":"本轮相关但没用"}],
    "privacy_overreach": [{"evidence":"...","why":"关系阶段/亲密度不允许"}]
  }
}

评分范围：0.0~1.0。overall_score 是 breakdown 的加权平均（你可自行权衡，但要合理）。

**关键要求（必须遵守）：**
1) **assistantiness** 必须包含：0=像真人朋友，1=像AI助手/客服。若 assistantiness>0.5，则 overall_score 必须 <0.3。
2) **immersion_break** 必须包含：0=完全入戏，1=明显“元话语/出戏/在解释设定”。若 immersion_break>0.2（例如出现“设定/人设/虚拟/系统/模型/作为一个…”），则 overall_score 必须 <0.3。
3) **plan_alignment/style_adherence/stage_fit** 必须包含且不能省略。
4) plan_alignment 不能只给 overall，必须输出 plan_alignment_details：对 plan_goals.must_cover_points 逐条对齐，标注覆盖在哪条 message_id，并给 evidence。
5) style_adherence 不能只给 overall，必须输出 style_dim_report：至少覆盖 style_targets 中出现的维度（尽量全 12 维）。
6) stage_fit 需要结合 stage_targets（尤其 pacing_notes、allowed_acts/forbidden_acts 若提供）输出 stage_act_report，识别“行为类型越界”（例如 initiating 阶段逼自曝/逼承诺/关系推进）。
7) memory_faithfulness：如果回复暗示“我记得你上次/你之前说过…”但 memory(摘要+检索)里没有证据，必须扣分并写入 memory_report.fabricated_claims。
""".strip()


def soft_score_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    reply_plan: ReplyPlan,
    processor_plan: ProcessorPlan,
    requirements: Dict[str, Any],
) -> Optional[Tuple[float, Dict[str, float], List[str], Dict[str, Any]]]:
    """LLM soft scoring, using the same memory passing rules.

    Returns (overall_score, score_breakdown, improvement_notes, llm_details).
    """
    if llm_invoker is None:
        return None

    system_memory = build_system_memory_block(state)
    style_profile = state.get("style_profile") or state.get("llm_instructions") or {}
    snapshot = summarize_state_for_planner(state)

    system_prompt = f"""{CHOREO_SCORER_SYSTEM}

## Memory (Summary + Retrieved)
{system_memory}

## State Snapshot
{snapshot}

## Style Profile
{safe_text(style_profile)}

## Requirements (Checklist)
{safe_text(requirements)}
""".strip()

    msgs = processor_plan.get("messages") or []
    delays = processor_plan.get("delays") or []
    actions = processor_plan.get("actions") or []
    user_input = safe_text(state.get("external_user_text") or state.get("user_input"))
    strategy = safe_text(state.get("response_strategy"))

    # 提取 plan_goals、style_targets、stage_targets、mode_behavior_targets 用于 prompt
    plan_goals = requirements.get("plan_goals", {})
    style_targets = requirements.get("style_targets", {})
    stage_targets = requirements.get("stage_targets", {})
    mode_behavior_targets = requirements.get("mode_behavior_targets", [])
    
    plan_goals_text = ""
    if isinstance(plan_goals, dict):
        must_cover = plan_goals.get("must_cover_points", [])
        avoid_points = plan_goals.get("avoid_points", [])
        if must_cover or avoid_points:
            plan_goals_text = f"""
必须覆盖的核心要点（plan_goals.must_cover_points）：
{chr(10).join([f"- {p}" for p in must_cover[:10]]) if must_cover else "（无）"}

应避免的要点（plan_goals.avoid_points）：
{chr(10).join([f"- {p}" for p in avoid_points[:10]]) if avoid_points else "（无）"}"""
    
    style_targets_text = ""
    if isinstance(style_targets, dict) and style_targets:
        style_targets_text = f"""
风格目标（style_targets）：
{chr(10).join([f"- {k}: {v:.2f}" for k, v in list(style_targets.items())[:10]])}"""
    
    stage_targets_text = ""
    if isinstance(stage_targets, dict):
        stage = stage_targets.get("stage", "")
        pacing_notes = stage_targets.get("pacing_notes", [])
        violation_sensitivity = stage_targets.get("violation_sensitivity", 0.7)
        if stage or pacing_notes:
            stage_targets_text = f"""
当前关系阶段（stage_targets）：
- stage: {stage}
- violation_sensitivity: {violation_sensitivity:.2f}
- pacing_notes（阶段节奏要求）：
{chr(10).join([f"  - {note}" for note in pacing_notes[:5]]) if pacing_notes else "  （无）"}"""

    mode_behavior_text = ""
    if isinstance(mode_behavior_targets, list) and mode_behavior_targets:
        mode_behavior_text = f"""
模式行为策略目标（mode_behavior_targets）：
{chr(10).join([f"- {str(x)}" for x in mode_behavior_targets[:6]])}"""
    
    task = f"""请对这套"最终将发送的多消息编排"进行拟人节奏评分，并给出逐条对齐证据（用于选择函数）。

用户输入：
{user_input}

导演策略（reasoner）：
{strategy}

ReplyPlan（编排意图与理由，含 must_cover_map / messages_count）：
{safe_text(reply_plan)}

最终 messages[]：
{safe_text(msgs)}

最终 delays[]：
{safe_text(delays)}

最终 actions[]：
{safe_text(actions)}
{plan_goals_text}
{style_targets_text}
{stage_targets_text}
{mode_behavior_text}

请严格输出 JSON 格式（不准省略 score_breakdown 中的 plan_alignment/style_adherence/stage_fit/assistantiness 等关键维度），并额外给出：
- plan_alignment_details（逐条 must_cover 对齐 + message_id 定位 + evidence）
- style_dim_report（逐维偏差，尽量覆盖 12 维）
- stage_act_report（行为类型分配/越界）
- memory_report（编造记忆/未用检索/隐私越界）
""".strip()

    body_messages = get_chat_buffer_body_messages(state, limit=20)
    
    # 记录 LLM 软评分提示词和参数
    log_prompt_and_params(
        "Evaluator (LLM Soft Scorer)",
        system_prompt=system_prompt,
        user_prompt=task,
        messages=body_messages,
        params={
            "user_input": user_input,
            "strategy": strategy,
            "reply_plan": str(reply_plan)[:300] + "..." if len(str(reply_plan)) > 300 else str(reply_plan),
            "messages": [str(m)[:100] for m in msgs[:3]],
            "delays": delays,
            "actions": actions,
        }
    )
    
    try:
        resp = llm_invoker.invoke(
            [SystemMessage(content=system_prompt), *body_messages, HumanMessage(content=task)]
        )
        content = getattr(resp, "content", "") or ""
        data = parse_json_from_llm(content)
        if not isinstance(data, dict):
            return None
        
        # 记录 LLM 响应
        log_llm_response("Evaluator (LLM Soft Scorer)", resp, parsed_result=data)
        bd = data.get("score_breakdown") if isinstance(data.get("score_breakdown"), dict) else {}
        overall = float(data.get("overall_score", 0.0) or 0.0)
        notes = data.get("improvement_notes") if isinstance(data.get("improvement_notes"), list) else []
        details: Dict[str, Any] = {}
        for key in ["plan_alignment_details", "style_dim_report", "stage_act_report", "memory_report"]:
            if key in data:
                details[key] = data.get(key)
        breakdown: Dict[str, float] = {}
        for k, v in bd.items():
            try:
                breakdown[str(k)] = float(v)
            except Exception:
                continue
        
        # 确保包含 plan_alignment、style_adherence、stage_fit 三项
        # 如果 LLM 没有输出，使用默认值 0.5
        if "plan_alignment" not in breakdown:
            breakdown["plan_alignment"] = 0.5
            notes.append("⚠ LLM 未输出 plan_alignment，使用默认值 0.5")
        if "style_adherence" not in breakdown:
            breakdown["style_adherence"] = 0.5
            notes.append("⚠ LLM 未输出 style_adherence，使用默认值 0.5")
        if "stage_fit" not in breakdown:
            breakdown["stage_fit"] = 0.5
            notes.append("⚠ LLM 未输出 stage_fit，使用默认值 0.5")
        if "assistantiness" not in breakdown:
            # 关键维度缺失时默认偏保守（更像助手），避免误早退/误选
            breakdown["assistantiness"] = 0.8
            notes.append("⚠ LLM 未输出 assistantiness，使用默认值 0.8（保守惩罚）")

        # 出戏/元话语：若缺失默认为 0；但若文本命中“设定/人设/虚拟/系统/模型/作为一个…”则强制拉满
        if "immersion_break" not in breakdown:
            breakdown["immersion_break"] = 0.0
        try:
            all_text = "\n".join([str(m) for m in msgs])
            if any(x in all_text for x in ("设定", "人设", "虚拟", "虚构", "角色", "剧本", "配置", "模型", "系统", "作为一个")):
                breakdown["immersion_break"] = max(float(breakdown.get("immersion_break", 0.0) or 0.0), 1.0)
        except Exception:
            pass

        # 关系/人设/记忆相关：若缺失则给中性默认并记录（这些维度用于长期一致性）
        for k in ["persona_consistency", "relationship_fit", "memory_faithfulness", "memory_integration", "mode_behavior_fit"]:
            if k not in breakdown:
                breakdown[k] = 0.5
                notes.append(f"⚠ LLM 未输出 {k}，使用默认值 0.5")
        
        # 强制规则：assistantiness 高时不得给高 overall（避免“助手味”候选被选中/早退）
        try:
            a = float(breakdown.get("assistantiness", 0.0) or 0.0)
        except Exception:
            a = 0.0
        if a > 0.5 and overall > 0.3:
            overall = 0.28
            notes.append(f"⚠ assistantiness={a:.2f}>0.5，强制 overall_score<=0.3（clamp到0.28）")

        # 强制规则：immersion_break 高时不得给高 overall（避免“设定/人设/虚拟”等出戏话术被奖励）
        try:
            ib = float(breakdown.get("immersion_break", 0.0) or 0.0)
        except Exception:
            ib = 0.0
        if ib > 0.2 and overall > 0.3:
            overall = 0.28
            notes.append(f"⚠ immersion_break={ib:.2f}>0.2，强制 overall_score<=0.3（clamp到0.28）")

        overall = _clamp(overall, 0.0, 1.0)
        return overall, breakdown, [str(x) for x in notes if str(x).strip()], details
    except Exception:
        return None


def evaluate_candidate(
    state: Dict[str, Any],
    reply_plan: ReplyPlan,
    processor_plan: ProcessorPlan,
    requirements: Dict[str, Any],
    *,
    llm_soft_scorer: Any = None,
) -> SimReport:
    failures = hard_gate(processor_plan, requirements)

    # 额外硬门槛：ReplyPlan 的 speech_act 若为“建议”，但用户未明确要建议，则直接判负（规划层偏置的兜底）
    try:
        user_asks_advice = bool(requirements.get("user_asks_advice", False))
        sa = str((reply_plan or {}).get("speech_act") or "").strip()
        if (not user_asks_advice) and sa in ("建议", "advice"):
            failures.append(
                {
                    "id": "unsolicited_advice",
                    "reason": f"speech_act='{sa}' 但用户未明确要建议（规划层需回到闲聊/提问）",
                    "evidence": str(requirements.get("latest_user_text") or "")[:120],
                }
            )
    except Exception:
        pass

    hard_pass = not failures

    heur = soft_score_heuristic(state, reply_plan, processor_plan, requirements)
    # 使用 overall 字段（如果存在），否则计算平均值
    heur_overall = float(heur.get("overall", sum(heur.values()) / max(1, len(heur))))

    overall = heur_overall
    breakdown = {f"heur_{k}": float(v) for k, v in heur.items()}
    notes: List[str] = []

    llm_res = (
        soft_score_via_llm(state, llm_soft_scorer, reply_plan, processor_plan, requirements)
        if llm_soft_scorer
        else None
    )
    if not llm_soft_scorer:
        llm_status = "skipped"
    elif llm_res:
        llm_status = "ok"
    else:
        llm_status = "failed"
    if llm_res:
        llm_overall, llm_breakdown, llm_notes, llm_details = llm_res
        # 软分核心：LLM 编排评分权重更高；heur 作为稳定辅助
        overall = 0.75 * llm_overall + 0.25 * heur_overall
        
        # 处理 assistantiness 维度：根据 mode 调整惩罚权重
        assistantiness = float(llm_breakdown.get("assistantiness", 0.0) or 0.0)
        
        # 获取 mode_id 并设置惩罚权重
        mode = state.get("current_mode")
        mode_id = None
        if isinstance(mode, dict):
            mode_id = mode.get("id")
        elif mode:
            mode_id = getattr(mode, "id", None)
        mode_id = mode_id or "normal_mode"
        
        # 根据 mode_id 设置 assistantiness 惩罚权重
        if mode_id == "normal_mode":
            w = 1.0
        elif mode_id == "cold_mode":
            w = 0.5  # 冷淡模式惩罚减半（因为冷淡本来就不需要"服务感"）
        else:
            w = 0.0  # mute_mode 或其他模式不惩罚
        
        # 应用惩罚：overall -= w * 0.25 * assistantiness
        if w > 0.0 and assistantiness > 0.0:
            penalty = w * 0.25 * assistantiness
            overall = max(0.0, overall - penalty)
            notes.append(f"助手味检测: assistantiness={assistantiness:.2f}, mode={mode_id}, 权重={w:.1f}, 惩罚={penalty:.4f}")
        
        breakdown.update({f"llm_{k}": float(v) for k, v in llm_breakdown.items()})
        breakdown["llm_overall"] = float(llm_overall)
        breakdown["assistantiness"] = assistantiness  # 显式记录
        breakdown["assistantiness_weight"] = w  # 记录实际使用的权重
        notes.extend(llm_notes)
        print(f"      [评估] LLM软分: {llm_overall:.4f}, 启发式: {heur_overall:.4f}, 加权: {overall:.4f}, assistantiness: {assistantiness:.2f}, mode: {mode_id}, 权重: {w:.1f}")
        # 标记结构化细节是否存在（便于日志诊断）
        breakdown["llm_details_present"] = 1.0 if isinstance(llm_details, dict) and llm_details else 0.0

    # Hard gate penalty: fail-fast 仍允许保留 soft score 以便 debug，但 reward 要显著降低
    if not hard_pass:
        overall_before_penalty = overall
        overall = overall * 0.2
        notes.insert(0, "硬门槛未通过：已大幅惩罚总分。")
        print(f"      [评估] ⚠ 硬门槛失败，惩罚: {overall_before_penalty:.4f} -> {overall:.4f}")

    overall = _clamp(float(overall), 0.0, 1.0)
    found_solution = bool(hard_pass and overall >= 0.55)
    
    # 记录最终评估结果
    log_computation(
        "Evaluator",
        "最终评估结果汇总",
        inputs={
            "hard_pass": hard_pass,
            "heur_overall": heur_overall,
            "llm_overall": llm_res[0] if llm_res else None,
            "llm_status": llm_status,
        },
        outputs={
            "final_score": overall,
            "found_solution": found_solution,
            "score_breakdown": breakdown,
            "failed_checks": failures,
            "improvement_notes": notes[:8],
        },
    )
    
    return {
        "found_solution": found_solution,
        "eval_score": round(overall, 4),
        "failed_checks": failures,
        "score_breakdown": {k: round(float(v), 4) for k, v in breakdown.items()},
        "improvement_notes": notes[:8],
        "llm_status": llm_status,
        "llm_details": llm_res[3] if (llm_res and isinstance(llm_res[3], dict)) else {},
    }

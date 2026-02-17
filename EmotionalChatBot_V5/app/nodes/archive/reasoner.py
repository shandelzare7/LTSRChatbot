"""Reasoner 节点：内容层和对话行为层规划（不包含表达层，表达层由 Style 负责）。"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from utils.llm_json import parse_json_from_llm
from utils.prompt_helpers import format_mind_rules, format_relationship_for_llm, format_stage_act_for_llm
from utils.tracing import trace_if_enabled
from utils.detailed_logging import log_prompt_and_params, log_llm_response

from app.state import AgentState


def _safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """安全地从嵌套字典中获取值。"""
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, {})
        else:
            return default
    return current if current != {} else default


def _pick_latest_user_text(state: Dict[str, Any], chat_buffer: List[BaseMessage]) -> str:
    """提取最新用户消息：优先 user_input，否则从 chat_buffer 找最后一条 HumanMessage。"""
    user_input = state.get("user_input", "").strip()
    if user_input:
        return user_input
    
    # 从 chat_buffer 逆序找第一条 HumanMessage
    for msg in reversed(chat_buffer):
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", str(msg)) or ""
            if content.strip():
                return content.strip()
        elif hasattr(msg, "type"):
            msg_type = getattr(msg, "type", "").lower()
            if "human" in msg_type or "user" in msg_type:
                content = getattr(msg, "content", str(msg)) or ""
                if content.strip():
                    return content.strip()
    
    return ""


def _format_detection_signals(detection_signals: Dict[str, Any]) -> str:
    """格式化 detection_signals 供 LLM 阅读。"""
    if not detection_signals:
        return "（无检测信号）"
    
    parts = []
    
    # composite
    composite = detection_signals.get("composite") or {}
    if composite:
        parts.append("组合指标：")
        parts.append(f"  - goodwill: {composite.get('goodwill', 0):.2f}")
        parts.append(f"  - conflict_eff: {composite.get('conflict_eff', 0):.2f}")
        parts.append(f"  - provocation: {composite.get('provocation', 0):.2f}")
        parts.append(f"  - pressure: {composite.get('pressure', 0):.2f}")
    
    # meta
    meta = detection_signals.get("meta") or {}
    if meta:
        parts.append("元信息：")
        parts.append(f"  - target_is_assistant: {meta.get('target_is_assistant', 0)}")
        parts.append(f"  - quoted_or_reported_speech: {meta.get('quoted_or_reported_speech', 0)}")
    
    # stage_ctx
    stage_ctx = detection_signals.get("stage_ctx") or {}
    if stage_ctx:
        parts.append("阶段语境（关系模式越界检测）：")
        for key, val in stage_ctx.items():
            if isinstance(val, (int, float)) and val > 0:
                parts.append(f"  - {key}: {val:.2f}")
    
    # trace（少量关键信号）
    trace = detection_signals.get("trace") or {}
    if trace:
        confusion = trace.get("confusion", 0)
        if confusion > 0.3:
            parts.append(f"  - confusion (trace): {confusion:.2f}")
    
    return "\n".join(parts) if parts else "（无显著信号）"


def _should_generate_alt_plan(detection_signals: Dict[str, Any], latest_user_text: str) -> bool:
    """判断是否需要生成备选 plan。"""
    # uncert/confusion 高
    trace = detection_signals.get("trace") or {}
    confusion = trace.get("confusion", 0)
    if isinstance(confusion, (int, float)) and confusion > 0.55:
        return True
    
    # meta.target_is_assistant == 0（可能在说第三方）
    meta = detection_signals.get("meta") or {}
    if meta.get("target_is_assistant", 1) == 0:
        return True
    
    # latest_user_text 明显多义（简单启发式：包含多个问号、或包含"还是"/"或者"）
    if latest_user_text:
        text_lower = latest_user_text.lower()
        if latest_user_text.count("?") + latest_user_text.count("？") > 1 or "还是" in text_lower or "或者" in text_lower or "要么" in text_lower:
            return True
    
    return False


def reasoner_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    [Reasoner Node] - 内容层和对话行为层规划
    
    功能：
    1. 接收 Inner Monologue 节点写入的内心独白、直觉、关系滤镜
    2. 生成 response_plan（1-2个）：做什么事、需要什么信息、输出什么事实/观点、检索什么、评估标准、停止条件
    3. 不包含表达层内容（语气、温度、长度等，由 Style 负责）
    """
    
    llm = config["configurable"].get("llm_model")
    
    # ==========================================
    # 1. 输入提取
    # ==========================================
    bot_name = state.get("bot_basic_info", {}).get("name", "Bot")
    
    # recent_dialogue_context（最近对话）
    chat_buffer = state.get("chat_buffer") or state.get("messages", [])[-10:]
    
    # latest_user_text（修复：fallback 到 chat_buffer）
    # 外部通道优先：避免 internal prompt/debug 污染 user_input 后影响导演规划
    latest_user_text = (state.get("external_user_text") or "").strip() or _pick_latest_user_text(state, chat_buffer)
    
    recent_dialogue = []
    for msg in chat_buffer:
        if isinstance(msg, (HumanMessage, AIMessage)):
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            content = getattr(msg, "content", str(msg)) or ""
            recent_dialogue.append(f"{role}: {content}")
        else:
            content = getattr(msg, "content", str(msg)) or ""
            recent_dialogue.append(f"Message: {content}")
    recent_dialogue_context = "\n".join(recent_dialogue[-10:])  # 最近10条
    
    # detection_signals
    detection_signals = state.get("detection_signals") or {}
    detection_summary = _format_detection_signals(detection_signals)
    
    # 判断是否需要备选 plan
    need_alt_plan = _should_generate_alt_plan(detection_signals, latest_user_text)
    
    # mode（只传 mode_id，不传 monologue_instruction，避免污染分工）
    current_mode = state.get("current_mode")
    mode_id = "normal_mode"
    mode_behavior = ""
    if current_mode:
        if isinstance(current_mode, dict):
            mode_id = current_mode.get("id", "normal_mode")
        elif hasattr(current_mode, "id"):
            mode_id = current_mode.id
    
    # mode 行为约束（从 mode.behavior_contract 读取，只影响"是否回应/是否进入拉扯"，不影响语气）
    if current_mode and hasattr(current_mode, "behavior_contract"):
        contract = current_mode.behavior_contract
        if hasattr(contract, "notes") and contract.notes:
            mode_behavior = "; ".join(contract.notes[:2])  # 取前两条 notes
        else:
            mode_behavior = f"模式: {mode_id}"
    else:
        # 向后兼容：使用默认映射
        mode_behavior_map = {
            "normal_mode": "正常回应",
            "cold_mode": "冷淡回应，少解释少建议",
            "mute_mode": "不回应或极简终止",
        }
        mode_behavior = mode_behavior_map.get(mode_id, "正常回应")
    
    # knapp_stage（怎么演：仅阶段角色/目标/策略，供规划用）
    knapp_stage_id = state.get("current_stage", "initiating")
    knapp_stage_desc = format_stage_act_for_llm(knapp_stage_id)
    
    # retrieval_context_summary（记忆检索摘要，修复：统一 str 化）
    conversation_summary = state.get("conversation_summary") or ""
    retrieved_memories = state.get("retrieved_memories") or []
    retrieval_context_summary = ""
    # 记忆卫生：过滤"自称助手/AI"的旧记忆，避免错误召回把导演计划推向助手模板
    def _looks_like_assistant_identity(s: str) -> bool:
        s = (s or "").strip().lower()
        if not s:
            return False
        patterns = [
            r"我\s*是[\s\S]{0,24}(ai|人工智能|智能助手|机器人助手|chatbot|聊天助手|助手)",
            r"(我叫|我是|叫我)[\s\S]{0,18}(一个|位)?[\s\S]{0,18}(ai|人工智能|智能助手|机器人助手|chatbot|聊天助手|助手)",
            r"是一个聊天助手",
        ]
        import re as _re
        return any(_re.search(p, s) for p in patterns)

    if conversation_summary and not _looks_like_assistant_identity(str(conversation_summary)):
        retrieval_context_summary += f"对话摘要：{conversation_summary}\n"
    if retrieved_memories:
        # 修复：统一 str 化，避免 join 报错
        memory_lines_raw = [str(x) for x in retrieved_memories[:8]]
        memory_lines = [ln for ln in memory_lines_raw if ln.strip() and not _looks_like_assistant_identity(ln)]
        if memory_lines:
            retrieval_context_summary += f"检索到的记忆：\n" + "\n".join(memory_lines[:5])
    
    # relationship_state（format_relationship_for_llm 输出的是 0-1，所以 prompt 说 0-1）
    relationship_state = state.get("relationship_state") or {}
    relationship_detail = format_relationship_for_llm(relationship_state)
    
    # emotion_state（修复：展示 busyness）
    mood_state = state.get("mood_state") or {}
    pleasure = mood_state.get("pleasure", 0.0)
    arousal = mood_state.get("arousal", 0.0)
    dominance = mood_state.get("dominance", 0.0)
    busyness = mood_state.get("busyness", 0.0)
    # PAD 如果是 -1..1 范围，显示时标注
    emotion_state = f"P(愉悦): {pleasure:.2f}, A(唤醒): {arousal:.2f}, D(支配): {dominance:.2f}, Busy(繁忙): {busyness:.2f}"
    
    # inner_monologue
    inner_monologue = state.get("inner_monologue") or "（无内心独白）"
    intuition_thought = state.get("intuition_thought") or "（无直觉思考）"
    relationship_filter = state.get("relationship_filter") or "（无关系滤镜）"
    
    # constraints/policy flags
    logic_constraints = format_mind_rules(state)
    
    # ==========================================
    # 2. System Prompt Construction
    # ==========================================
    bot_basic_info_raw = state.get("bot_basic_info") or {}
    bot_persona_raw = state.get("bot_persona") or {}

    # 身份卫生：如果 DB/本地存了"助手/AI"字样的人设字段，这里先过滤掉，避免上游被脏数据牵引
    def _sanitize_obj(x):
        import re as _re
        pats = [
            r"(ai|人工智能|智能助手|机器人助手|chatbot|聊天助手|助手)",
        ]
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if isinstance(v, (dict, list)):
                    out[k] = _sanitize_obj(v)
                else:
                    s = str(v) if v is not None else ""
                    if any(_re.search(p, s.lower()) for p in pats):
                        continue
                    out[k] = v
            return out
        if isinstance(x, list):
            out_list = []
            for it in x:
                if isinstance(it, (dict, list)):
                    out_list.append(_sanitize_obj(it))
                else:
                    s = str(it) if it is not None else ""
                    if any(_re.search(p, s.lower()) for p in pats):
                        continue
                    out_list.append(it)
            return out_list
        return x

    bot_basic_info = _sanitize_obj(bot_basic_info_raw)
    bot_persona = _sanitize_obj(bot_persona_raw)

    system_prompt = f"""# Role: 对话导演（只写分镜/动作清单，不写台词）

你是 {bot_name} 的「对话导演」。你只产出"本轮要做哪些动作、按什么顺序、需要检索什么、怎么验收"，不产出任何具体措辞、语气、长度指令。

## 身份约束（非常重要）
- 你在制定 core_points / success_criteria 时，**禁止把 {bot_name} 定义成"助手/AI/chatbot/客服"**，也禁止要求回复里出现"我可以帮你…"这类服务口吻。
- 当用户问"你是谁/你是什么/你能做什么"时：请按 bot_basic_info/bot_persona 的人设事实来组织回答（例如名字/地区/职业/爱好/背景），把它当作"人物自我介绍"，不是"产品说明"。
- 若检索摘要里出现"{bot_name} 是聊天助手/我可以帮你…"等旧片段：视为低质量模板记忆，不要把它写进 core_points。

## 分工原则

**你负责（内容层 + 对话行为层）：**
- 这轮要做什么事（回答/澄清/设边界/反击/修复/转移/结束）
- 需要哪些信息（缺哪些槽、要问什么）
- 要输出哪些事实/观点（核心要点、论证骨架、反问点）
- 要不要检索、检索什么（给 LATS 的结构化 search spec）
- 怎么判定"做得好"（给 critic 的结构化评估 rubric）
- 什么时候停、什么时候改问问题（stop / fallback）


## 输入信息

### 1. 最新用户消息
{latest_user_text if latest_user_text else "（无用户消息）"}

### 2. 近期对话上下文（最近10条）
{recent_dialogue_context if recent_dialogue_context else "（无近期对话）"}

### 3. 检测信号
{detection_summary}

### 4. 当前模式
模式ID: {mode_id}
行为约束: {mode_behavior}
（注意：mode 只影响"是否回应/是否进入拉扯"，不影响语气。语气由 Style 节点决定。）

### 4.1 Bot 硬身份（bot_basic_info）
{bot_basic_info}

### 4.2 Bot 人设（bot_persona）
{bot_persona}

### 5. Knapp 关系阶段
{knapp_stage_desc}

### 6. 检索上下文摘要
{retrieval_context_summary if retrieval_context_summary else "（无检索内容）"}

### 7. 关系状态（6维，0–1）
{relationship_detail}

### 8. 情绪状态（PAD + Busy）
{emotion_state}
（注意：情绪状态仅供参考，用于理解当前心境，不用于决定语气。语气由 Style 节点决定。）

### 9. 内心独白（来自 Inner Monologue 节点）
{inner_monologue}

直觉思考：{intuition_thought}
关系滤镜：{relationship_filter}

### 10. 逻辑约束与规则
{logic_constraints}

## 你的任务

生成 response_plan（主 plan + 可选备选 plan）。

**生成规则：**
- **只出 1 个 plan**：默认情况（意图明确、情况简单、确定度高）
- **出 2 个 plan**：当以下任一成立时
  - uncert/confusion 高（trace.confusion > 0.55）
  - meta.target_is_assistant == 0（可能在说第三方）
  - latest_user_text 明显多义（包含多个问号、或包含"还是"/"或者"）

当前是否需要备选 plan：{"是" if need_alt_plan else "否"}

## response_plan 字段说明（结构化，供 LATS 直接使用）

每个 plan 必须包含以下字段：

1. **id**: 计划ID（字符串，如 "P1", "P2"）
2. **weight**: 权重（数字，0-1，主 plan 通常 0.7-1.0，备选 plan 0.2-0.3）
3. **action**: 要做什么事（字符串，如 "回答"/"澄清"/"设边界"/"反击"/"修复"/"转移"/"结束"）
4. **information_needs**: 需要哪些信息（字符串，描述缺哪些槽、要问什么问题）
5. **core_points**: 要输出哪些事实/观点（字符串，核心要点、论证骨架、反问点）
6. **search_spec**: 检索需求（结构化对象，供 LATS 直接使用）：
   {{
     "enabled": true/false,
     "query_seeds": ["关键词1", "关键词2"],
     "must_cover": ["必须检索到的信息点1", "必须检索到的信息点2"],
     "optional_topics": ["可选话题1", "可选话题2"]
   }}
7. **evaluation_rubric**: 评估标准（结构化对象，供 Critic 直接使用）：
   {{
     "success_criteria": ["必须包含的要点1", "必须包含的要点2"],
     "failure_modes": ["不能出现的错误1", "不能出现的错误2"],
     "quality_threshold": 0.7
   }}
8. **stop_conditions**: 停止条件（字符串，如 "已回答核心问题"）
9. **fallback_conditions**: fallback 条件（字符串，如 "如果用户继续追问，改为反问澄清"）

## 禁止事项

**不要包含以下内容：**
- "语气要冷/要温柔/要幽默/要简短/要长一点/不要表情包"
- "用讽刺口吻/用高冷口吻/用亲昵称呼"
- "写一句话""控制在10字"这种长度硬约束
-禁止输出任何具体措辞示例（哪怕一两句），只输出要点与动作。

**可以包含：**
- "不展开解释""不进入拉扯""只做边界声明+一个反问"（这是内容结构，不是语气）
- "只回答一个核心点""只问一个澄清问题"
- "拒绝该请求并说明理由"（内容行为）

## 输出格式（JSON）

{{
  "user_intent": "简要分析用户意图（字符串）",
  "plans": [
    {{
      "id": "P1",
      "weight": 0.75,
      "action": "回答",
      "information_needs": "需要确认用户的具体需求",
      "core_points": "解释核心概念，提供示例，反问用户是否理解",
      "search_spec": {{
        "enabled": true,
        "query_seeds": ["核心概念", "示例"],
        "must_cover": ["定义", "应用场景"],
        "optional_topics": ["相关话题"]
      }},
      "evaluation_rubric": {{
        "success_criteria": ["包含核心概念定义", "提供至少一个示例"],
        "failure_modes": ["过于抽象", "偏离主题"],
        "quality_threshold": 0.7
      }},
      "stop_conditions": "已回答核心问题，用户表示理解",
      "fallback_conditions": "用户继续追问时，改为反问澄清具体疑问点"
    }}
  ]
}}

**注意：**
- weight 必须是数字（number），其它字段按 schema 类型（list/object/string）
- 如果不需要备选 plan，plans 数组只放一个元素
- weight 总和应该接近 1.0
- search_spec 和 evaluation_rubric 必须是结构化对象，不要用字符串
"""

    # ==========================================
    # 3. LLM Execution & Parsing
    # ==========================================
    
    log_prompt_and_params(
        "Reasoner",
        system_prompt=system_prompt[:2000],  # 截断以便日志
        user_prompt="生成 response_plan",
        params={
            "user_input": latest_user_text[:100],
            "mode_id": mode_id,
            "knapp_stage": knapp_stage_id,
            "has_inner_monologue": bool(inner_monologue),
            "need_alt_plan": need_alt_plan,
        }
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="请根据上述信息生成 response_plan。")
        ])
        content = getattr(response, "content", "") or ""
        result = parse_json_from_llm(content)
        if not isinstance(result, dict):
            result = None
        
        if result:
            log_llm_response("Reasoner", response, parsed_result={"user_intent": result.get("user_intent"), "plans_count": len(result.get("plans", []))})
    except Exception as e:
        result = None
        print(f"[Reasoner] LLM 异常: {e}")
    
    # ==========================================
    # 4. 默认值处理与修复
    # ==========================================
    if not result:
        result = {
            "user_intent": "常规对话",
            "plans": [
                {
                    "id": "P1",
                    "weight": 1.0,
                    "action": "回答",
                    "information_needs": "无特殊需求",
                    "core_points": "根据对话上下文自然回应",
                    "search_spec": {"enabled": False, "query_seeds": [], "must_cover": [], "optional_topics": []},
                    "evaluation_rubric": {"success_criteria": ["回应自然、相关"], "failure_modes": [], "quality_threshold": 0.7},
                    "stop_conditions": "已回答用户问题",
                    "fallback_conditions": "用户继续追问时澄清"
                }
            ]
        }
    
    # 确保 plans 格式正确
    plans = result.get("plans", [])
    if not plans:
        plans = [{
            "id": "P1",
            "weight": 1.0,
            "action": "回答",
            "information_needs": "无",
            "core_points": "自然回应",
            "search_spec": {"enabled": False, "query_seeds": [], "must_cover": [], "optional_topics": []},
            "evaluation_rubric": {"success_criteria": ["自然相关"], "failure_modes": [], "quality_threshold": 0.7},
            "stop_conditions": "已回答",
            "fallback_conditions": "继续追问时澄清"
        }]
    
    # 修复：weight 可能是字符串，统一转为数字
    for p in plans:
        weight_raw = p.get("weight", 0.5)
        if isinstance(weight_raw, str):
            try:
                p["weight"] = float(weight_raw)
            except (ValueError, TypeError):
                p["weight"] = 0.5
        elif not isinstance(weight_raw, (int, float)):
            p["weight"] = 0.5
        else:
            p["weight"] = float(weight_raw)
    
    # 归一化 weight
    total_weight = sum(p.get("weight", 0.5) for p in plans)
    if total_weight > 0:
        for p in plans:
            p["weight"] = p.get("weight", 0.5) / total_weight
    
    # 确保 search_spec 和 evaluation_rubric 是结构化对象
    for p in plans:
        if not isinstance(p.get("search_spec"), dict):
            p["search_spec"] = {"enabled": False, "query_seeds": [], "must_cover": [], "optional_topics": []}
        if not isinstance(p.get("evaluation_rubric"), dict):
            p["evaluation_rubric"] = {"success_criteria": [], "failure_modes": [], "quality_threshold": 0.7}
        
        # 字段级归一化：search_spec
        search_spec = p.get("search_spec", {})
        # enabled: bool（处理字符串 "true"/"false"）
        enabled_raw = search_spec.get("enabled", False)
        if isinstance(enabled_raw, str):
            search_spec["enabled"] = enabled_raw.lower() in ("true", "1", "yes", "on")
        else:
            search_spec["enabled"] = bool(enabled_raw)
        # query_seeds: list[str]
        query_seeds_raw = search_spec.get("query_seeds", [])
        if not isinstance(query_seeds_raw, list):
            search_spec["query_seeds"] = []
        else:
            search_spec["query_seeds"] = [str(x) for x in query_seeds_raw if x]
        # must_cover: list[str]
        must_cover_raw = search_spec.get("must_cover", [])
        if not isinstance(must_cover_raw, list):
            search_spec["must_cover"] = []
        else:
            search_spec["must_cover"] = [str(x) for x in must_cover_raw if x]
        # optional_topics: list[str]
        optional_topics_raw = search_spec.get("optional_topics", [])
        if not isinstance(optional_topics_raw, list):
            search_spec["optional_topics"] = []
        else:
            search_spec["optional_topics"] = [str(x) for x in optional_topics_raw if x]
        
        # 字段级归一化：evaluation_rubric
        eval_rubric = p.get("evaluation_rubric", {})
        # success_criteria: list[str]
        success_criteria_raw = eval_rubric.get("success_criteria", [])
        if not isinstance(success_criteria_raw, list):
            eval_rubric["success_criteria"] = []
        else:
            eval_rubric["success_criteria"] = [str(x) for x in success_criteria_raw if x]
        # failure_modes: list[str]
        failure_modes_raw = eval_rubric.get("failure_modes", [])
        if not isinstance(failure_modes_raw, list):
            eval_rubric["failure_modes"] = []
        else:
            eval_rubric["failure_modes"] = [str(x) for x in failure_modes_raw if x]
        # quality_threshold: float
        threshold_raw = eval_rubric.get("quality_threshold", 0.7)
        try:
            eval_rubric["quality_threshold"] = float(threshold_raw)
        except (TypeError, ValueError):
            eval_rubric["quality_threshold"] = 0.7
    
    # 强制执行 need_alt_plan 逻辑（防止 LLM 不听话）
    if not need_alt_plan and len(plans) > 1:
        plans = plans[:1]
    # 如果 need_alt_plan=True 但只有1个plan，允许（不强制要求2个）
    
    result["plans"] = plans
    
    # ==========================================
    # 5. 返回状态更新
    # ==========================================
    
    # 兼容旧字段 response_strategy（从主 plan 提取）
    main_plan = plans[0] if plans else {}
    response_strategy = f"行动: {main_plan.get('action', '回答')}. 核心要点: {main_plan.get('core_points', '自然回应')}"
    
    print(f"[Reasoner] 生成 {len(plans)} 个 plan，主 plan: {main_plan.get('action', 'N/A')}")
    
    return {
        "user_intent": result.get("user_intent", "常规对话"),
        "response_plan": result,  # 完整 plan 结构
        "response_strategy": response_strategy,  # 兼容旧字段
    }


# -------------------------------------------------------------------
# LangGraph 兼容包装器
# -------------------------------------------------------------------

def create_reasoner_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """
    创建 Reasoner 节点：内容层和对话行为层规划。
    """

    def _ensure_defaults(s: Dict[str, Any]) -> Dict[str, Any]:
        s = dict(s)
        s.setdefault("user_input", "")
        bot = dict(s.get("bot_basic_info") or {})
        bot.setdefault("name", "Bot")
        s["bot_basic_info"] = bot
        
        # 修复：mood_state 默认值改为中性 0.5（PAD 范围是 -1..1，但这里用 0-1 范围的中性值）
        # 如果系统期望 -1..1，则用 0.0；如果期望 0..1，则用 0.5
        # 根据 emotion_update.py，PAD 是 -1..1，但计算时映射到 0..1，所以这里默认用 0.0（-1..1 的中性）
        # 但根据用户要求，应该用 0.5（0..1 的中性），所以如果系统内部是 -1..1，需要映射
        mood = dict(s.get("mood_state") or {})
        # 假设系统内部 PAD 是 -1..1，中性是 0.0
        mood.setdefault("pleasure", 0.0)  # -1..1 的中性
        mood.setdefault("arousal", 0.0)   # -1..1 的中性
        mood.setdefault("dominance", 0.0) # -1..1 的中性
        mood.setdefault("busyness", 0.0)  # 0..1，默认不忙
        s["mood_state"] = mood
        
        # relationship_state：系统内部统一使用 0-1 范围，默认中性值 0.5
        rel = dict(s.get("relationship_state") or {})
        rel.setdefault("closeness", 0.5)   # 0-1 范围的中性值
        rel.setdefault("trust", 0.5)       # 0-1 范围的中性值
        rel.setdefault("liking", 0.5)      # 0-1 范围的中性值
        rel.setdefault("respect", 0.5)     # 0-1 范围的中性值
        rel.setdefault("warmth", 0.5)      # 0-1 范围的中性值
        rel.setdefault("power", 0.5)       # 0-1 范围的中性值
        s["relationship_state"] = rel
        
        s.setdefault("current_stage", "initiating")
        s.setdefault("current_mode", None)
        return s

    @trace_if_enabled(
        name="Reasoner",
        run_type="chain",
        tags=["node", "thinking", "reasoner"],
        metadata={"state_outputs": ["response_plan", "response_strategy", "user_intent"]},
    )
    def node(state: Dict[str, Any]) -> dict:
        safe = _ensure_defaults(state)
        out = reasoner_node(safe, {"configurable": {"llm_model": llm_invoker}})
        
        # 兼容：deep_reasoning_trace（保留旧字段）
        monologue_raw = safe.get("inner_monologue", "") or ""
        strategy_raw = out.get("response_strategy", "") or ""
        
        def _to_text(x: Any) -> str:
            if x is None:
                return ""
            if isinstance(x, str):
                return x
            try:
                return json.dumps(x, ensure_ascii=False)
            except Exception:
                return str(x)
        
        monologue = _to_text(monologue_raw).strip()
        strategy = _to_text(strategy_raw).strip()
        
        trace_text = (monologue + ("\n\nStrategy:\n" + strategy if strategy else "")).strip()
        out["deep_reasoning_trace"] = {"reasoning": trace_text, "enabled": bool(trace_text)}
        
        return out

    return node

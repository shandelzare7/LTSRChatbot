from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re

from app.state import RequirementsChecklist
from utils.yaml_loader import get_project_root


def _norm_str(x: Any) -> str:
    return (str(x) if x is not None else "").strip()


# “用户是否明确在要建议/教程”的最低配判定：用于抑制无请求的科普/建议（助手味来源之一）
_ASK_ADVICE_PAT = re.compile(
    r"(怎么|如何|咋|怎样|教我|求教|请教|建议|推荐|该不该|要不要|能不能|可以不可以|帮我|帮忙|指导|步骤|教程|总结一下|方案|策略|怎么做)",
    re.IGNORECASE,
)


def _user_asks_for_advice(text: str) -> bool:
    s = _norm_str(text)
    if not s:
        return False
    # 问候/寒暄不算“要建议”
    if len(s) <= 12 and re.match(r"^\s*(hi|hello|hey|你好|您好|嗨|哈喽|在吗|早上好|中午好|晚上好|晚安)\s*[!！。.]?\s*$", s, re.IGNORECASE):
        return False
    return bool(_ASK_ADVICE_PAT.search(s))

# 沉浸破坏词：一旦出现在回复中，用户会立刻意识到“在扮演/在解释配置”
# 这组词应当被当作高权重违禁（hard gate 直接淘汰候选）。
_IMMERSION_BREAK_FORBIDDEN: List[str] = [
    "设定",
    "人设",
    "虚拟",
    "虚构",
    "角色",
    "剧本",
    "配置",
    "模型",
    "系统",
    "作为一个",
]


def build_requirements(mode: Any, reasoner_plan: Any) -> RequirementsChecklist:
    """
    根据 mode 动态生成 requirements 基础参数，然后合并 reasoner 的 must_have/forbidden/safety_notes（如果 policy 允许）。
    
    Args:
        mode: PsychoMode 对象或字典
        reasoner_plan: reasoner 输出的 response_plan（包含 search_spec 和 evaluation_rubric）
    
    Returns:
        RequirementsChecklist 对象
    """
    # 1. 获取 mode_id
    mode_id = None
    if isinstance(mode, dict):
        mode_id = mode.get("id")
    elif mode:
        mode_id = getattr(mode, "id", None)
    
    mode_id = mode_id or "normal_mode"
    
    # 2. 根据 mode_id 设置基础参数
    if mode_id == "normal_mode":
        max_messages = 3
        min_first_len = 8
        must_have_policy = "soft"
    elif mode_id == "cold_mode":
        max_messages = 1
        min_first_len = 1
        must_have_policy = "none"
    elif mode_id == "mute_mode":
        max_messages = 0  # 或 1，根据需求
        min_first_len = 0
        must_have_policy = "none"
    else:
        # 默认 fallback
        max_messages = 3
        min_first_len = 8
        must_have_policy = "soft"
    
    # 3. 从 mode 读取 requirements_policy 和 lats_budget（覆盖基础参数）
    requirements_policy = None
    lats_budget = None
    disallowed = []
    allow_short_reply = False
    allow_empty_reply = False
    must_have_min_coverage = 0.75
    
    if mode:
        if isinstance(mode, dict):
            requirements_policy = mode.get("requirements_policy")
            lats_budget = mode.get("lats_budget")
            disallowed = list(mode.get("disallowed", []))
        else:
            if hasattr(mode, "requirements_policy"):
                requirements_policy = mode.requirements_policy
            if hasattr(mode, "lats_budget"):
                lats_budget = mode.lats_budget
            if hasattr(mode, "disallowed"):
                disallowed = list(mode.disallowed) if isinstance(mode.disallowed, list) else []
    
    # 从 requirements_policy 读取策略（覆盖基础参数）
    if requirements_policy:
        if isinstance(requirements_policy, dict):
            must_have_policy = str(requirements_policy.get("must_have_policy", must_have_policy))
            must_have_min_coverage = float(requirements_policy.get("must_have_min_coverage", must_have_min_coverage))
            allow_short_reply = bool(requirements_policy.get("allow_short_reply", False))
            allow_empty_reply = bool(requirements_policy.get("allow_empty_reply", False))
        else:
            if hasattr(requirements_policy, "must_have_policy"):
                must_have_policy = str(requirements_policy.must_have_policy)
            if hasattr(requirements_policy, "must_have_min_coverage"):
                must_have_min_coverage = float(requirements_policy.must_have_min_coverage)
            if hasattr(requirements_policy, "allow_short_reply"):
                allow_short_reply = bool(requirements_policy.allow_short_reply)
            if hasattr(requirements_policy, "allow_empty_reply"):
                allow_empty_reply = bool(requirements_policy.allow_empty_reply)
    
    # 从 lats_budget 读取消息长度限制（覆盖基础参数）
    if lats_budget:
        if isinstance(lats_budget, dict):
            max_messages = int(lats_budget.get("max_messages", max_messages))
            min_first_len = int(lats_budget.get("min_first_len", min_first_len))
            max_message_len = int(lats_budget.get("max_message_len", 220))
        else:
            if hasattr(lats_budget, "max_messages"):
                max_messages = int(lats_budget.max_messages)
            if hasattr(lats_budget, "min_first_len"):
                min_first_len = int(lats_budget.min_first_len)
            if hasattr(lats_budget, "max_message_len"):
                max_message_len = int(lats_budget.max_message_len)
            else:
                max_message_len = 220
    else:
        max_message_len = 220
    
    # 4. 初始化 must_have / forbidden / safety_notes
    must_have: List[str] = []
    forbidden: List[str] = []
    safety_notes: List[str] = []
    
    # 将 mode.disallowed 加入 forbidden
    forbidden.extend(disallowed)

    # 额外：沉浸破坏词作为高权重违禁
    forbidden.extend(_IMMERSION_BREAK_FORBIDDEN)
    
    # 基础安全兜底
    safety_notes.extend([
        "不得自称AI/模型/系统",
        "不得输出违法/自残/暴力等危险指导",
        "尊重边界，不进行性骚扰或强迫",
    ])
    
    # 5. 从 reasoner_plan 合并 must_have / forbidden / safety_notes（如果 policy 允许）
    if reasoner_plan and must_have_policy != "none":
        # 提取主 plan（weight 最高的）
        plans = []
        if isinstance(reasoner_plan, dict):
            plans = reasoner_plan.get("plans", [])
        elif isinstance(reasoner_plan, list):
            plans = reasoner_plan
        
        if plans:
            # 找到 weight 最高的 plan
            main_plan = max(plans, key=lambda p: float(p.get("weight", 0) or 0))
            
            # 从 search_spec.must_cover 提取 must_have
            search_spec = main_plan.get("search_spec", {})
            if isinstance(search_spec, dict):
                must_cover = search_spec.get("must_cover", [])
                if isinstance(must_cover, list):
                    must_have.extend([str(x) for x in must_cover if x])
            
            # 从 evaluation_rubric.success_criteria 提取 must_have
            eval_rubric = main_plan.get("evaluation_rubric", {})
            if isinstance(eval_rubric, dict):
                success_criteria = eval_rubric.get("success_criteria", [])
                if isinstance(success_criteria, list):
                    must_have.extend([str(x) for x in success_criteria if x])
            
            # 从 evaluation_rubric.failure_modes 提取 forbidden/safety_notes
            if isinstance(eval_rubric, dict):
                failure_modes = eval_rubric.get("failure_modes", [])
                if isinstance(failure_modes, list):
                    forbidden.extend([str(x) for x in failure_modes if x])
    
    # 6. 从 mode.critic_criteria 读取评估标准（用于记录）
    criteria_list: List[str] = []
    if mode:
        if isinstance(mode, dict):
            crit = mode.get("critic_criteria", {})
            if isinstance(crit, dict):
                criteria_list = list(crit.get("focus", []))
        elif hasattr(mode, "critic_criteria"):
            crit = mode.critic_criteria
            if hasattr(crit, "focus"):
                criteria_list = list(crit.focus) if isinstance(crit.focus, list) else []
    
    # 7. 构建返回对象
    stage = ""  # 将在 compile_requirements 中从 state 读取
    first_rule = "第一条必须先回应用户/先给态度或结论，不能只铺垫。"
    
    return {
        "must_have": must_have,
        "forbidden": forbidden,
        "safety_notes": safety_notes,
        "mode_critic_criteria": criteria_list,
        "first_message_rule": first_rule,
        "max_messages": int(max_messages),
        "min_first_len": int(min_first_len),
        "max_message_len": int(max_message_len),
        "stage_pacing_notes": "",  # 将在 compile_requirements 中填充
        "must_have_policy": must_have_policy,
        "must_have_min_coverage": float(must_have_min_coverage),
        "allow_short_reply": bool(allow_short_reply),
        "allow_empty_reply": bool(allow_empty_reply),
    }


def _load_stage_config(stage_id: str) -> Dict[str, Any]:
    """加载 stage 配置文件"""
    try:
        root = get_project_root()
        stage_file = Path(root) / "config" / "stages" / f"{stage_id}.yaml"
        if stage_file.exists():
            with open(stage_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[Requirements] 加载 stage 配置失败: {e}")
    return {}


def _default_stage_act_targets(stage_id: str) -> Tuple[List[str], List[str]]:
    """
    返回 (allowed_acts, forbidden_acts) 的最低配默认枚举。
    目的：让 stage_fit 不止是“规则提示”，而是可评估的行为类型约束。
    """
    s = _norm_str(stage_id or "experimenting")
    mapping: Dict[str, Tuple[List[str], List[str]]] = {
        "initiating": (
            ["answer", "clarify", "question", "light_tease", "small_talk"],
            ["deep_probe", "commitment_push", "intimacy_escalate"],
        ),
        "experimenting": (
            ["answer", "clarify", "question", "light_tease", "small_talk"],
            ["commitment_push", "intimacy_escalate"],
        ),
        "intensifying": (
            ["answer", "clarify", "question", "empathy", "self_disclosure", "light_tease"],
            ["commitment_push_hard", "intimacy_escalate_fast"],
        ),
        "circumscribing": (
            ["answer", "clarify", "boundary", "light_tease"],
            ["deep_probe", "commitment_push", "intimacy_escalate", "emotionally_overbearing"],
        ),
        "avoiding": (
            ["answer", "boundary", "clarify"],
            ["deep_probe", "commitment_push", "intimacy_escalate"],
        ),
        "terminating": (
            ["boundary", "answer", "closing"],
            ["intimacy_escalate", "commitment_push"],
        ),
    }
    return mapping.get(s, (["answer", "clarify", "question"], ["commitment_push", "intimacy_escalate"]))


def _extract_mode_behavior_targets(mode: Any) -> List[str]:
    """
    取 mode 的最低配“对话行为策略目标”（不是语气词），用于 planner 与 scorer。
    """
    if mode is None:
        return []
    # 1) 来自 behavior_contract.notes
    try:
        if isinstance(mode, dict):
            bc = mode.get("behavior_contract") or {}
            notes = bc.get("notes") if isinstance(bc, dict) else None
            if isinstance(notes, list):
                return [str(x).strip() for x in notes if str(x).strip()][:6]
        if hasattr(mode, "behavior_contract"):
            bc = getattr(mode, "behavior_contract", None)
            notes = getattr(bc, "notes", None)
            if isinstance(notes, list):
                return [str(x).strip() for x in notes if str(x).strip()][:6]
    except Exception:
        pass

    # 2) fallback: 按 mode_id 给最短策略枚举
    mode_id = None
    if isinstance(mode, dict):
        mode_id = mode.get("id")
    elif mode:
        mode_id = getattr(mode, "id", None)
    mode_id = _norm_str(mode_id) or "normal_mode"

    fallback: Dict[str, List[str]] = {
        "normal_mode": ["自然回应", "不要助手味", "先回应再解释（若需要）"],
        "cold_mode": ["允许很短", "不安抚不解释", "只回应一个点/一句话结束"],
        "mute_mode": ["尽量少说或不说", "不进入拉扯", "避免情绪化升级"],
    }
    return fallback.get(mode_id, ["自然回应", "不要助手味"])


_ASSISTANTISH_POINT_PATTERNS = [
    r"(聊天助手|智能助手|助手|客服|chatbot)",
    r"(我可以帮你|我能帮你|我可以为你|我能为你|有什么可以帮你|需要我帮你)",
    r"(解答问题|提供信息|为您服务|随时咨询|祝您使用愉快|感谢您的使用)",
]


def _sanitize_plan_goal_points(points: List[str]) -> List[str]:
    """
    计划目标卫生：防止 plan_goals.must_cover_points 被“助手/产品说明”脏要点污染。
    这些要点一旦进入 must_cover，会把 ReplyPlanner 推向“先满足指标（当助手）”而非拟人化。
    """
    out: List[str] = []
    for p in points or []:
        s = _norm_str(p)
        if not s:
            continue
        low = s.lower()
        if any(re.search(pat, low) for pat in _ASSISTANTISH_POINT_PATTERNS):
            continue
        # 过短/空泛的也不要（避免 must_cover 变成噪声）
        if len(s) < 2:
            continue
        if s not in out:
            out.append(s)
    return out[:10]


def compile_requirements(state: Dict[str, Any]) -> RequirementsChecklist:
    """将 reasoner/style/mode 中分散的约束编译为 checklist（硬门槛 + must-have/forbidden）。"""
    mode = state.get("current_mode")
    reasoner_plan = state.get("response_plan")
    
    # 使用 build_requirements 生成基础 requirements
    requirements = build_requirements(mode, reasoner_plan)
    # mode 行为策略目标（进入 planner/scorer 的硬结构约束，而不是只给 evaluator 看）
    requirements["mode_behavior_targets"] = _extract_mode_behavior_targets(mode)
    
    # 填充 stage_pacing_notes（需要从 state 读取）
    stage = _norm_str(state.get("current_stage") or "experimenting")
    requirements["stage_pacing_notes"] = f"节奏需匹配 stage={stage}：越亲密可更自然碎片化；avoiding/stagnating 允许更冷淡更慢，但首条仍要可用。"
    
    # ==========================================
    # 1. plan_goals: 默认应尽量“空”（避免把导演的泛化动作说明误当成 must_cover，导致任务导向/助手味）
    #    仅当 reasoner.search_spec.must_cover 明确且能在用户原话中找到时，才进入 must_cover_points。
    # ==========================================
    plan_goals: Dict[str, Any] = {
        "must_cover_points": [],
        "avoid_points": [],
    }
    
    # 从 reasoner_plan 提取 avoid_points（failure_modes），以及“可选 must_cover”（严格过滤）
    user_text = _norm_str(state.get("external_user_text") or state.get("user_input") or "")
    # 传给 evaluator/hard_gate 的“用户是否明确要建议”开关（P0：无请求的建议/教程要硬挡）
    requirements["latest_user_text"] = user_text
    requirements["user_asks_advice"] = bool(_user_asks_for_advice(user_text))
    if not requirements["user_asks_advice"]:
        # 规划层先验：未被请求时，不要主动“建议/教程/步骤”
        try:
            mbt = list(requirements.get("mode_behavior_targets") or [])
            mbt.append("未被请求时不要给建议/步骤/教程；优先闲聊、呼应对方、抛1个轻问题。")
            requirements["mode_behavior_targets"] = [str(x).strip() for x in mbt if str(x).strip()][:8]
        except Exception:
            pass
    if reasoner_plan:
        plans = []
        if isinstance(reasoner_plan, dict):
            plans = reasoner_plan.get("plans", [])
        elif isinstance(reasoner_plan, list):
            plans = reasoner_plan

        if plans:
            main_plan = max(plans, key=lambda p: float(p.get("weight", 0) or 0))

            # 1) avoid_points: 来自 failure_modes（不参与“硬塞主题”，仅用于避免明显失败）
            eval_rubric = main_plan.get("evaluation_rubric", {})
            if isinstance(eval_rubric, dict):
                failure_modes = eval_rubric.get("failure_modes", [])
                if isinstance(failure_modes, list):
                    plan_goals["avoid_points"] = [str(x).strip() for x in failure_modes if str(x).strip()][:10]

            # 2) must_cover_points: 仅当 search_spec.must_cover 存在，且能在用户原话中命中（避免“问候也硬聊编程”）
            search_spec = main_plan.get("search_spec", {})
            must_cover = []
            if isinstance(search_spec, dict):
                mc = search_spec.get("must_cover", [])
                if isinstance(mc, list):
                    must_cover = [str(x).strip() for x in mc if str(x).strip()]
            if must_cover and user_text:
                filtered = [p for p in must_cover if p in user_text]
                plan_goals["must_cover_points"] = _sanitize_plan_goal_points(filtered)

    requirements["plan_goals"] = plan_goals
    
    # ==========================================
    # 2. style_targets: 来自 Style 的 12 维目标
    # ==========================================
    style_targets: Dict[str, float] = {}
    
    style_output = state.get("style")
    if style_output:
        # style_output 可能是 dict 或对象
        if isinstance(style_output, dict):
            # 提取 12 维目标（不包括 gate 和 derived）
            style_dimensions = [
                "verbal_length",
                "social_distance",
                "tone_temperature",
                "emotional_display",
                "wit_and_humor",
                "non_verbal_cues",
                "self_disclosure",
                "topic_adherence",
                "initiative",
                "advice_style",
                "subjectivity",
                "memory_hook",
            ]
            for dim in style_dimensions:
                val = style_output.get(dim)
                if val is not None:
                    try:
                        style_targets[dim] = float(val)
                    except (TypeError, ValueError):
                        pass
        elif hasattr(style_output, "__dict__"):
            # 如果是对象，尝试从属性读取
            for dim in ["verbal_length", "social_distance", "tone_temperature", "emotional_display",
                        "wit_and_humor", "non_verbal_cues", "self_disclosure", "topic_adherence",
                        "initiative", "advice_style", "subjectivity", "memory_hook"]:
                if hasattr(style_output, dim):
                    val = getattr(style_output, dim)
                    if val is not None:
                        try:
                            style_targets[dim] = float(val)
                        except (TypeError, ValueError):
                            pass
    
    # 如果没有提取到，使用默认值（0.5 表示中性）
    if not style_targets:
        style_targets = {
            "verbal_length": 0.5,
            "social_distance": 0.5,
            "tone_temperature": 0.5,
            "emotional_display": 0.5,
            "wit_and_humor": 0.5,
            "non_verbal_cues": 0.5,
        }
    
    requirements["style_targets"] = style_targets
    
    # ==========================================
    # 3. stage_targets: 来自 Knapp 阶段
    # ==========================================
    allowed_acts, forbidden_acts = _default_stage_act_targets(stage)
    stage_targets: Dict[str, Any] = {
        "stage": stage,
        "pacing_notes": [],
        "violation_sensitivity": 0.75,  # 默认值：不应因 stage_ctx 缺失而变成 0
        "allowed_acts": allowed_acts,
        "forbidden_acts": forbidden_acts,
    }
    
    # 加载 stage 配置（怎么演：优先从 act 块取，兼容旧版顶层）
    stage_config = _load_stage_config(stage)
    if stage_config:
        act = stage_config.get("act") or {}
        strategy = act.get("strategy") or stage_config.get("strategy") or []
        if isinstance(strategy, list):
            stage_targets["pacing_notes"] = [str(s).strip() for s in strategy if str(s).strip()]
        elif isinstance(strategy, str):
            stage_targets["pacing_notes"] = [strategy.strip()]
        
        stage_goal = act.get("stage_goal") or stage_config.get("stage_goal") or ""
        if stage_goal:
            stage_targets["pacing_notes"].append(f"阶段目标: {stage_goal}")

        sa = act.get("allowed_acts") or stage_config.get("allowed_acts")
        sf = act.get("forbidden_acts") or stage_config.get("forbidden_acts")
        if isinstance(sa, list) and sa:
            stage_targets["allowed_acts"] = [str(x).strip() for x in sa if str(x).strip()][:12]
        if isinstance(sf, list) and sf:
            stage_targets["forbidden_acts"] = [str(x).strip() for x in sf if str(x).strip()][:12]
    
    # 从 detection_signals.stage_ctx 计算 violation_sensitivity
    detection_signals = state.get("detection_signals", {})
    stage_ctx = detection_signals.get("stage_ctx", {})
    if isinstance(stage_ctx, dict):
        max_violation = max([float(v) for v in stage_ctx.values() if isinstance(v, (int, float))], default=0.0)
        # 只允许“更敏感”，不允许被覆盖成 0
        base = float(stage_targets.get("violation_sensitivity", 0.75) or 0.75)
        stage_targets["violation_sensitivity"] = min(1.0, max(base, max_violation))
    
    requirements["stage_targets"] = stage_targets

    # ==========================================
    # 4. task_planner 输出：tasks_for_lats / task_budget_max / word_budget（LATS 之前节点写入）
    # ==========================================
    requirements["tasks_for_lats"] = state.get("tasks_for_lats") or []
    requirements["task_budget_max"] = int(state.get("task_budget_max", 2) or 2)
    word_budget = int(state.get("word_budget", 60) or 60)
    requirements["word_budget"] = word_budget
    
    # 根据 word_budget 动态调整 max_messages
    # word_budget > 40 时允许 4-5 条消息
    base_max_messages = requirements.get("max_messages", 3)
    if word_budget > 40:
        # 根据 word_budget 调整：40-60 允许 4 条，>60 允许 5 条
        if word_budget > 60:
            requirements["max_messages"] = min(5, base_max_messages + 2)
        else:
            requirements["max_messages"] = min(4, base_max_messages + 1)
    
    return requirements

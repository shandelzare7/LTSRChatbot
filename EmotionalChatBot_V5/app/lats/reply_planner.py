from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.state import ReplyPlan
from src.schemas import ReplyPlannerSingle, ReplyPlannerCandidates
from utils.external_text import strip_candidate_prefix
from utils.llm_json import parse_json_from_llm
from utils.detailed_logging import log_prompt_and_params, log_llm_response

from app.lats.prompt_utils import (
    build_style_profile,
    build_system_memory_block,
    get_chat_buffer_body_messages_with_time_slices,
    safe_text,
)
from app.services.llm import set_current_node, reset_current_node
from utils.time_context import build_time_context_block, TIME_SLICE_BEHAVIOR_RULES


# ✅ 更稳健的 yaml_loader 导入：避免“一个函数导入失败就把三个全置 None”
try:
    import utils.yaml_loader as _yaml_loader
except Exception as e:
    _yaml_loader = None
    print(
        f"[ReplyPlanner] ⚠ cannot import utils.yaml_loader: {type(e).__name__}: {str(e)[:160]}",
        flush=True,
    )

load_stage_by_id = getattr(_yaml_loader, "load_stage_by_id", None) if _yaml_loader else None
load_content_moves = getattr(_yaml_loader, "load_content_moves", None) if _yaml_loader else None
load_pure_content_transformations = getattr(_yaml_loader, "load_pure_content_transformations", None) if _yaml_loader else None


def _full_logs() -> bool:
    return str(os.getenv("LTSR_FULL_PROMPT_LOG") or os.getenv("BOT2BOT_FULL_LOGS") or "").strip() in (
        "1",
        "true",
        "yes",
        "on",
    )


# 限制回复计划生成的最大 token，避免输出过长被截断导致 LengthFinishReasonError。可通过 LTSR_REPLY_PLAN_MAX_TOKENS 覆盖（默认 8192）
def _reply_plan_max_tokens() -> int:
    try:
        v = (os.getenv("LTSR_REPLY_PLAN_MAX_TOKENS") or "").strip()
        if v:
            return max(1024, min(16384, int(v)))
    except Exception:
        pass
    return 8192


def _planner_frequency_presence_penalty() -> Tuple[float, float]:
    """ReplyPlanner 使用的 frequency_penalty / presence_penalty，优先从 graph_llm_config 读取（由 graph.py 设置）。"""
    try:
        from app.core import graph_llm_config as _glc
        return (
            getattr(_glc, "PLANNER_FREQUENCY_PENALTY", 0.4),
            getattr(_glc, "PLANNER_PRESENCE_PENALTY", 0.5),
        )
    except Exception:
        return (0.4, 0.5)


def _env_int_clamped(key: str, default: int, *, min_v: int, max_v: int) -> int:
    try:
        raw = (os.getenv(key) or "").strip()
        if raw:
            v = int(raw)
            return max(min_v, min(max_v, v))
    except Exception:
        pass
    return max(min_v, min(max_v, default))


def _truncate_text(s: str, max_chars: int, *, head_ratio: float = 0.78) -> str:
    s = (s or "").strip()
    if not s or max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    head = max(1, int(max_chars * head_ratio))
    tail = max_chars - head - 1
    if tail <= 0:
        return s[: max_chars - 1].rstrip() + "…"
    return s[:head].rstrip() + "…" + s[-tail:].lstrip()


def _safe_text_limited(obj: Any, max_chars: int) -> str:
    try:
        return _truncate_text(safe_text(obj), max_chars=max_chars)
    except Exception:
        return _truncate_text(str(obj), max_chars=max_chars)


def _normalize_pure_content_transformations(raw: Any) -> List[Dict[str, Any]]:
    """
    兼容 loader 返回：
    - list[dict]（理想）
    - {"pure_content_transformations": [...]}（常见 YAML 顶层 dict）
    - 其他/异常 → []
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("pure_content_transformations") or []
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for x in raw:
        if isinstance(x, dict):
            out.append(x)
    return out


def _normalize_pure_content_move_action_text(tag: str, action: str) -> str:
    """
    修复「物理锚定」与写作规则冲突：
    - 允许身体动作/停顿/语气等“可撤销”描写
    - 避免硬编环境客观事实（光线/温度/声音/地点等）
    """
    t = (tag or "").strip()
    a = (action or "").strip()
    if t in ("物理锚定", "Physical Anchoring"):
        extra = "注意：不要硬编具体地点/温度/声音等客观环境事实；优先用可撤销的身体动作/停顿/语气描写来做锚定。"
        if extra not in a:
            a = (a.rstrip() + "\n" + extra).strip()
    return a


# Prompt block 长度控制（减少噪声，提高注意力）
PROMPT_BG_MAX_CHARS = _env_int_clamped("LTSR_PROMPT_BG_MAX_CHARS", 2400, min_v=400, max_v=12000)
PROMPT_MONOLOGUE_MAX_CHARS = _env_int_clamped("LTSR_PROMPT_MONOLOGUE_MAX_CHARS", 1200, min_v=200, max_v=8000)
PROMPT_STYLE_MAX_CHARS = _env_int_clamped("LTSR_PROMPT_STYLE_MAX_CHARS", 1600, min_v=200, max_v=8000)
PROMPT_MEMORY_MAX_CHARS = _env_int_clamped("LTSR_PROMPT_MEMORY_MAX_CHARS", 12000, min_v=800, max_v=60000)
PROMPT_TIME_MAX_CHARS = _env_int_clamped("LTSR_PROMPT_TIME_MAX_CHARS", 2800, min_v=200, max_v=12000)


# -------------------------
# 全局无差别概率降低提问：软惩罚（伯努利开关）
# -------------------------
def _question_prob() -> float:
    """
    LTSR_QUESTION_PROB: 0~1
    表示“本次生成倾向允许问句”的概率（软惩罚，不做硬限制）。
    - 1.0：不降低
    - 0.5：约一半调用倾向不问（默认）
    """
    raw = (os.getenv("LTSR_QUESTION_PROB") or "").strip()
    if not raw:
        return 0.5
    try:
        p = float(raw)
        if p != p:
            return 0.5
        return max(0.0, min(1.0, p))
    except Exception:
        return 0.5


def _sample_askq_flags(k: int) -> List[int]:
    """
    逐条采样 ASKQ（0/1），不依赖“是否需要澄清”等任何条件。
    - 0：倾向不用问句
    - 1：倾向可以带问句
    """
    p = _question_prob()
    k = max(1, int(k))
    return [1 if random.random() < p else 0 for _ in range(k)]


def _render_askq_meta(k: int, flags: List[int], *, content_move_tag: Optional[str] = None) -> str:
    if not flags:
        return ""
    if int(k) <= 1:
        return f"ASKQ={flags[0]}"
    if content_move_tag and int(k) == 3:
        a, b, c = (flags + [1, 1, 1])[:3]
        return f"ASKQ(light,medium,strong)={a},{b},{c}"
    return f"ASKQ_LIST={flags[: int(k)]}"


# -------------------------
# stage / strategy
# -------------------------
def _get_stage_prompts(state: Dict[str, Any]) -> Tuple[str, str]:
    distribution_prompt = ""
    intent_prompt = ""
    if load_stage_by_id is None:
        return distribution_prompt, intent_prompt
    stage_id = str(state.get("current_stage") or "experimenting").strip()
    try:
        stage_config = load_stage_by_id(stage_id)
        prompts = (stage_config or {}).get("prompts") or {}
        if isinstance(prompts, dict):
            distribution_prompt = (prompts.get("distribution_prompt") or "").strip()
            intent_prompt = (prompts.get("intent_goal") or "").strip()
            if not intent_prompt:
                intent_prompt = (prompts.get("intent_prompt") or "").strip()
        if not intent_prompt and isinstance(prompts, dict):
            intent_prompt = (prompts.get("strategy_prompt") or "").strip()
        if not intent_prompt:
            act = (stage_config or {}).get("act") or {}
            sp = act.get("system_prompt") or ""
            if isinstance(sp, str) and sp.strip():
                intent_prompt = sp.strip()
    except Exception:
        pass
    return distribution_prompt, intent_prompt


def _get_current_strategy_prompt(state: Dict[str, Any]) -> str:
    cur = state.get("current_strategy")
    if not isinstance(cur, dict):
        return ""
    prompt = cur.get("prompt")
    return (prompt or "").strip() if isinstance(prompt, str) else ""


def _select_user_profile(state: Dict[str, Any]) -> Any:
    full_profile = state.get("user_inferred_profile") or state.get("user_profile") or {}
    selected_keys = state.get("selected_profile_keys") or []
    if selected_keys and isinstance(full_profile, dict):
        return {k: full_profile[k] for k in selected_keys if k in full_profile}
    return full_profile


# -------------------------
# required tasks
# -------------------------
def _task_to_user_instruction(task: Any) -> str:
    if not isinstance(task, dict):
        t = safe_text(task).strip()
        return t if t else "完成系统指定任务"

    for key in ("user_facing_instruction", "instruction", "must_do", "directive", "goal", "description", "title"):
        val = task.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    action = (task.get("action") or task.get("task_name") or task.get("task_type") or "").strip()
    action_l = str(action).lower()
    mapping = {
        "ask_user_name": "回复中必须明确询问对方的姓名或称呼",
        "ask_user_age": "回复中必须明确询问对方的年龄",
        "ask_user_occupation": "回复中必须明确询问对方的职业/身份",
        "ask_user_location": "回复中必须明确询问对方所在城市/地区",
    }
    if action_l in mapping:
        return mapping[action_l]

    bits: List[str] = []
    for k in ("id", "name", "field", "target_field", "question", "prompt"):
        v = task.get(k)
        if isinstance(v, str) and v.strip():
            if k in ("question", "prompt"):
                return v.strip()
            bits.append(v.strip())
    if bits:
        return " / ".join(bits[:3])
    return "完成系统指定任务"


def _extract_required_tasks(requirements: Any) -> List[str]:
    if not isinstance(requirements, dict):
        return []
    tasks = requirements.get("tasks_for_lats")
    if not isinstance(tasks, list):
        return []

    required: List[str] = []
    for t in tasks:
        if isinstance(t, dict):
            is_urgent = bool(t.get("is_urgent")) or (str(t.get("task_type") or "").lower() == "urgent")
            if is_urgent:
                required.append(_task_to_user_instruction(t))

    seen = set()
    out: List[str] = []
    for x in required:
        key = (x or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


# -------------------------
# content_move actions (仅改词汇/表述适配 gpt-4.1-mini)
# -------------------------
DEFAULT_CONTENT_MOVE_ACTION: Dict[str, str] = {
    "DRILL_DOWN": "ASK_FOR_DETAILS",
    "EXTRACT_PATTERN": "GENERALIZE_PATTERN",
    "GIVE_ANALOGY": "GIVE_ONE_ANALOGY",
    "PROGRESS_NEXT": "CHECKPOINT_AND_NEXT",
    "PROPOSE_CAUSE": "PROPOSE_CAUSAL_CHAIN",
    "WHAT_IF": "COUNTERFACTUAL_IF_THEN",
    "DIAGNOSE_BLOCKER": "NAME_THE_BLOCKER",
    "CLARIFY_TERMS": "CONFIRM_INTERPRETATION",
}


def _lookup_content_move_action(tag: str) -> str:
    t = (tag or "").strip().upper()
    if not t:
        return ""
    if t == "FREE":
        return "FREEFORM"

    if load_content_moves is not None:
        try:
            moves = load_content_moves() or []
            for m in moves:
                mt = str((m or {}).get("tag") or "").strip().upper()
                if mt == t:
                    act = str((m or {}).get("action") or (m or {}).get("action_en") or "").strip()
                    if act:
                        return act
        except Exception:
            pass

    return DEFAULT_CONTENT_MOVE_ACTION.get(t, "")


def _normalize_action_for_gpt41mini(tag: str, action: str) -> str:
    t = (tag or "").strip().upper()
    if t == "PROGRESS_NEXT":
        return "CHECKPOINT_AND_NEXT"
    if t == "CLARIFY_TERMS":
        return "CONFIRM_INTERPRETATION"
    return (action or "").strip()


# -------------------------
# prompt constants
# -------------------------
STRICT_STRATEGY_REMINDER = "⚠️必须严格遵守【当前策略】的硬约束与意图，不得违背。"

MANDATORY_RULES_NAMES = """必须遵守以下部分（按下方顺序）。若规则之间冲突，则按本列表从上到下的优先顺序执行（靠前的优先于靠后的）：
- 当前策略
- 时间与会话上下文
- 阶段意图与行为准则
- 风格说明（style）
- 输出格式
""".strip()


def _build_system_prompt_b(
    *,
    bot_name: str,
    user_name: str,
    bot_basic_info: Dict[str, Any],
    bot_persona: Dict[str, Any],
    user_basic_info: Dict[str, Any],
    user_profile_selected: Any,
    system_memory: str,
    style_profile: Any,
    requirements: Any,
    required_tasks: List[str],
    k: int = 1,
    intent_prompt: str = "",
    strategies_prompt: str = "",
) -> Dict[str, str]:
    header = f"你是 {bot_name}，正在和 {user_name} 对话。"

    identity_hard = (
        f"【身份硬约束】你的名字是且仅是「{bot_name}」。"
        "回复中不得自称或使用他人名字（例如不得说「我叫XXX」「你可以叫我XXX」除非 XXX 就是你本人名字）。"
        "若需自报姓名，必须且只能使用你的名字。"
    )

    background = f"""
【背景信息（只用于生成，不要照抄给用户）】
- bot_basic_info：{_safe_text_limited(bot_basic_info, PROMPT_BG_MAX_CHARS // 3)}
- bot_persona：{_safe_text_limited(bot_persona, PROMPT_BG_MAX_CHARS // 3)}
- user_basic_info：{_safe_text_limited(user_basic_info, PROMPT_BG_MAX_CHARS // 3)}
- user_profile（本轮选中字段）：{_safe_text_limited(user_profile_selected, PROMPT_BG_MAX_CHARS // 3)}
""".strip()

    memory_block = f"【memory（摘要 + 检索）】\n{_truncate_text(system_memory, PROMPT_MEMORY_MAX_CHARS)}".strip()

    style_block = (
        "【风格说明（style 节点 6 维参数：FORMALITY/POLITENESS/WARMTH/CERTAINTY/CHAT_MARKERS/EXPRESSION_MODE）】\n"
        + _safe_text_limited(style_profile, PROMPT_STYLE_MAX_CHARS)
    ).strip()

    intent_block = f"【阶段核心目的】\n{intent_prompt}".strip() if intent_prompt else ""
    strategy_block = f"【当前策略（本轮回调策略）】\n{strategies_prompt}".strip() if strategies_prompt else ""

    required_tasks_block = ""
    if required_tasks:
        lines = [f"- {str(t).strip()}" for t in required_tasks if str(t).strip()]
        if lines:
            required_tasks_block = "【必须完成任务列表】\n" + "\n".join(lines)

    askq_soft_block = """【提问软惩罚（全局无差别随机）】
- 用户消息末尾会给出 ASKQ 元数据（0/1），它是系统随机采样的“提问倾向”，与你觉得是否需要澄清无关。
- ASKQ=0：倾向不提问，用陈述/建议/推进来表达。
- ASKQ=1：倾向允许提问（但也不必强行问）。
- 不要在最终回复中提及 ASKQ。
""".strip()

    # ✅ 修复物理锚定冲突：增加明确例外说明（仍禁止硬编客观事实）
    writing_rules = f"""【写作要求（生成给用户看的自然回复）】
- 更真实、自然、像人一样的发消息，真实的对话往往没有那么长，说多了反而假
- 避免客服模板句式或“出戏说明”（例如“作为一个模型/根据设定/我可以为你提供…”）。
- TIME_* 标记为元数据，不要复述；不要输出精确时间戳（除非用户明确问）。
- 不要输出你的推理过程或“内心独白”，只输出给用户看的最终回复。

{TIME_SLICE_BEHAVIOR_RULES}

- 若最后一条用户消息包含 CONTENT_OP / CONTENT_OP_ACTION：它只表示“内容推进方向/动作意图”，不是固定模板；语气与措辞仍服从 style_profile。
- 不要在最终回复中提及 CONTENT_OP / CONTENT_OP_ACTION / light / medium / strong 等元标签。
- 不要编造可被当作客观事实的现实环境细节（光线/温度/声音/地点/具体经历等），除非这些信息已在上下文明确给出。
- 例外：当 CONTENT_OP=物理锚定（Physical Anchoring）时，允许使用“非事实性、可撤销”的轻度锚定描写（如身体动作/停顿/语气/“仿佛能想象到…”），但仍禁止硬编具体地点、具体温度数值、具体声音来源等。
""".strip()

    if k <= 1:
        schema_block = """【输出 JSON schema（只输出 JSON，不要额外文字）】
必须输出一个 JSON 对象，形如：
{"reply": "<你将发送给用户的完整回复>"}
""".strip()
    else:
        schema_block = f"""【输出 JSON schema（只输出 JSON，不要额外文字）】
必须输出一个 JSON 对象，形如：
{{"candidates":[{{"reply":"..."}}, ...]}}
- candidates 至少 1 条，最多 {int(k)} 条（尽量接近 {int(k)} 条）
- 每条 reply 都必须是“可直接发送给用户”的完整回复
""".strip()

    return {
        "header": header,
        "identity_hard": identity_hard,
        "background": background,
        "memory_block": memory_block,
        "style_block": style_block,
        "intent_block": intent_block,
        "strategy_block": strategy_block,
        "required_tasks_block": required_tasks_block,
        "askq_soft_block": askq_soft_block,
        "writing_rules": writing_rules,
        "schema_block": schema_block,
    }


def _planner_sampling_for_round(gen_round: int) -> Tuple[float, float]:
    """ReplyPlanner 的 temperature/top_p 统一从 graph_llm_config 读取（由 graph.py 设置），与 gen_round 无关。"""
    try:
        from app.core import graph_llm_config as _glc
        return (
            getattr(_glc, "PLANNER_TEMPERATURE", 1.1),
            getattr(_glc, "PLANNER_TOP_P", 0.95),
        )
    except Exception:
        return (1.1, 0.95)


def _parse_planner_response(data: Dict[str, Any], k: int) -> List[ReplyPlan]:
    out: List[ReplyPlan] = []

    def _one_plan(c: Any) -> str:
        raw = ""
        if isinstance(c, dict) and isinstance(c.get("reply"), str):
            raw = c.get("reply", "").strip()
        elif isinstance(c, dict) and isinstance(c.get("messages"), list) and c["messages"]:
            m0 = c["messages"][0]
            if isinstance(m0, str):
                raw = m0.strip()
            elif isinstance(m0, dict):
                raw = str(m0.get("content") or "").strip()
            else:
                raw = str(m0).strip()
        return strip_candidate_prefix(raw) if raw else ""

    if k <= 1:
        reply = (data.get("reply") or "").strip() if isinstance(data.get("reply"), str) else ""
        if not reply and isinstance(data.get("messages"), list) and data["messages"]:
            m0 = data["messages"][0]
            reply = _one_plan({"reply": m0} if isinstance(m0, str) else m0)
        else:
            reply = strip_candidate_prefix(reply) if reply else ""
        if reply:
            out.append({"reply": reply})  # type: ignore[typeddict-item]
    else:
        cands = data.get("candidates")
        if not isinstance(cands, list) and isinstance(data.get("reply"), str) and data.get("reply", "").strip():
            cands = [{"reply": data["reply"].strip()}]
        if isinstance(cands, list):
            for c in cands[: int(k)]:
                reply = _one_plan(c) if isinstance(c, dict) else ""
                if reply:
                    out.append({"reply": reply})  # type: ignore[list-item,typeddict-item]

    return out


def _invoke_planner_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    *,
    k: int = 1,
    content_move_text: Optional[str] = None,
    content_move_tag: Optional[str] = None,
    content_move_action: Optional[str] = None,
    global_guidelines: Optional[str] = None,
    gen_round: int = 0,
    user_message_only: bool = False,
) -> List[ReplyPlan]:
    if llm_invoker is None:
        return []

    system_memory = build_system_memory_block(state)
    style_profile = build_style_profile(state)
    requirements = state.get("requirements") or {}
    required_tasks = _extract_required_tasks(requirements)

    bot_basic_info = state.get("bot_basic_info") or {}
    bot_persona = state.get("bot_persona") or {}
    user_basic_info = state.get("user_basic_info") or {}
    user_profile_selected = _select_user_profile(state)

    bot_name = safe_text((bot_basic_info or {}).get("name") or "Bot").strip() or "Bot"
    user_name = safe_text((user_basic_info or {}).get("name") or "不知道姓名的人").strip() or "不知道姓名的人"

    _, intent_prompt = _get_stage_prompts(state)
    strategies_prompt = _get_current_strategy_prompt(state)

    parts = _build_system_prompt_b(
        bot_name=bot_name,
        user_name=user_name,
        bot_basic_info=bot_basic_info,
        bot_persona=bot_persona,
        user_basic_info=user_basic_info,
        user_profile_selected=user_profile_selected,
        system_memory=system_memory,
        style_profile=style_profile,
        requirements=requirements,
        required_tasks=required_tasks,
        k=int(k),
        intent_prompt=intent_prompt,
        strategies_prompt=strategies_prompt,
    )

    time_context_block = _truncate_text(build_time_context_block(state), PROMPT_TIME_MAX_CHARS)
    user_input = safe_text(state.get("external_user_text") or state.get("user_input"))
    monologue = safe_text(state.get("inner_monologue") or "")
    monologue_block = (
        "【内心动机】（只当参考，不要照抄）：\n" + _truncate_text(monologue, PROMPT_MONOLOGUE_MAX_CHARS)
        if monologue.strip()
        else ""
    )

    askq_flags = _sample_askq_flags(int(k))
    askq_meta = _render_askq_meta(int(k), askq_flags, content_move_tag=str(content_move_tag or "").strip() or None)

    # 本轮 content_op
    op_tag = ""
    op_action = ""
    if content_move_tag and str(content_move_tag).strip():
        op_tag = str(content_move_tag).strip()
        raw_action = (content_move_action or "").strip() or _lookup_content_move_action(op_tag)
        op_action = _normalize_action_for_gpt41mini(op_tag, raw_action) or raw_action

    content_op_hint_block = ""
    if op_tag:
        if op_tag.upper() == "FREE":
            content_op_hint_block = "【本轮 CONTENT_OP】FREE（自由发挥）"
        else:
            content_op_hint_block = (
                "【本轮 CONTENT_OP】\n"
                f"CONTENT_OP={op_tag}\n"
                f"CONTENT_OP_ACTION={op_action}"
            )

    system_blocks: List[str] = []
    system_blocks.append(STRICT_STRATEGY_REMINDER)
    system_blocks.append(parts["header"])
    system_blocks.append(parts["identity_hard"])
    system_blocks.append(parts["background"])
    system_blocks.append(MANDATORY_RULES_NAMES)

    if parts["strategy_block"]:
        system_blocks.append(parts["strategy_block"])

    system_blocks.append(parts["style_block"])
    if monologue_block:
        system_blocks.append(monologue_block)

    system_blocks.append(parts["memory_block"])

    if parts.get("required_tasks_block"):
        system_blocks.append(parts["required_tasks_block"])

    system_blocks.append("【时间与会话上下文】\n" + time_context_block)

    if parts["intent_block"]:
        system_blocks.append(parts["intent_block"])

    if content_op_hint_block:
        system_blocks.append(content_op_hint_block)

    if global_guidelines and isinstance(global_guidelines, str) and global_guidelines.strip():
        system_blocks.append("【全局指导原则】\n" + global_guidelines.strip())

    system_blocks.append(parts["askq_soft_block"])
    system_blocks.append(parts["writing_rules"])
    system_blocks.append(parts["schema_block"])

    system_prompt = "\n\n".join(system_blocks)
    print(
        f"[ReplyPlanner] system_len={len(system_prompt)} user_len={len(user_input)} "
        f"required_tasks={len(required_tasks)} k={int(k)} askq_p={_question_prob():.3f} askq={askq_flags[:min(3,len(askq_flags))]}"
    )

    # user 消息
    if user_message_only:
        last_user_content = user_input
    elif op_tag and op_tag.strip():
        tag = op_tag.strip()
        if tag.upper() == "FREE":
            last_user_content = (
                "CONTENT_OP=FREE\n"
                "CONTENT_OP_ACTION=FREEFORM\n"
                "请基于对方消息生成 3 条候选回复，候选之间必须明显不同（内容角度/推进方式不同），不能只是同义改写。\n"
                "不要在正文里写任何标签或编号。\n"
                f"对方消息：{user_input}"
            )
        else:
            last_user_content = (
                f"CONTENT_OP={tag}\n"
                f"CONTENT_OP_ACTION={op_action}\n"
                "请基于对方消息生成 3 条候选回复，并按 light → medium → strong 的顺序排列。\n"
                "候选之间必须明显不同（内容角度/推进方式不同），不能只是同义改写。\n"
                "不要在正文里写任何标签或编号。\n"
                f"对方消息：{user_input}"
            )
    else:
        user_parts: List[str] = []
        if content_move_text and content_move_text.strip():
            user_parts.append("【本轮生成策略（content_move）】\n" + content_move_text.strip())
        user_parts.append("本轮用户输入在最后一条用户消息中，请根据其生成回复。")
        user_parts.append(user_input)
        last_user_content = "\n\n".join(p for p in user_parts if p.strip())

    last_user_content = (last_user_content or "").strip() + "\n\n" + STRICT_STRATEGY_REMINDER

    if askq_meta:
        last_user_content += "\n" + askq_meta

    if (not user_message_only) and op_tag:
        if op_tag.upper() == "FREE":
            last_user_content += "\nCONTENT_OP=FREE\nCONTENT_OP_ACTION=FREEFORM"
        else:
            last_user_content += f"\nCONTENT_OP={op_tag}\nCONTENT_OP_ACTION={op_action}"

    body_messages = get_chat_buffer_body_messages_with_time_slices(state, limit=20)

    log_name = "ReplyPlanGen" if int(k) <= 1 else "ReplyPlanGen (Candidates)"
    log_prompt_and_params(
        log_name,
        system_prompt=system_prompt,
        user_prompt=last_user_content,
        messages=body_messages,
        params={
            "gen_round": gen_round,
            "k": int(k),
            "temperature": _planner_sampling_for_round(gen_round)[0],
            "top_p": _planner_sampling_for_round(gen_round)[1],
            "frequency_penalty": _planner_frequency_presence_penalty()[0],
            "presence_penalty": _planner_frequency_presence_penalty()[1],
            "has_content_move": bool(content_move_text or content_move_tag),
            "has_global_guidelines": bool(global_guidelines),
            "required_tasks": required_tasks,
            "content_move_tag": op_tag,
            "content_move_action": op_action,
            "askq_prob": _question_prob(),
            "askq_flags": askq_flags,
        },
    )

    try:
        temperature, top_p = _planner_sampling_for_round(gen_round)
        freq_penalty, pres_penalty = _planner_frequency_presence_penalty()
        messages = [SystemMessage(content=system_prompt), *body_messages, HumanMessage(content=last_user_content)]
        data = None
        resp = None

        if hasattr(llm_invoker, "with_structured_output"):
            schema = ReplyPlannerSingle if int(k) <= 1 else ReplyPlannerCandidates
            structured = llm_invoker.with_structured_output(schema)

            log_name_ctx = "ReplyPlanGen" if int(k) <= 1 else "ReplyPlanGenCandidates"
            tok = set_current_node(log_name_ctx)
            try:
                obj = structured.invoke(
                    messages,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=freq_penalty,
                    presence_penalty=pres_penalty,
                    max_tokens=_reply_plan_max_tokens(),
                )
                data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
            except Exception as e:
                reset_current_node(tok)
                err_name, err_msg = type(e).__name__, str(e)
                if "LengthFinishReasonError" in err_name or "length limit" in err_msg.lower():
                    print(f"  [ReplyPlanner] ⚠ 输出达到长度上限被截断，无法解析: {err_name}", flush=True)
                    return []
                print(f"  [ReplyPlanner] ⚠ structured_output 调用异常: {err_name}: {err_msg[:200]}", flush=True)
                raise
            reset_current_node(tok)
        else:
            resp = llm_invoker.invoke(
                messages,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=freq_penalty,
                presence_penalty=pres_penalty,
                max_tokens=_reply_plan_max_tokens(),
            )
            content = getattr(resp, "content", "") or ""
            data = parse_json_from_llm(content)

        if not isinstance(data, dict):
            content = getattr(resp, "content", "") or "" if resp is not None else ""
            preview = (content[:400] + "...") if len(content) > 400 else content
            print(f"  [ReplyPlanner] ⚠ JSON解析失败: {preview}", flush=True)
            clean = (content or "").strip()
            if clean and len(clean) < 4000:
                data = {"reply": clean} if int(k) <= 1 else {"candidates": [{"reply": clean}]}
                print("  [ReplyPlanner] 已用原文作为单条 reply 兜底", flush=True)
            else:
                return []

        log_llm_response(
            log_name,
            resp if resp is not None else "(structured_output)",
            parsed_result=data if (int(k) <= 1 or _full_logs()) else {"candidates": len(data.get("candidates") or [])},
        )

        plans = _parse_planner_response(data, int(k))
        if int(k) <= 1:
            print("  [计划生成] ✓ reply=1条" if plans else "  [计划生成] ⚠ reply 为空")
        else:
            print(f"  [计划生成] ✓ candidates={len(plans)}条")
        return plans

    except Exception as e:
        import traceback
        err_type, err_msg = type(e).__name__, str(e)
        print(f"  [ReplyPlanner] ❌ 异常: {err_type}: {err_msg[:80]}")
        traceback.print_exc()
        return []


def plan_reply_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    *,
    global_guidelines: Optional[str] = None,
    gen_round: int = 0,
    user_message_only: bool = False,
) -> Optional[ReplyPlan]:
    plans = _invoke_planner_llm(
        state,
        llm_invoker,
        k=1,
        global_guidelines=global_guidelines,
        gen_round=gen_round,
        user_message_only=user_message_only,
    )
    return plans[0] if plans else None


def plan_reply_candidates_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    *,
    k: int = 10,
    content_move_text: Optional[str] = None,
    global_guidelines: Optional[str] = None,
    gen_round: int = 0,
) -> List[ReplyPlan]:
    return _invoke_planner_llm(
        state,
        llm_invoker,
        k=int(k),
        content_move_text=content_move_text,
        global_guidelines=global_guidelines,
        gen_round=gen_round,
    )


# LATS V3：5 路并行 = 1 FREE + 4 路各 1 个 move（来自 inner_monologue 的 selected_content_move_ids）；每路三档 → 共 15 候选。
CANDIDATE_27_DEGREES = ("light", "medium", "strong")


def _one_content_move_gen(
    state: Dict[str, Any],
    llm_invoker: Any,
    slot_index: int,
    tag: str,
    action: str = "",
) -> List[Dict[str, Any]]:
    plans = _invoke_planner_llm(
        state,
        llm_invoker,
        k=3,
        content_move_tag=tag,
        content_move_action=action,
        gen_round=0,
    )
    base_id = slot_index * 3
    out: List[Dict[str, Any]] = []
    for i, rp in enumerate(plans[:3]):
        reply = (rp or {}).get("reply") or ""
        if isinstance(reply, str) and reply.strip():
            out.append(
                {
                    "id": base_id + i,
                    "tag": tag,
                    "action": action or ("FREEFORM" if tag.upper() == "FREE" else ""),
                    "degree": CANDIDATE_27_DEGREES[i] if i < len(CANDIDATE_27_DEGREES) else "medium",
                    "reply": reply.strip(),
                }
            )
    return out


def plan_reply_27_via_content_moves(
    state: Dict[str, Any],
    llm_invoker: Any,
) -> List[Dict[str, Any]]:
    """5 路并行：1 FREE + 4 路各传 1 个 move（从 state.selected_content_move_ids 取名称+动作）；每路三档 → 共 15 候选。"""
    if llm_invoker is None:
        return []

    # ✅ 诊断：yaml_loader 是否可用
    if load_pure_content_transformations is None:
        print(
            "  [ReplyPlanner] ⚠ load_pure_content_transformations=None（utils.yaml_loader 导入失败或缺失该函数）",
            flush=True,
        )

    selected_ids: List[int] = []
    raw = state.get("selected_content_move_ids") or []
    for x in raw[:4]:
        if x is None:
            continue
        try:
            selected_ids.append(int(x))
        except (TypeError, ValueError):
            continue

    # 诊断：日志中可见「8 选 4」是否传入（stdout 被 bottobot 重定向到 log）
    print(f"  [ReplyPlanner] selected_content_move_ids from state: {raw!r} -> resolved ids: {selected_ids!r}", flush=True)

    id_to_move: Dict[int, Dict[str, Any]] = {}
    if load_pure_content_transformations:
        try:
            raw_moves = load_pure_content_transformations()
            moves = _normalize_pure_content_transformations(raw_moves)
            if not moves:
                print("  [ReplyPlanner] ⚠ pure_content_transformations 为空（yaml 读取到了但列表为空？）", flush=True)
            for m in moves:
                mid = m.get("id")
                if mid is not None:
                    id_to_move[int(mid)] = m
        except Exception as e:
            print(
                f"  [ReplyPlanner] ⚠ load_pure_content_transformations failed: {type(e).__name__}: {str(e)[:160]}",
                flush=True,
            )

    tasks: List[Tuple[int, str, str]] = []
    # 前 4 路：每路只传 1 个 move（selected_ids[i] → slot i 的 name+action）
    for slot_index, move_id in enumerate(selected_ids):
        m = id_to_move.get(move_id)
        if not m:
            continue
        name = (m.get("name") or "").strip() or "UNKNOWN"
        action = (m.get("content_operation") or "").strip() or ""
        action = _normalize_pure_content_move_action_text(name, action)  # ✅ 修复物理锚定冲突
        tasks.append((slot_index, name, action))

    # 第 5 路：自由
    free_slot = len(tasks)
    tasks.append((free_slot, "FREE", "FREEFORM"))

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(5, 16)) as ex:
        futs = {
            ex.submit(_one_content_move_gen, state, llm_invoker, slot_index, tag, action): (slot_index, tag, action)
            for slot_index, tag, action in tasks
        }
        for fut in as_completed(futs):
            try:
                results.extend(fut.result())
            except Exception as e:
                slot_index, tag, action = futs[fut]
                print(f"  [ReplyPlanner] content_move slot={slot_index} tag={tag} action={action} 异常: {e}", flush=True)

    results.sort(key=lambda x: int(x.get("id", 0)))
    n_expected = len(tasks) * 3
    if len(results) < n_expected:
        print(f"  [ReplyPlanner] 5 路仅得到 {len(results)} 条候选（预期共 {n_expected}）", flush=True)
    return results
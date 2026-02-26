"""安全层 + 守门层节点（新架构第一道关）。

合并了旧版三个模块的功能：
1. 注入攻击 / 脱角色指令检测（utils/security.py）
2. HighStakes Router（anti_ai_defense, boundary_defense, yielding_apology,
   reasonable_assistance, physical_limitation_refusal）

原则：宁可漏报不可误报。只有非常确定时才触发。
触发后下游走 fast_safety_reply → processor，跳过独白/生成/judge 流水线。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.lats.prompt_utils import safe_text
from app.state import AgentState
from src.schemas import SafetyOutput, StrategyRouterOutput
from utils.detailed_logging import log_prompt_and_params
from utils.llm_json import parse_json_from_llm
from utils.prompt_helpers import format_relationship_for_llm, format_stage_for_llm, stage_to_knapp_index
from utils.security import detect_manipulation_attempts, detect_injection_attempt
from utils.tracing import trace_if_enabled
from utils.yaml_loader import load_strategies

logger = logging.getLogger(__name__)

# HighStakes 策略 id 集合（同旧 strategy_routers.py）
HIGH_STAKES_IDS = frozenset({
    "anti_ai_defense",
    "boundary_defense",
    "yielding_apology",
    "reasonable_assistance",
    "physical_limitation_refusal",
})

RECENT_DIALOGUE_LAST_N = 30
RECENT_MSG_CONTENT_MAX = 500
LATEST_USER_TEXT_MAX = 800
RECENT_DIALOGUE_CHARS = 2500


def _is_user_message(m: Any) -> bool:
    t = getattr(m, "type", "") or ""
    return "human" in t.lower() or "user" in t.lower()


def _gather_context(state: Dict[str, Any]) -> Dict[str, Any]:
    raw_buffer = state.get("chat_buffer") or state.get("messages") or []
    chat_buffer = list(raw_buffer)[-RECENT_DIALOGUE_LAST_N:]

    latest_user_text_raw = (state.get("user_input") or "").strip()
    if not latest_user_text_raw and chat_buffer:
        for m in reversed(chat_buffer):
            if _is_user_message(m):
                latest_user_text_raw = (getattr(m, "content", "") or str(m)).strip()
                break
        if not latest_user_text_raw:
            latest_user_text_raw = "（无用户新句）"
    latest_user_text = (latest_user_text_raw or "（无用户消息）")[:LATEST_USER_TEXT_MAX]

    lines: List[str] = []
    for m in chat_buffer:
        role = "Human" if _is_user_message(m) else "AI"
        content = (getattr(m, "content", "") or str(m)).strip()
        if len(content) > RECENT_MSG_CONTENT_MAX:
            content = content[:RECENT_MSG_CONTENT_MAX] + "…"
        lines.append(f"{role}: {content}")
    recent_dialogue = "\n".join(lines) if lines else "（无历史对话）"

    stage_id = str(state.get("current_stage") or "initiating")
    relationship_state = state.get("relationship_state") or {}
    rel_desc = format_relationship_for_llm(relationship_state)
    stage_desc = format_stage_for_llm(stage_id, include_judge_hints=False)

    bot_basic_info = state.get("bot_basic_info") or {}
    user_basic_info = state.get("user_basic_info") or {}

    return {
        "latest_user_text": latest_user_text,
        "latest_user_text_raw": latest_user_text_raw,
        "recent_dialogue": recent_dialogue,
        "rel_desc": rel_desc,
        "stage_desc": stage_desc,
        "bot_basic_info": bot_basic_info,
        "user_basic_info": user_basic_info,
        "stage_id": stage_id,
    }


def _rule_based_check(latest_user_text: str) -> Optional[str]:
    """规则层（不调用 LLM）：检测注入 / 操控尝试，返回 strategy_id 或 None。"""
    try:
        manipulation_flags = detect_manipulation_attempts(latest_user_text)
        is_injection, _ = detect_injection_attempt(latest_user_text)
        if any(manipulation_flags.values()) or is_injection:
            logger.info("[Safety] 规则层检测到注入/操控")
            return "anti_ai_defense"
    except Exception as e:
        logger.warning("[Safety] 规则层检测异常: %s", e)
    return None


def _build_conditions_block(filtered: List[Dict[str, Any]]) -> str:
    if not filtered:
        return "（当前阶段无可用策略，请输出 triggered=false。）"
    lines: List[str] = []
    for s in filtered:
        name = s.get("name") or s.get("id") or ""
        trigger = (s.get("trigger") or "").strip()
        lines.append(f"- **{name}**（id: {s.get('id')}）：{trigger}")
    return "\n".join(lines)


def _filter_strategies(strategies: List[Dict[str, Any]], stage_index: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in strategies or []:
        if not isinstance(s, dict):
            continue
        sid = s.get("id")
        if sid not in HIGH_STAKES_IDS:
            continue
        stages = s.get("knapp_stages")
        if isinstance(stages, list) and stage_index in stages:
            out.append(s)
    return out


def _llm_check(
    llm_invoker: Any,
    ctx: Dict[str, Any],
    strategies: List[Dict[str, Any]],
    stage_index: int,
) -> Optional[str]:
    """LLM 层：在过滤后的 HIGH_STAKES 策略中做路由，返回 strategy_id 或 None。"""
    filtered = _filter_strategies(strategies, stage_index)
    if not filtered:
        return None

    conditions_block = _build_conditions_block(filtered)
    system_prompt = "\n".join([
        "你是高危意图分类器。判断当前用户消息是否命中下列高危情形之一。",
        "宁可漏报，不可误报。仅当非常确定时才输出命中 id，否则一律 triggered=false。",
        "",
        "## 检测条件",
        conditions_block,
        "",
        "（输出格式由系统约束：triggered=true/false，strategy_id 为命中的 id 或 null。）",
    ])

    user_content = f"""## 背景
- 关系：{ctx['rel_desc'][:400]}
- 阶段：{ctx['stage_desc'][:300]}

## 历史对话（保留最近部分）
{(ctx['recent_dialogue'] or '')[-RECENT_DIALOGUE_CHARS:]}

## 当前用户消息
{ctx['latest_user_text']}

请判断是否命中检测条件。"""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    log_prompt_and_params("Safety/LLM", messages=messages)
    try:
        if hasattr(llm_invoker, "with_structured_output"):
            try:
                structured = llm_invoker.with_structured_output(SafetyOutput)
                obj = structured.invoke(messages)
                if obj.triggered and obj.strategy_id in HIGH_STAKES_IDS:
                    return obj.strategy_id
                return None
            except Exception:
                pass
        # fallback: parse JSON
        msg = llm_invoker.invoke(messages)
        raw = (getattr(msg, "content", "") or str(msg)).strip()
        parsed = parse_json_from_llm(raw)
        if isinstance(parsed, dict):
            if parsed.get("triggered") and parsed.get("strategy_id") in HIGH_STAKES_IDS:
                return str(parsed["strategy_id"])
    except Exception as e:
        logger.warning("[Safety] LLM 检测异常: %s", e)
    return None


def create_safety_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """创建安全层 + 守门层节点。"""

    @trace_if_enabled(
        name="Safety",
        run_type="chain",
        tags=["node", "safety"],
        metadata={"state_outputs": ["safety_triggered", "safety_strategy_id"]},
    )
    def safety_node(state: AgentState) -> dict:
        ctx = _gather_context(state)

        # 1. 规则层（快速，无 LLM）
        rule_hit = _rule_based_check(ctx["latest_user_text_raw"])
        if rule_hit:
            logger.info("[Safety] 触发（规则层）: %s", rule_hit)
            return {"safety_triggered": True, "safety_strategy_id": rule_hit}

        # 2. LLM 层（HIGH_STAKES 路由）
        if llm_invoker is not None:
            try:
                strategies = load_strategies()
                stage_index = stage_to_knapp_index(state.get("current_stage"))
                llm_hit = _llm_check(llm_invoker, ctx, strategies, stage_index)
                if llm_hit:
                    logger.info("[Safety] 触发（LLM 层）: %s", llm_hit)
                    return {"safety_triggered": True, "safety_strategy_id": llm_hit}
            except Exception as e:
                logger.warning("[Safety] LLM 层异常: %s", e)

        logger.info("[Safety] 未触发")
        return {"safety_triggered": False, "safety_strategy_id": None}

    return safety_node

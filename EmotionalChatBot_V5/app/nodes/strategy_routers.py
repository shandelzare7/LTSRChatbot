"""
策略路由三节点：按当前 Knapp 阶段动态过滤策略，调用 gpt-4o-mini 判定命中；下游 strategy_resolver 合并结果写入 current_strategy。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage

from app.lats.prompt_utils import safe_text
from app.state import AgentState
from src.schemas import StrategyRouterOutput
from utils.llm_json import parse_json_from_llm
from utils.prompt_helpers import format_relationship_for_llm, format_stage_for_llm, stage_to_knapp_index
from utils.tracing import trace_if_enabled
from utils.yaml_loader import load_strategies


RECENT_DIALOGUE_LAST_N = 30
RECENT_MSG_CONTENT_MAX = 500
LATEST_USER_TEXT_MAX = 800
RECENT_DIALOGUE_CHARS = 2500  # 截断时保留「最近」的字符数（保留结尾、不保留开头）

# 各节点策略 id 集合（HighStakes 共 5 个）
HIGH_STAKES_IDS = {"anti_ai_defense", "boundary_defense", "yielding_apology", "reasonable_assistance", "physical_limitation_refusal"}
EMOTIONAL_GAME_IDS = {"shit_test_counter", "co_rumination", "passive_aggression", "deflection", "flirting_banter"}
FORM_RHYTHM_IDS = {"tldr_refusal", "micro_reaction", "clarification", "detail_nitpicking"}

BUSYNESS_THRESHOLD = 0.6  # >= 则 Bot 状态为「忙碌」


def _is_user_message(m: Any) -> bool:
    t = getattr(m, "type", "") or ""
    return "human" in t.lower() or "user" in t.lower()


def _gather_router_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """从 state 抽取：当轮用户消息、历史对话、关系/阶段描述、PAD、Bot 状态与文本长度（供节点 C）。"""
    raw_buffer = state.get("chat_buffer") or state.get("messages") or []
    chat_buffer = list(raw_buffer)[-RECENT_DIALOGUE_LAST_N:]

    stage_id = str(state.get("current_stage") or "initiating")
    relationship_state = state.get("relationship_state") or {}
    mood_state = state.get("mood_state") or {}

    latest_user_text_raw = (state.get("user_input") or "").strip()
    if not latest_user_text_raw and chat_buffer:
        for m in reversed(chat_buffer):
            if _is_user_message(m):
                latest_user_text_raw = (getattr(m, "content", "") or str(m)).strip()
                break
        if not latest_user_text_raw:
            latest_user_text_raw = "（无用户新句）"
    latest_user_text = (latest_user_text_raw or "（无用户消息）")[:LATEST_USER_TEXT_MAX]
    text_len = len(latest_user_text_raw or "")

    lines: List[str] = []
    for m in chat_buffer:
        role = "Human" if _is_user_message(m) else "AI"
        content = (getattr(m, "content", "") or str(m)).strip()
        if len(content) > RECENT_MSG_CONTENT_MAX:
            content = content[:RECENT_MSG_CONTENT_MAX] + "…"
        lines.append(f"{role}: {content}")
    recent_dialogue = "\n".join(lines) if lines else "（无历史对话）"

    rel_desc = format_relationship_for_llm(relationship_state)
    stage_desc = format_stage_for_llm(stage_id, include_judge_hints=False)

    pleasure = mood_state.get("pleasure")
    arousal = mood_state.get("arousal")
    dominance = mood_state.get("dominance")
    busyness = float(mood_state.get("busyness", 0) or 0)
    pad_str = f"pleasure={pleasure}, arousal={arousal}, dominance={dominance}, busyness={busyness}"

    bot_state = "忙碌" if busyness >= BUSYNESS_THRESHOLD else "空闲"

    bot_basic_info = state.get("bot_basic_info") or {}
    user_basic_info = state.get("user_basic_info") or {}

    return {
        "latest_user_text": latest_user_text,
        "recent_dialogue": recent_dialogue,
        "rel_desc": rel_desc,
        "stage_desc": stage_desc,
        "pad_str": pad_str,
        "bot_state": bot_state,
        "text_len": text_len,
        "bot_basic_info": bot_basic_info,
        "user_basic_info": user_basic_info,
    }


def _filter_strategies_for_stage(
    strategies: List[Dict[str, Any]],
    allowed_ids: set,
    stage_index: int,
) -> List[Dict[str, Any]]:
    """只保留 id 在 allowed_ids 且 knapp_stages 包含 stage_index 的策略。"""
    out: List[Dict[str, Any]] = []
    for s in strategies or []:
        if not isinstance(s, dict):
            continue
        sid = s.get("id")
        if sid not in allowed_ids:
            continue
        stages = s.get("knapp_stages")
        if isinstance(stages, list) and stage_index in stages:
            out.append(s)
    return out


def _build_conditions_block(filtered: List[Dict[str, Any]]) -> str:
    """将过滤后的策略的 name + trigger 拼成「检测条件」列表。"""
    if not filtered:
        return "（当前阶段无可用策略，请输出 null。）"
    lines: List[str] = []
    for s in filtered:
        name = s.get("name") or s.get("id") or ""
        trigger = (s.get("trigger") or "").strip()
        lines.append(f"- **{name}**（id: {s.get('id')}）：{trigger}")
    return "\n".join(lines)


def _invoke_router(
    llm_invoker: Any,
    system_prompt: str,
    user_content: str,
    allowed_ids: set,
) -> str | None:
    """调用 LLM，解析 JSON 中的 hit；若非法或不在 allowed_ids 则返回 None。"""
    if not llm_invoker:
        return None
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    try:
        hit = None
        if hasattr(llm_invoker, "with_structured_output"):
            try:
                structured = llm_invoker.with_structured_output(StrategyRouterOutput)
                obj = structured.invoke(messages)
                hit = getattr(obj, "hit", None)
            except Exception:
                hit = None
        if hit is None:
            msg = llm_invoker.invoke(messages)
            raw = (getattr(msg, "content", "") or str(msg)).strip()
            parsed = parse_json_from_llm(raw)
            if isinstance(parsed, dict):
                hit = parsed.get("hit")
        if hit is not None:
            hit = str(hit).strip()
        if hit and hit in allowed_ids:
            return hit
    except Exception as e:
        logger.warning("[StrategyRouter] LLM 解析异常: %s", e)
    return None


def _create_router_node(
    router_name: str,
    state_key: str,
    strategy_ids: set,
    role_desc: str,
    rule_line: str,
    llm_invoker: Any,
    extra_system_fn: Callable[[Dict[str, Any]], str] | None = None,
) -> Callable[[AgentState], dict]:
    """通用路由节点工厂。extra_system_fn 用于节点 C 注入 Bot 状态与文本长度。"""

    @trace_if_enabled(
        name=router_name,
        run_type="chain",
        tags=["node", "strategy_router"],
        metadata={"state_outputs": [state_key]},
    )
    def node(state: AgentState) -> dict:
        stage_index = stage_to_knapp_index(state.get("current_stage"))
        strategies = load_strategies()
        filtered = _filter_strategies_for_stage(strategies, strategy_ids, stage_index)

        if not filtered:
            logger.info("[%s] 当前阶段 stage_index=%s 无可用策略，输出 None", router_name, stage_index)
            return {state_key: None}

        ctx = _gather_router_context(state)
        conditions_block = _build_conditions_block(filtered)

        system_parts = [
            f"你是{role_desc}。",
            "请结合「背景信息」「本轮用户消息」「聊天记录」与「当前情绪 PAD」判断是否命中下列任一情况。",
            "",
            "## 检测条件（仅当命中时输出对应 id，否则输出 null）",
            conditions_block,
            "",
            rule_line,
            "",
            "**硬性规则：宁可漏报，不可误报。仅当非常确定命中时才输出 id，否则输出 null。不要因为列表里有策略就必须选一个。**",
            "",
            "（输出格式由系统约束：hit 为命中的策略 id 或 null。）",
        ]
        if extra_system_fn:
            system_parts.insert(2, extra_system_fn(ctx))
        system_prompt = "\n".join(system_parts)

        # 背景/关系/阶段/Bot/用户为摘要字段，取前 N 字（与「聊天记录保留结尾」约定区分）
        user_content = f"""## 背景信息
- 关系状态：{ctx['rel_desc'][:600]}
- 阶段：{ctx['stage_desc'][:400]}
- Bot 信息：{safe_text(ctx['bot_basic_info'])[:300]}
- 用户信息：{safe_text(ctx['user_basic_info'])[:300]}

## 当前情绪 PAD
{ctx['pad_str']}

## 聊天记录（最近 {RECENT_DIALOGUE_LAST_N} 条，保留结尾）
{(ctx['recent_dialogue'] or '')[-RECENT_DIALOGUE_CHARS:]}

## 本轮用户消息
{ctx['latest_user_text']}

请判断是否命中上述检测条件之一；若命中输出对应 id，否则输出 null。"""

        hit = _invoke_router(llm_invoker, system_prompt, user_content, strategy_ids)
        if hit:
            logger.info("[%s] 命中: %s", router_name, hit)
        else:
            logger.info("[%s] 未命中 (hit=None)", router_name)
        return {state_key: hit}

    return node


def create_router_high_stakes_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """节点 A：高危与核心诉求。策略：anti_ai_defense, boundary_defense, yielding_apology, reasonable_assistance, physical_limitation_refusal（共 5 个）。"""
    return _create_router_node(
        router_name="Router/HighStakes",
        state_key="router_high_stakes",
        strategy_ids=HIGH_STAKES_IDS,
        role_desc="高危意图与核心诉求分类器，优先排查用户输入是否命中安全底线、Bot 自身错误修复、以及正经求助",
        rule_line="判定规则：若命中输出对应 id，否则严格输出 null。",
        llm_invoker=llm_invoker,
    )


def create_router_emotional_game_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """节点 B：高阶情感与博弈。策略：shit_test_counter, co_rumination, passive_aggression, deflection, flirting_banter。"""
    return _create_router_node(
        router_name="Router/EmotionalGame",
        state_key="router_emotional_game",
        strategy_ids=EMOTIONAL_GAME_IDS,
        role_desc="深层情感博弈嗅探器，分析是否命中情感陷阱、极端情绪、冷暴力与躲避",
        rule_line="判定规则：若命中输出对应 id，如果是普通情绪则输出 null。",
        llm_invoker=llm_invoker,
    )


def _form_rhythm_extra_system(ctx: Dict[str, Any]) -> str:
    return f"已知当前 Bot 状态为 **{ctx['bot_state']}**，本轮用户消息 **文本长度为 {ctx['text_len']} 字**。请据此判断是否命中下列情况。"


def create_router_form_rhythm_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """节点 C：对话形态与节奏。策略：tldr_refusal, micro_reaction, clarification, detail_nitpicking。"""
    return _create_router_node(
        router_name="Router/FormRhythm",
        state_key="router_form_rhythm",
        strategy_ids=FORM_RHYTHM_IDS,
        role_desc="对话节奏与形态打断器，处理废话、无意义输入、过载信息及找借口开溜",
        rule_line="判定规则：若命中输出对应 id，若句子清晰且值得正常聊则输出 null。",
        llm_invoker=llm_invoker,
        extra_system_fn=_form_rhythm_extra_system,
    )

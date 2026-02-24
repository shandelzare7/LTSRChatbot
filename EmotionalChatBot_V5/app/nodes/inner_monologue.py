"""内心独白节点：产出内心反应文本 + 选择相关的 inferred_profile 键（不负责预算）。"""
from __future__ import annotations

import logging
import random
from typing import Any, Callable, Dict, List, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.lats.prompt_utils import safe_text
from utils.prompt_helpers import format_relationship_for_llm, format_stage_for_llm
from utils.tracing import trace_if_enabled
from utils.llm_json import parse_json_from_llm
from src.schemas import InnerMonologueOutput
from utils.yaml_loader import load_pure_content_transformations

from app.state import AgentState

logger = logging.getLogger(__name__)

# 内心独白：多句/短段均可，只表达倾向/意愿/态度，不写步骤；字数上限大幅放宽
INNER_MONOLOGUE_MAX_CHARS = 400

# 与 Detection 对齐：当轮消息≤800 字，历史每条≤500 字，最近 30 条
LATEST_USER_TEXT_MAX = 800
RECENT_MSG_CONTENT_MAX = 500
RECENT_DIALOGUE_LAST_N = 30  # ✅ 修复：原代码里引用但未定义，会导致 NameError 然后被吞错兜底

# 当轮可选的 content move 最多返回 4 个
SELECTED_CONTENT_MOVE_IDS_MAX = 4


def _is_user_message(m: BaseMessage) -> bool:
    t = getattr(m, "type", "") or ""
    return "human" in t.lower() or "user" in t.lower()


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


def _normalize_cm_action_for_prompt(name: str, action: str) -> str:
    """
    仅用于“给 LLM 看”的 content_operation 文案修正，避免与 ReplyPlanner 写作规则硬冲突。
    物理锚定：强调用可撤销的身体动作/停顿/语气锚定，避免硬编环境事实。
    """
    n = (name or "").strip()
    a = (action or "").strip()
    if n in ("物理锚定", "Physical Anchoring"):
        extra = "注意：避免硬编具体环境事实（光线/温度/声音/地点）；优先用可撤销的身体动作/停顿/语气来锚定。"
        if extra not in a:
            a = (a + "\n" + extra).strip()
    return a


def _gather_context_for_monologue(state: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """从 state 抽取：当轮用户消息、历史对话、关系状态描述、关系阶段描述。"""
    chat_buffer: List[BaseMessage] = list(
        state.get("chat_buffer") or state.get("messages", [])[-RECENT_DIALOGUE_LAST_N:]
    )
    stage_id = str(state.get("current_stage") or "initiating")
    relationship_state = state.get("relationship_state") or {}

    # 当轮用户消息（≤800 字）
    latest_user_text_raw = (state.get("user_input") or "").strip()
    if not latest_user_text_raw and chat_buffer:
        last_msg = chat_buffer[-1]
        latest_user_text_raw = (getattr(last_msg, "content", "") or str(last_msg)).strip()
        if not _is_user_message(last_msg):
            latest_user_text_raw = latest_user_text_raw or "（无用户新句）"
    latest_user_text = (latest_user_text_raw or "（无用户消息）")[:LATEST_USER_TEXT_MAX]

    # 历史对话（最近 30 条，每条 content≤500 字）
    lines: List[str] = []
    for m in chat_buffer:
        role = "Human" if _is_user_message(m) else "AI"
        content = (getattr(m, "content", "") or str(m)).strip()
        if len(content) > RECENT_MSG_CONTENT_MAX:
            content = content[:RECENT_MSG_CONTENT_MAX] + "…"
        lines.append(f"{role}: {content}")
    recent_dialogue_context = "\n".join(lines) if lines else "（无历史对话）"

    rel_for_llm = format_relationship_for_llm(relationship_state)
    stage_desc = format_stage_for_llm(stage_id, include_judge_hints=True)

    return latest_user_text, recent_dialogue_context, rel_for_llm, stage_desc


def create_inner_monologue_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """创建内心独白节点：产出 inner_monologue 文本 + selected_profile_keys + selected_content_move_ids（最多 4 个）。"""

    @trace_if_enabled(
        name="Inner Monologue",
        run_type="chain",
        tags=["node", "inner_monologue", "perception"],
        metadata={"state_outputs": ["inner_monologue", "selected_profile_keys", "selected_content_move_ids"]},
    )
    def inner_monologue_node(state: AgentState) -> dict:
        monologue, selected_keys, selected_content_move_ids = _generate_monologue(state, llm_invoker)

        monologue = (monologue or "按常理接话即可。").strip()
        if len(monologue) > INNER_MONOLOGUE_MAX_CHARS:
            monologue = monologue[:INNER_MONOLOGUE_MAX_CHARS]

        # 当轮选中的 content move（最多 4 个）：打 logger 便于排查与统计
        id_to_name: Dict[int, str] = {}
        try:
            raw = load_pure_content_transformations()
            moves = _normalize_pure_content_transformations(raw)
            for m in moves:
                mid = m.get("id")
                if mid is not None:
                    id_to_name[int(mid)] = (m.get("name") or "").strip() or str(mid)
        except Exception as e:
            logger.exception("[InnerMonologue] load_pure_content_transformations failed in logger mapping: %s", e)

        names = [id_to_name.get(i, str(i)) for i in selected_content_move_ids]
        logger.info(
            "[InnerMonologue] selected_content_move_ids=%s (%s) selected_profile_keys=%s monologue_len=%d",
            selected_content_move_ids,
            ", ".join(names) if names else "（无）",
            selected_keys,
            len(monologue),
        )
        # 同时 print 以便 bottobot 重定向 stdout 时写入同一 log 文件
        print(
            f"[InnerMonologue] selected_content_move_ids={selected_content_move_ids!r} "
            f"({', '.join(names) if names else '（无）'}) "
            f"selected_profile_keys={selected_keys!r} monologue_len={len(monologue)}",
            flush=True,
        )

        return {
            "inner_monologue": monologue,
            "selected_profile_keys": selected_keys,
            "selected_content_move_ids": selected_content_move_ids,
        }

    return inner_monologue_node


def _generate_monologue(
    state: AgentState,
    llm_invoker: Any,
) -> Tuple[str, List[str], List[int]]:
    """LLM 生成内心独白 + 选择相关的 inferred_profile 键 + 选择最多 4 个当轮可执行的 content move id。"""
    empty_ids: List[int] = []
    if llm_invoker is None:
        return "按常理接话即可。", [], empty_ids

    content: str = ""  # 用于异常兜底时的 fallback
    try:
        latest_user_text, recent_dialogue_context, rel_for_llm, stage_desc = _gather_context_for_monologue(state)

        # 加载 content_moves.yaml 中的 pure_content_transformations（名称 + 动作），供 LLM 选 4 个当轮可执行
        transformations: List[Dict[str, Any]] = []
        try:
            raw = load_pure_content_transformations()
            transformations = _normalize_pure_content_transformations(raw)
        except Exception as e:
            logger.exception("[InnerMonologue] load_pure_content_transformations failed: %s", e)
            transformations = []

        if not transformations:
            logger.warning(
                "[InnerMonologue] pure_content_transformations is empty; selected_content_move_ids will likely be []"
            )

        valid_ids = {int(m["id"]) for m in transformations if isinstance(m, dict) and m.get("id") is not None}

        content_moves_block = ""
        if transformations:
            # 打乱顺序再传入，避免列表位置带来的选择偏差
            shuffled = list(transformations)
            random.shuffle(shuffled)
            lines_cm = []
            for m in shuffled:
                mid = m.get("id")
                name = (m.get("name") or "").strip() or "（未命名）"
                action = (m.get("content_operation") or "").strip() or "（无）"
                action = _normalize_cm_action_for_prompt(name, action)
                lines_cm.append(f"- id: {mid}, 名称: {name}, 动作: {action}")
            content_moves_block = "\n".join(lines_cm)

        # Bot / User 身份信息（与 reply_planner 对齐）
        bot_basic_info = state.get("bot_basic_info") or {}
        bot_persona = state.get("bot_persona") or ""
        user_basic_info = state.get("user_basic_info") or {}
        bot_name = safe_text((bot_basic_info or {}).get("name") or "Bot").strip() or "Bot"
        user_name_raw = safe_text((user_basic_info or {}).get("name") or "").strip()
        user_name = user_name_raw if user_name_raw else "你不知道对方的名字"

        # 收集 inferred_profile 的所有键名
        inferred_profile = state.get("user_inferred_profile") or {}
        profile_keys = sorted(inferred_profile.keys()) if isinstance(inferred_profile, dict) else []
        profile_keys_str = ", ".join(profile_keys) if profile_keys else "（暂无）"

        part3_section = ""
        if content_moves_block:
            part3_section = f"""
## 第三部分：选择当轮可执行的 content move (selected_content_move_ids)
以下是可用的「内容推进」标签（id + 名称 + 动作）：
{content_moves_block}
- 请从中选出**最多 {SELECTED_CONTENT_MOVE_IDS_MAX} 个**适合**本轮对话**可以执行的 id（只填 id 数字，如 [1, 3, 5]）。
- 只选与本轮用户消息和语境相匹配的标签，不要凑满 4 个；若没有合适的可返回空数组 []。
"""

        sys = f"""你是 {bot_name}。你正在和 {user_name} 对话。
下面给你一段「历史对话」和「当轮用户消息」作为正文，请以你（{bot_name}）的第一人称视角，根据正文语境输出三部分内容，给下游执行用，不给用户看。

## Identity (Bot & User)
bot_basic_info: {safe_text(bot_basic_info)}
bot_persona: {safe_text(bot_persona)}
user_basic_info: {safe_text(user_basic_info)}

## 第一部分：内心独白 (monologue)
- 字数**最多 {INNER_MONOLOGUE_MAX_CHARS} 个字符（中文按字计）**，可多句、可短段，但不得超出此上限。
- 以你（{bot_name}）的第一人称视角表达倾向/意愿/态度（如：懒得问、想挡一下、想接球但别太热情、对这句话有点意外但可以接），不要步骤、不要策略清单。

## 第二部分：选择相关的用户画像键 (selected_profile_keys)
以下是当前已有的用户推断画像键名列表：
[{profile_keys_str}]
- 从中选出 0~5 个与当前对话语境**相关**的键名（用于后续生成回复时参考）。
- 如果没有相关键或列表为空，返回空数组 []。
- 只选键名，不要自己编造不存在的键。
{part3_section}
## 关系状态（0–1）
{rel_for_llm}

## 关系阶段与越界提示
{stage_desc}

## 输出格式（Response format，必须严格遵守）
必须输出且仅输出一个 JSON 对象，包含且仅包含以下三个键（与 schema 一致，便于解析）：
- **monologue** (string)：内心独白文本，可为空字符串。
- **selected_profile_keys** (array of string)：选中的画像键名列表，0~5 个，无则 []。
- **selected_content_move_ids** (array of integer)：当轮可执行的 content move 的 id 列表，最多 4 个，如 [1, 3, 5, 7]，无则 []。
不要输出其他键或 Markdown 代码块标记，只输出上述 JSON。
"""
        user_body = f"""【历史对话】（最近 {RECENT_DIALOGUE_LAST_N} 条）
{recent_dialogue_context}

【当轮用户消息】
{latest_user_text}

请根据上述对话与当轮用户消息，按【输出格式】输出内心独白、选中的画像键以及当轮可执行的 content move id（三键：monologue, selected_profile_keys, selected_content_move_ids）。"""

        data = None
        if hasattr(llm_invoker, "with_structured_output"):
            try:
                structured = llm_invoker.with_structured_output(InnerMonologueOutput)
                obj = structured.invoke([SystemMessage(content=sys), HumanMessage(content=user_body)])
                data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
            except Exception as e:
                logger.exception("[InnerMonologue] structured_output failed, fallback to json parse: %s", e)
                data = None

        if data is None:
            msg = llm_invoker.invoke([SystemMessage(content=sys), HumanMessage(content=user_body)])
            content = (getattr(msg, "content", "") or str(msg)).strip()
            data = parse_json_from_llm(content)

        if isinstance(data, dict):
            monologue = str(data.get("monologue") or "").strip().strip("\"'")
            raw_keys = data.get("selected_profile_keys") or []
            selected = [
                str(k)
                for k in raw_keys
                if isinstance(k, str) and k.strip() and k.strip() in profile_keys
            ][:5]

            raw_ids = data.get("selected_content_move_ids") or []
            selected_ids = [
                int(x)
                for x in raw_ids
                if x is not None
                and (
                    isinstance(x, int)
                    or (isinstance(x, float) and int(x) == x)
                    or (isinstance(x, str) and str(x).strip().isdigit())
                )
            ]
            selected_ids = [x for x in selected_ids if x in valid_ids][:SELECTED_CONTENT_MOVE_IDS_MAX]

            # 只要解析成功就返回（含 selected_ids），不要仅因 monologue 为空就丢弃 8 选 4 结果
            monologue_out = (monologue or "按常理接话即可。").strip()[:INNER_MONOLOGUE_MAX_CHARS]
            return monologue_out, selected, selected_ids

        # JSON 解析失败时，将整个输出作为独白文本
        fallback_text = (content or "").strip().strip("\"'")
        if fallback_text:
            return fallback_text[:INNER_MONOLOGUE_MAX_CHARS], [], empty_ids

    except Exception as e:
        # ✅ 关键：不要吞错
        logger.exception("[InnerMonologue] _generate_monologue failed: %s", e)

    return "按常理接话即可。", [], empty_ids
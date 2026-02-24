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

INNER_MONOLOGUE_MAX_CHARS = 400

LATEST_USER_TEXT_MAX = 800
RECENT_MSG_CONTENT_MAX = 500
RECENT_DIALOGUE_LAST_N = 30  # ✅ 修复：原先引用但未定义
SELECTED_CONTENT_MOVE_IDS_MAX = 4
SELECTED_CONTENT_MOVE_IDS_REQUIRED = 4
FALLBACK_CM_PRIORITY = [1, 2, 5, 7, 3, 6, 4, 8]

# ✅ 注意力优化：每个 move 的动作描述不要太长（避免“8 选 4”时模型注意力分散）
CM_ACTION_MAX_CHARS = 220


def _is_user_message(m: BaseMessage) -> bool:
    t = getattr(m, "type", "") or ""
    return "human" in t.lower() or "user" in t.lower()


def _normalize_pure_content_transformations(raw: Any) -> List[Dict[str, Any]]:
    """兼容 loader 返回 list 或 {'pure_content_transformations': [...]}。"""
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("pure_content_transformations") or []
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, dict)]


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    if not s or n <= 0:
        return ""
    return s if len(s) <= n else (s[: max(1, n - 1)].rstrip() + "…")


def _normalize_cm_action_for_prompt(name: str, action: str) -> str:
    """
    仅用于“给 LLM 看”的 content_operation 文案修正，避免与 ReplyPlanner 写作规则硬冲突。
    物理锚定：强调用可撤销的身体动作/停顿/语气锚定，避免硬编环境事实。
    """
    n = (name or "").strip()
    a = (action or "").strip()
    if n in ("物理锚定", "Physical Anchoring"):
        extra = "注意：避免硬编具体环境事实（光线/温度/声音/地点）；优先用可撤销的身体动作/停顿/语气锚定。"
        if extra not in a:
            a = (a + "\n" + extra).strip()
    return a


def _gather_context_for_monologue(state: Dict[str, Any]) -> Tuple[str, str, str, str]:
    chat_buffer: List[BaseMessage] = list(
        state.get("chat_buffer") or state.get("messages", [])[-RECENT_DIALOGUE_LAST_N:]
    )
    stage_id = str(state.get("current_stage") or "initiating")
    relationship_state = state.get("relationship_state") or {}

    latest_user_text_raw = (state.get("user_input") or "").strip()
    if not latest_user_text_raw and chat_buffer:
        last_msg = chat_buffer[-1]
        latest_user_text_raw = (getattr(last_msg, "content", "") or str(last_msg)).strip()
        if not _is_user_message(last_msg):
            latest_user_text_raw = latest_user_text_raw or "（无用户新句）"
    latest_user_text = (latest_user_text_raw or "（无用户消息）")[:LATEST_USER_TEXT_MAX]

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


def _pad_to_exact_4(selected: List[int], valid_ids: List[int]) -> List[int]:
    """保证返回恰好 4 个 id：先去重保序，不足则按 FALLBACK_CM_PRIORITY 与 valid_ids 补齐。"""
    out: List[int] = []
    seen: set = set()
    for x in selected:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    valid_set = set(valid_ids)
    for x in FALLBACK_CM_PRIORITY:
        if len(out) >= SELECTED_CONTENT_MOVE_IDS_REQUIRED:
            break
        if x in valid_set and x not in seen:
            seen.add(x)
            out.append(x)
    for x in valid_ids:
        if len(out) >= SELECTED_CONTENT_MOVE_IDS_REQUIRED:
            break
        if x not in seen:
            seen.add(x)
            out.append(x)
    if not out and valid_ids:
        out = [valid_ids[0]]
    while len(out) < SELECTED_CONTENT_MOVE_IDS_REQUIRED:
        out.append(out[-1] if out else 1)
    return out[:SELECTED_CONTENT_MOVE_IDS_REQUIRED]


def create_inner_monologue_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
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

        # 诊断日志：把 id→name 映射打出来
        id_to_name: Dict[int, str] = {}
        try:
            moves = _normalize_pure_content_transformations(load_pure_content_transformations())
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
    valid_ids_list = list(range(1, 9))
    if llm_invoker is None:
        return "按常理接话即可。", [], _pad_to_exact_4([], list(range(1, 9)))

    content: str = ""
    try:
        latest_user_text, recent_dialogue_context, rel_for_llm, stage_desc = _gather_context_for_monologue(state)

        valid_ids_list = list(range(1, 9))  # 兜底，后续会被 transformations 覆盖
        # ✅ 加载 moves（防御式）
        try:
            transformations = _normalize_pure_content_transformations(load_pure_content_transformations())
        except Exception as e:
            logger.exception("[InnerMonologue] load_pure_content_transformations failed: %s", e)
            transformations = []

        valid_ids = {int(m["id"]) for m in transformations if m.get("id") is not None}
        valid_ids_list = sorted(valid_ids) if valid_ids else list(range(1, 9))

        # ✅ 注意力优化：只给 LLM “id+name+简短动作”，避免 8 个长段落稀释注意力
        content_moves_block = ""
        if transformations:
            shuffled = list(transformations)
            random.shuffle(shuffled)
            lines_cm = []
            for m in shuffled:
                mid = m.get("id")
                name = (m.get("name") or "").strip() or "（未命名）"
                action = (m.get("content_operation") or "").strip() or "（无）"
                action = _normalize_cm_action_for_prompt(name, action)
                action = _truncate(action, CM_ACTION_MAX_CHARS)
                lines_cm.append(f"- id: {mid} | {name} | 动作要点: {action}")
            content_moves_block = "\n".join(lines_cm)

        bot_basic_info = state.get("bot_basic_info") or {}
        bot_persona = state.get("bot_persona") or ""
        user_basic_info = state.get("user_basic_info") or {}
        bot_name = safe_text((bot_basic_info or {}).get("name") or "Bot").strip() or "Bot"
        user_name_raw = safe_text((user_basic_info or {}).get("name") or "").strip()
        user_name = user_name_raw if user_name_raw else "你不知道对方的名字"

        inferred_profile = state.get("user_inferred_profile") or {}
        profile_keys = sorted(inferred_profile.keys()) if isinstance(inferred_profile, dict) else []
        profile_keys_str = ", ".join(profile_keys) if profile_keys else "（暂无）"

        part3_section = ""
        if content_moves_block:
            part3_section = f"""
## 第三部分：选择当轮可执行的 content move (selected_content_move_ids)
可用 content moves（id | 名称 | 动作要点）：
{content_moves_block}

要求：
- 你必须从候选中**选择最佳的确切 4 个** id，输出恰好 4 个，不要多也不要少。
- id 必须来自上面候选列表中的 id（不要编造），且**不要重复**。
- 选择标准：按“最适合本轮语境、最能推进对话”排序，取前 4 个。只填数字数组，如 [3,1,5,7]。
"""

        sys = f"""你是 {bot_name}。你正在和 {user_name} 对话。
下面给你「历史对话」和「当轮用户消息」，请输出给下游执行用（不给用户看）。

## 第一部分：内心独白 (monologue)
- ≤{INNER_MONOLOGUE_MAX_CHARS} 字符；第一人称；只写倾向/意愿/态度；不要步骤/清单。

## 第二部分：选择相关的用户画像键 (selected_profile_keys)
可选键名：
[{profile_keys_str}]
- 选 0~5 个；只选存在的键名；无则 []。
{part3_section}
## 关系状态（0–1）
{rel_for_llm}

## 关系阶段与越界提示
{stage_desc}

## 输出格式（必须严格 JSON）
仅输出一个 JSON 对象，且只包含：
- monologue: string
- selected_profile_keys: string[]
- selected_content_move_ids: int[]
不要输出其他键/不要 Markdown。
"""

        user_body = f"""【历史对话】（最近 {RECENT_DIALOGUE_LAST_N} 条）
{recent_dialogue_context}

【当轮用户消息】
{latest_user_text}
"""

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
            selected_keys = [
                str(k) for k in raw_keys
                if isinstance(k, str) and k.strip() and k.strip() in profile_keys
            ][:5]

            raw_ids = data.get("selected_content_move_ids") or []
            selected_ids = [
                int(x) for x in raw_ids
                if x is not None and (
                    isinstance(x, int)
                    or (isinstance(x, float) and int(x) == x)
                    or (isinstance(x, str) and str(x).strip().isdigit())
                )
            ]
            selected_ids = [x for x in selected_ids if x in valid_ids][:SELECTED_CONTENT_MOVE_IDS_MAX]
            selected_ids = _pad_to_exact_4(selected_ids, valid_ids_list)

            monologue_out = (monologue or "按常理接话即可。").strip()[:INNER_MONOLOGUE_MAX_CHARS]
            return monologue_out, selected_keys, selected_ids

        fallback_text = (content or "").strip().strip("\"'")
        if fallback_text:
            return fallback_text[:INNER_MONOLOGUE_MAX_CHARS], [], _pad_to_exact_4([], valid_ids_list)

    except Exception as e:
        logger.exception("[InnerMonologue] _generate_monologue failed: %s", e)

    return "按常理接话即可。", [], _pad_to_exact_4([], valid_ids_list)
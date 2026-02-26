"""结构化提取节点：从内心独白中提取信号，并完成 move 和 profile_key 选择。

这是独白之后的第一步，单次 LLM 调用（~800-1200 tok）：
- emotion_tag / attitude / momentum_delta / topic_appeal / subtext_guess
- selected_profile_keys（替代旧版 inner_monologue 里的 profile key 选择）
- selected_content_move_ids（2-4 个，替代旧版 inner_monologue 里的 move 选择）
"""
from __future__ import annotations

import logging
import random
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.lats.prompt_utils import safe_text
from app.state import AgentState
from src.schemas import MonologueExtractOutput
from utils.detailed_logging import log_prompt_and_params, log_llm_response
from utils.llm_json import parse_json_from_llm
from utils.tracing import trace_if_enabled
from utils.yaml_loader import load_pure_content_transformations

logger = logging.getLogger(__name__)

SELECTED_CONTENT_MOVE_IDS_MAX = 4
SELECTED_CONTENT_MOVE_IDS_MIN = 2
FALLBACK_CM_PRIORITY = [1, 2, 5, 7, 3, 6, 4, 8]
CM_ACTION_MAX_CHARS = 160


def _normalize_pure_content_transformations(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("pure_content_transformations") or []
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, dict)]


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: max(1, n - 1)].rstrip() + "…"


def _pad_to_range(selected: List[int], valid_ids: List[int]) -> List[int]:
    """确保返回 2-4 个不重复 id。"""
    out: List[int] = []
    seen: set = set()
    for x in selected:
        if x not in seen and x in set(valid_ids):
            seen.add(x)
            out.append(x)
    for x in FALLBACK_CM_PRIORITY:
        if len(out) >= SELECTED_CONTENT_MOVE_IDS_MAX:
            break
        if x in set(valid_ids) and x not in seen:
            seen.add(x)
            out.append(x)
    for x in valid_ids:
        if len(out) >= SELECTED_CONTENT_MOVE_IDS_MAX:
            break
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out[:SELECTED_CONTENT_MOVE_IDS_MAX] if len(out) >= SELECTED_CONTENT_MOVE_IDS_MIN else out


def create_extract_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """创建结构化提取节点。"""

    @trace_if_enabled(
        name="Monologue/Extract",
        run_type="chain",
        tags=["node", "extract"],
        metadata={"state_outputs": ["monologue_extract"]},
    )
    def extract_node(state: AgentState) -> dict:
        monologue = (state.get("inner_monologue") or "").strip()
        if not monologue:
            return {"monologue_extract": _default_extract()}

        result = _run_extract(state, monologue, llm_invoker)
        logger.info(
            "[Extract] emotion_tag=%s attitude=%s momentum_delta=%.2f topic_appeal=%.1f move_ids=%s",
            result.get("emotion_tag"),
            result.get("attitude"),
            result.get("momentum_delta", 0.0),
            result.get("topic_appeal", 5.0),
            result.get("selected_content_move_ids"),
        )
        return {"monologue_extract": result}

    return extract_node


def _default_extract() -> Dict[str, Any]:
    return {
        "emotion_tag": "平静",
        "attitude": "被动应付",
        "momentum_delta": 0.0,
        "topic_appeal": 5.0,
        "subtext_guess": "",
        "selected_profile_keys": [],
        "selected_content_move_ids": [1, 2, 5, 7],
        "inferred_gender": None,
    }


def _gender_unknown(state: AgentState) -> bool:
    """True 当且仅当 user_basic_info.gender 为空（需要推断）。"""
    g = str((state.get("user_basic_info") or {}).get("gender") or "").strip()
    return not g


def _run_extract(state: AgentState, monologue: str, llm_invoker: Any) -> Dict[str, Any]:
    # 加载 content moves
    try:
        transformations = _normalize_pure_content_transformations(load_pure_content_transformations())
    except Exception as e:
        logger.warning("[Extract] load_pure_content_transformations failed: %s", e)
        transformations = []

    valid_ids = [int(m["id"]) for m in transformations if m.get("id") is not None]
    valid_ids_list = sorted(valid_ids) if valid_ids else list(range(1, 9))

    # 构建 content moves 列表（随机打乱，避免首位偏差）
    moves_for_prompt: List[str] = []
    if transformations:
        shuffled = list(transformations)
        random.shuffle(shuffled)
        for m in shuffled:
            mid = m.get("id")
            name = (m.get("name") or "").strip()
            action = _truncate((m.get("content_operation") or "").strip(), CM_ACTION_MAX_CHARS)
            moves_for_prompt.append(f"- id:{mid} {name} | {action}")
    moves_block = "\n".join(moves_for_prompt) if moves_for_prompt else "（无可用 move）"

    # 用户画像键
    inferred_profile = state.get("user_inferred_profile") or {}
    profile_keys = sorted(inferred_profile.keys()) if isinstance(inferred_profile, dict) else []
    profile_keys_str = ", ".join(profile_keys) if profile_keys else "（暂无）"

    need_gender = _gender_unknown(state)
    gender_field = (
        '\n  "inferred_gender": null,  // 从对话/独白中推断用户性别（"男"/"女"/"其他"），无法判断则 null'
        if need_gender else
        '\n  "inferred_gender": null,  // 性别已知，保持 null'
    )
    gender_instruction = (
        "\n\n## 用户性别推断（本次需要）\n"
        "用户性别尚未记录。请综合独白内容、用户措辞、称呼方式等线索，推断用户性别（男/女/其他），"
        "填入 inferred_gender；若完全无法判断则填 null。"
        if need_gender else ""
    )

    system_content = f"""你是分析助手，请从下面的「内心独白」中提取结构化信息，严格输出 JSON。

## 可选 content move（2-4个）
{moves_block}

## 可选 profile key（0-5个）
[{profile_keys_str}]
{gender_instruction}
## 输出格式（仅输出此 JSON，不要其他内容）
{{
  "emotion_tag": "情绪标签，一两个词，如 心疼/烦躁/期待/无聊/开心/纠结",
  "attitude": "对用户的态度倾向，如 主动配合/被动应付/想转移话题/好奇/享受/排斥",
  "momentum_delta": 0.0,  // 冲量变化 -1.0~+1.0，正=想继续，负=想结束
  "topic_appeal": 5.0,    // 话题吸引力 0-10
  "subtext_guess": "对用户潜台词的一句猜测，无则空字符串",
  "selected_profile_keys": [],  // 0-5个存在的 profile key
  "selected_content_move_ids": [1, 2]{gender_field}
}}"""

    user_content = f"内心独白：\n{monologue}"

    messages = [SystemMessage(content=system_content), HumanMessage(content=user_content)]
    log_prompt_and_params("Extract", messages=messages)

    default = _default_extract()
    try:
        data = None
        if hasattr(llm_invoker, "with_structured_output"):
            try:
                structured = llm_invoker.with_structured_output(MonologueExtractOutput)
                obj = structured.invoke(messages)
                data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
            except Exception as e:
                logger.warning("[Extract] structured_output failed: %s", e)
                data = None

        if data is None:
            msg = llm_invoker.invoke(messages)
            raw = (getattr(msg, "content", "") or str(msg)).strip()
            data = parse_json_from_llm(raw)

        if not isinstance(data, dict):
            return default

        # 验证并归一化
        emotion_tag = str(data.get("emotion_tag") or "平静").strip()
        attitude = str(data.get("attitude") or "被动应付").strip()

        try:
            momentum_delta = float(data.get("momentum_delta", 0.0))
            momentum_delta = max(-1.0, min(1.0, momentum_delta))
        except (TypeError, ValueError):
            momentum_delta = 0.0

        try:
            topic_appeal = float(data.get("topic_appeal", 5.0))
            topic_appeal = max(0.0, min(10.0, topic_appeal))
        except (TypeError, ValueError):
            topic_appeal = 5.0

        subtext_guess = str(data.get("subtext_guess") or "").strip()

        # profile keys 验证
        raw_keys = data.get("selected_profile_keys") or []
        selected_keys = [k for k in raw_keys if isinstance(k, str) and k.strip() in profile_keys][:5]

        # move ids 验证
        raw_ids = data.get("selected_content_move_ids") or []
        selected_ids_raw = []
        for x in raw_ids:
            try:
                v = int(float(x))
                if v in valid_ids_list:
                    selected_ids_raw.append(v)
            except (TypeError, ValueError):
                pass
        selected_ids = _pad_to_range(selected_ids_raw, valid_ids_list)

        # 构建 move 详情用于日志
        move_details = []
        for mid in selected_ids:
            for m in transformations:
                if m.get("id") == mid:
                    move_details.append(f"id:{mid} {m.get('name', '')} | {(m.get('content_operation') or '')[:80]}")
                    break

        result_for_log = {
            "emotion_tag": emotion_tag,
            "attitude": attitude,
            "momentum_delta": momentum_delta,
            "topic_appeal": topic_appeal,
            "subtext_guess": subtext_guess,
            "selected_profile_keys": selected_keys,
            "selected_content_move_ids": selected_ids,
            "move_details": move_details,
        }
        log_llm_response("Extract", "(parsed)", parsed_result=result_for_log)

        # inferred_gender：只在性别未知时才采用（防止覆盖已知性别）
        inferred_gender: Optional[str] = None
        if need_gender:
            raw_g = str(data.get("inferred_gender") or "").strip()
            if raw_g in ("男", "女", "其他"):
                inferred_gender = raw_g

        return {
            "emotion_tag": emotion_tag,
            "attitude": attitude,
            "momentum_delta": momentum_delta,
            "topic_appeal": topic_appeal,
            "subtext_guess": subtext_guess,
            "selected_profile_keys": selected_keys,
            "selected_content_move_ids": selected_ids,
            "inferred_gender": inferred_gender,
        }

    except Exception as e:
        logger.exception("[Extract] failed: %s", e)
        return default

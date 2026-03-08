"""结构化提取节点：从内心独白中提取信号，并完成 move 和 profile_key 选择。

这是独白之后的第一步，单次 LLM 调用（~600-900 tok）：
- emotion_tag / bot_stance / topic_appeal
- selected_profile_keys（2-5个，驱动 generate 的画像注入）
- selected_content_move_ids（2-4个，驱动并行生成路由）
- inferred_gender（仅在性别未知时填写）
"""
from __future__ import annotations

import logging
import os
import random
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.prompts.prompt_utils import safe_text
from app.state import AgentState
from src.schemas import MonologueExtractOutput
from utils.detailed_logging import log_prompt_and_params, log_llm_response
from utils.llm_json import parse_json_from_llm
from utils.tracing import trace_if_enabled
from utils.yaml_loader import load_pure_content_transformations

logger = logging.getLogger(__name__)

SELECTED_CONTENT_MOVE_IDS_MAX = 4
SELECTED_CONTENT_MOVE_IDS_MIN = 2
# 兜底顺序：轻重交替，避免连续锁死在深挖型
FALLBACK_CM_PRIORITY = [1, 7, 3, 2, 5, 8, 4, 6]
CM_ACTION_MAX_CHARS = 160

_VALID_STANCES = {"supportive", "exploratory", "self_sharing", "redirecting", "challenging"}


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
        if os.getenv("ABLATION_MODE"):
            # 对照组：不做结构化提取，返回空 move_ids（generate 将只走 FREE 路）
            ablation_extract = _default_extract()
            ablation_extract["selected_content_move_ids"] = []
            return {"monologue_extract": ablation_extract}
        monologue = (state.get("inner_monologue") or "").strip()
        if not monologue:
            return {"monologue_extract": _default_extract()}

        result = _run_extract(state, monologue, llm_invoker)
        logger.info(
            "[Extract] emotion_tag=%s bot_stance=%s topic_appeal=%.1f move_ids=%s",
            result.get("emotion_tag"),
            result.get("bot_stance"),
            result.get("topic_appeal", 5.0),
            result.get("selected_content_move_ids"),
        )
        return {"monologue_extract": result}

    return extract_node


def _default_extract() -> Dict[str, Any]:
    return {
        "emotion_tag": "平静",
        "bot_stance": "supportive",
        "topic_appeal": 5.0,
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

    # 近期 bot 发言（供 topic_appeal 语义重复感知用）
    def _is_user_msg(m: Any) -> bool:
        t = getattr(m, "type", "") or ""
        return "human" in t.lower() or "user" in t.lower()

    _chat_buf = list(state.get("chat_buffer") or state.get("messages") or [])
    _recent_bot_lines = [
        (getattr(m, "content", "") or str(m)).strip()[:80]
        for m in _chat_buf[-12:]
        if not _is_user_msg(m) and (getattr(m, "content", "") or str(m)).strip()
    ][-3:]
    _recent_bot_block = ""
    if len(_recent_bot_lines) >= 2:
        _recent_bot_block = (
            "\n\n## 近期你自己说过的内容\n"
            + "\n".join(f"- {t}" for t in _recent_bot_lines)
            + "\n（评分参考：若当前独白话题与以上内容高度重叠，说明话题已充分探索，topic_appeal 应适当偏低）"
        )

    system_content = f"""你是分析助手，请从下面的「内心独白」中提取结构化信息，严格输出 JSON。

## 可选 content move
{moves_block}

**选 move 规则**：
- 选 4 个**互补**的 move，深挖型（id=1,5,6）和轻短型（id=2,7,8）至少各有 1 个，避免全是同一方向。
- 每次根据独白内容和话题氛围**主动换用**不同 move，不要每轮都选同一组。
- 所有 id 必须来自上方列表，不要编造。

## 可选 profile key（0-5个）
[{profile_keys_str}]
{gender_instruction}
## bot_stance 说明（本轮沟通立场，五选一）
- supportive：认同/共情，回应对方情绪或立场
- exploratory：追问深挖，对话题有兴趣想了解更多细节
- self_sharing：主动分享，对方的话触发了想说自己的事（经历/感受/近况），或纯粹想聊聊自己
- redirecting：温和转移，当前话题已无聊/重复，想引向其他方向
- challenging：轻度挑战，用反问或不同视角推动对话（只在关系较亲密时用）

**区分 self_sharing vs redirecting**：
- self_sharing：可以是因为对方话题有共鸣 OR 有独立欲望 → 想说自己的；不一定排斥当前话题
- redirecting：当前话题让我提不起劲 → 想换掉；明确想离开当前话题

## 输出格式（仅输出此 JSON，不要其他内容）
{{
  "emotion_tag": "一两个词，从独白中自然提炼，不限于固定列表——可以是任何真实情绪描述",
  "bot_stance": "supportive",  // supportive/exploratory/self_sharing/redirecting/challenging 五选一
  "topic_appeal": 5.0,    // 话题吸引力 0-10（若话题与近期发言高度重复，应偏低）
  "selected_profile_keys": [],  // 0-5个存在的 profile key，用于本轮回复中自然融入对方信息
  "selected_content_move_ids": [1, 2]{gender_field}
}}"""

    user_content = f"内心独白：\n{monologue}{_recent_bot_block}"

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

        raw_stance = str(data.get("bot_stance") or "supportive").strip().lower()
        bot_stance = raw_stance if raw_stance in _VALID_STANCES else "supportive"

        try:
            topic_appeal = float(data.get("topic_appeal", 5.0))
            topic_appeal = max(0.0, min(10.0, topic_appeal))
        except (TypeError, ValueError):
            topic_appeal = 5.0

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
            "bot_stance": bot_stance,
            "topic_appeal": topic_appeal,
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
            "bot_stance": bot_stance,
            "topic_appeal": topic_appeal,
            "selected_profile_keys": selected_keys,
            "selected_content_move_ids": selected_ids,
            "inferred_gender": inferred_gender,
        }

    except Exception as e:
        logger.exception("[Extract] failed: %s", e)
        return default

"""结构化提取节点：从内心独白中提取信号，并通过加权算法完成 move 选择。

LLM 只输出轻量分类（emotion_tag / bot_stance / topic_appeal / user_act / profile_keys / gender），
move 选择由 7 维加权算法（user_act / urgency / hostility / engagement / momentum / closeness / recent_moves）完成。
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
CM_ACTION_MAX_CHARS = 160

_VALID_STANCES = {"supportive", "exploratory", "self_sharing", "redirecting", "challenging"}
_VALID_USER_ACTS = {"asking", "venting", "deciding", "pushing_back", "phatic"}

# Move 类别映射
_MOVE_CATEGORY = {
    1: "要信息", 2: "要信息", 3: "要信息",
    4: "给反应", 5: "给反应",
    6: "给内容", 7: "给内容", 8: "给内容",
    9: "给分析", 10: "给分析", 11: "给分析",
    12: "给方向", 13: "给方向",
}

# ════════════════════════════════════
# Move 选择：7 维加权 → 抽 4 个
# ════════════════════════════════════

# 维度1：user_act
_USER_ACT_RULES: Dict[str, Dict[str, list]] = {
    "asking":       {"+2": [4, 9],  "+1": [6, 3],  "-1": [5, 2]},
    "venting":      {"+2": [5, 2],  "+1": [4, 1],  "-2": [12, 13]},
    "deciding":     {"+2": [12, 10], "+1": [11, 3]},
    "pushing_back": {"+2": [2, 4],  "+1": [9],     "blocked": [13]},
    "phatic":       {"+2": [8, 6],  "+1": [1],     "-2": [9, 10, 11]},
}

# 维度2：urgency（from detection.urgency 0-10）
_URGENCY_RULES: Dict[str, Dict[str, list]] = {
    "low":  {},
    "mid":  {},
    "high": {"+2": [5, 2], "+1": [4], "-2": [9, 10, 13], "-1": [12, 11]},
}

# 维度3：hostility（from detection.hostility_level 0-10）
_HOSTILITY_RULES: Dict[str, Dict[str, list]] = {
    "safe": {},
    "hot":  {"+2": [2, 5], "-2": [13, 12, 9], "blocked": [10, 11]},
}

# 维度4：engagement（from detection.engagement_level 0-10）
_ENGAGEMENT_RULES: Dict[str, Dict[str, list]] = {
    "low":  {"+2": [8, 6], "+1": [1, 4], "-2": [9, 10, 11], "blocked": [13]},
    "mid":  {},
    "high": {"+1": [9, 10, 13, 7]},
}

# 维度5：momentum（from conversation_momentum 0-1）
_MOMENTUM_RULES: Dict[str, Dict[str, list]] = {
    "cold": {"+2": [1, 8, 6], "blocked": [13, 11], "-1": [12]},
    "warm": {},
    "hot":  {"+1": [9, 10, 13, 12]},
}

# 维度6：closeness（from relationship_state.closeness 0-1）
_CLOSENESS_RULES: Dict[str, Dict[str, list]] = {
    "stranger": {"+1": [1, 2, 4], "blocked": [6, 7], "-1": [13]},
    "familiar": {"+1": [6, 7]},
    "close":    {"+1": [5, 13, 6, 12]},
}


# ── bucket 函数 ──

def _bucket_urgency(v: float) -> str:
    if v < 4:
        return "low"
    if v < 7:
        return "mid"
    return "high"


def _bucket_hostility(v: float) -> str:
    return "hot" if v >= 4 else "safe"


def _bucket_engagement(v: float) -> str:
    if v < 3:
        return "low"
    if v < 7:
        return "mid"
    return "high"


def _bucket_momentum(v: float) -> str:
    if v < 0.3:
        return "cold"
    if v < 0.7:
        return "warm"
    return "hot"


def _bucket_closeness(v: float) -> str:
    if v < 0.3:
        return "stranger"
    if v < 0.6:
        return "familiar"
    return "close"


def _apply_rules(weights: Dict[int, float], blocked: set, rules: Dict[str, list]):
    """应用一组规则到权重和 blocked 集合。"""
    for key, ids in rules.items():
        if key == "blocked":
            blocked.update(ids)
        elif key == "+2":
            for mid in ids:
                weights[mid] = weights.get(mid, 0) + 2
        elif key == "+1":
            for mid in ids:
                weights[mid] = weights.get(mid, 0) + 1
        elif key == "-1":
            for mid in ids:
                weights[mid] = weights.get(mid, 0) - 1
        elif key == "-2":
            for mid in ids:
                weights[mid] = weights.get(mid, 0) - 2


def select_moves(
    user_act: str,
    state: AgentState,
    valid_ids: List[int],
) -> List[int]:
    """7 维加权浮动选择 4 个 move。"""
    weights: Dict[int, float] = {mid: 0.0 for mid in valid_ids}
    blocked: set = set()

    # 维度1：user_act
    _apply_rules(weights, blocked, _USER_ACT_RULES.get(user_act, {}))

    # 维度2-3：urgency / hostility（from detection）
    detection = state.get("detection") or {}
    urgency = float(detection.get("urgency", 0))
    hostility = float(detection.get("hostility_level", 0))
    _apply_rules(weights, blocked, _URGENCY_RULES.get(_bucket_urgency(urgency), {}))
    _apply_rules(weights, blocked, _HOSTILITY_RULES.get(_bucket_hostility(hostility), {}))

    # 维度4：engagement
    engagement = float(detection.get("engagement_level", 5))
    _apply_rules(weights, blocked, _ENGAGEMENT_RULES.get(_bucket_engagement(engagement), {}))

    # 维度5：momentum
    momentum = 0.5
    try:
        momentum = float(state.get("conversation_momentum", 0.5))
    except (TypeError, ValueError):
        pass
    _apply_rules(weights, blocked, _MOMENTUM_RULES.get(_bucket_momentum(momentum), {}))

    # 维度6：closeness
    rel = state.get("relationship_state") or {}
    closeness = float(rel.get("closeness", 0.3))
    _apply_rules(weights, blocked, _CLOSENESS_RULES.get(_bucket_closeness(closeness), {}))

    # 维度7：recent_moves
    history: List[List[int]] = list(state.get("recent_move_history") or [])
    if history:
        last_round = history[-1] if len(history) >= 1 else []
        for mid in last_round:
            weights[mid] = weights.get(mid, 0) - 3
        if len(history) >= 2:
            for mid in history[-2]:
                weights[mid] = weights.get(mid, 0) - 1

    # 去掉 blocked
    pool = [(mid, w) for mid, w in weights.items() if mid not in blocked]
    if not pool:
        pool = [(mid, w) for mid, w in weights.items()]
    pool.sort(key=lambda x: -x[1])

    # 抽选：权重最高必选，剩下随机抽
    selected: List[int] = []
    if pool:
        selected.append(pool[0][0])

    remaining = [(mid, w) for mid, w in pool if mid != selected[0]] if selected else list(pool)

    pos_pool = [mid for mid, w in remaining if w >= 0]
    if len(pos_pool) >= 3:
        selected.extend(random.sample(pos_pool, 3))
    else:
        relaxed = [mid for mid, w in remaining if w >= -1]
        selected.extend(pos_pool)
        need = 3 - len(pos_pool)
        extra = [mid for mid in relaxed if mid not in set(selected)]
        selected.extend(random.sample(extra, min(need, len(extra))))

    # 去重，保证 4 个
    seen: set = set()
    out: List[int] = []
    for mid in selected:
        if mid not in seen:
            seen.add(mid)
            out.append(mid)
    for mid, _ in pool:
        if len(out) >= 4:
            break
        if mid not in seen:
            seen.add(mid)
            out.append(mid)

    return out[:4]


def _normalize_pure_content_transformations(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("moves") or raw.get("pure_content_transformations") or []
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, dict)]


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
            ablation_extract = _default_extract()
            ablation_extract["selected_content_move_ids"] = []
            return {"monologue_extract": ablation_extract}
        monologue = (state.get("inner_monologue") or "").strip()
        if not monologue:
            return {"monologue_extract": _default_extract()}

        result = _run_extract(state, monologue, llm_invoker)
        selected_ids = result.get("selected_content_move_ids") or []

        # 更新 recent_move_history（保留最近 5 轮）
        history: List[List[int]] = list(state.get("recent_move_history") or [])
        history.append(selected_ids)
        history = history[-5:]

        logger.info(
            "[Extract] emotion_tag=%s bot_stance=%s user_act=%s move_ids=%s",
            result.get("emotion_tag"),
            result.get("bot_stance"),
            result.get("user_act"),
            selected_ids,
        )
        return {
            "monologue_extract": result,
            "recent_move_history": history,
        }

    return extract_node


def _default_extract() -> Dict[str, Any]:
    return {
        "emotion_tag": "平静",
        "bot_stance": "supportive",
        "topic_appeal": 5.0,
        "selected_profile_keys": [],
        "selected_content_move_ids": [1, 4, 8, 6],
        "user_act": "venting",
        "inferred_gender": None,
    }


def _gender_unknown(state: AgentState) -> bool:
    g = str((state.get("user_basic_info") or {}).get("gender") or "").strip()
    return not g


def _run_extract(state: AgentState, monologue: str, llm_invoker: Any) -> Dict[str, Any]:
    # 加载 content moves（仅用于日志，不注入 LLM prompt）
    try:
        transformations = _normalize_pure_content_transformations(load_pure_content_transformations())
    except Exception as e:
        logger.warning("[Extract] load_pure_content_transformations failed: %s", e)
        transformations = []

    valid_ids = [int(m["id"]) for m in transformations if m.get("id") is not None]
    valid_ids_list = sorted(valid_ids) if valid_ids else list(range(1, 14))

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

    system_content = f"""你是分析助手，请从下面的「内心独白」和用户消息中提取结构化信息，严格输出 JSON。

## 可选 profile key（0-5个）
[{profile_keys_str}]
{gender_instruction}
## bot_stance 说明（本轮沟通立场，五选一）
- supportive：认同/共情，回应对方情绪或立场
- exploratory：追问深挖，对话题有兴趣想了解更多细节
- self_sharing：主动分享，对方的话触发了想说自己的事（经历/感受/近况），或纯粹想聊聊自己
- redirecting：温和转移，当前话题已无聊/重复，想引向其他方向
- challenging：轻度挑战，用反问或不同视角推动对话（只在关系较亲密时用）

## user_act 说明（用户本条消息的行为类型，五选一）
- asking：用户在提问（问事实、问意见、问原因）
- venting：用户在倾诉/表达情绪（讲事情、抒发感受、吐槽）
- deciding：用户在做决策/求建议（纠结某件事、问该怎么办）
- pushing_back：用户在反驳/质疑/表达不满（不同意你说的、怼回来）
- phatic：低能量寒暄/维持性回复（嗯嗯、哈哈、还好吧、表情包式回应）

## 输出格式（仅输出此 JSON，不要其他内容）
{{
  "emotion_tag": "一两个词，从独白中自然提炼",
  "bot_stance": "supportive",
  "topic_appeal": 5.0,
  "selected_profile_keys": [],
  "user_act": "venting"{gender_field}
}}"""

    _user_input = safe_text(state.get("user_input") or "").strip()
    user_content = f"用户消息：\n{_user_input}\n\n内心独白：\n{monologue}{_recent_bot_block}"

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

        # user_act 验证
        user_act = str(data.get("user_act", "venting")).strip().lower()
        if user_act not in _VALID_USER_ACTS:
            user_act = "venting"

        # 用加权算法选 move
        selected_ids = select_moves(user_act, state, valid_ids_list)

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
            "user_act": user_act,
            "selected_content_move_ids": selected_ids,
            "move_details": move_details,
        }
        log_llm_response("Extract", "(parsed)", parsed_result=result_for_log)

        # inferred_gender
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
            "user_act": user_act,
            "inferred_gender": inferred_gender,
        }

    except Exception as e:
        logger.exception("[Extract] failed: %s", e)
        return default

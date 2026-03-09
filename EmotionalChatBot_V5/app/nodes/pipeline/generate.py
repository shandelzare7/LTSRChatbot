"""并行生成节点（新架构核心：替代 LATS）。

5 路并发（2-4 个 move 路 + 1 FREE 路），每路用 Qwen API n=4 参数一次性拿到 4 个候选。
延迟 ≈ max(单路延迟)，总候选数 ≤ 20。Judge 节点从所有候选中选最优。

共享输入：独白全文 + style 指令 + 角色信息 + 8-12轮对话 + 用户消息
各路差异：move 动作描述（FREE 路无）
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.prompts.prompt_utils import format_style_as_param_list, safe_text
from app.state import AgentState
from utils.detailed_logging import log_prompt_and_params
from utils.time_context import _parse_ts, _to_local
from utils.tracing import trace_if_enabled
from utils.yaml_loader import load_pure_content_transformations

_WEEKDAY_ZH = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")

logger = logging.getLogger(__name__)

RECENT_DIALOGUE_LAST_N = 12
RECENT_MSG_CONTENT_MAX = 600
LATEST_USER_TEXT_MAX = 800
CANDIDATES_PER_ROUTE = 4


def _is_user_message(m: Any) -> bool:
    t = getattr(m, "type", "") or ""
    return "human" in t.lower() or "user" in t.lower()


def _momentum_to_direction(momentum: float) -> str:
    """将 conversation_momentum 值映射为信息密度 + 深度指令（不指定具体手法，由 bot 自由选择）。"""
    if momentum >= 0.80:
        return "饱满 [+2]：回复信息密度高，深入话题的某个具体细节、真实经历或新角度，避免重复已说过的词句，让对话有充足空间延续"
    elif momentum >= 0.60:
        return "延展 [+1]：在对方内容基础上补充具体细节或个人真实感受，内容充实，不刻意收尾"
    elif momentum >= 0.40:
        return "对等 [0]：提供与对方信息量相当的回复，保持当前节奏，不主动延展也不缩减"
    elif momentum >= 0.20:
        return "简短 [-1]：只做基础承接，语气平淡，回复偏短，不主动提供额外信息"
    else:
        return "终结 [-2]：给出陈述性结论，内容简短，语气趋于冷淡，不延展话题"



def _normalize_pure_content_transformations(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("moves") or raw.get("pure_content_transformations") or []
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, dict)]


def _load_move_descriptions() -> Dict[int, Dict[str, str]]:
    """返回 {move_id: {name, description}} 字典。"""
    try:
        transformations = _normalize_pure_content_transformations(load_pure_content_transformations())
        return {
            int(m["id"]): {"name": (m.get("name") or "").strip(), "desc": (m.get("content_operation") or "").strip()}
            for m in transformations
            if m.get("id") is not None
        }
    except Exception as e:
        logger.warning("[Generate] 加载 content moves 失败: %s", e)
        return {}


def _build_dialogue_context(state: AgentState) -> str:
    chat_buffer: List[BaseMessage] = list(
        state.get("chat_buffer") or state.get("messages", [])[-RECENT_DIALOGUE_LAST_N * 2:]
    )[-RECENT_DIALOGUE_LAST_N * 2:]

    lines: List[str] = []
    for m in chat_buffer:
        role = "Human" if _is_user_message(m) else "AI"
        content = (getattr(m, "content", "") or str(m)).strip()
        if len(content) > RECENT_MSG_CONTENT_MAX:
            content = content[:RECENT_MSG_CONTENT_MAX] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "（无历史对话）"


_TASK_HINT = {
    "ask_user_name":       "对方叫什么名字",
    "ask_user_age":        "对方多大",
    "ask_user_occupation": "对方做什么工作",
    "ask_user_location":   "对方在哪个城市",
}


def _build_basic_info_task_block(state: AgentState) -> str:
    """构建 free 路由专用的基础信息问询提示。前期温和，后期强硬。"""
    pending = list(
        (state.get("relationship_assets") or {}).get("session_basic_info_pending_task_ids") or []
    )
    completed = set(state.get("completed_task_ids") or [])
    active = [t for t in pending if t not in completed and t in _TASK_HINT]
    if not active:
        return ""

    turn = int(state.get("turn_count_in_session") or 0)
    items = [_TASK_HINT[t] for t in active[:2]]

    if turn >= 8:
        header = "## 你还不知道的事（已经聊了很久了，这轮回复里必须自然地问出来）"
    elif turn >= 4:
        header = "## 你还不知道的事（找个自然的时机顺便问一下）"
    else:
        header = "## 你还不知道的事（不急，聊天中合适了再带出来）"

    return header + "\n" + "\n".join(f"- {h}" for h in items) + "\n"



def _build_profile_block(state: AgentState, selected_keys: List[str]) -> str:
    """根据 selected_profile_keys 从 user_inferred_profile 提取相关条目，生成提示块。"""
    if not selected_keys:
        return ""
    profile = state.get("user_inferred_profile") or {}
    if not isinstance(profile, dict):
        return ""
    items = [
        f"- {k}：{safe_text(str(profile[k]))[:60]}"
        for k in selected_keys
        if k in profile and profile[k]
    ]
    if not items:
        return ""
    return "## 关于对方的已知信息（自然融入，不要生硬列举）\n" + "\n".join(items) + "\n"


def _build_messages_for_route(
    state: AgentState,
    move_desc: Optional[str],
    move_name: Optional[str],
    dialogue_context: str,
    style_text: str,
    monologue: str,
    bot_name: str,
    user_name: str,
    task_hint: str = "",
    profile_block: str = "",
) -> List[Any]:
    """为单路生成构建 messages 列表。"""
    user_input = (state.get("user_input") or "").strip()[:LATEST_USER_TEXT_MAX]

    # 人设信息
    bot_basic_info = state.get("bot_basic_info") or {}
    bot_persona = state.get("bot_persona") or ""
    persona_text = ""
    if bot_persona:
        persona_text = f"\n## 你的人设\n{safe_text(str(bot_persona))[:500]}"

    # Move 约束（不暴露动作名，避免 LLM 照抄）
    move_block = ""
    if move_desc:
        move_block = f"""
## 本次回复的内容约束
{move_desc}
（自然融入，不要提及这条约束本身）
"""

    # 字数上限：1/2 * 对方消息长度 + 35 * momentum；下限 1，硬上限 60 字
    momentum = float(state.get("conversation_momentum") or 0.5)
    _other_len = len(user_input or "")
    max_chars = min(60, int(_other_len * 0.5) + int(35 * momentum))
    min_chars = 1

    direction = _momentum_to_direction(momentum)
    direction_block = f"## 当前对话意愿（信息密度）\n{direction}\n"

    # 外部素材块：bot 生活事件 + 资讯话题
    # 始终提供，由 LLM 根据独白自主决定是否引入，不做 active/passive 条件切换
    daily_topics_list = list(state.get("daily_topics") or [])
    bot_recent_list = list(state.get("bot_recent_activities") or [])
    external_context_block = ""
    if bot_recent_list or daily_topics_list:
        parts: list[str] = ["\n## 你最近的生活 & 知道的事（独白里如果有想聊的冲动，可以顺势带进来）"]
        if bot_recent_list:
            parts.append("【你最近的事】")
            parts.extend(f"- {t}" for t in bot_recent_list[:5])
        if daily_topics_list:
            parts.append("【你听到/看到的】")
            parts.extend(f"- {t}" for t in daily_topics_list[:5])
        external_context_block = "\n".join(parts) + "\n"
    daily_topics_block = external_context_block  # 保持变量名兼容

    # 当前日期与星期（回答「今天星期几」等事实问题时必须使用此项，不要猜测）
    time_fact_block = ""
    try:
        now = _parse_ts(state.get("current_time"))
        if now is not None:
            local_now = _to_local(now)
            wd = _WEEKDAY_ZH[local_now.weekday()]
            time_fact_block = f"## 当前日期与星期\n{local_now.year}年{local_now.month}月{local_now.day}日 {wd}\n\n"
    except Exception:
        pass

    # 搜索结果事实块：当有外部搜索结果时，直接注入生成提示词，确保事实被使用
    ext_knowledge = (state.get("retrieved_external_knowledge") or "").strip()
    ext_facts_block = ""
    if ext_knowledge:
        ext_facts_block = (
            f"## 你刚查到的事实（对方在问这个，回复中必须自然地包含这些信息）\n"
            f"⚠️ 以下是实时搜索到的最新事实，**必须在回复中体现核心信息**，用你自己的话自然地说出来。\n"
            f"{ext_knowledge}\n\n"
        )

    system_content = f"""你是 {bot_name}。你正在和 {user_name} 对话。
{persona_text}
{profile_block}
{time_fact_block}{ext_facts_block}## 你的内心活动（情绪/态度/意愿）——用于调节回复基调，不是要说出口的内容
{monologue}

{direction_block}
## 写作风格参数
{style_text}
{daily_topics_block}
{move_block}{task_hint}## 回复规则
- 第一人称，你就是这个人在社交软件上聊天，说话口语自然，不要暴露是AI
- 回复要和内心独白的情绪/态度保持一致，**信息密度指令优先于其他规则**
- 如果上面有「你刚查到的事实」，必须在回复中自然地融入核心信息，不要回避或含糊——对方在问你这个问题，你要直接回答
- 不要使用"作为AI"、"根据您的请求"等客服套话
- 自然、有个性，符合写作风格参数
- 每条消息开头应自然变化，不要以固定词语重复起句
- **严禁**：任何括号形式的动作描写，如（戳戳脸颊）[轻轻拍你]——聊天消息里不会出现这种写法
- **严禁一切文学化表达**（违反即废稿，零容忍）：
  ① 文学性修辞=禁止：禁止比喻、拟人、排比、对偶、夸张、通感、借代、反复等文学性修辞
  ② 文学性=zero，散文感=zero：禁止散文化句子、金句、哲理感悟、诗意总结
  ③ 抒情性=zero：禁止抒情，不要"感叹人生"、不要"总结情感"、不要升华
  ④ 书面语体=禁止：只用口语，禁止书面语。"我觉得挺好的"可以，"这便是最好的答案"不行
  ⑤ 意象=禁止：禁止一切意象化描写，如"光"、"风"、"雨"、"路"等用作隐喻
  ⑥ 押韵=禁止：句尾不允许押韵、对仗、节奏工整
  你是普通人在微信上打字，不是在写作文。说人话，短句，口语，不完整也没关系。
  × "旧东西才肯说真话" ← 金句，禁止
  × "人和猫之间，靠的都是这种不刻意的记得" ← 散文化总结，禁止
  × "因为不急着赶路，才听得见" ← 文学性抒情，禁止
  × "偷偷留住那点安静" ← 意象化，禁止
  × "像在偷听别人的故事" ← 比喻，禁止
  ✓ "哈哈那挺好的" ✓ "你说的对，我也这么觉得" ✓ "行吧，回头再说"
- 回复直接输出，不要任何前缀或格式标记
"""

    user_content = f"""## 历史对话（最近 {RECENT_DIALOGUE_LAST_N} 条）
{dialogue_context}

## 当前用户消息
{user_input or '（空）'}

请严格按照上方「写作风格参数」的全部要求，写出你（{bot_name}）的回复（**字数限制：{min_chars}-{max_chars} 字，社交软件聊天风格，说完就停**）："""

    return [SystemMessage(content=system_content), HumanMessage(content=user_content)]


async def _generate_route(
    llm_gen: Any,
    messages: List[Any],
    move_id: Optional[int],
    route_label: str,
    max_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """单路异步生成，llm_gen 构建时已通过 n=4 传入批量参数；不 bind(max_tokens) 以保留 agenerate，改在调用时传参。"""
    extra_kw: Dict[str, Any] = {}
    if max_tokens is not None:
        extra_kw["max_tokens"] = max_tokens
    candidates: List[Dict[str, Any]] = []
    try:
        if hasattr(llm_gen, "agenerate"):
            result = await llm_gen.agenerate([messages], **extra_kw)
            gens = result.generations[0] if result.generations else []
            for gen in gens:
                text = ""
                if hasattr(gen, "text"):
                    text = str(gen.text).strip()
                elif hasattr(gen, "message"):
                    text = (getattr(gen.message, "content", "") or "").strip()
                if text:
                    candidates.append({"move_id": move_id, "route": route_label, "text": text})
        elif hasattr(llm_gen, "ainvoke"):
            msg = await llm_gen.ainvoke(messages, **extra_kw)
            text = (getattr(msg, "content", "") or str(msg)).strip()
            if text:
                candidates.append({"move_id": move_id, "route": route_label, "text": text})

    except Exception as e:
        logger.exception("[Generate] route=%s 异常: %s", route_label, e)

    logger.info("[Generate] route=%s candidates=%d", route_label, len(candidates))
    return candidates


def create_generate_node(llm_gen: Any) -> Callable[[AgentState], Any]:
    """创建并行生成节点（async）。"""

    @trace_if_enabled(
        name="Response/Generate",
        run_type="chain",
        tags=["node", "generate"],
        metadata={"state_outputs": ["generation_candidates"]},
    )
    async def generate_node(state: AgentState) -> Dict[str, Any]:
        extract = state.get("monologue_extract") or {}
        move_ids: List[int] = list(extract.get("selected_content_move_ids") or [])[:4]
        monologue = (state.get("inner_monologue") or "按常理接话即可。").strip()

        # 加载 move 描述
        move_map = _load_move_descriptions()

        # 构建共享上下文
        bot_basic_info = state.get("bot_basic_info") or {}
        user_basic_info = state.get("user_basic_info") or {}
        bot_name = safe_text((bot_basic_info or {}).get("name") or "Bot").strip() or "Bot"
        user_name_raw = safe_text((user_basic_info or {}).get("name") or "").strip()
        user_name = user_name_raw if user_name_raw else "对方"

        dialogue_context = _build_dialogue_context(state)
        style_dict = state.get("style") or {}
        style_text = format_style_as_param_list(style_dict) or "（默认风格）"

        # 根据 momentum 计算 max_tokens（硬约束，强制截断超长输出）
        # 中文约 1-1.5 token/字，加 10 token 余量给标点和空格
        momentum = float(state.get("conversation_momentum") or 0.5)
        _max_chars = min(60, int(11 + 34 * momentum))
        _max_tokens = int(_max_chars * 4) + 40    # 宽松安全网，不干扰正常生成
        logger.info("[Generate] momentum=%.2f max_chars=%d max_tokens=%d", momentum, _max_chars, _max_tokens)

        # ── ABLATION_MODE：跳过多路 Move 生成 + Judge，单次 LLM 直接生成 ──
        if os.getenv("ABLATION_MODE"):
            _extract = dict(state.get("monologue_extract") or {})
            _selected_profile_keys: List[str] = list(_extract.get("selected_profile_keys") or [])
            _profile_block = _build_profile_block(state, _selected_profile_keys)
            _task_hint = _build_basic_info_task_block(state)
            free_msgs = _build_messages_for_route(
                state, None, None, dialogue_context, style_text, monologue, bot_name, user_name,
                task_hint=_task_hint, profile_block=_profile_block,
            )
            logger.info("[Generate][ABLATION] 单次生成，跳过 Move 多路 + Judge")
            try:
                msg = await llm_gen.ainvoke(free_msgs, max_tokens=_max_tokens)
                text = (getattr(msg, "content", "") or str(msg)).strip()
            except Exception as e:
                logger.exception("[Generate][ABLATION] 生成异常: %s", e)
                text = ""
            return {"generation_candidates": [{"move_id": None, "route": "ablation_direct", "text": text}]}

        # ── 正常模式：多路并行生成 ──

        # 提取共享的 extract 信号
        _extract = dict(state.get("monologue_extract") or {})
        _bot_stance = str(_extract.get("bot_stance") or "")
        _selected_profile_keys: List[str] = list(_extract.get("selected_profile_keys") or [])
        _profile_block = _build_profile_block(state, _selected_profile_keys)

        # 为每路构建任务（同时收集路由信息供日志使用）
        route_infos: List[tuple] = []  # (label, mid, name, desc, msgs)
        tasks = []

        # Move 路（2-4 个）
        for mid in move_ids:
            move_info = move_map.get(mid, {})
            move_name = move_info.get("name", f"move_{mid}")
            move_desc = move_info.get("desc", "")
            msgs = _build_messages_for_route(
                state, move_desc, move_name, dialogue_context, style_text, monologue, bot_name, user_name,
                profile_block=_profile_block,
            )
            label = f"move_{mid}"
            route_infos.append((label, mid, move_name, move_desc, msgs))
            tasks.append(_generate_route(llm_gen, msgs, mid, label, max_tokens=_max_tokens))

        # FREE 路（无 move 约束，附带基础信息问询任务提示）
        _task_hint = _build_basic_info_task_block(state)
        free_msgs = _build_messages_for_route(
            state, None, None, dialogue_context, style_text, monologue, bot_name, user_name,
            task_hint=_task_hint, profile_block=_profile_block,
        )
        route_infos.append(("free", None, "FREE", "", free_msgs))
        tasks.append(_generate_route(llm_gen, free_msgs, None, "free", max_tokens=_max_tokens))

        # 日志：路由配置概要
        logger.info("[Generate] ===== 生成路由配置 =====")
        logger.info("  选中 move_ids: %s", move_ids)
        for label, mid, name, desc, _ in route_infos:
            desc_short = desc[:80] + "…" if len(desc) > 80 else (desc or "无约束")
            logger.info("  路由 %-12s | %s | %s", label, name, desc_short)
        logger.info("[Generate] =================================")

        # 日志：第一路完整提示词（其他路仅 move_block 不同，不重复输出）
        if route_infos:
            first_label, _, first_name, _, first_msgs = route_infos[0]
            logger.info("[Generate] ===== 提示词（%s 路，其他路仅 move_block 不同）=====", first_label)
            log_prompt_and_params(f"Generate/{first_label}", messages=first_msgs)

        # 并行执行所有路
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_candidates: List[Dict[str, Any]] = []
        for (label, mid, name, desc, _), r in zip(route_infos, results):
            if isinstance(r, list):
                all_candidates.extend(r)
            elif isinstance(r, Exception):
                logger.warning("[Generate] 路由 %s 异常: %s", label, r)

        # 日志：所有候选全文（按路由分组）；logger 会写入会话 log（WebChatLogHandler）
        logger.info("[Generate] ===== 全部候选（按路由）=====")
        for (label, mid, name, desc, _), r in zip(route_infos, results):
            candidates_in_route = r if isinstance(r, list) else []
            n_in_route = len(candidates_in_route)
            logger.info("  【路由 %s】(%s): %d 个候选", label, name, n_in_route)
            for i, c in enumerate(candidates_in_route):
                text = (c.get("text") or "").strip()
                logger.debug("    [%d] %s", i, text)
        logger.info("[Generate] 总计 %d 个候选，%d 路", len(all_candidates), len(tasks))
        logger.info("[Generate] =============================")

        # 如果所有路都失败，产出一个空候选避免 judge 崩溃
        if not all_candidates:
            all_candidates = [{"move_id": None, "route": "fallback", "text": ""}]

        return {"generation_candidates": all_candidates}

    return generate_node

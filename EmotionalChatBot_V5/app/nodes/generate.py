"""并行生成节点（新架构核心：替代 LATS）。

5 路并发（2-4 个 move 路 + 1 FREE 路），每路用 Qwen API n=4 参数一次性拿到 4 个候选。
延迟 ≈ max(单路延迟)，总候选数 ≤ 20。Judge 节点从所有候选中选最优。

共享输入：独白全文 + style 指令 + 角色信息 + 8-12轮对话 + 用户消息
各路差异：move 动作描述（FREE 路无）
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.lats.prompt_utils import format_style_as_param_list, safe_text
from app.state import AgentState
from utils.detailed_logging import log_prompt_and_params
from utils.tracing import trace_if_enabled
from utils.yaml_loader import load_pure_content_transformations

logger = logging.getLogger(__name__)

RECENT_DIALOGUE_LAST_N = 12
RECENT_MSG_CONTENT_MAX = 600
LATEST_USER_TEXT_MAX = 800
CANDIDATES_PER_ROUTE = 4


def _is_user_message(m: Any) -> bool:
    t = getattr(m, "type", "") or ""
    return "human" in t.lower() or "user" in t.lower()


def _momentum_to_direction(momentum: float) -> str:
    """将 conversation_momentum 值映射为信息密度指令（不指定具体手法，由 bot 自由选择）。"""
    if momentum >= 0.80:
        return "饱满 [+2]：回复信息密度高，自然带出更多细节、看法或相关内容，让对话有充足空间延续"
    elif momentum >= 0.60:
        return "延展 [+1]：在对方内容基础上补充细节或个人感受，内容充实，不刻意收尾"
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
        raw = raw.get("pure_content_transformations") or []
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


def _build_messages_for_route(
    state: AgentState,
    move_desc: Optional[str],
    move_name: Optional[str],
    dialogue_context: str,
    style_text: str,
    monologue: str,
    bot_name: str,
    user_name: str,
) -> List[Any]:
    """为单路生成构建 messages 列表。"""
    user_input = (state.get("user_input") or "").strip()[:LATEST_USER_TEXT_MAX]

    # 人设信息
    bot_basic_info = state.get("bot_basic_info") or {}
    bot_persona = state.get("bot_persona") or ""
    persona_text = ""
    if bot_persona:
        persona_text = f"\n## 你的人设\n{safe_text(str(bot_persona))[:500]}"

    # Move 约束
    move_block = ""
    if move_desc:
        move_block = f"""
## 本次回复的内容约束（必须融入，不要照抄）
动作名：{move_name}
具体要求：{move_desc}
"""

    # 计算动态字数限制（根据 momentum）
    momentum = float(state.get("conversation_momentum") or 0.5)
    # 微信聊天合理范围：低动量真短，高动量真长
    # momentum=0.2→5-40, 0.4→10-65, 0.5→15-82, 0.7→20-110, 1.0→25-150
    max_chars = int(15 + 135 * momentum)
    min_chars = max(3, int(5 + 20 * momentum))

    direction = _momentum_to_direction(momentum)
    direction_block = f"## 当前对话意愿（信息密度）\n{direction}\n"

    # 外部素材块：bot 生活事件 + 资讯话题（如话题合适，可自然带入，不强制）
    daily_topics_list = list(state.get("daily_topics") or [])
    bot_recent_list = list(state.get("bot_recent_activities") or [])
    external_context_block = ""
    if bot_recent_list or daily_topics_list:
        parts: list[str] = ["\n## 你最近的生活 & 知道的事（话题合适时可自然带入，不必强行）"]
        if bot_recent_list:
            parts.append("【你最近的事】")
            parts.extend(f"- {t}" for t in bot_recent_list[:5])
        if daily_topics_list:
            parts.append("【你听到/看到的】")
            parts.extend(f"- {t}" for t in daily_topics_list[:5])
        external_context_block = "\n".join(parts) + "\n"
    daily_topics_block = external_context_block  # 保持变量名兼容

    system_content = f"""你是 {bot_name}，一个拟人化的聊天伙伴。你正在和 {user_name} 对话。
{persona_text}

## 你的内心独白（刚才的真实想法，供参考，不要直接说出）
{monologue}

{direction_block}
## 写作风格参数
{style_text}
{daily_topics_block}
{move_block}
## 回复规则
- 第一人称，你就是这个人在发微信，说话口语自然，不要暴露是AI
- 回复要和内心独白的情绪/态度保持一致，**信息密度指令优先于其他规则**
- 不要使用"作为AI"、"根据您的请求"等客服套话
- 自然、有个性，符合写作风格参数
- **字数限制（momentum={momentum:.2f}）**：{min_chars}-{max_chars} 字，微信聊天风格，说完就停
- **严禁**：诗意表达、押韵、排比句、散文化抒情、比喻堆叠——这不是文学创作，是真实聊天
- **严禁**：括号内的动作描写，如（戳戳脸颊）（轻轻拍你）——聊天消息里不会出现这种写法
- 回复直接输出，不要任何前缀或格式标记
"""

    user_content = f"""## 历史对话（最近 {RECENT_DIALOGUE_LAST_N} 条）
{dialogue_context}

## 当前用户消息
{user_input or '（空）'}

请直接写出你（{bot_name}）的回复："""

    return [SystemMessage(content=system_content), HumanMessage(content=user_content)]


async def _generate_route(
    llm_gen: Any,
    messages: List[Any],
    move_id: Optional[int],
    route_label: str,
) -> List[Dict[str, Any]]:
    """单路异步生成，llm_gen 构建时已通过 model_kwargs n=4 传入批量参数。"""
    candidates: List[Dict[str, Any]] = []
    try:
        if hasattr(llm_gen, "agenerate"):
            result = await llm_gen.agenerate([messages])
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
            msg = await llm_gen.ainvoke(messages)
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

        # 为每路构建任务（同时收集路由信息供日志使用）
        route_infos: List[tuple] = []  # (label, mid, name, desc, msgs)
        tasks = []

        # Move 路（2-4 个）
        for mid in move_ids:
            move_info = move_map.get(mid, {})
            move_name = move_info.get("name", f"move_{mid}")
            move_desc = move_info.get("desc", "")
            msgs = _build_messages_for_route(
                state, move_desc, move_name, dialogue_context, style_text, monologue, bot_name, user_name
            )
            label = f"move_{mid}"
            route_infos.append((label, mid, move_name, move_desc, msgs))
            tasks.append(_generate_route(llm_gen, msgs, mid, label))

        # FREE 路（无 move 约束）
        free_msgs = _build_messages_for_route(
            state, None, None, dialogue_context, style_text, monologue, bot_name, user_name
        )
        route_infos.append(("free", None, "FREE", "", free_msgs))
        tasks.append(_generate_route(llm_gen, free_msgs, None, "free"))

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

        # 日志：所有候选全文（按路由分组）
        logger.info("[Generate] ===== 全部候选（按路由）=====")
        for (label, mid, name, desc, _), r in zip(route_infos, results):
            candidates_in_route = r if isinstance(r, list) else []
            logger.info("  【路由 %s】(%s): %d 个候选", label, name, len(candidates_in_route))
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

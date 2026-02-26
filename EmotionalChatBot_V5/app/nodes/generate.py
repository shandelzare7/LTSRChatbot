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

from app.lats.prompt_utils import build_system_memory_block, format_style_as_param_list, safe_text
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
    """将 conversation_momentum 值映射为对话方向指令（对应旧版 base_momentum 策略）。"""
    if momentum >= 0.80:
        return "创造/延续 [+2]：主动抛出具有探讨空间的问题或发散相关新话题，给对方充足的接话切入点"
    elif momentum >= 0.60:
        return "延展 [+1]：主动拓展话题边界，分享丰富细节或个人见解，让对话更有深度"
    elif momentum >= 0.40:
        return "维系 [0]：提供与对方信息量对等的回复，维持当前节奏，不刻意延展也不缩减"
    elif momentum >= 0.20:
        return "收敛 [-1]：被动承接上文，语气平淡，让对方继续主导，自己不主动延展"
    else:
        return "阻断 [-2]：给出终结话题的陈述性结论，明显降低热情，不鼓励对方继续此话题"


def _big_five_to_constraints(state: AgentState) -> str:
    """将极端 Big Five 值转为具体行为约束（确定性映射，非人设描述）。

    只在维度明显偏极端（低于 0.35 或高于 0.65）时注入，避免对中间值过度干预。
    """
    bf = state.get("bot_big_five") or {}
    try:
        A = float(bf.get("agreeableness", 0.5))
        E = float(bf.get("extraversion", 0.5))
        O = float(bf.get("openness", 0.5))
        N = float(bf.get("neuroticism", 0.5))
    except (TypeError, ValueError):
        return ""

    constraints: List[str] = []
    if A < 0.35:
        constraints.append("不要完全认同对方的每句话，可以轻描淡写、带点保留，甚至不接这个话题")
    if E < 0.35:
        constraints.append("话少简短是你的常态，不要主动抛出多个追问")
    if O < 0.35:
        constraints.append("用具体日常的词语，不要用比喻或意象堆叠")
    if N > 0.65:
        constraints.append("允许直接流露情绪波动，不需要维持表面平稳")

    if not constraints:
        return ""
    lines = "\n".join(f"- {c}" for c in constraints)
    return f"## 人格行为约束（优先级高于风格参数）\n{lines}"


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
    memory_block: str,
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
    max_chars = int(20 + 20 * momentum)  # 基础 20 + momentum 增幅
    min_chars = max(1, int(10 * momentum))  # 最少保留基础

    # 从 monologue_extract 读取情绪/态度信号（增强生成方向）
    extract = state.get("monologue_extract") or {}
    emotion_tag = str(extract.get("emotion_tag") or "").strip()
    attitude = str(extract.get("attitude") or "").strip()
    direction = _momentum_to_direction(momentum)

    direction_block = f"""## 当前对话意愿指令（必须体现在回复中）
- **对话方向**：{direction}
- **回复态度**：{attitude or "正常接话"}（来自独白，体现在语气和措辞上）
- **当前情绪**：{emotion_tag or "平静"}（决定语气基调：开心→轻快活泼，烦躁→简短带情绪，心疼→温柔带担忧，低落→沉静克制）
"""

    personality_constraints = _big_five_to_constraints(state)
    personality_block = f"\n{personality_constraints}\n" if personality_constraints else ""

    system_content = f"""你是 {bot_name}，一个拟人化的聊天伙伴。你正在和 {user_name} 对话。
{persona_text}

## 你的内心独白（刚才的真实想法，供参考，不要直接说出）
{monologue}

{direction_block}
## 写作风格参数
{style_text}
{personality_block}
## 记忆
{memory_block}
{move_block}
## 回复规则
- 第一人称，你就是这个人在发微信，说话口语自然，不要暴露是AI
- 回复要和内心独白的情绪/态度保持一致，**对话方向指令优先于其他规则**
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
        memory_block = build_system_memory_block(state)
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
                state, move_desc, move_name, dialogue_context, memory_block, style_text, monologue, bot_name, user_name
            )
            label = f"move_{mid}"
            route_infos.append((label, mid, move_name, move_desc, msgs))
            tasks.append(_generate_route(llm_gen, msgs, mid, label))

        # FREE 路（无 move 约束）
        free_msgs = _build_messages_for_route(
            state, None, None, dialogue_context, memory_block, style_text, monologue, bot_name, user_name
        )
        route_infos.append(("free", None, "FREE", "", free_msgs))
        tasks.append(_generate_route(llm_gen, free_msgs, None, "free"))

        # 日志：路由配置概要
        print("[Generate] ===== 生成路由配置 =====")
        print(f"  选中 move_ids: {move_ids}")
        for label, mid, name, desc, _ in route_infos:
            desc_short = desc[:80] + "…" if len(desc) > 80 else (desc or "无约束")
            print(f"  路由 {label:<12} | {name} | {desc_short}")
        print("[Generate] =================================")

        # 日志：第一路完整提示词（其他路仅 move_block 不同，不重复输出）
        if route_infos:
            first_label, _, first_name, _, first_msgs = route_infos[0]
            print(f"[Generate] ===== 提示词（{first_label} 路，其他路仅 move_block 不同）=====")
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
        print("[Generate] ===== 全部候选（按路由）=====")
        for (label, mid, name, desc, _), r in zip(route_infos, results):
            candidates_in_route = r if isinstance(r, list) else []
            print(f"\n  【路由 {label}】({name}): {len(candidates_in_route)} 个候选")
            for i, c in enumerate(candidates_in_route):
                text = (c.get("text") or "").strip()
                print(f"    [{i}] {text}")
        print(f"\n[Generate] 总计 {len(all_candidates)} 个候选，{len(tasks)} 路")
        print("[Generate] =============================")

        # 如果所有路都失败，产出一个空候选避免 judge 崩溃
        if not all_candidates:
            all_candidates = [{"move_id": None, "route": "fallback", "text": ""}]

        return {"generation_candidates": all_candidates}

    return generate_node

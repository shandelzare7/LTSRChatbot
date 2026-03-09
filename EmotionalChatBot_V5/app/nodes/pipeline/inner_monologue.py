"""内心独白节点（新架构核心模块）：产出角色的真实内在反应文本。

改动说明（V2 版本）：
- 只输出纯文本独白，不做结构化提取（profile_keys / move_ids 移入 monologue_extraction 节点）
- 输入改进：使用 state_to_text 将 PAD/busy/momentum/relationship 转成有感染力的文本
- 字数目标 400-600，让独白自然流出
- Prompt 重点：被触发的感受，不是"我应该怎么回"，而是"我心里真实的翻涌"
- 历史窗口：10 轮（对 RAG 检索结果的关联度更好）
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Dict, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.prompts.prompt_utils import safe_text
from utils.tracing import trace_if_enabled
from utils.state_to_text import convert_state_to_context_text
from datetime import datetime, timezone
from utils.time_context import _parse_ts, get_day_part, _to_local, _time_gap_readable
from src.schemas import InnerMonologueOutput

from app.state import AgentState

logger = logging.getLogger(__name__)

INNER_MONOLOGUE_MAX_CHARS = 2000  # 软目标 600-1200，硬截断防溢出
LATEST_USER_TEXT_MAX = 800
RECENT_MSG_CONTENT_MAX = 200  # 每条对话只显示 200 字，减少噪声
RECENT_DIALOGUE_LAST_N = 15   # 保留 15 轮，保证上下文连贯


def _is_user_message(m: BaseMessage) -> bool:
    t = getattr(m, "type", "") or ""
    return "human" in t.lower() or "user" in t.lower()


def _build_user_profile_summary(state: AgentState) -> str:
    """从user_inferred_profile生成简短的用户画像总结。"""
    user_profile = state.get("user_inferred_profile") or {}
    if not user_profile:
        return "（关于他的了解还不多）"

    lines = []
    # 取最多5条关键信息
    for key, value in list(user_profile.items())[:5]:
        if isinstance(value, (str, int, float)):
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) if lines else "（关于他的了解还不多）"


def _gather_context_for_monologue(state: dict) -> Dict[str, str]:
    """收集内心独白所需的所有上下文信息。"""
    chat_buffer: List[BaseMessage] = list(
        state.get("chat_buffer") or state.get("messages", [])[-RECENT_DIALOGUE_LAST_N:]
    )

    # 最新用户消息
    latest_user_text_raw = (state.get("user_input") or "").strip()
    if not latest_user_text_raw and chat_buffer:
        last_msg = chat_buffer[-1]
        latest_user_text_raw = (getattr(last_msg, "content", "") or str(last_msg)).strip()
    latest_user_text = (latest_user_text_raw or "（无用户消息）")[:LATEST_USER_TEXT_MAX]

    # 近期对话历史
    lines: List[str] = []
    for m in chat_buffer[-RECENT_DIALOGUE_LAST_N:]:
        role = "User" if _is_user_message(m) else "Bot"
        content = (getattr(m, "content", "") or str(m)).strip()
        if len(content) > RECENT_MSG_CONTENT_MAX:
            content = content[:RECENT_MSG_CONTENT_MAX] + "…"
        lines.append(f"{role}: {content}")
    recent_dialogue_context = "\n".join(lines) if lines else "（无历史对话）"

    # 角色名字和用户名字
    bot_basic_info = state.get("bot_basic_info") or {}
    user_basic_info = state.get("user_basic_info") or {}
    bot_name = safe_text((bot_basic_info or {}).get("name") or "Bot").strip() or "Bot"
    user_name_raw = safe_text((user_basic_info or {}).get("name") or "").strip()
    user_name = user_name_raw if user_name_raw else "（你还不知道对方的名字）"

    # 检索到的相关记忆（top 3-5）
    retrieved_memories: List[str] = list(state.get("retrieved_memories") or [])
    mem_block = ""
    if retrieved_memories:
        mem_lines = [f"- {m[:150]}" for m in retrieved_memories[:5]]
        mem_block = "## 被唤起的记忆\n" + "\n".join(mem_lines)

    # 用户代词（根据已知性别）
    user_gender = str((state.get("user_basic_info") or {}).get("gender") or "").strip()
    user_pronoun = "他" if user_gender == "男" else ("她" if user_gender == "女" else "对方")

    # 未完成的基础信息问询任务
    pending_tasks = list((state.get("relationship_assets") or {}).get("session_basic_info_pending_task_ids") or [])
    completed_task_ids = set(state.get("completed_task_ids") or [])
    active_tasks = [t for t in pending_tasks if t not in completed_task_ids]
    _TASK_HINT = {
        "ask_user_name":       "还不知道对方叫什么——今天聊的时候找个时机问一下，很自然地插进话里就好",
        "ask_user_age":        "还不知道对方年龄，话题合适的话带出来",
        "ask_user_occupation": "还不知道对方做什么工作，感兴趣就问",
        "ask_user_location":   "还不知道对方在哪个城市",
    }
    task_block = ""
    if active_tasks:
        hints = [_TASK_HINT[t] for t in active_tasks if t in _TASK_HINT]
        if hints:
            turn_count = int(state.get("turn_count_in_session") or 0)
            # 前几轮用更强烈的标题，让 bot 真正意识到这是今天的目标
            header = "## 你今天想搞清楚的事（趁这次聊找机会自然带出）" if turn_count <= 8 else "## 你心里记着但还不了解的事"
            task_block = header + "\n" + "\n".join(f"- {h}" for h in hints[:2])

    # 话题欲望钩子：双通道触发
    # 通道 1（appeal 低）：话题本身无聊 → 想换
    # 通道 2（内容停滞）：最近 bot 发言 n-gram 词汇多样性下降 → 原地转圈 → 想换
    # 两个通道各自独立，取较大触发概率，话题切换后自动复位
    import random as _random

    def _content_stagnation(msgs: List[str], n: int = 2) -> float:
        """TTR（type-token ratio）反转：词汇多样性越低 → 停滞分越高（0-1）。"""
        total, union = 0, set()
        for msg in msgs:
            cleaned = msg.replace(" ", "")
            grams = [cleaned[i:i + n] for i in range(len(cleaned) - n + 1)]
            total += len(grams)
            union.update(grams)
        if total == 0:
            return 0.0
        return max(0.0, 1.0 - len(union) / total)

    topic_shift_hook = ""
    turn_count = int(state.get("turn_count_in_session") or 0)
    has_external_topics = bool(state.get("bot_recent_activities") or state.get("daily_topics"))
    if has_external_topics and turn_count >= 2:
        # --- 通道 1：appeal 低 ---
        _prev_extract = dict(state.get("monologue_extract") or {})
        try:
            _topic_appeal = float(_prev_extract.get("topic_appeal", 5.0))
        except (TypeError, ValueError):
            _topic_appeal = 5.0
        _appeal_factor = max(0.0, (6.0 - _topic_appeal) / 6.0)
        _turn_factor = min(1.0, turn_count * 0.10)
        _appeal_prob = _appeal_factor * _turn_factor * 0.85

        # --- 通道 2：内容停滞（取最近 20 条 bot 发言，覆盖分段消息） ---
        _bot_msgs = [
            (getattr(m, "content", "") or str(m)).strip()
            for m in chat_buffer[-20:]
            if not _is_user_message(m) and (getattr(m, "content", "") or str(m)).strip()
        ]
        _stagnation = _content_stagnation(_bot_msgs)  # 0-1，越高越重复
        # stagnation > 0.20 时开始触发，0.37 时触发概率约 75%，线性增长
        _stagnation_prob = min(0.75, max(0.0, (_stagnation - 0.20) * 4.5))

        _final_prob = max(_appeal_prob, _stagnation_prob)
        if _random.random() < _final_prob:
            if _stagnation > 0.35:
                topic_shift_hook = "- 你隐约觉得你们最近说的东西开始转圈了，心里有点想说点不一样的"
            else:
                topic_shift_hook = "- 你心里有点想聊点别的，最近自己的一些事说不定可以带进来"

    # --- 时间感知（方案B：自然第二人称，独立 block） ---
    _WEEKDAY_ZH = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
    time_block_text = ""
    try:
        now = _parse_ts(state.get("current_time"))
        if now is None:
            now = datetime.now(timezone.utc)
        local_now = _to_local(now)
        _, part_label = get_day_part(now)
        wd = _WEEKDAY_ZH[local_now.weekday()]
        time_str = f"现在是{local_now.year}年{local_now.month}月{local_now.day}日 {wd}{part_label}（约{local_now.hour}:{local_now.minute:02d}）"
        gap_sec = state.get("seconds_since_last_message")
        if gap_sec is not None and isinstance(gap_sec, (int, float)) and gap_sec >= 60:
            gap_readable = _time_gap_readable(int(gap_sec))
            time_str += f"。你们上次聊天是{gap_readable}前"
        time_block_text = time_str
    except Exception:
        pass

    # 对话摘要（长期记忆，比 retrieved_memories 更宏观）
    conversation_summary = (state.get("conversation_summary") or "").strip()
    summary_block = ""
    if conversation_summary:
        summary_block = f"## 你对之前聊天的大致印象\n{conversation_summary[:500]}"

    # 上次会话精华摘要（跨 session 回忆）
    session_summary_raw = (state.get("session_summary") or "").strip()
    session_summary_block = ""
    if session_summary_raw:
        session_summary_block = f"## 上次你们聊了什么\n{session_summary_raw[:500]}"

    # 关系变化趋势（上一轮的关系维度变化方向）
    rel_deltas = state.get("relationship_deltas_applied") or {}
    rel_trend_block = ""
    if isinstance(rel_deltas, dict):
        _DIM_ZH = {
            "closeness": "亲密感", "trust": "信任", "liking": "好感",
            "respect": "尊重", "attractiveness": "吸引力", "power": "对方的主导感",
        }
        trend_parts: list[str] = []
        for dim in ("closeness", "trust", "liking", "respect", "attractiveness", "power"):
            delta = rel_deltas.get(dim)
            if not isinstance(delta, (int, float)) or abs(delta) < 0.005:
                continue
            zh = _DIM_ZH.get(dim, dim)
            if delta > 0.03:
                trend_parts.append(f"{zh}明显上升")
            elif delta > 0:
                trend_parts.append(f"{zh}略有上升")
            elif delta < -0.03:
                trend_parts.append(f"{zh}明显下降")
            else:
                trend_parts.append(f"{zh}略有下降")
        if trend_parts:
            rel_trend_block = "## 你感觉到的关系变化\n上一轮对话后：" + "，".join(trend_parts)

    # 外部搜索结果（由 knowledge_fetcher 写入）
    ext_knowledge = (state.get("retrieved_external_knowledge") or "").strip()
    ext_knowledge_block = ""
    if ext_knowledge:
        ext_knowledge_block = (
            f"## 你刚查到的最新信息（优先于你的旧记忆）\n"
            f"⚠️ 以下是刚刚实时搜索到的结果，代表当前最新事实。如果你脑海中的印象与搜索结果不同，**以搜索结果为准**。\n"
            f"{ext_knowledge}"
        )

    # Detection 客观信号
    detection = state.get("detection") or {}
    det_block = ""
    if detection:
        hostility = detection.get("hostility_level", 0)
        engagement = detection.get("engagement_level", 5)
        urgency = detection.get("urgency", 5)
        stage_pacing = detection.get("stage_pacing", "正常")
        det_block = (
            f"## {user_pronoun}说这句话的语境\n"
            f"- 敌意程度：{hostility}/10（0无，10很强）\n"
            f"- 信息量/投入度：{engagement}/10（0冷淡，10很活跃）\n"
            f"- 紧迫感：{urgency}/10（0可延后，10很紧急）\n"
            f"- 关系节奏：{stage_pacing}"
        )

    # 转换动态状态为文本（PAD、busy、momentum、relationship）
    state_text_dict = convert_state_to_context_text(state)

    # 组合后的"你现在的状态"块
    current_state_block = f"""## 你现在的状态
- 身体/心理感受：{state_text_dict["pad_state"]}
- 忙碌度/注意力：{state_text_dict["busy_text"]}
- 聊天意愿：{state_text_dict["momentum_text"]}
- 对关系的感受：{state_text_dict["relationship_narrative"]}
- 关系阶段：{state_text_dict["stage_narrative"]}"""

    return {
        "bot_name": bot_name,
        "user_name": user_name,
        "user_pronoun": user_pronoun,
        "latest_user_text": latest_user_text,
        "recent_dialogue": recent_dialogue_context,
        "persona": safe_text(str(state.get("bot_persona") or ""))[:600],
        "memories": mem_block,
        "summary": summary_block,
        "detection": det_block,
        "current_state": current_state_block,
        "time_block_text": time_block_text,
        "session_summary_block": session_summary_block,
        "rel_trend_block": rel_trend_block,
        "user_profile_summary": _build_user_profile_summary(state),
        "task_block": task_block,
        "topic_shift_hook": topic_shift_hook,
        "ext_knowledge_block": ext_knowledge_block,
    }


def create_inner_monologue_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """创建内心独白节点。"""

    @trace_if_enabled(
        name="Inner Monologue",
        run_type="chain",
        tags=["node", "inner_monologue", "core"],
        metadata={"state_outputs": ["inner_monologue"]},
    )
    def inner_monologue_node(state: AgentState) -> dict:
        monologue = _generate_monologue(state, llm_invoker)
        monologue = (monologue or "按常理接话即可。").strip()

        # 防溢出截断，但不硬限
        if len(monologue) > INNER_MONOLOGUE_MAX_CHARS:
            monologue = monologue[:INNER_MONOLOGUE_MAX_CHARS]

        logger.info("[InnerMonologue] generated, length=%d chars", len(monologue))
        return {"inner_monologue": monologue}

    return inner_monologue_node


def _generate_monologue(state: AgentState, llm_invoker: Any) -> str:
    """生成纯内心独白（不再选move、不再选profile_keys）。"""
    if llm_invoker is None:
        return "按常理接话即可。"

    try:
        ctx = _gather_context_for_monologue(state)
        bot_name = ctx["bot_name"]
        user_name = ctx["user_name"]

        user_pronoun = ctx["user_pronoun"]
        task_block = ctx.get("task_block", "")
        topic_shift_hook = ctx.get("topic_shift_hook", "")

        # 新式提示词：重点是"被触发的感受"而不是"下一步怎么办"
        ext_knowledge_block = ctx.get("ext_knowledge_block", "")
        time_block_text = ctx.get("time_block_text", "")
        summary_block = ctx.get("summary", "")
        session_summary_block = ctx.get("session_summary_block", "")
        rel_trend_block = ctx.get("rel_trend_block", "")

        time_block = f"## 时间\n{time_block_text}" if time_block_text else ""

        system_prompt = f"""你是 {bot_name}。

## 你这个人
{ctx['persona']}

{time_block}

{ctx['current_state']}

{rel_trend_block + chr(10) if rel_trend_block else ""}{ctx['detection']}

{session_summary_block + chr(10) if session_summary_block else ""}{summary_block + chr(10) if summary_block else ""}{ctx['memories']}

{ext_knowledge_block + chr(10) if ext_knowledge_block else ""}## 关于 {user_name}
{ctx['user_profile_summary']}

{task_block + chr(10) if task_block else ""}## 最近的对话
{ctx['recent_dialogue']}

---

## 你现在的任务
写出你（{bot_name}）在这一刻的内心独白（**600-1200 字**，意识流，不列条目）。

不是分析{user_pronoun}说了什么，不是规划你该怎么回。
就是你看到{user_pronoun}的消息时，脑子里涌上来的真实想法。

你会想到什么？
- 这句话触到了你什么，还是根本没触及？
- 你有没有想起{user_pronoun}之前做过的什么事？
- 你对{user_pronoun}这句话的真实反应是什么？
- 你猜{user_pronoun}为什么这么说，背后是什么？
- 你想靠近{user_pronoun}还是想推开{user_pronoun}？
- 有没有什么小欲望在蠢蠢欲动？
- 你们现在聊的方向，是让你越聊越有劲，还是有点转不出去的感觉？
{topic_shift_hook}

允许你：
- 跑题、反复纠缠在一个感受上
- 心里想的和将来要说的不一样
- 有矛盾、有防御、有不敢说出口的欲望
- 一会儿理性一会儿纯情绪

不要：
- 分析{user_pronoun}的意图（除非你真的在猜测）
- 规划你的回复方式或措辞
- 列清单或分条目
- 输出 JSON 或其他结构化格式
- 用诗意、文学化的语言写独白——你是普通人在心里想事情，不是在写散文

就是意识流。你的思绪。此刻。
"""

        user_prompt = f"""{user_pronoun}（{user_name}）刚刚对你说：

"{ctx['latest_user_text']}"

---

写出你现在的内心独白。"""

        # 调用 LLM
        try:
            if hasattr(llm_invoker, "with_structured_output"):
                try:
                    structured = llm_invoker.with_structured_output(InnerMonologueOutput)
                    obj = structured.invoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ])
                    result = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
                    return str(result.get("monologue") or "").strip()
                except Exception as e:
                    logger.warning("[InnerMonologue] structured_output failed, fallback to plain text: %s", e)
        except Exception:
            pass

        # Fallback: 纯文本调用
        msg = llm_invoker.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        content = (getattr(msg, "content", "") or str(msg)).strip()
        return content

    except Exception as e:
        logger.exception("[InnerMonologue] failed to generate monologue: %s", e)
        return "按常理接话即可。"

"""内心独白节点：生成 intuition_thought、关系滤镜(rel_str)、inner_monologue。放在 detection 与 reasoner 之间。"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from utils.tracing import trace_if_enabled
from utils.detailed_logging import log_prompt_and_params, log_llm_response
from utils.llm_json import parse_json_from_llm
from utils.prompt_helpers import format_relationship_for_llm, format_stage_for_llm

from app.state import AgentState
from app.lats.prompt_utils import sanitize_memory_text, filter_retrieved_memories

# 写作禁令：保证内心独白是「感受/见闻」而非「策略」
INNER_MONOLOGUE_FORBIDDEN = (
    "禁止出现：我决定 / 我打算 / 我应该 / 必须 / 接下来我会 / 我将 / 策略 / 计划 / 先…再…；"
    "以及任何显式指令句（「要…」「请…」「务必…」）。"
)
INNER_MONOLOGUE_ALLOWED = (
    "允许且鼓励：见闻（对方语气、用词、节奏、是否敷衍/挑衅）；"
    "感受（烦、冷、厌、无聊、被冒犯、想压一头、懒得解释）；"
    "身体/状态（忙、疲惫、注意力不够、耐心低）；"
    "关系滤镜（对 TA 的熟悉/信任/尊重在这一刻的主观体验，是「感觉」不是更新长期值）。"
)
INNER_MONOLOGUE_FORMAT = (
    "4–8 句，第一人称现在时，像镜头/内心OS，但不下指令。"
    "最后一句可以是「情绪立场」，不是「行动计划」。"
)


def create_inner_monologue_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """创建内心独白节点：用 state 信息调用 LLM 生成纯感受/见闻式内心独白，写入 inner_monologue。"""

    @trace_if_enabled(
        name="Inner Monologue",
        run_type="chain",
        tags=["node", "inner_monologue", "perception"],
        metadata={"state_outputs": ["inner_monologue", "intuition_thought", "relationship_filter"]},
    )
    def _to_lc_message(m: BaseMessage) -> BaseMessage:
        """确保是 LangChain HumanMessage/AIMessage，便于 LLM 区分说话人。"""
        if isinstance(m, (HumanMessage, AIMessage)):
            return m
        content = getattr(m, "content", str(m))
        t = getattr(m, "type", "") or ""
        if "human" in t.lower() or "user" in t.lower():
            return HumanMessage(content=content)
        return AIMessage(content=content)

    def inner_monologue_node(state: AgentState) -> dict:
        bot_basic = state.get("bot_basic_info") or {}
        bot_big_five = state.get("bot_big_five") or {}
        bot_persona = state.get("bot_persona") or {}
        user_basic = state.get("user_basic_info") or {}
        user_inferred = state.get("user_inferred_profile") or {}
        conversation_summary = sanitize_memory_text((state.get("conversation_summary") or "").strip())
        retrieved_memories = filter_retrieved_memories(state.get("retrieved_memories") or [])
        mood_state = state.get("mood_state") or {}
        relationship_state = state.get("relationship_state") or {}
        chat_buffer: List[BaseMessage] = list(state.get("chat_buffer") or state.get("messages", [])[-20:])

        bot_name = bot_basic.get("name", "我") if isinstance(bot_basic, dict) else getattr(bot_basic, "name", "我")
        mood_str = ""
        if isinstance(mood_state, dict):
            mood_str = f"愉悦:{mood_state.get('pleasure', 0):.1f}, 激动:{mood_state.get('arousal', 0):.1f}, 强势感:{mood_state.get('dominance', 0):.1f}"
        # 6 维关系：加载详细数值说明并格式化为 LLM 可读（含区间语义）
        rel_for_llm = format_relationship_for_llm(relationship_state) if isinstance(relationship_state, dict) else "（未提供）"
        # Knapp 阶段描述
        current_stage = state.get("current_stage", "initiating")
        stage_desc = format_stage_for_llm(current_stage) if isinstance(current_stage, str) else "（未提供）"

        # Bot 全部信息（你是具体那个人，不是"系统"）
        bot_info_lines = [f"- 你是 **{bot_name}**（以下是你的人设与状态，用第一人称感受/见闻，不输出策略或决策）："]
        if isinstance(bot_basic, dict):
            bot_info_lines.append("- **身份：** " + json.dumps(bot_basic, ensure_ascii=False))
        if isinstance(bot_big_five, dict):
            bot_info_lines.append("- **大五人格：** " + json.dumps(bot_big_five, ensure_ascii=False))
        if isinstance(bot_persona, dict):
            bot_info_lines.append("- **人设（attributes/collections/lore）：** " + json.dumps(bot_persona, ensure_ascii=False))
        bot_info_str = "\n".join(bot_info_lines)

        # User 全部信息
        user_info_lines = ["- **当前对话用户（你正在和 TA 对话）：**"]
        if isinstance(user_basic, dict):
            user_info_lines.append("  - 显性信息: " + json.dumps(user_basic, ensure_ascii=False))
        if isinstance(user_inferred, dict):
            user_info_lines.append("  - 推断侧写: " + json.dumps(user_inferred, ensure_ascii=False))
        user_info_str = "\n".join(user_info_lines)

        summary_str = conversation_summary if conversation_summary else "（无近期摘要）"
        memories_str = "\n".join([str(x) for x in retrieved_memories]) if retrieved_memories else "（无相关记忆）"

        system_content = f"""# 你是谁
{bot_info_str}

# 对方是谁
{user_info_str}

# 近期记忆摘要
{summary_str}

# 检索到的相关记忆
{memories_str}

# 当前情绪与关系数值（仅作参考，用于写 relationship_filter）
- 情绪（PAD）：{mood_str or "（未提供）"}
- 关系维度（0–1，含区间语义说明，用于理解当前关系阶段与边界）：
{rel_for_llm}

# 当前关系阶段（Knapp 阶段）
{stage_desc}

# 本步你要输出三样（全部为「感受/见闻」，禁止策略与决策）
1. **intuition_thought**：对用户最后一句话的简短直觉（一两句）。
2. **relationship_filter**：关系滤镜。此刻你对 TA 的主观体验（一两句），是「感觉」不是更新数据。
3. **inner_monologue**：内心独白，4–8 句，第一人称现在时，像镜头/内心OS。

# 写作禁令（强约束）
{INNER_MONOLOGUE_FORBIDDEN}

# 允许出现的内容
{INNER_MONOLOGUE_ALLOWED}

# inner_monologue 格式
{INNER_MONOLOGUE_FORMAT}

# 输出格式（仅输出一个 JSON，不要其他文字）
{{
  "intuition_thought": "一两句直觉",
  "relationship_filter": "一两句此刻对 TA 的主观关系感受",
  "inner_monologue": "4–8 句内心独白，第一人称现在时"
}}"""

        # 近期对话：完整 Human/AI 消息列表传给 LLM，不拼成一段字符串
        conv_messages = [_to_lc_message(m) for m in chat_buffer]
        task_message = HumanMessage(
            content="请根据上面这段对话，以你的身份输出一个 JSON：包含 intuition_thought、relationship_filter、inner_monologue。只输出 JSON，不要其他文字。"
        )
        messages_to_invoke: List[BaseMessage] = [SystemMessage(content=system_content), *conv_messages, task_message]

        log_prompt_and_params(
            "Inner Monologue",
            system_prompt=system_content[:1500],
            user_prompt=f"[对话条数: {len(conv_messages)}] + 输出 JSON 指令",
            params={"bot_name": bot_name, "conv_len": len(conv_messages)},
        )
        try:
            msg = llm_invoker.invoke(messages_to_invoke)
            content = (getattr(msg, "content", "") or str(msg)).strip()
            result = parse_json_from_llm(content)
            if isinstance(result, dict):
                intuition_thought = (result.get("intuition_thought") or "").strip()[:500]
                relationship_filter = (result.get("relationship_filter") or "").strip()[:500]
                inner_monologue = (result.get("inner_monologue") or "").strip()[:2000]
            else:
                intuition_thought = ""
                relationship_filter = ""
                inner_monologue = "（此刻没什么特别想法，按常理接话即可。）"
            log_llm_response("Inner Monologue", msg, parsed_result={"intuition_thought": intuition_thought[:80], "relationship_filter": relationship_filter[:80]})
            print("[Inner Monologue] 已生成 intuition_thought、relationship_filter、inner_monologue。")
        except Exception as e:
            print(f"[Inner Monologue] 生成异常: {e}，使用占位。")
            intuition_thought = ""
            relationship_filter = ""
            inner_monologue = "（此刻没什么特别想法，按常理接话即可。）"

        return {
            "inner_monologue": inner_monologue,
            "intuition_thought": intuition_thought,
            "relationship_filter": relationship_filter,
        }
    return inner_monologue_node

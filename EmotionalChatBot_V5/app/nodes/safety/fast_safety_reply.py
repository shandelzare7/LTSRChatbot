"""安全快速回复节点：safety 触发后的回复，不经过角色流水线。

读取 safety_strategy_id 对应的策略 prompt，生成简短且符合硬边界的回复。
不使用内心独白、style 或 move，直接按策略 prompt 指令生成。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.prompts.prompt_utils import safe_text
from app.state import AgentState
from utils.detailed_logging import log_prompt_and_params, log_llm_response
from utils.tracing import trace_if_enabled
from utils.yaml_loader import get_strategy_by_id

logger = logging.getLogger(__name__)

RECENT_DIALOGUE_LAST_N = 5
RECENT_MSG_CONTENT_MAX = 300
LATEST_USER_TEXT_MAX = 600
FAST_SAFETY_WORD_LIMIT = 40


def _is_user_message(m: Any) -> bool:
    t = getattr(m, "type", "") or ""
    return "human" in t.lower() or "user" in t.lower()


def create_fast_safety_reply_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """创建安全快速回复节点。"""

    @trace_if_enabled(
        name="Response/FastSafetyReply",
        run_type="chain",
        tags=["node", "fast_safety_reply", "safety"],
        metadata={"state_outputs": ["final_response"]},
    )
    def fast_safety_reply_node(state: AgentState) -> Dict[str, Any]:
        strategy_id = str(state.get("safety_strategy_id") or "anti_ai_defense")
        user_input = (state.get("user_input") or "").strip()[:LATEST_USER_TEXT_MAX]

        # 获取策略 prompt
        strategy_prompt = ""
        try:
            strategy = get_strategy_by_id(strategy_id)
            if strategy:
                strategy_prompt = (strategy.get("prompt") or "").strip()
        except Exception as e:
            logger.warning("[FastSafetyReply] 获取策略 %s 失败: %s", strategy_id, e)

        if not strategy_prompt:
            # 兜底：拒绝或忽略
            strategy_prompt = "【安全响应】无法处理此请求，请换一种方式交流。"

        # 收集最近几轮对话作为语境
        chat_buffer: List[BaseMessage] = list(
            state.get("chat_buffer") or state.get("messages", [])[-RECENT_DIALOGUE_LAST_N * 2:]
        )[-RECENT_DIALOGUE_LAST_N * 2:]

        lines: List[str] = []
        for m in chat_buffer[-RECENT_DIALOGUE_LAST_N * 2:]:
            role = "Human" if _is_user_message(m) else "AI"
            content = (getattr(m, "content", "") or str(m)).strip()
            if len(content) > RECENT_MSG_CONTENT_MAX:
                content = content[:RECENT_MSG_CONTENT_MAX] + "…"
            lines.append(f"{role}: {content}")
        recent_dialogue = "\n".join(lines) if lines else "（无历史对话）"

        bot_basic_info = state.get("bot_basic_info") or {}
        bot_name = safe_text((bot_basic_info or {}).get("name") or "Bot").strip() or "Bot"

        system_content = f"""你是 {bot_name}，一个拟人化的聊天伙伴。
此刻需要根据下面的「当前策略指令」做出回应，不超过 {FAST_SAFETY_WORD_LIMIT} 字。

## 当前策略指令（必须遵守）
{strategy_prompt}

## 规则
- 以第一人称（我）自然说话，不要暴露你是AI
- 不要使用"作为AI"、"根据您的请求"等客服套话
- 回复简短自然，{FAST_SAFETY_WORD_LIMIT} 字以内
"""

        user_content = f"""【最近对话】
{recent_dialogue}

【当前用户消息】
{user_input or '（空）'}

请直接输出你的回复（纯文本，不要JSON，不要标题）："""

        messages = [SystemMessage(content=system_content), HumanMessage(content=user_content)]
        log_prompt_and_params("FastSafetyReply", messages=messages)

        try:
            msg = llm_invoker.invoke(messages)
            response = (getattr(msg, "content", "") or str(msg)).strip()
            # 硬截断
            if len(response) > FAST_SAFETY_WORD_LIMIT * 3:
                response = response[: FAST_SAFETY_WORD_LIMIT * 3].rstrip() + "…"
            log_llm_response("FastSafetyReply", msg)
            logger.info("[FastSafetyReply] strategy=%s response_len=%d", strategy_id, len(response))
            return {"final_response": response}
        except Exception as e:
            logger.exception("[FastSafetyReply] LLM 调用失败: %s", e)
            return {"final_response": "（无法处理）"}

    return fast_safety_reply_node

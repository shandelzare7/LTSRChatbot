"""Judge 节点：从并行生成的所有候选中，选出最符合内心独白的那条。

评判标准：与独白的情绪/态度/意愿最匹配，而非「最自然」或「最长」。
输入：inner_monologue（独白全文）+ generation_candidates（所有路候选）
输出：final_response（中选文本）+ judge_result（评分详情）
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.state import AgentState
from src.schemas import JudgeOutput
from utils.detailed_logging import log_prompt_and_params, log_llm_response
from utils.tracing import trace_if_enabled

logger = logging.getLogger(__name__)

JUDGE_RECENT_DIALOGUE_N = 5
JUDGE_MAX_CANDIDATE_CHARS = 200  # 每条候选展示的最大字符数


def _is_user_message(m: Any) -> bool:
    t = getattr(m, "type", "") or ""
    return "human" in t.lower() or "user" in t.lower()


def _build_dialogue_snippet(state: AgentState) -> str:
    """取最近 N 轮对话（只用于 Judge 语境，不需要完整历史）。"""
    chat_buffer: List[Any] = list(
        state.get("chat_buffer") or state.get("messages", [])[-JUDGE_RECENT_DIALOGUE_N * 2:]
    )[-JUDGE_RECENT_DIALOGUE_N * 2:]
    lines: List[str] = []
    for m in chat_buffer:
        role = "Human" if _is_user_message(m) else "AI"
        content = (getattr(m, "content", "") or str(m)).strip()[:200]
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "（无历史对话）"


def _format_candidates(candidates: List[Dict[str, Any]]) -> str:
    """将候选列表格式化为编号文本块。"""
    lines: List[str] = []
    for i, c in enumerate(candidates):
        text = (c.get("text") or "").strip()
        if len(text) > JUDGE_MAX_CANDIDATE_CHARS:
            text = text[:JUDGE_MAX_CANDIDATE_CHARS] + "…"
        route = c.get("route", "?")
        lines.append(f"[{i}] ({route}) {text}")
    return "\n".join(lines) if lines else "（无候选）"


def create_judge_node(llm_judge: Any) -> Callable[[AgentState], Dict[str, Any]]:
    """创建 Judge 节点。"""

    @trace_if_enabled(
        name="Response/Judge",
        run_type="chain",
        tags=["node", "judge"],
        metadata={"state_outputs": ["final_response", "judge_result"]},
    )
    def judge_node(state: AgentState) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = list(state.get("generation_candidates") or [])
        monologue: str = (state.get("inner_monologue") or "按常理接话即可。").strip()
        user_input: str = (state.get("user_input") or "").strip()

        # 过滤掉空文本候选
        valid_candidates = [c for c in candidates if (c.get("text") or "").strip()]
        if not valid_candidates:
            logger.warning("[Judge] 无有效候选，返回空回复")
            return {"final_response": "", "judge_result": {"winner_index": -1, "justification": "无有效候选"}}

        # 只有 1 个候选时直接返回，无需 LLM
        if len(valid_candidates) == 1:
            logger.info("[Judge] 只有 1 个候选，直接返回")
            return {
                "final_response": valid_candidates[0]["text"].strip(),
                "judge_result": {"winner_index": 0, "justification": "唯一候选"},
            }

        # 日志：所有候选全文（评审前展示，不截断）
        print("[Judge] ===== 输入候选全文 =====")
        for i, c in enumerate(valid_candidates):
            text = (c.get("text") or "").strip()
            route = c.get("route", "?")
            print(f"  [{i}] ({route}) {text}")
        print("[Judge] ===========================")

        dialogue_snippet = _build_dialogue_snippet(state)
        candidates_text = _format_candidates(valid_candidates)
        n = len(valid_candidates)

        system_content = f"""你是回复质量评审官。你的任务是从 {n} 条候选回复中，选出最符合角色当前内心独白的那条。

评判标准：
1. 情绪/态度与独白一致：独白里透露的情绪（如心疼、烦躁、期待）和对用户的态度，候选回复是否如实体现？
2. 不是最"正确"或最"自然"的回复，而是最像这个角色"此刻"会说的话。

注意：
- 不要选最长的
- 不要选最礼貌的
- 要选最「人味」、最贴近独白心境的
- 输出 winner_index（候选列表中的下标 0..{n-1}）和简短 justification
"""

        user_content = f"""## 当前用户消息
{user_input or '（空）'}

## 最近对话（参考语境）
{dialogue_snippet}

## 角色内心独白（评判核心依据）
{monologue}

## 候选回复列表（格式：[序号] (路由标签) 文本）
{candidates_text}

请选出最符合独白心境的那条，输出 winner_index 和 justification："""

        messages = [SystemMessage(content=system_content), HumanMessage(content=user_content)]
        log_prompt_and_params("Judge", messages=messages)

        winner_index = 0
        justification = ""

        try:
            result = None
            if hasattr(llm_judge, "with_structured_output"):
                try:
                    structured = llm_judge.with_structured_output(JudgeOutput)
                    result = structured.invoke(messages)
                except Exception as e:
                    logger.warning("[Judge] structured_output failed: %s，回退 fallback", e)
                    result = None

            if result is None:
                # fallback：直接 invoke，解析 JSON
                from utils.llm_json import parse_json_from_llm
                msg = llm_judge.invoke(messages)
                raw = (getattr(msg, "content", "") or str(msg)).strip()
                parsed = parse_json_from_llm(raw)
                if isinstance(parsed, dict):
                    result = parsed

            if result is not None:
                if hasattr(result, "model_dump"):
                    result = result.model_dump()
                elif hasattr(result, "dict"):
                    result = result.dict()
                if isinstance(result, dict):
                    idx = int(result.get("winner_index", 0))
                    winner_index = max(0, min(idx, n - 1))
                    justification = str(result.get("justification", ""))

        except Exception as e:
            logger.exception("[Judge] 评判异常，使用默认第 0 条: %s", e)

        winner = valid_candidates[winner_index]
        final_text = winner.get("text", "").strip()

        print(f"[Judge] ===== 评审结果 =====")
        print(f"  winner_index : {winner_index}")
        print(f"  route        : {winner.get('route', '?')}")
        print(f"  justification: {justification}")
        print(f"  winner_text  : {final_text}")
        print(f"[Judge] ====================")

        return {
            "final_response": final_text,
            "judge_result": {"winner_index": winner_index, "justification": justification},
        }

    return judge_node

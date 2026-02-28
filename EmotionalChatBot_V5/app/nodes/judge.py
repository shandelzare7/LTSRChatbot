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


def _extract_char_ngrams(text: str, n: int = 3) -> set:
    """提取字符级 n-gram（适合中文短语重复检测）。"""
    cleaned = text.replace(" ", "")
    return {cleaned[i:i+n] for i in range(len(cleaned) - n + 1)}


def _compute_repetition_ratio(candidate_text: str, recent_bot_texts: List[str], n: int = 3) -> float:
    """计算候选文本与近期 bot 发言的字符 n-gram 重叠率（0-1）。"""
    if not recent_bot_texts or not candidate_text:
        return 0.0
    candidate_ngrams = _extract_char_ngrams(candidate_text, n)
    if not candidate_ngrams:
        return 0.0
    recent_ngrams: set = set()
    for t in recent_bot_texts:
        recent_ngrams |= _extract_char_ngrams(t, n)
    return len(candidate_ngrams & recent_ngrams) / len(candidate_ngrams)


def _format_candidates(candidates: List[Dict[str, Any]]) -> str:
    """将候选列表格式化为编号文本块。"""
    lines: List[str] = []
    for i, c in enumerate(candidates):
        text = (c.get("text") or "").strip()
        if len(text) > JUDGE_MAX_CANDIDATE_CHARS:
            text = text[:JUDGE_MAX_CANDIDATE_CHARS] + "…"
        route = c.get("route", "?")
        lines.append(f"[{i}] {text}")
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
            msg = "[Judge] 无有效候选，返回空回复"
            logger.warning(msg)
            print(msg)  # 写入会话 log（web 轮次里 stdout 被重定向到 web_chat_log）
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

        # 外部素材摘要（一行，让 judge 知道哪些话题是"合理来源"）
        _bot_recent = list(state.get("bot_recent_activities") or [])
        _topics = list(state.get("daily_topics") or [])
        _ctx_items = [t[:30] for t in (_bot_recent[:3] + _topics[:2]) if t]
        external_ctx_line = "、".join(_ctx_items) if _ctx_items else ""

        # 重复短语检测：提取近期 bot 发言 + 最近一条对方（Human）发言
        # 原因：bot-to-bot 场景中，"Human" 消息就是对方 bot 刚说的话，
        # 需要防止当前 bot 原样复述对方的词句
        _chat_buf = list(state.get("chat_buffer") or state.get("messages", []))
        _recent_bot_texts = [
            (getattr(m, "content", "") or str(m)).strip()
            for m in _chat_buf[-8:]
            if not _is_user_message(m)
        ][-4:]  # 最多看最近 2 轮 bot 自己的发言
        # 加入最近 1-2 条 Human（对方）消息，防止直接复述对方
        _recent_human_texts = [
            (getattr(m, "content", "") or str(m)).strip()
            for m in _chat_buf[-4:]
            if _is_user_message(m)
        ][-2:]
        _recent_bot_texts = _recent_bot_texts + _recent_human_texts
        _repetition_warnings: List[str] = []
        for _i, _c in enumerate(valid_candidates):
            _ratio = _compute_repetition_ratio(_c.get("text", ""), _recent_bot_texts)
            if _ratio > 0.45:
                _repetition_warnings.append(f"[{_i}] 重复率 {_ratio:.0%}")

        _rep_block = (
            "\n\n⚠️ 重复短语警告（以下候选与近期发言存在较高短语重叠，若其他维度相近请优先选择更新颖的候选）：\n"
            + "\n".join(_repetition_warnings)
        ) if _repetition_warnings else ""

        system_content = f"""你是有常识和丰富经验的语言学家，现担任评审官。你的任务是从 {n} 条候选回复中，选出最符合当前情景和上下文的那条。

评判标准（优先级从高到低）：
1. **情景契合度**：候选回复是否与用户刚说的话自然衔接？是否合理回应了对方的内容和当前对话节奏？
2. **内容新鲜度**：候选回复是否引入了新的信息、角度或感受？避免选出复述或过度呼应刚刚已说过词句的回复。
3. **情绪基调吻合**：候选回复的基调是否与角色的内心独白（情绪/态度/意愿）大体吻合？

核心原则：**要「人味」，不要「写作感」**
- 人味 = 像真人社交软件里会打出来的话：短、口语、不刻意漂亮、有时不完整也没关系。
- 写作感 = 像写文章/作文：比喻堆叠、排比、金句、散文化抒情、句子过长或过于工整。
- **宁可选短而口语、像随口说的，也不要选「写得好」但像散文/金句的。**
- 不要选最长的；**不要因为某条更有分析感、解释性、深度或文采就选它**——分析腔、文采不等于情景契合，且违反「人味」。
- 若某条候选出现明显违规（诗意表达、比喻堆叠、排比句、散文化、超长句），即使其他维度尚可，也应排除，优先选更「像聊天」的那条。
- **如果某条回复自然引入了角色的日常话题或生活动态（见下方"可用素材"），且整体情绪基调与独白一致，这是正常聊天行为——可正向评价；但若是用来回避用户的核心问题，则不加分。**
- 输出 winner_index（候选列表中的下标 0..{n-1}）和简短 justification
{_rep_block}"""

        ctx_line = f"\n## 可用日常素材（角色可能引入，供参考）\n{external_ctx_line}\n" if external_ctx_line else ""
        user_content = f"""## 当前用户消息
{user_input or '（空）'}

## 最近对话（参考语境）
{dialogue_snippet}
{ctx_line}
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

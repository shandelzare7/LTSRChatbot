"""Monologue Extraction 节点：从独白文本中提取结构化信号。

这个节点在 inner_monologue 之后，基于生成好的独白文本，
用一次 LLM 调用提取：
- selected_move_ids：2-4 个相关的 content move
- selected_profile_keys：0-5 个相关的用户画像键
- emotion_tag：主要情绪标签
- momentum_delta：对 momentum 的调整建议（-0.15 到 +0.15）

这样将多任务分离，使独白生成纯粹，信号提取专门。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from utils.tracing import trace_if_enabled
from src.schemas import MonologueExtractOutput
from app.state import AgentState

logger = logging.getLogger(__name__)


def create_monologue_extraction_node(llm_extract: Any) -> Callable[[AgentState], Dict[str, Any]]:
    """创建从独白提取结构化信号的节点。"""

    @trace_if_enabled(
        name="Monologue Extraction",
        run_type="chain",
        tags=["node", "extraction"],
        metadata={"state_outputs": ["selected_content_move_ids", "selected_profile_keys", "monologue_emotion", "monologue_momentum_delta"]},
    )
    def extract_node(state: AgentState) -> Dict[str, Any]:
        """从独白中提取结构化信号。"""
        monologue = (state.get("inner_monologue") or "").strip()
        content_moves = state.get("content_moves") or {}
        user_profile_keys = list((state.get("user_inferred_profile") or {}).keys())

        if not monologue:
            logger.warning("[MonologueExtraction] 无独白文本，返回默认值")
            return {
                "selected_content_move_ids": [1],
                "selected_profile_keys": [],
                "monologue_emotion": "无感",
                "monologue_momentum_delta": 0.0,
            }

        # 构建 content move 列表，用于 LLM 选择
        move_options = []
        for move_id, move_info in content_moves.items():
            if isinstance(move_info, dict):
                name = move_info.get("name", "")
                operation = move_info.get("content_operation", "")
                move_options.append(f"[{move_id}] {name}: {operation}")
            else:
                move_options.append(f"[{move_id}] {move_info}")

        move_list_text = "\n".join(move_options) if move_options else "（无可用move）"

        # 构建提示词
        system_prompt = """你是一个结构化信号提取器。

基于角色的内心独白文本，你需要提取以下信号：

1. selected_move_ids：2-4 个最适合当前独白的 content move id
   （move 是不同的对话内容操作类型，比如追问细节、联想类比、自我暴露等）

2. selected_profile_keys：0-5 个被激活的用户画像键
   （哪些关于用户的已知信息在这一刻被触发/关联到了？）

3. primary_emotion：当前的主要情绪标签
   （如：期待、烦躁、心软、警惕、无聊、开心、委屈）

4. momentum_delta：对对话参与意愿的修正
   （-0.15 ~ +0.15 之间，正数表示想聊，负数表示想结束）

评判逻辑：
- 从独白中读出角色此刻想用什么方式表达（对应 move）
- 读出独白中显露的情感信号
- 估计这个反应是增强还是削弱了继续对话的意愿
- 找出被勾起的关于用户的记忆或认知（对应 profile keys）
"""

        user_prompt = f"""## 角色的内心独白
{monologue}

## 可用的 Content Move（对话内容操作类型）
{move_list_text}

## 可用的用户画像键
{', '.join(user_profile_keys) if user_profile_keys else '（无）'}

---

基于上面的独白，请提取结构化信号。
返回 JSON：
{{
    "selected_move_ids": [int, ...],  // 2-4 个 move id
    "selected_profile_keys": [str, ...],  // 0-5 个 profile key
    "primary_emotion": "string",  // 单个情绪标签
    "momentum_delta": float  // -0.15 to +0.15
}}
"""

        result = _call_llm_for_extraction(llm_extract, system_prompt, user_prompt)

        return {
            "selected_content_move_ids": result.get("selected_move_ids", [1]),
            "selected_profile_keys": result.get("selected_profile_keys", []),
            "monologue_emotion": result.get("primary_emotion", ""),
            "monologue_momentum_delta": result.get("momentum_delta", 0.0),
        }

    return extract_node


def _call_llm_for_extraction(llm_extract: Any, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """调用 LLM 进行结构化提取。"""
    try:
        # 优先用 structured output
        if hasattr(llm_extract, "with_structured_output"):
            try:
                structured = llm_extract.with_structured_output(MonologueExtractOutput)
                obj = structured.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ])
                if hasattr(obj, "model_dump"):
                    return obj.model_dump()
                elif hasattr(obj, "dict"):
                    return obj.dict()
            except Exception as e:
                logger.warning("[MonologueExtraction] structured_output failed: %s, fallback to plain text", e)

        # Fallback: 纯文本 + JSON 解析
        msg = llm_extract.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        content = (getattr(msg, "content", "") or str(msg)).strip()

        # 尝试解析 JSON
        if content.startswith("{"):
            import json
            try:
                return json.loads(content)
            except Exception:
                pass

        # 如果都失败，返回默认值
        logger.warning("[MonologueExtraction] 无法从 LLM 提取结构化信号")
        return {
            "selected_move_ids": [1],
            "selected_profile_keys": [],
            "primary_emotion": "",
            "momentum_delta": 0.0,
        }

    except Exception as e:
        logger.exception("[MonologueExtraction] 提取失败: %s", e)
        return {
            "selected_move_ids": [1],
            "selected_profile_keys": [],
            "primary_emotion": "",
            "momentum_delta": 0.0,
        }

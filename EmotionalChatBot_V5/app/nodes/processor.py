"""
Relationship Processor (语义处理器)

职责：
- 读取用户输入与上下文，输出结构化的 `processor_output`
- 提供给后续流水线：Updater(资产) -> Evolver(情感分数) -> StageManager(阶段)

输出字段（写入 state['processor_output']）：
- topic_category: str
- spt_level: int (1-4)
- is_intellectually_deep: bool
- base_deltas: Dict[str,int]  # 6 维基础变化（-3..3）
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.state import AgentState
from src.schemas import RelationshipAnalysis
from src.prompts.relationship import build_analyzer_prompt
from utils.tracing import trace_if_enabled


REL_DIMS = ("closeness", "trust", "liking", "respect", "warmth", "power")


def _ensure_defaults(state: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(state)
    s.setdefault("user_input", "")
    s.setdefault("relationship_state", {"closeness": 0, "trust": 0, "liking": 0, "respect": 0, "warmth": 0, "power": 50})
    s.setdefault("mood_state", {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0, "busyness": 0.0})
    s.setdefault("current_stage", "initiating")
    return s


def _analysis_to_processor_output(analysis: RelationshipAnalysis) -> Dict[str, Any]:
    deltas = analysis.deltas.model_dump() if hasattr(analysis.deltas, "model_dump") else analysis.deltas.dict()
    base_deltas = {k: int(deltas.get(k, 0)) for k in REL_DIMS}
    return {
        "topic_category": str(getattr(analysis, "topic_category", "general") or "general"),
        "spt_level": int(getattr(analysis, "self_disclosure_depth_level", 1) or 1),
        "is_intellectually_deep": bool(getattr(analysis, "is_intellectually_deep", False)),
        "base_deltas": base_deltas,
        # 可选：给调试用
        "detected_signals": list(getattr(analysis, "detected_signals", []) or []),
        "thought_process": str(getattr(analysis, "thought_process", "") or ""),
    }


def create_processor_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    @trace_if_enabled(
        name="Relationship/Processor",
        run_type="chain",
        tags=["node", "relationship", "processor"],
        metadata={"state_outputs": ["processor_output"]},
    )
    def node(state: AgentState) -> dict:
        safe = _ensure_defaults(state)
        sys_prompt = build_analyzer_prompt(safe)
        user_msg = safe.get("user_input") or ""

        # LLM 输出：严格 JSON（复用 relationship prompt/schema）
        try:
            resp = llm_invoker.invoke(
                [SystemMessage(content=sys_prompt), HumanMessage(content=user_msg)]
            )
            raw = getattr(resp, "content", str(resp))
            data = json.loads(str(raw).strip())
        except Exception as e:
            data = {
                "thought_process": f"Fallback: parse error ({e}); assume neutral.",
                "detected_signals": [],
                "topic_category": "general",
                "self_disclosure_depth_level": 1,
                "is_intellectually_deep": False,
                "deltas": {k: 0 for k in REL_DIMS},
            }

        # Pydantic 校验
        try:
            analysis = RelationshipAnalysis.model_validate(data)  # pydantic v2
        except Exception:
            analysis = RelationshipAnalysis.parse_obj(data)  # type: ignore[attr-defined]

        return {"processor_output": _analysis_to_processor_output(analysis)}

    return node

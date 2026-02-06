"""
6维关系演化引擎 (Relationship Engine)

动静分离：
- 静态的“信号判断标准”放在 `config/relationship_signals.yaml`
- 动态的 State 由本模块处理

双层处理：
- Node 1 (Analyzer): LLM 接收完整上下文 + YAML 标准，输出 JSON Deltas
- Node 2 (Updater): Python 应用阻尼公式（边际收益递减 + 背叛惩罚）更新 relationship_state

同时保留一个末尾 Memory Writer（原 evolver 行为），供 graph 在 processor 之后记录回复文本。
"""

import json
from typing import TYPE_CHECKING, Any, Callable, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from app.state import AgentState
from src.schemas import RelationshipAnalysis
from src.prompts.relationship import build_analyzer_prompt
from utils.tracing import trace_if_enabled

if TYPE_CHECKING:
    from app.services.memory.base import MemoryBase


REL_DIMS = ("closeness", "trust", "liking", "respect", "warmth", "power")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _ensure_relationship_defaults(state: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(state)
    rel = dict(s.get("relationship_state") or {})
    rel.setdefault("closeness", 50.0)
    rel.setdefault("trust", 50.0)
    rel.setdefault("liking", 50.0)
    rel.setdefault("respect", 50.0)
    rel.setdefault("warmth", 50.0)
    rel.setdefault("power", 50.0)
    s["relationship_state"] = rel

    s.setdefault("mood_state", {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0, "busyness": 0.0})
    s.setdefault("current_stage", "experimenting")
    s.setdefault("user_input", s.get("user_input") or "")
    return s


def calculate_damped_delta(current_score: float, raw_delta: int) -> float:
    """
    阻尼公式：实现边际收益递减和背叛惩罚。
    - 正向：越高越难涨
    - 负向：高信任/高亲密被破坏时更痛（背叛惩罚）
    """
    try:
        cs = float(current_score)
    except Exception:
        cs = 50.0
    rd = int(raw_delta)

    if rd > 0:
        if cs >= 90:
            return rd * 0.1
        if cs >= 60:
            return rd * 0.5
        return rd * 1.0

    if rd < 0:
        if cs >= 80:
            return rd * 1.5
        return rd * 1.0

    return 0.0


# -------------------------------------------------------------------
# Node 1: Analyzer (LLM) —— 只分析，不改分
# -------------------------------------------------------------------

def create_relationship_analyzer_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    @trace_if_enabled(
        name="Relationship/Analyzer",
        run_type="chain",
        tags=["node", "relationship", "analyzer"],
        metadata={"state_outputs": ["latest_relationship_analysis", "relationship_deltas"]},
    )
    def node(state: AgentState) -> dict:
        safe = _ensure_relationship_defaults(state)
        sys_prompt = build_analyzer_prompt(safe)
        user_msg = safe.get("user_input") or ""

        # LLM 输出：严格 JSON
        try:
            resp = llm_invoker.invoke(
                [SystemMessage(content=sys_prompt), HumanMessage(content=user_msg)]
            )
            raw = getattr(resp, "content", str(resp))
            data = json.loads(str(raw).strip())
        except Exception as e:
            print(f"[Relationship Analyzer] parse error: {e}")
            data = {
                "thought_process": "Fallback: unable to parse model output; assume neutral.",
                "detected_signals": [],
                "deltas": {k: 0 for k in REL_DIMS},
            }

        # Pydantic 校验
        try:
            analysis = RelationshipAnalysis.model_validate(data)  # pydantic v2
        except Exception:
            # pydantic v1 兼容
            analysis = RelationshipAnalysis.parse_obj(data)  # type: ignore[attr-defined]

        analysis_dict = analysis.model_dump() if hasattr(analysis, "model_dump") else analysis.dict()
        deltas_dict = analysis.deltas.model_dump() if hasattr(analysis.deltas, "model_dump") else analysis.deltas.dict()

        return {
            "latest_relationship_analysis": analysis_dict,
            "relationship_deltas": deltas_dict,  # raw deltas（-3..3）
        }

    return node


# -------------------------------------------------------------------
# Node 2: Updater (Math) —— 应用阻尼公式，更新 relationship_state
# -------------------------------------------------------------------

def create_relationship_updater_node() -> Callable[[AgentState], dict]:
    @trace_if_enabled(
        name="Relationship/Updater",
        run_type="chain",
        tags=["node", "relationship", "updater"],
        metadata={"state_outputs": ["relationship_state", "relationship_deltas_applied"]},
    )
    def node(state: AgentState) -> dict:
        safe = _ensure_relationship_defaults(state)

        rel: Dict[str, float] = dict(safe.get("relationship_state") or {})
        raw_deltas = safe.get("relationship_deltas") or {}

        applied: Dict[str, float] = {}
        for dim in REL_DIMS:
            score = float(rel.get(dim, 50.0))
            raw_val = raw_deltas.get(dim, 0)
            try:
                raw_int = int(raw_val)
            except Exception:
                raw_int = 0

            if raw_int == 0:
                applied[dim] = 0.0
                continue

            real_change = float(calculate_damped_delta(score, raw_int))
            new_score = _clamp(score + real_change, 0.0, 100.0)
            rel[dim] = round(new_score, 2)
            applied[dim] = round(real_change, 3)

        return {
            "relationship_state": rel,
            "relationship_deltas_applied": applied,
        }

    return node


# -------------------------------------------------------------------
# Memory Writer（原 evolver 行为）：processor 之后记录最终回复片段
# -------------------------------------------------------------------

def create_memory_writer_node(memory_service: "MemoryBase") -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        user_id = state.get("user_id") or "default_user"
        segments = state.get("final_segments", [])
        final_text = " ".join(segments) if segments else state.get("draft_response", "")
        if final_text:
            memory_service.append_memory(
                user_id,
                f"Bot 回复: {final_text[:200]}",
                meta={"source": "memory_writer"},
            )
        return {}

    return node


# backward-compatible alias (旧 graph 若仍调用 create_evolver_node)
def create_evolver_node(memory_service: "MemoryBase") -> Callable[[AgentState], dict]:
    return create_memory_writer_node(memory_service)

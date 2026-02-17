"""【编排层】构建 LangGraph：节点与边（含并行思考与 Critic 循环）。"""
from __future__ import annotations

import asyncio
import inspect
import os
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Literal

from langgraph.graph import END, StateGraph

from app.core.engine import PsychoEngine
from app.core.mode_base import PsychoMode
from app.state import AgentState
from app.nodes.loader import create_loader_node

# 从 detection.py 文件直接导入（在 nodes/ 目录下，与 detection/ 文件夹同名）
import importlib.util
from pathlib import Path

_detection_file = Path(__file__).parent / "nodes" / "detection.py"
_spec = importlib.util.spec_from_file_location("detection_module", _detection_file)
_detection_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_detection_module)

create_detection_node = _detection_module.create_detection_node
from app.nodes.security_check import create_security_check_node
from app.nodes.inner_monologue import create_inner_monologue_node
from app.nodes.style import create_style_node
from app.nodes.task_planner import create_task_planner_node
from app.nodes.lats_search import create_lats_search_node
from app.nodes.processor import create_processor_node
from app.nodes.final_validator import create_final_validator_node
from app.nodes.evolver import create_evolver_node
from app.nodes.stage_manager import create_stage_manager_node
from app.nodes.memory_manager import create_memory_manager_node
from app.nodes.memory_writer import create_memory_writer_node
from app.nodes.security_response import create_security_response_node
from app.services.llm import get_llm, llm_stats_diff, llm_stats_snapshot, set_current_node, reset_current_node
from app.services.memory import MockMemory
from utils.yaml_loader import get_project_root, load_modes_from_dir


def _no_reply_handler(state: dict) -> dict:
    """word_budget=0 时短路：不调用 LATS，直接产出空回复并结束图。"""
    return {
        "final_response": "",
        "processor_plan": {"messages": []},
        "reply_plan": {"messages": []},
    }


def _after_security_handler(state: dict) -> dict:
    """安全检查通过后的汇合占位节点（no-op），用于 fan-out 并行分支。"""
    return {}


def _load_modes() -> list[PsychoMode]:
    root = get_project_root()
    modes_dir = root / "config" / "modes"
    raw = load_modes_from_dir(modes_dir)
    return [PsychoMode.model_validate(m) for m in raw]


def build_graph(
    *,
    llm: Any = None,
    llm_fast: Any = None,
    llm_judge: Any = None,
    memory_service: Any = None,
    modes: Optional[List[PsychoMode]] = None,
    entry_point: Literal["loader", "evolver"] = "loader",
    end_at: Literal["memory_writer", "final_validator"] = "memory_writer",
) -> Any:
    """
    构建 LangGraph。
    - llm / memory_service / modes 为 None 时使用默认（MockMemory、_load_modes、get_llm()）。
    """
    root = get_project_root()
    # Role-based LLM routing:
    # - main: planner / generation
    # - fast: detection / analyzer / memory write
    # - judge: LATS soft scorer
    llm = llm or get_llm(role="main")
    llm_fast = llm_fast or get_llm(role="fast")
    llm_judge = llm_judge or get_llm(role="judge")
    # Processor LLM (segmentation + delay). Default to gpt-4o-mini (priority tier on official OpenAI endpoint).
    processor_role = (os.getenv("LTSR_PROCESSOR_LLM_ROLE") or "fast").strip().lower() or "fast"
    processor_model = (os.getenv("LTSR_PROCESSOR_LLM_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"
    try:
        processor_temp = float((os.getenv("LTSR_PROCESSOR_LLM_TEMPERATURE") or "0.2").strip() or "0.2")
    except Exception:
        processor_temp = 0.2
    llm_processor = get_llm(role=processor_role, model=processor_model, temperature=processor_temp)
    memory_service = memory_service or MockMemory()
    modes = modes or _load_modes()
    engine = PsychoEngine(modes=modes, llm_invoker=llm)

    def _truthy(v: str | None) -> bool:
        return str(v or "").strip().lower() in {"1", "true", "yes", "y", "on"}

    enable_step_profile = _truthy(os.getenv("LTSR_PROFILE_STEPS"))
    enable_call_log = _truthy(os.getenv("LTSR_LLM_CALL_LOG"))

    def _wrap_node(name: str, fn: Any) -> Any:
        # If neither step profiling nor per-call logging is enabled, keep original function.
        if not enable_step_profile and not enable_call_log:
            return fn

        def _ensure_profile(state: Any) -> dict:
            try:
                prof = state.get("_profile") if isinstance(state, dict) else None
                if not isinstance(prof, dict):
                    prof = {}
                prof.setdefault("nodes", [])
                return prof
            except Exception:
                return {"nodes": []}

        async def _call_async(state: Any) -> Any:
            tok = set_current_node(name)
            prof = _ensure_profile(state)
            before = llm_stats_snapshot(node_name=name)
            t0 = time.perf_counter()
            try:
                out = await fn(state)
            finally:
                reset_current_node(tok)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            after = llm_stats_snapshot(node_name=name)
            prof["nodes"].append(
                {
                    "name": name,
                    "dt_ms": round(dt_ms, 2),
                    "llm_delta": llm_stats_diff(before, after),
                }
            )
            if isinstance(out, dict):
                out["_profile"] = prof
            return out

        def _call_sync(state: Any) -> Any:
            tok = set_current_node(name)
            prof = _ensure_profile(state)
            before = llm_stats_snapshot(node_name=name)
            t0 = time.perf_counter()
            try:
                out = fn(state)
            finally:
                reset_current_node(tok)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            after = llm_stats_snapshot(node_name=name)
            prof["nodes"].append(
                {
                    "name": name,
                    "dt_ms": round(dt_ms, 2),
                    "llm_delta": llm_stats_diff(before, after),
                }
            )
            if isinstance(out, dict):
                out["_profile"] = prof
            return out

        # Some nodes are sync, others are async (DB/memory). Preserve the type.
        if inspect.iscoroutinefunction(fn):
            return _call_async
        return _call_sync

    loader_node = create_loader_node(memory_service)
    security_check_node = create_security_check_node(llm_fast)
    detection_node = create_detection_node(llm)
    security_response_node = create_security_response_node(llm_fast)
    inner_monologue_node = create_inner_monologue_node(llm)
    style_node = create_style_node(llm)
    task_planner_node = create_task_planner_node(llm_fast)
    lats_node = create_lats_search_node(llm, llm_soft_scorer=llm_judge)
    processor_node = create_processor_node(llm_processor)
    final_validator_node = create_final_validator_node()
    evolver_node = create_evolver_node(llm_fast)
    stage_manager_node = create_stage_manager_node()
    memory_manager_node = create_memory_manager_node(llm_fast)
    memory_writer_node = create_memory_writer_node(memory_service)

    workflow = StateGraph(AgentState)

    if entry_point == "loader":
        # Full (or fast-return) graph: loader -> ... -> final_validator -> (optional tail) -> END
        workflow.add_node("loader", _wrap_node("loader", loader_node))
        workflow.add_node("security_check", _wrap_node("security_check", security_check_node))
        workflow.add_node("detection", _wrap_node("detection", detection_node))
        # Security short-circuit node (must be registered before edges reference it)
        workflow.add_node("security_response", _wrap_node("security_response", security_response_node))
        workflow.add_node("after_security", _wrap_node("after_security", _after_security_handler))
        workflow.add_node("inner_monologue", _wrap_node("inner_monologue", inner_monologue_node))
        workflow.add_node("style", _wrap_node("style", style_node))
        workflow.add_node("task_planner", _wrap_node("task_planner", task_planner_node))
        workflow.add_node("lats_search", _wrap_node("lats_search", lats_node))
        workflow.add_node("processor", _wrap_node("processor", processor_node))
        workflow.add_node("final_validator", _wrap_node("final_validator", final_validator_node))
        workflow.add_node("evolver", _wrap_node("evolver", evolver_node))
        workflow.add_node("stage_manager", _wrap_node("stage_manager", stage_manager_node))
        workflow.add_node("memory_manager", _wrap_node("memory_manager", memory_manager_node))
        workflow.add_node("memory_writer", _wrap_node("memory_writer", memory_writer_node))

        workflow.set_entry_point("loader")
        workflow.add_edge("loader", "security_check")

        # ✅ 路由：security_check 先短路；通过后再并行执行 detection + inner_monologue
        def _route_after_security_check(state: dict) -> str:
            sc = state.get("security_check") or {}
            if sc.get("needs_security_response", False):
                return "security_response"
            return "after_security"

        workflow.add_conditional_edges(
            "security_check",
            _route_after_security_check,
            {
                "security_response": "security_response",
                "after_security": "after_security",
            },
        )
        
        # ✅ 安全响应节点直接结束（跳过 LATS 等流程）
        workflow.add_edge("security_response", END)

        # 并行分支：after_security fan-out -> (detection, inner_monologue) -> join at style
        workflow.add_edge("after_security", "detection")
        workflow.add_edge("after_security", "inner_monologue")
        workflow.add_edge("detection", "style")
        workflow.add_edge("inner_monologue", "style")
        workflow.add_edge("style", "task_planner")

        def _route_after_task_planner(state: dict) -> str:
            if state.get("no_reply") or (state.get("word_budget") == 0):
                return "no_reply"
            return "lats_search"

        workflow.add_node("no_reply_handler", _wrap_node("no_reply_handler", _no_reply_handler))
        workflow.add_conditional_edges("task_planner", _route_after_task_planner, {"no_reply": "no_reply_handler", "lats_search": "lats_search"})
        workflow.add_edge("no_reply_handler", END)
        workflow.add_edge("lats_search", "processor")
        workflow.add_edge("processor", "final_validator")

        if end_at == "final_validator":
            # Fast return: stop as soon as the final reply is validated.
            workflow.add_edge("final_validator", END)
        else:
            workflow.add_edge("final_validator", "evolver")
            workflow.add_edge("evolver", "stage_manager")
            workflow.add_edge("stage_manager", "memory_manager")
            workflow.add_edge("memory_manager", "memory_writer")
            workflow.add_edge("memory_writer", END)

        return workflow.compile()

    if entry_point == "evolver":
        # Tail graph: evolver -> stage_manager -> memory_manager -> memory_writer -> END
        workflow.add_node("evolver", _wrap_node("evolver", evolver_node))
        workflow.add_node("stage_manager", _wrap_node("stage_manager", stage_manager_node))
        workflow.add_node("memory_manager", _wrap_node("memory_manager", memory_manager_node))
        workflow.add_node("memory_writer", _wrap_node("memory_writer", memory_writer_node))
        workflow.set_entry_point("evolver")
        workflow.add_edge("evolver", "stage_manager")
        workflow.add_edge("stage_manager", "memory_manager")
        workflow.add_edge("memory_manager", "memory_writer")
        workflow.add_edge("memory_writer", END)
        return workflow.compile()

    raise ValueError(f"unsupported entry_point: {entry_point}")

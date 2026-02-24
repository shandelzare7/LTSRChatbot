"""【编排层】构建 LangGraph：节点与边（含并行思考与 Critic 循环）。"""
from __future__ import annotations

import asyncio
import inspect
import os
import time
from typing import Any, Callable, List, Optional, Literal

from langgraph.graph import END, StateGraph

from app.state import AgentState
from app.nodes.loader import create_loader_node

from app.nodes.detection import create_detection_node
from app.nodes.inner_monologue import create_inner_monologue_node
from app.nodes.style import create_style_node
from app.nodes.task_planner import create_task_planner_node
from app.nodes.lats_search import create_lats_search_node
from app.nodes.fast_reply import create_fast_reply_node
from app.nodes.processor import create_processor_node
from app.nodes.evolver import create_evolver_node
from app.nodes.stage_manager import create_stage_manager_node
from app.nodes.memory_manager import create_memory_manager_node
from app.nodes.memory_writer import create_memory_writer_node
from app.nodes.strategy_resolver import create_strategy_resolver_node
from app.nodes.strategy_routers import (
    create_router_high_stakes_node,
    create_router_emotional_game_node,
    create_router_form_rhythm_node,
)
from app.services.llm import get_llm, llm_stats_diff, llm_stats_snapshot, set_current_node, reset_current_node
from app.services.memory import MockMemory


def _no_reply_handler(state: dict) -> dict:
    """no_reply 时短路：不调用 LATS，直接产出空回复并结束图。"""
    return {
        "final_response": "",
        "reply_plan": {"messages": []},
    }


def _skip_reply_handler(state: dict) -> dict:
    """动量过低时：不输出回复，但产出空回复并进入 evolver 等后续节点（更新状态、写记忆等）。"""
    return {
        "final_response": "",
        "reply_plan": {"messages": []},
    }


def _after_security_handler(state: dict) -> dict:
    """安全检查通过后的汇合占位节点（no-op），用于 fan-out 并行分支。"""
    return {}


def build_graph(
    *,
    llm: Any = None,
    llm_fast: Any = None,
    llm_judge: Any = None,
    memory_service: Any = None,
    entry_point: Literal["loader", "evolver"] = "loader",
    end_at: Literal["memory_writer", "processor"] = "memory_writer",
) -> Any:
    """
    构建 LangGraph。
    - llm / memory_service 为 None 时使用默认（MockMemory、get_llm()）。
    - Mode 行为策略（心理模式）已归档，图中不再加载 modes 或使用 PsychoEngine。
    """
    # LLM 分配：evolver、LATS 的 27 候选生成 用 fast；LATS 单模型评估用 main；fast_reply 用 main。
    # 按节点设 temperature：detection 0.1、task_planner 0.15、fast_reply 0.55；
    # processor 0.3、evolver 0.18、memory_manager 0.1、三 router 0.05；reply_planner 27 候选见下。
    from app.core import graph_llm_config as _glc
    _glc.PLANNER_TEMPERATURE = 1.1
    _glc.PLANNER_TOP_P = 0.95
    _glc.PLANNER_FREQUENCY_PENALTY = 0.4
    _glc.PLANNER_PRESENCE_PENALTY = 0.5
    llm = llm or get_llm(role="main")
    llm_fast = llm_fast or get_llm(role="fast")
    llm_detection = get_llm(role="fast", temperature=0.1)
    llm_task_planner = get_llm(role="fast", temperature=0.15)
    llm_fast_reply = get_llm(role="main", temperature=0.55)
    llm_planner_27 = get_llm(role="fast", model="gpt-4.1-mini", temperature=_glc.PLANNER_TEMPERATURE)
    llm_processor = get_llm(role="fast", temperature=0.3)
    llm_evolver = get_llm(role="fast", temperature=0.18)
    llm_memory_manager = get_llm(role="fast", temperature=0.1)
    llm_routers = get_llm(role="fast", temperature=0.05)
    memory_service = memory_service or MockMemory()

    def _truthy(v: str | None) -> bool:
        return str(v or "").strip().lower() in {"1", "true", "yes", "y", "on"}

    enable_step_profile = _truthy(os.getenv("LTSR_PROFILE_STEPS"))
    enable_call_log = _truthy(os.getenv("LTSR_LLM_CALL_LOG"))

    def _wrap_node(name: str, fn: Any) -> Any:
        # 始终设置节点上下文，便于 [LLM_ELAPSED] 等 log 打出正确 node 名。
        do_profile = enable_step_profile or enable_call_log

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
            try:
                if not do_profile:
                    return await fn(state)
                prof = _ensure_profile(state)
                before = llm_stats_snapshot(node_name=name)
                t0 = time.perf_counter()
                try:
                    out = await fn(state)
                finally:
                    pass
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
            finally:
                reset_current_node(tok)

        def _call_sync(state: Any) -> Any:
            tok = set_current_node(name)
            try:
                if not do_profile:
                    return fn(state)
                prof = _ensure_profile(state)
                before = llm_stats_snapshot(node_name=name)
                t0 = time.perf_counter()
                try:
                    out = fn(state)
                finally:
                    pass
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
            finally:
                reset_current_node(tok)

        if inspect.iscoroutinefunction(fn):
            return _call_async
        return _call_sync

    loader_node = create_loader_node(memory_service)
    detection_node = create_detection_node(llm_detection)   # fast, temperature=0.1
    inner_monologue_node = create_inner_monologue_node(llm_fast)
    style_node = create_style_node(None)  # 纯计算，不调用 LLM
    task_planner_node = create_task_planner_node(llm_task_planner)  # fast, temperature=0.15
    lats_node = create_lats_search_node(
        llm_fast,
        llm_soft_scorer=llm_fast,
        llm_gen=llm_planner_27,   # 27 候选生成（reply_planner）；temperature/top_p 由 graph_llm_config 定义，reply_planner 按 gen_round 读取
        llm_eval=llm,             # 单模型评估用 main
    )
    fast_reply_node = create_fast_reply_node(llm_fast_reply)  # main, temperature=0.55
    processor_node = create_processor_node(llm_processor)
    evolver_node = create_evolver_node(llm_evolver)
    stage_manager_node = create_stage_manager_node()
    memory_manager_node = create_memory_manager_node(llm_memory_manager)
    memory_writer_node = create_memory_writer_node(memory_service)
    router_high_stakes_node = create_router_high_stakes_node(llm_routers)
    router_emotional_game_node = create_router_emotional_game_node(llm_routers)
    router_form_rhythm_node = create_router_form_rhythm_node(llm_routers)
    strategy_resolver_node = create_strategy_resolver_node()

    workflow = StateGraph(AgentState)

    if entry_point == "loader":
        # Full (or fast-return) graph: loader -> ... -> processor -> (optional tail) -> END
        workflow.add_node("loader", _wrap_node("loader", loader_node))
        workflow.add_node("detection", _wrap_node("detection", detection_node))
        workflow.add_node("after_security", _wrap_node("after_security", _after_security_handler))
        workflow.add_node("inner_monologue", _wrap_node("inner_monologue", inner_monologue_node))
        workflow.add_node("router_high_stakes", _wrap_node("router_high_stakes", router_high_stakes_node))
        workflow.add_node("router_emotional_game", _wrap_node("router_emotional_game", router_emotional_game_node))
        workflow.add_node("router_form_rhythm", _wrap_node("router_form_rhythm", router_form_rhythm_node))
        workflow.add_node("style", _wrap_node("style", style_node))
        workflow.add_node("strategy_resolver", _wrap_node("strategy_resolver", strategy_resolver_node))
        workflow.add_node("task_planner", _wrap_node("task_planner", task_planner_node))
        workflow.add_node("lats_search", _wrap_node("lats_search", lats_node))
        workflow.add_node("fast_reply", _wrap_node("fast_reply", fast_reply_node))
        workflow.add_node("processor", _wrap_node("processor", processor_node))
        workflow.add_node("evolver", _wrap_node("evolver", evolver_node))
        workflow.add_node("stage_manager", _wrap_node("stage_manager", stage_manager_node))
        workflow.add_node("memory_manager", _wrap_node("memory_manager", memory_manager_node))
        workflow.add_node("memory_writer", _wrap_node("memory_writer", memory_writer_node))

        workflow.set_entry_point("loader")
        workflow.add_edge("loader", "after_security")

        # 并行分支 -> strategy_resolver -> style -> task_planner；路由在 task_planner 之后，使 fast_reply 也能吃到任务
        workflow.add_edge("after_security", "detection")
        workflow.add_edge("after_security", "inner_monologue")
        workflow.add_edge("after_security", "router_high_stakes")
        workflow.add_edge("after_security", "router_emotional_game")
        workflow.add_edge("after_security", "router_form_rhythm")
        workflow.add_edge("detection", "strategy_resolver")
        workflow.add_edge("inner_monologue", "strategy_resolver")
        workflow.add_edge("router_high_stakes", "strategy_resolver")
        workflow.add_edge("router_emotional_game", "strategy_resolver")
        workflow.add_edge("router_form_rhythm", "strategy_resolver")
        workflow.add_edge("strategy_resolver", "style")
        workflow.add_edge("style", "task_planner")

        def _route_after_task_planner(state: dict) -> str:
            # skip_reply（strategy_resolver 动量过低）与 no_reply（task_planner 本轮不回复）统一走 skip_reply_handler -> evolver
            if state.get("skip_reply") or state.get("no_reply"):
                return "skip_reply"
            # 策略为 route_path=="fast" 或前两条回复总用时 < 阈值（force_fast_route）时走 fast_reply
            if state.get("force_fast_route"):
                return "fast_reply"
            cur = state.get("current_strategy") or {}
            if cur.get("route_path") == "fast":
                return "fast_reply"
            return "lats_search"

        workflow.add_node("skip_reply_handler", _wrap_node("skip_reply_handler", _skip_reply_handler))
        # 从 task_planner 分支到 fast_reply / lats_search，保证 fast_reply 也能吃到 task_planner 产出的任务；skip_reply 与 no_reply 统一走 skip_reply_handler
        workflow.add_conditional_edges(
            "task_planner",
            _route_after_task_planner,
            {
                "skip_reply": "skip_reply_handler",
                "fast_reply": "fast_reply",
                "lats_search": "lats_search",
            },
        )
        workflow.add_edge("skip_reply_handler", "evolver")
        workflow.add_edge("fast_reply", "processor")
        workflow.add_edge("lats_search", "processor")

        if end_at == "processor":
            # Fast return: stop as soon as the processor outputs the reply.
            workflow.add_edge("processor", END)
        else:
            workflow.add_edge("processor", "evolver")
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

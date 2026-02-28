"""【编排层】构建 LangGraph：以内心独白为核心的新流水线。

新架构流程：
  loader
    └─→ safety ──(triggered)──→ fast_safety_reply ─┐
              │(normal)                              │
              ↓                                      │
        after_safety (fan-out)                       │
          ├─→ detection                              │
          └─→ state_prep                             │
                │ (fan-in → monologue_join)          │
                ↓                                    │
            inner_monologue                          │
                ↓                                    │
             extract                              │
              ┌────┴────┐                            │
              ↓         ↓                            │
          state_update  style                        │
              └────┬────┘                            │
                   ↓                                 │
             generate_join (fan-in)                  │
                   │                                 │
                   ↓                                 │
               generate (async 5-route × n=4)        │
                   │                                 │
                   ↓                                 │
                judge ◄──────────────────────────────┘
                   │
                   ↓
               processor
                   │
       ┌───────────┴────────────┐
       ↓                        ↓
   evolver              (END if end_at=processor)
       │
  stage_manager
       │
 memory_manager
       │
 memory_writer ──→ END
"""
from __future__ import annotations

import inspect
import os
import time
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from app.state import AgentState
from app.nodes.loader import create_loader_node
from app.nodes.safety import create_safety_node
from app.nodes.fast_safety_reply import create_fast_safety_reply_node
from app.nodes.detection import create_detection_node
from app.nodes.state_prep import create_state_prep_node
from app.nodes.inner_monologue import create_inner_monologue_node
from app.nodes.extract import create_extract_node
from app.nodes.state_update import create_state_update_node
from app.nodes.style import create_style_node
from app.nodes.generate import create_generate_node
from app.nodes.judge import create_judge_node
from app.nodes.processor import create_processor_node
from app.nodes.evolver import create_evolver_node
from app.nodes.stage_manager import create_stage_manager_node
from app.nodes.memory_manager import create_memory_manager_node
from app.nodes.memory_writer import create_memory_writer_node
from app.nodes.knowledge_fetcher import create_knowledge_fetcher_node
from app.services.llm import get_llm, llm_stats_diff, llm_stats_snapshot, set_current_node, reset_current_node
from app.services.memory import MockMemory


def _noop_handler(_state: dict) -> dict:
    """无操作汇合节点（fan-out → fan-in 之间的占位）。"""
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
    构建新架构 LangGraph。
    - llm / memory_service 为 None 时使用默认（MockMemory、get_llm()）。
    """
    # ── LLM 分配 ──────────────────────────────────────────────────────────────
    llm = llm or get_llm(role="main")
    llm_fast = llm_fast or get_llm(role="fast")
    llm_judge = llm_judge or get_llm(role="main")

    llm_safety = get_llm(role="fast", temperature=0.05)
    llm_detection = get_llm(role="fast", temperature=0.1)
    llm_monologue = get_llm(role="main")
    llm_extract = get_llm(role="fast", temperature=0.1)
    llm_processor = get_llm(role="fast", temperature=0.3)
    llm_evolver = get_llm(role="fast", temperature=0.18)
    llm_memory_manager = get_llm(role="fast", temperature=0.1)
    llm_fast_safety_reply = get_llm(role="main", temperature=0.55)

    # Generate LLM：仅从 config/llm_models.yaml 的 generate 读取，改模型只改那一处；API Key 仍用 .env
    from utils.yaml_loader import load_llm_models_config
    _llm_models = load_llm_models_config()
    _gen_cfg = (_llm_models.get("generate") or {}) if isinstance(_llm_models.get("generate"), dict) else {}
    _gen_model = (_gen_cfg.get("model") or "").strip() or "qwen-plus-2024-12-20"
    _gen_base_url = (_gen_cfg.get("base_url") or "").strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    _gen_api_key = (os.getenv("LTSR_GEN_API_KEY") or "").strip() or None
    if _gen_model and "qwen" in _gen_model.lower():
        _gen_api_key = _gen_api_key or (os.getenv("QWEN_API_KEY") or "").strip() or None
    _gen_temperature = float(_gen_cfg.get("temperature", 1.0)) if _gen_cfg else 1.0
    _gen_top_p = float(_gen_cfg.get("top_p", 0.95)) if _gen_cfg else 0.95
    _gen_penalty = float(_gen_cfg.get("presence_penalty", 0.3)) if _gen_cfg else 0.3
    _gen_n = int(_gen_cfg.get("n", 4)) if _gen_cfg else 4
    llm_gen = get_llm(
        role="fast",
        model=_gen_model,
        api_key=_gen_api_key,
        base_url=_gen_base_url or None,
        temperature=_gen_temperature,
        top_p=_gen_top_p,
        presence_penalty=_gen_penalty,
        n=_gen_n,
    )

    memory_service = memory_service or MockMemory()

    def _truthy(v: str | None) -> bool:
        return str(v or "").strip().lower() in {"1", "true", "yes", "y", "on"}

    enable_step_profile = _truthy(os.getenv("LTSR_PROFILE_STEPS"))
    enable_call_log = _truthy(os.getenv("LTSR_LLM_CALL_LOG"))

    def _wrap_node(name: str, fn: Any) -> Any:
        """为节点函数包装计时/LLM 统计逻辑，兼容 async。"""
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
                out = await fn(state)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                after = llm_stats_snapshot(node_name=name)
                prof["nodes"].append(
                    {"name": name, "dt_ms": round(dt_ms, 2), "llm_delta": llm_stats_diff(before, after)}
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
                out = fn(state)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                after = llm_stats_snapshot(node_name=name)
                prof["nodes"].append(
                    {"name": name, "dt_ms": round(dt_ms, 2), "llm_delta": llm_stats_diff(before, after)}
                )
                if isinstance(out, dict):
                    out["_profile"] = prof
                return out
            finally:
                reset_current_node(tok)

        if inspect.iscoroutinefunction(fn):
            return _call_async
        return _call_sync

    # ── 节点实例化 ─────────────────────────────────────────────────────────────
    loader_node = create_loader_node(memory_service)
    safety_node = create_safety_node(llm_safety)
    fast_safety_reply_node = create_fast_safety_reply_node(llm_fast_safety_reply)
    detection_node = create_detection_node(llm_detection)
    state_prep_node = create_state_prep_node()            # 纯代码，无 LLM
    inner_monologue_node = create_inner_monologue_node(llm_monologue)
    extract_node = create_extract_node(llm_extract)
    state_update_node = create_state_update_node()        # 纯代码，无 LLM
    style_node = create_style_node(None)                  # 纯代码，无 LLM
    generate_node = create_generate_node(llm_gen)         # async 并行生成
    judge_node = create_judge_node(llm_judge)
    processor_node = create_processor_node(llm_processor)
    evolver_node = create_evolver_node(llm_evolver)
    stage_manager_node = create_stage_manager_node()
    memory_manager_node = create_memory_manager_node(llm_memory_manager)
    memory_writer_node = create_memory_writer_node(memory_service)
    knowledge_fetcher_node = create_knowledge_fetcher_node()   # 纯代码，无 LLM

    # ── 图构建 ─────────────────────────────────────────────────────────────────
    workflow = StateGraph(AgentState)

    if entry_point == "loader":
        # ── 注册所有节点 ────────────────────────────────────────────────────
        workflow.add_node("loader",            _wrap_node("loader",            loader_node))
        workflow.add_node("safety",            _wrap_node("safety",            safety_node))
        workflow.add_node("fast_safety_reply", _wrap_node("fast_safety_reply", fast_safety_reply_node))
        workflow.add_node("after_safety",      _wrap_node("after_safety",      _noop_handler))
        workflow.add_node("detection",         _wrap_node("detection",         detection_node))
        workflow.add_node("state_prep",        _wrap_node("state_prep",        state_prep_node))
        workflow.add_node("monologue_join",    _wrap_node("monologue_join",    _noop_handler))
        workflow.add_node("inner_monologue",   _wrap_node("inner_monologue",   inner_monologue_node))
        workflow.add_node("extract",           _wrap_node("extract",           extract_node))
        workflow.add_node("state_update",      _wrap_node("state_update",      state_update_node))
        workflow.add_node("style",             _wrap_node("style",             style_node))
        workflow.add_node("generate_join",     _wrap_node("generate_join",     _noop_handler))
        workflow.add_node("generate",          _wrap_node("generate",          generate_node))
        workflow.add_node("judge",             _wrap_node("judge",             judge_node))
        workflow.add_node("processor",         _wrap_node("processor",         processor_node))
        workflow.add_node("evolver",           _wrap_node("evolver",           evolver_node))
        workflow.add_node("stage_manager",     _wrap_node("stage_manager",     stage_manager_node))
        workflow.add_node("memory_manager",    _wrap_node("memory_manager",    memory_manager_node))
        workflow.add_node("memory_writer",     _wrap_node("memory_writer",     memory_writer_node))
        workflow.add_node("knowledge_fetcher", _wrap_node("knowledge_fetcher", knowledge_fetcher_node))

        workflow.set_entry_point("loader")

        # loader → safety
        workflow.add_edge("loader", "safety")

        # safety → conditional: triggered → fast_safety_reply, normal → after_safety
        def _route_safety(state: dict) -> str:
            return "fast_safety_reply" if state.get("safety_triggered") else "after_safety"

        workflow.add_conditional_edges(
            "safety",
            _route_safety,
            {"fast_safety_reply": "fast_safety_reply", "after_safety": "after_safety"},
        )

        # after_safety → fan-out: detection + state_prep（并行）
        workflow.add_edge("after_safety", "detection")
        workflow.add_edge("after_safety", "state_prep")

        # detection + state_prep → fan-in: monologue_join
        workflow.add_edge("detection",  "monologue_join")
        workflow.add_edge("state_prep", "monologue_join")

        # monologue_join → conditional: knowledge_gap → knowledge_fetcher → inner_monologue
        #                              no gap → inner_monologue 直接
        def _route_knowledge(state: dict) -> str:
            detection = state.get("detection") or {}
            return "knowledge_fetcher" if detection.get("knowledge_gap") else "inner_monologue"

        workflow.add_conditional_edges(
            "monologue_join",
            _route_knowledge,
            {"knowledge_fetcher": "knowledge_fetcher", "inner_monologue": "inner_monologue"},
        )
        workflow.add_edge("knowledge_fetcher", "inner_monologue")
        workflow.add_edge("inner_monologue",   "extract")

        # extract → fan-out: state_update + style（并行）
        workflow.add_edge("extract", "state_update")
        workflow.add_edge("extract", "style")

        # state_update + style → fan-in: generate_join → generate → judge
        workflow.add_edge("state_update",  "generate_join")
        workflow.add_edge("style",         "generate_join")
        workflow.add_edge("generate_join", "generate")
        workflow.add_edge("generate",      "judge")

        # fast_safety_reply 和 judge 都进 processor（两个路径的汇合）
        workflow.add_edge("fast_safety_reply", "processor")
        workflow.add_edge("judge",             "processor")

        if end_at == "processor":
            workflow.add_edge("processor", END)
        else:
            workflow.add_edge("processor",      "evolver")
            workflow.add_edge("evolver",        "stage_manager")
            workflow.add_edge("stage_manager",  "memory_manager")
            workflow.add_edge("memory_manager", "memory_writer")
            workflow.add_edge("memory_writer",  END)

        return workflow.compile()

    if entry_point == "evolver":
        # Tail graph: evolver → stage_manager → memory_manager → memory_writer → END
        workflow.add_node("evolver",        _wrap_node("evolver",        evolver_node))
        workflow.add_node("stage_manager",  _wrap_node("stage_manager",  stage_manager_node))
        workflow.add_node("memory_manager", _wrap_node("memory_manager", memory_manager_node))
        workflow.add_node("memory_writer",  _wrap_node("memory_writer",  memory_writer_node))
        workflow.set_entry_point("evolver")
        workflow.add_edge("evolver",        "stage_manager")
        workflow.add_edge("stage_manager",  "memory_manager")
        workflow.add_edge("memory_manager", "memory_writer")
        workflow.add_edge("memory_writer",  END)
        return workflow.compile()

    raise ValueError(f"unsupported entry_point: {entry_point}")

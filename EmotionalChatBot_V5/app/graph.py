"""【编排层】构建 LangGraph：节点与边（含并行思考与 Critic 循环）。"""
from pathlib import Path
from typing import Any, Callable

from langgraph.graph import END, StateGraph

from app.core.engine import PsychoEngine
from app.core.mode_base import PsychoMode
from app.state import AgentState
from app.nodes.loader import create_loader_node
from app.nodes.monitor import create_monitor_node
from app.nodes.reasoner import create_reasoner_node
from app.nodes.style import create_style_node
from app.nodes.generator import create_generator_node
from app.nodes.critic import check_critic_result, create_critic_node
from app.nodes.processor import create_processor_node
from app.nodes.evolver import create_evolver_node
from app.services.llm import get_llm
from app.services.memory import MockMemory
from utils.yaml_loader import get_project_root, load_modes_from_dir


def _load_modes() -> list[PsychoMode]:
    root = get_project_root()
    modes_dir = root / "config" / "modes"
    raw = load_modes_from_dir(modes_dir)
    return [PsychoMode.model_validate(m) for m in raw]


def _thinking_node(
    reasoner_fn: Callable[[AgentState], dict],
    styler_fn: Callable[[AgentState], dict],
) -> Callable[[AgentState], dict]:
    """并行思考：合并 Reasoner 与 Styler 的输出（单节点内顺序执行以简化）。"""

    def node(state: AgentState) -> dict:
        out = {}
        out.update(reasoner_fn(state))
        out.update(styler_fn(state))
        return out

    return node


def build_graph(
    *,
    llm: Any = None,
    memory_service: Any = None,
    modes: list[PsychoMode] | None = None,
) -> Any:
    """
    构建 LangGraph。
    - llm / memory_service / modes 为 None 时使用默认（MockMemory、_load_modes、get_llm()）。
    """
    root = get_project_root()
    llm = llm or get_llm()
    memory_service = memory_service or MockMemory()
    modes = modes or _load_modes()
    engine = PsychoEngine(modes=modes, llm_invoker=llm)

    loader_node = create_loader_node(memory_service)
    monitor_node = create_monitor_node(engine)
    reasoner_node = create_reasoner_node(llm)
    style_node = create_style_node(llm)
    thinking_node = _thinking_node(reasoner_node, style_node)
    generator_node = create_generator_node(llm)
    critic_node = create_critic_node(llm)
    processor_node = create_processor_node()
    evolver_node = create_evolver_node(memory_service)

    workflow = StateGraph(AgentState)

    workflow.add_node("loader", loader_node)
    workflow.add_node("monitor", monitor_node)
    workflow.add_node("thinking", thinking_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("refiner", generator_node)
    workflow.add_node("processor", processor_node)
    workflow.add_node("evolver", evolver_node)

    workflow.set_entry_point("loader")
    workflow.add_edge("loader", "monitor")
    workflow.add_edge("monitor", "thinking")
    workflow.add_edge("thinking", "generator")
    workflow.add_edge("generator", "critic")
    workflow.add_conditional_edges(
        "critic",
        check_critic_result,
        {"pass": "processor", "retry": "refiner"},
    )
    workflow.add_edge("refiner", "critic")
    workflow.add_edge("processor", "evolver")
    workflow.add_edge("evolver", END)

    return workflow.compile()

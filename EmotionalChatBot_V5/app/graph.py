"""【编排层】构建 LangGraph：节点与边（含并行思考与 Critic 循环）。"""
from pathlib import Path
from typing import Any, Callable, List, Optional

from langgraph.graph import END, StateGraph

from app.core.engine import PsychoEngine
from app.core.mode_base import PsychoMode
from app.state import AgentState
from app.nodes.loader import create_loader_node
from app.nodes.detection.boundary import create_boundary_node
from app.nodes.detection.sarcasm import create_sarcasm_node
from app.nodes.detection.confusion import create_confusion_node

# 从 detection.py 文件直接导入（在 nodes/ 目录下，与 detection/ 文件夹同名）
import importlib.util
from pathlib import Path

_detection_file = Path(__file__).parent / "nodes" / "detection.py"
_spec = importlib.util.spec_from_file_location("detection_module", _detection_file)
_detection_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_detection_module)

create_detection_node = _detection_module.create_detection_node
route_by_detection = _detection_module.route_by_detection
from app.nodes.monitor import create_monitor_node
from app.nodes.reasoner import create_reasoner_node
from app.nodes.style import create_style_node
from app.nodes.generator import create_generator_node
from app.nodes.critic import check_critic_result, create_critic_node
from app.nodes.processor import create_processor_node
from app.nodes.evolver import (
    create_memory_writer_node,
    create_relationship_analyzer_node,
    create_relationship_updater_node,
)
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
    modes: Optional[List[PsychoMode]] = None,
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
    detection_node = create_detection_node(llm)
    monitor_node = create_monitor_node(engine)
    relationship_analyzer_node = create_relationship_analyzer_node(llm)
    relationship_updater_node = create_relationship_updater_node()
    reasoner_node = create_reasoner_node(llm)
    style_node = create_style_node(llm)
    thinking_node = _thinking_node(reasoner_node, style_node)
    generator_node = create_generator_node(llm)
    critic_node = create_critic_node(llm)
    processor_node = create_processor_node()
    memory_writer_node = create_memory_writer_node(memory_service)
    boundary_node = create_boundary_node(llm)
    sarcasm_node = create_sarcasm_node(llm)
    confusion_node = create_confusion_node(llm)

    workflow = StateGraph(AgentState)

    # 添加所有节点
    workflow.add_node("loader", loader_node)
    workflow.add_node("detection", detection_node)
    workflow.add_node("monitor", monitor_node)
    workflow.add_node("relationship_analyzer", relationship_analyzer_node)
    workflow.add_node("relationship_updater", relationship_updater_node)
    workflow.add_node("thinking", thinking_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("refiner", generator_node)
    workflow.add_node("processor", processor_node)
    workflow.add_node("memory_writer", memory_writer_node)
    workflow.add_node("boundary", boundary_node)
    workflow.add_node("sarcasm", sarcasm_node)
    workflow.add_node("confusion", confusion_node)

    # 设置入口点
    workflow.set_entry_point("loader")
    
    # 主流程：loader -> detection
    workflow.add_edge("loader", "detection")
    
    # 检测节点条件路由：根据检测结果导向不同节点
    workflow.add_conditional_edges(
        "detection",
        route_by_detection,
        {
            "normal": "monitor",      # NORMAL -> 正常流程（monitor -> thinking -> generator）
            "creepy": "boundary",     # CREEPY -> 防御节点
            "sarcasm": "sarcasm",     # KY/BORING -> 冷淡节点
            "confusion": "confusion"   # CRAZY -> 困惑节点
        }
    )
    
    # 正常流程：monitor -> relationship_analyzer -> relationship_updater -> thinking -> generator -> critic -> processor -> memory_writer
    workflow.add_edge("monitor", "relationship_analyzer")
    workflow.add_edge("relationship_analyzer", "relationship_updater")
    workflow.add_edge("relationship_updater", "thinking")
    workflow.add_edge("thinking", "generator")
    workflow.add_edge("generator", "critic")
    workflow.add_conditional_edges(
        "critic",
        check_critic_result,
        {"pass": "processor", "retry": "refiner"},
    )
    workflow.add_edge("refiner", "critic")
    workflow.add_edge("processor", "memory_writer")
    workflow.add_edge("memory_writer", END)
    
    # 特殊处理节点：直接结束（因为它们已经设置了 final_response）
    workflow.add_edge("boundary", END)
    workflow.add_edge("sarcasm", END)
    workflow.add_edge("confusion", END)

    return workflow.compile()

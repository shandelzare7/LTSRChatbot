"""【编排层】构建 LangGraph：节点与边（含并行思考与 Critic 循环）。"""
from pathlib import Path
from typing import Any, Callable, List, Optional

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
from app.nodes.inner_monologue import create_inner_monologue_node
from app.nodes.mode_manager import create_mode_manager_node
from app.nodes.emotion_update import create_emotion_update_node
from app.nodes.reasoner import create_reasoner_node
from app.nodes.memory_retriever import create_memory_retriever_node
from app.nodes.style import create_style_node
from app.nodes.generator import create_generator_node
from app.nodes.critic import check_critic_result, create_critic_node
from app.nodes.lats_search import create_lats_search_node
from app.nodes.processor import create_processor_node
from app.nodes.final_validator import create_final_validator_node
from app.nodes.evolver import create_evolver_node
from app.nodes.stage_manager import create_stage_manager_node
from app.nodes.memory_manager import create_memory_manager_node
from app.nodes.memory_writer import create_memory_writer_node
from app.services.llm import get_llm
from app.services.memory import MockMemory
from utils.yaml_loader import get_project_root, load_modes_from_dir


def _load_modes() -> list[PsychoMode]:
    root = get_project_root()
    modes_dir = root / "config" / "modes"
    raw = load_modes_from_dir(modes_dir)
    return [PsychoMode.model_validate(m) for m in raw]


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
    mode_manager_node = create_mode_manager_node(modes)
    inner_monologue_node = create_inner_monologue_node(llm)
    emotion_update_node = create_emotion_update_node()
    reasoner_node = create_reasoner_node(llm)
    memory_retriever_node = create_memory_retriever_node(memory_service)
    style_node = create_style_node(llm)
    generator_node = create_generator_node(llm)  # legacy / fallback
    critic_node = create_critic_node(llm)  # legacy / fallback
    lats_node = create_lats_search_node(llm)
    processor_node = create_processor_node(llm)
    final_validator_node = create_final_validator_node()
    evolver_node = create_evolver_node(llm)
    stage_manager_node = create_stage_manager_node()
    memory_manager_node = create_memory_manager_node(llm)
    memory_writer_node = create_memory_writer_node(memory_service)

    workflow = StateGraph(AgentState)

    # 添加所有节点
    workflow.add_node("loader", loader_node)
    workflow.add_node("detection", detection_node)
    workflow.add_node("mode_manager", mode_manager_node)
    workflow.add_node("inner_monologue", inner_monologue_node)
    workflow.add_node("emotion_update", emotion_update_node)
    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node("memory_retriever", memory_retriever_node)
    workflow.add_node("style", style_node)
    # legacy nodes kept for fallback / experimentation
    workflow.add_node("generator", generator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("refiner", generator_node)
    # new LATS node
    workflow.add_node("lats_search", lats_node)
    workflow.add_node("processor", processor_node)
    workflow.add_node("final_validator", final_validator_node)
    workflow.add_node("evolver", evolver_node)
    workflow.add_node("stage_manager", stage_manager_node)
    workflow.add_node("memory_manager", memory_manager_node)
    workflow.add_node("memory_writer", memory_writer_node)

    # 设置入口点
    workflow.set_entry_point("loader")
    
    # 主流程：loader -> detection
    workflow.add_edge("loader", "detection")
    
    # 检测节点后直接进入 mode_manager（不再分支）
    workflow.add_edge("detection", "mode_manager")
    # mode_manager 之后直接进入 inner_monologue（不再分支）
    workflow.add_edge("mode_manager", "inner_monologue")
    workflow.add_edge("inner_monologue", "emotion_update")
    workflow.add_edge("emotion_update", "reasoner")

    # 正常流程（LATS）：
    # reasoner -> memory_retriever -> style -> lats_search -> processor -> final_validator -> evolver -> stage_manager -> memory_manager -> memory_writer
    workflow.add_edge("reasoner", "memory_retriever")
    workflow.add_edge("memory_retriever", "style")
    workflow.add_edge("style", "lats_search")
    workflow.add_edge("lats_search", "processor")
    workflow.add_edge("processor", "final_validator")
    workflow.add_edge("final_validator", "evolver")
    workflow.add_edge("evolver", "stage_manager")
    workflow.add_edge("stage_manager", "memory_manager")
    workflow.add_edge("memory_manager", "memory_writer")
    workflow.add_edge("memory_writer", END)

    return workflow.compile()

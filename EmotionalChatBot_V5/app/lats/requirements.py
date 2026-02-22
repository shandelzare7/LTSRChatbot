"""
LATS 用「需求清单」：只保留 style（自然语言）、stage_targets、tasks_for_lats、task_budget_max。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.state import RequirementsChecklist
from utils.yaml_loader import get_project_root, load_stage_by_id


def _norm_str(x: Any) -> str:
    return (str(x) if x is not None else "").strip()


STAGE_TO_INDEX = {
    "initiating": 1, "experimenting": 2, "intensifying": 3, "integrating": 4, "bonding": 5,
    "differentiating": 6, "circumscribing": 7, "stagnating": 8, "avoiding": 9, "terminating": 10,
}


def _load_stage_config(stage_id: str) -> Dict[str, Any]:
    """从 config/stages.yaml 按 stage_id 读取单阶段配置（支持多文档 --- 格式）。"""
    try:
        return load_stage_by_id(stage_id) or {}
    except Exception as e:
        print(f"[Requirements] 加载 stage 配置失败: {e}")
    return {}


def _default_stage_act_targets(stage_id: str) -> Tuple[List[str], List[str]]:
    s = _norm_str(stage_id or "experimenting")
    mapping: Dict[str, Tuple[List[str], List[str]]] = {
        "initiating": (["answer", "clarify", "question", "light_tease", "small_talk"],
                       ["deep_probe", "commitment_push", "intimacy_escalate"]),
        "experimenting": (["answer", "clarify", "question", "light_tease", "small_talk"],
                         ["commitment_push", "intimacy_escalate"]),
        "intensifying": (["answer", "clarify", "question", "empathy", "self_disclosure", "light_tease"],
                         ["commitment_push_hard", "intimacy_escalate_fast"]),
        "circumscribing": (["answer", "clarify", "boundary", "light_tease"],
                          ["deep_probe", "commitment_push", "intimacy_escalate", "emotionally_overbearing"]),
        "avoiding": (["answer", "boundary", "clarify"],
                     ["deep_probe", "commitment_push", "intimacy_escalate"]),
        "terminating": (["boundary", "answer", "closing"],
                       ["intimacy_escalate", "commitment_push"]),
    }
    return mapping.get(s, (["answer", "clarify", "question"], ["commitment_push", "intimacy_escalate"]))


def _build_stage_targets(state: Dict[str, Any], stage: str) -> Dict[str, Any]:
    allowed, forbidden = _default_stage_act_targets(stage)
    stage_targets: Dict[str, Any] = {
        "stage": stage,
        "pacing_notes": [],
        "violation_sensitivity": 0.75,
        "allowed_acts": allowed,
        "forbidden_acts": forbidden,
    }
    cfg = _load_stage_config(stage)
    if cfg:
        act = cfg.get("act") or {}
        goal = act.get("stage_goal") or cfg.get("stage_goal") or ""
        if goal:
            stage_targets["pacing_notes"] = [f"阶段目标: {goal}"]

    detection = state.get("detection") or {}
    pacing = (detection.get("stage_pacing") or "正常").strip()
    max_violation = 0.8 if pacing in ("过分亲密", "过分生疏") else 0.0
    base = float(stage_targets.get("violation_sensitivity", 0.75) or 0.75)
    stage_targets["violation_sensitivity"] = min(1.0, max(base, max_violation))
    return stage_targets


def compile_requirements(state: Dict[str, Any]) -> RequirementsChecklist:
    """只产出：style_instructions（style 节点自然语言）、stage_targets、tasks_for_lats、task_budget_max。"""
    stage = _norm_str(state.get("current_stage") or "experimenting")
    style_instructions = state.get("llm_instructions")
    if style_instructions is None:
        style_instructions = ""
    if not isinstance(style_instructions, str):
        style_instructions = str(style_instructions) if style_instructions else ""

    return {
        "style_instructions": style_instructions,
        "stage_targets": _build_stage_targets(state, stage),
        "tasks_for_lats": state.get("tasks_for_lats") or [],
        "task_budget_max": int(state.get("task_budget_max", 2) or 2),
    }

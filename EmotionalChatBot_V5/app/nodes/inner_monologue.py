"""内心独白节点：基于 detection 感知结果 + mood + relationship，产出 1 句内心反应与投入预算（word_budget / task_budget_max）。"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from utils.tracing import trace_if_enabled
from utils.detailed_logging import log_prompt_and_params, log_llm_response
from utils.llm_json import parse_json_from_llm

from app.state import AgentState

# 内心独白：1 句，≤25 字，只表达倾向/意愿/态度，不写步骤
INNER_MONOLOGUE_MAX_CHARS = 25


def _word_budget_from_rules(state: Dict[str, Any]) -> int:
    """
    规则：hostile/overstep 很高且 mood/关系支撑不足 → 0；
    hostile/overstep/low_effort 较高 → 1–5；否则按 stage × willingness 给正常预算（如 60）。
    """
    scores = state.get("detection_scores") or state.get("detection_signals", {}).get("scores") or {}
    stage_judge = state.get("detection_stage_judge") or state.get("detection_signals", {}).get("stage_judge") or {}
    mood = state.get("mood_state") or {}
    rel = state.get("relationship_state") or {}

    hostile = float(scores.get("hostile", 0) or 0)
    overstep = float(scores.get("overstep", 0) or 0)
    low_effort = float(scores.get("low_effort", 0) or 0)
    friendly = float(scores.get("friendly", 0) or 0)

    # 关系/情绪支撑：信任、亲密、愉悦、强势感
    trust = float(rel.get("trust", 0.5) or 0.5)
    closeness = float(rel.get("closeness", 0.5) or 0.5)
    pleasure = float(mood.get("pleasure", 0) or 0)
    dominance = float(mood.get("dominance", 0) or 0)
    support = (trust + closeness) / 2 + 0.3 * (pleasure + dominance) / 2
    support = max(0.0, min(1.0, support))

    # 高敌意或高越界，且支撑不足 → 不回复
    if (hostile >= 0.7 or overstep >= 0.7) and support < 0.4:
        return 0
    # 敌意/越界/敷衍较高 → 极短回复 1–5 字
    if hostile >= 0.5 or overstep >= 0.6 or (low_effort >= 0.6 and friendly < 0.3):
        return 5
    # 低投入但有一定友好 → 短回复
    if low_effort >= 0.5:
        return 15
    # 正常：按阶段 cap × 意愿给预算（简化：40–60）
    direction = (stage_judge or {}).get("direction") or "none"
    if direction in ("too_fast", "control_or_binding", "betrayal_or_attack"):
        return 25
    return 60


def _task_budget_max_from_rules(state: Dict[str, Any]) -> int:
    """本轮允许完成的任务数 0–2。高压力/低意愿时降低。"""
    scores = state.get("detection_scores") or state.get("detection_signals", {}).get("scores") or {}
    wb = _word_budget_from_rules(state)
    if wb == 0:
        return 0
    hostile = float(scores.get("hostile", 0) or 0)
    overstep = float(scores.get("overstep", 0) or 0)
    if hostile >= 0.5 or overstep >= 0.5:
        return 1
    return 2


def create_inner_monologue_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """创建内心独白节点：基于 detection 输出 + mood + relationship，产出 1 句独白 + word_budget + task_budget_max。"""

    @trace_if_enabled(
        name="Inner Monologue",
        run_type="chain",
        tags=["node", "inner_monologue", "perception"],
        metadata={"state_outputs": ["inner_monologue", "word_budget", "task_budget_max"]},
    )
    def inner_monologue_node(state: AgentState) -> dict:
        word_budget = _word_budget_from_rules(state)
        task_budget_max = _task_budget_max_from_rules(state)

        # 1 句内心独白（≤25 字）：倾向/意愿/态度，不给用户看，供 LATS 人格驱动
        detection_brief = state.get("detection_brief") or {}
        reaction_seed = (detection_brief.get("reaction_seed") or "").strip()
        scores = state.get("detection_scores") or {}
        stage_judge = state.get("detection_stage_judge") or {}
        direction = (stage_judge or {}).get("direction") or "none"

        # 规则 fallback 短句（不调用 LLM 也能有稳定输出）
        if word_budget == 0:
            inner_monologue = "不想接，先不回。"
        elif direction == "too_fast":
            inner_monologue = "这句有点越界，先挡一下。"
        elif float(scores.get("hostile", 0) or 0) >= 0.5:
            inner_monologue = "有点冲，少说为妙。"
        elif float(scores.get("low_effort", 0) or 0) >= 0.5:
            inner_monologue = "挺敷衍的，别追问。"
        elif reaction_seed and len(reaction_seed) <= INNER_MONOLOGUE_MAX_CHARS:
            inner_monologue = reaction_seed[:INNER_MONOLOGUE_MAX_CHARS]
        else:
            # 可选：用 LLM 生成 1 句（严格 ≤25 字）
            inner_monologue = _generate_one_sentence_monologue(state, llm_invoker, reaction_seed, scores, stage_judge)
            if not inner_monologue or len(inner_monologue) > INNER_MONOLOGUE_MAX_CHARS:
                inner_monologue = (inner_monologue or "按常理接话即可。")[:INNER_MONOLOGUE_MAX_CHARS]

        return {
            "inner_monologue": inner_monologue,
            "word_budget": word_budget,
            "task_budget_max": task_budget_max,
        }
    return inner_monologue_node


def _generate_one_sentence_monologue(
    state: AgentState,
    llm_invoker: Any,
    reaction_seed: str,
    scores: Dict[str, float],
    stage_judge: Dict[str, Any],
) -> str:
    """可选：LLM 生成 1 句内心独白（≤25 字），仅表达倾向/意愿，不写步骤。"""
    if llm_invoker is None:
        return "按常理接话即可。"
    try:
        direction = (stage_judge or {}).get("direction") or "none"
        sys = f"""你输出「此刻我想怎么表现」的一句话内心反应，给下游执行用，不给用户看。
要求：1 句中文，≤25 字；只表达倾向/意愿/态度（如：懒得问、想挡一下、想接球但别太热情），不要步骤、不要策略句。
当前感知：friendly={scores.get('friendly', 0):.1f} hostile={scores.get('hostile', 0):.1f} overstep={scores.get('overstep', 0):.1f} low_effort={scores.get('low_effort', 0):.1f}；stage direction={direction}。
reaction_seed（可选）：{reaction_seed or '无'}
只输出这一句话，不要 JSON 不要引号。"""
        user = "请输出一句≤25字的中文内心独白。"
        msg = llm_invoker.invoke([SystemMessage(content=sys), HumanMessage(content=user)])
        content = (getattr(msg, "content", "") or str(msg)).strip().strip('"\'')
        if content and len(content) <= INNER_MONOLOGUE_MAX_CHARS + 5:
            return content[:INNER_MONOLOGUE_MAX_CHARS]
    except Exception:
        pass
    return "按常理接话即可。"

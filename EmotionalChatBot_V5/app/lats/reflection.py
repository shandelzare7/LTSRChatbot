from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from utils.llm_json import parse_json_from_llm

from app.lats.prompt_utils import (
    build_style_profile,
    build_system_memory_block,
    get_chat_buffer_body_messages,
    safe_text,
    summarize_state_for_planner,
)


def failures_to_actionable_hints(
    failed_checks: Any,
    improvement_notes: Any = None,
    *,
    top_k: int = 2,
) -> str:
    """
    方案C：把 failed_checks 转成可执行、具体、可遵守的生成约束（而不是重复"你错了"）。
    - 只取 TopK，避免 prompt 膨胀
    - 强约束/可执行语句优先
    """
    if not isinstance(failed_checks, list) or not failed_checks:
        return ""

    hints: List[str] = []

    def _add(s: str):
        s = (s or "").strip()
        if s and s not in hints:
            hints.append(s)

    for f in failed_checks[: max(1, int(top_k))]:
        if not isinstance(f, dict):
            continue
        fid = str(f.get("id") or "").strip()
        # 将常见硬门槛映射为可执行修正
        if fid == "first_too_short":
            _add("第一条必须≥8字，且必须包含对用户问题的直接回应/明确态度/结论之一；禁止只用寒暄铺垫。")
        elif fid == "too_many_messages":
            _add("控制消息条数≤max_messages；优先合并相邻重复/补充说明，避免碎片化过度。")
        elif fid == "message_too_long":
            _add("单条不要超过 max_message_len；若内容多，用'先结论/态度 + 分点'把信息分配到前2条。")
        elif fid == "empty_message":
            _add("禁止输出空消息；每条必须是完整句/短语且语义闭合。")
        elif fid == "must_have_missing":
            _add("必须覆盖 requirements.must_have 中的关键点；把关键点放在前1-2条，不要拖到最后。")
        else:
            # 兜底：保留人类可读原因，但要求变成动作
            reason = str(f.get("reason") or "").strip()
            if reason:
                _add(f"避免触发失败检查：{reason}。请在生成时主动规避。")

    if isinstance(improvement_notes, list) and improvement_notes:
        for n in improvement_notes[:2]:
            s = str(n or "").strip()
            if s:
                _add(f"改进建议：{s}")

    if not hints:
        return ""
    return "\n".join([f"- {h}" for h in hints])


def update_failure_counter(counter: Counter[str], failed_checks: Any) -> None:
    if not isinstance(failed_checks, list):
        return
    for f in failed_checks:
        if isinstance(f, dict):
            fid = str(f.get("id") or "").strip()
            if fid:
                counter[fid] += 1


def should_trigger_global_guidelines(
    counter: Counter[str],
    *,
    min_count: int = 3,
    top_n: int = 2,
) -> List[Tuple[str, int]]:
    """方案A：检测重复失败模式，达到阈值才触发一次全局 guidelines 生成。"""
    if not counter:
        return []
    items = [(k, v) for k, v in counter.most_common(int(top_n)) if v >= int(min_count)]
    return items


REFLECTION_PATCH_SYSTEM = """你是"搜索树反思总结器"(ReflectionOnTree)。
目标：根据近期搜索中反复出现的失败模式，输出结构化的 patch，用于后续生成候选时避免重复踩坑。

**重要限制：**
- 禁止输出任何抽象道德指令（如"请更有帮助/更礼貌"）
- 禁止输出散文式口号（如"更有用、更清晰"）
- 只能输出以下 4 类结构化 patch：

1. **plan_patch**: 补漏 must_cover_points（内容缺点）
   - add_must_cover_points: 添加缺失的核心要点（不写台词，只写要点）
   - remove_must_cover_points: 移除不必要的要点

2. **style_patch**: 调整 style_targets（表达层修正）
   - 调整 verbal_length（例如：更短/更长）
   - 调整 social_distance（例如：更远/更近）
   - 调整 tone_temperature（例如：更冷/更暖）
   - 调整 emotional_display（例如：更少/更多）
   - 调整 wit_and_humor（例如：更少/更多）
   - 调整 non_verbal_cues（例如：更少/更多）
   - 格式：{"verbal_length": 0.15} 表示设置为 0.15，{"verbal_length": -0.1} 表示减少 0.1

3. **stage_patch**: 加强 stage_targets.pacing_notes 或提高 violation_sensitivity
   - add_pacing_notes: 添加阶段节奏要求（例如："不要突然过度亲密"）
   - adjust_violation_sensitivity: 调整 violation_sensitivity（例如：0.1 表示增加 0.1）

4. **search_patch**: 调整 query_seeds（检索没命中）
   - add_query_seeds: 添加检索关键词
   - remove_query_seeds: 移除无效关键词
   - strengthen_entities: 强化实体（需要检索的实体名称）

请严格输出 JSON 格式，不要其他文字。""".strip()


def build_reflection_patch_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    repeated_failures: List[Tuple[str, int]],
    *,
    examples: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    用 LLM 把"重复失败模式"转换为结构化 patch。
    
    返回格式：
    {
        "plan_patch": {
            "add_must_cover_points": ["要点1", "要点2"],
            "remove_must_cover_points": ["要点3"]
        },
        "style_patch": {
            "verbal_length": 0.15,  # 绝对值或相对值（负数表示减少）
            "social_distance": -0.1,  # 减少 0.1
            "tone_temperature": 0.2,  # 增加 0.2
            ...
        },
        "stage_patch": {
            "add_pacing_notes": ["不要突然过度亲密", "避免深挖隐私"],
            "adjust_violation_sensitivity": 0.1  # 增加 0.1
        },
        "search_patch": {
            "add_query_seeds": ["关键词1", "关键词2"],
            "remove_query_seeds": ["关键词3"],
            "strengthen_entities": ["实体1", "实体2"]
        }
    }
    """
    if llm_invoker is None or not repeated_failures:
        return {}
    
    # 获取 mode_id
    mode = state.get("current_mode")
    mode_id = None
    if isinstance(mode, dict):
        mode_id = mode.get("id")
    elif mode:
        mode_id = getattr(mode, "id", None)
    mode_id = mode_id or "normal_mode"
    
    requirements = state.get("requirements") or {}
    snapshot = summarize_state_for_planner(state)
    
    # 提取 plan_goals、style_targets、stage_targets 用于 prompt
    plan_goals = requirements.get("plan_goals", {})
    style_targets = requirements.get("style_targets", {})
    stage_targets = requirements.get("stage_targets", {})
    
    plan_goals_text = ""
    if isinstance(plan_goals, dict):
        must_cover = plan_goals.get("must_cover_points", [])
        avoid_points = plan_goals.get("avoid_points", [])
        if must_cover or avoid_points:
            plan_goals_text = f"""
## Plan Goals (当前)
must_cover_points: {must_cover[:10]}
avoid_points: {avoid_points[:10]}"""
    
    style_targets_text = ""
    if isinstance(style_targets, dict) and style_targets:
        style_targets_text = f"""
## Style Targets (当前)
{chr(10).join([f"{k}: {v:.2f}" for k, v in list(style_targets.items())[:10]])}"""
    
    stage_targets_text = ""
    if isinstance(stage_targets, dict):
        stage = stage_targets.get("stage", "")
        pacing_notes = stage_targets.get("pacing_notes", [])
        violation_sensitivity = stage_targets.get("violation_sensitivity", 0.7)
        if stage or pacing_notes:
            stage_targets_text = f"""
## Stage Targets (当前)
stage: {stage}
violation_sensitivity: {violation_sensitivity:.2f}
pacing_notes: {pacing_notes[:5]}"""

    system_prompt = f"""{REFLECTION_PATCH_SYSTEM}

## State Snapshot
{snapshot}

## Requirements (Checklist)
{safe_text(requirements)}
{plan_goals_text}
{style_targets_text}
{stage_targets_text}

## Current Mode
{mode_id}
""".strip()

    failure_lines = "\n".join([f"- {fid}: {cnt} 次" for fid, cnt in repeated_failures])
    ex = safe_text(examples) if examples else "（无）"

    task = f"""近期搜索中反复出现的失败模式如下：
{failure_lines}

可选示例（用于理解失败上下文）：
{ex}

请根据这些失败模式，输出结构化的 patch（JSON 格式）。

**重要：**
- 只输出可执行的结构化 patch，不要输出抽象的道德指令或散文式口号
- plan_patch: 补漏 must_cover_points（内容缺点），只写要点，不写台词
- style_patch: 调整 style_targets，使用绝对值（0.0-1.0）或相对值（负数表示减少）
- stage_patch: 加强 pacing_notes 或提高 violation_sensitivity
- search_patch: 调整 query_seeds（检索没命中时）

输出格式（JSON）：
{{
  "plan_patch": {{
    "add_must_cover_points": [],
    "remove_must_cover_points": []
  }},
  "style_patch": {{
    "verbal_length": 0.15,
    "social_distance": -0.1,
    "tone_temperature": 0.2
  }},
  "stage_patch": {{
    "add_pacing_notes": [],
    "adjust_violation_sensitivity": 0.1
  }},
  "search_patch": {{
    "add_query_seeds": [],
    "remove_query_seeds": [],
    "strengthen_entities": []
  }}
}}""".strip()

    body = get_chat_buffer_body_messages(state, limit=20)
    try:
        resp = llm_invoker.invoke([SystemMessage(content=system_prompt), *body, HumanMessage(content=task)])
        txt = getattr(resp, "content", "") or ""
        data = parse_json_from_llm(txt)
        
        if isinstance(data, dict):
            # 验证并清理 patch 结构
            patch: Dict[str, Any] = {}
            
            # 1. plan_patch: 补漏 must_cover_points（内容缺点）
            if "plan_patch" in data and isinstance(data["plan_patch"], dict):
                pp = data["plan_patch"]
                patch["plan_patch"] = {
                    "add_must_cover_points": [str(x) for x in pp.get("add_must_cover_points", []) if str(x).strip()],
                    "remove_must_cover_points": [str(x) for x in pp.get("remove_must_cover_points", []) if str(x).strip()],
                }
            
            # 2. style_patch: 调整 style_targets（表达层修正）
            if "style_patch" in data and isinstance(data["style_patch"], dict):
                sp = data["style_patch"]
                style_patch_clean: Dict[str, float] = {}
                # 支持的 style 维度
                style_dims = [
                    "verbal_length", "social_distance", "tone_temperature",
                    "emotional_display", "wit_and_humor", "non_verbal_cues",
                    "self_disclosure", "topic_adherence", "initiative",
                    "advice_style", "subjectivity", "memory_hook",
                ]
                for dim in style_dims:
                    if dim in sp:
                        try:
                            val = float(sp[dim])
                            # 限制在合理范围内（绝对值 0-1，相对值 -1 到 1）
                            if -1.0 <= val <= 1.0:
                                style_patch_clean[dim] = val
                        except (TypeError, ValueError):
                            pass
                if style_patch_clean:
                    patch["style_patch"] = style_patch_clean
            
            # 3. stage_patch: 加强 stage_targets.pacing_notes 或提高 violation_sensitivity
            if "stage_patch" in data and isinstance(data["stage_patch"], dict):
                stp = data["stage_patch"]
                stage_patch_clean: Dict[str, Any] = {}
                
                # add_pacing_notes
                if "add_pacing_notes" in stp:
                    notes = stp["add_pacing_notes"]
                    if isinstance(notes, list):
                        stage_patch_clean["add_pacing_notes"] = [str(x).strip() for x in notes if str(x).strip()]
                
                # adjust_violation_sensitivity
                if "adjust_violation_sensitivity" in stp:
                    try:
                        adj = float(stp["adjust_violation_sensitivity"])
                        # 限制在合理范围内（-1 到 1）
                        if -1.0 <= adj <= 1.0:
                            stage_patch_clean["adjust_violation_sensitivity"] = adj
                    except (TypeError, ValueError):
                        pass
                
                if stage_patch_clean:
                    patch["stage_patch"] = stage_patch_clean
            
            # 4. search_patch: 调整 query_seeds（检索没命中）
            if "search_patch" in data and isinstance(data["search_patch"], dict):
                sep = data["search_patch"]
                patch["search_patch"] = {
                    "add_query_seeds": [str(x) for x in sep.get("add_query_seeds", []) if str(x).strip()],
                    "remove_query_seeds": [str(x) for x in sep.get("remove_query_seeds", []) if str(x).strip()],
                    "strengthen_entities": [str(x) for x in sep.get("strengthen_entities", []) if str(x).strip()],
                }
            
            return patch
    except Exception as e:
        print(f"[Reflection] ⚠ Patch 生成失败: {e}")
    
    return {}


# 向后兼容：保留旧函数名，但改为调用新函数
def build_global_guidelines_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    repeated_failures: List[Tuple[str, int]],
    *,
    examples: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    向后兼容函数：将结构化 patch 转换为文本格式（用于旧代码）。
    新代码应直接使用 build_reflection_patch_via_llm。
    """
    patch = build_reflection_patch_via_llm(state, llm_invoker, repeated_failures, examples=examples)
    
    # 将 patch 转换为文本格式（仅用于向后兼容）
    lines = []
    
    if patch.get("plan_patch"):
        pp = patch["plan_patch"]
        if pp.get("add_must_cover_points"):
            lines.append(f"- 补漏要点：{', '.join(pp['add_must_cover_points'][:3])}")
        if pp.get("remove_must_cover_points"):
            lines.append(f"- 移除要点：{', '.join(pp['remove_must_cover_points'][:2])}")
    
    if patch.get("style_patch"):
        sp = patch["style_patch"]
        style_changes = [f"{k}={v:+.2f}" for k, v in list(sp.items())[:3]]
        if style_changes:
            lines.append(f"- 风格调整：{', '.join(style_changes)}")
    
    if patch.get("stage_patch"):
        stp = patch["stage_patch"]
        if stp.get("add_pacing_notes"):
            lines.append(f"- 阶段要求：{', '.join(stp['add_pacing_notes'][:2])}")
        if stp.get("adjust_violation_sensitivity") is not None:
            adj = stp["adjust_violation_sensitivity"]
            lines.append(f"- 越界敏感度：{adj:+.2f}")
    
    if patch.get("search_patch"):
        sep = patch["search_patch"]
        if sep.get("add_query_seeds"):
            lines.append(f"- 检索增强：添加关键词 {', '.join(sep['add_query_seeds'][:3])}")
        if sep.get("strengthen_entities"):
            lines.append(f"- 实体强化：{', '.join(sep['strengthen_entities'][:3])}")
    
    return "\n".join(lines).strip() if lines else ""

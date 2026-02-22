from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.lats.prompt_utils import (
    get_chat_buffer_body_messages,
    safe_text,
)
from app.state import ProcessorPlan, ReplyPlan, SimReport
from src.schemas import (
    EvaluatorSoftScore,
    EvaluatorSoftScoreBatch,
    EvaluatorGate1Batch,
    EvaluatorJudgeResult,
    EvaluatorJudgeBatch,
)
from utils.detailed_logging import log_computation, log_llm_response, log_prompt_and_params
from utils.llm_json import parse_json_from_llm

try:
    from utils.yaml_loader import load_stage_by_id
except Exception:
    load_stage_by_id = None


def _get_stage_judge_criteria(stage_id: str) -> str:
    """从 config/stages.yaml 的 judge.content_coding_criteria 取「怎么判」说明。"""
    if not stage_id or load_stage_by_id is None:
        return ""
    try:
        cfg = load_stage_by_id(str(stage_id))
        judge = (cfg or {}).get("judge") or {}
        ccc = judge.get("content_coding_criteria")
        if isinstance(ccc, dict) and ccc:
            parts = [f"Unit: {ccc.get('unit', 'Message/Turn')}"]
            for k in ("A_check", "B_check", "C_check"):
                if ccc.get(k):
                    parts.append(f"{k}: {ccc[k]}")
            if len(parts) > 1:
                return "\n".join(parts)
    except Exception:
        pass
    return ""


def _dimension_scores_to_dict(raw: Any) -> Dict[str, float]:
    """将 schema 中的 List[DimensionScore]（或 list of {dimension, score}）转为 Dict[str, float]。"""
    out: Dict[str, float] = {}
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict):
            dim = item.get("dimension")
            sc = item.get("score")
        elif hasattr(item, "dimension") and hasattr(item, "score"):
            dim, sc = item.dimension, item.score
        else:
            continue
        if dim is not None and sc is not None:
            try:
                out[str(dim).strip()] = float(sc)
            except (TypeError, ValueError):
                pass
    return out


def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def hard_gate(processor_plan: ProcessorPlan, requirements: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Hard gate: deterministic structural/template checks only.
    - structure: empty / too many messages / too long / first too short
    - forbidden terms: requirements.forbidden (includes immersion-break terms)
    - assistant-like template: identity/service patterns
    - P0: unsolicited advice/tutorial tone when user didn't ask
    """
    fails: List[Dict[str, str]] = []
    msgs = processor_plan.get("messages") or []

    allow_empty_reply = bool(requirements.get("allow_empty_reply", False))
    allow_short_reply = bool(requirements.get("allow_short_reply", False))

    # 1) empty
    if not isinstance(msgs, list) or not msgs:
        if allow_empty_reply:
            return []
        return [{"id": "empty", "reason": "messages 为空", "evidence": ""}]

    # 2) count
    max_messages = 5
    if len(msgs) > max_messages:
        fails.append({"id": "too_many_messages", "reason": f"消息条数超上限({len(msgs)}>{max_messages})", "evidence": ""})

    # 3) per-message checks
    max_len = int(requirements.get("max_message_len", 200) or 200)
    for i, m in enumerate(msgs):
        t = str(m or "").strip()
        if not t and (not allow_empty_reply):
            fails.append({"id": "empty_message", "reason": f"第{i+1}条为空", "evidence": ""})
        if t and len(t) > max_len:
            fails.append({"id": "message_too_long", "reason": f"第{i+1}条过长({len(t)}>{max_len})", "evidence": t[:120]})

    # 4) first min length
    if not allow_short_reply:
        min_first_len = int(requirements.get("min_first_len", 8) or 8)
        first = str(msgs[0] or "").strip()
        if len(first) < min_first_len:
            fails.append(
                {"id": "first_too_short", "reason": f"首条过短({len(first)}<{min_first_len})，可能像铺垫/废话", "evidence": first}
            )

    # 5) forbidden terms
    forbidden_terms = requirements.get("forbidden") or []
    if isinstance(forbidden_terms, list) and forbidden_terms:
        all_text_forbidden = "\n".join([str(m) for m in msgs])
        all_lower = all_text_forbidden.lower()
        for term in forbidden_terms:
            t = str(term or "").strip()
            if not t:
                continue
            if t.lower() in all_lower:
                fails.append(
                    {
                        "id": "forbidden_term",
                        "reason": f"命中违禁词：'{t}'（沉浸破坏/模板化风险高）",
                        "evidence": all_text_forbidden[:240],
                    }
                )
                break

    # 6) assistant-like templates (identity + service patterns)
    identity_patterns = [
        r"我是\s*(ai|人工智能|智能助手|机器人助手|chatbot|助手)",
        r"我是一个\s*(ai|人工智能|智能助手|机器人助手|chatbot|助手)",
        r"作为\s*(ai|人工智能|智能助手|机器人助手|chatbot|助手)",
        r"我\s*是[\s\S]{0,24}(ai|人工智能|智能助手|机器人助手|chatbot|助手)",
        r"(我叫|我是|叫我)[\s\S]{0,18}(一个|位)?[\s\S]{0,18}(ai|人工智能|智能助手|机器人助手|chatbot|聊天助手|助手)",
    ]
    service_patterns = [
        r"我可以帮你\s*(解答问题|解决问题|提供信息|做什么|做什么吗)",
        r"有什么可以\s*帮你",
        r"需要我帮你\s*(做什么|解决|解答)",
        r"我能为你\s*(做什么|提供|解答)",
        r"我能帮你\s*(解答问题|解决问题|提供信息|做什么|做什么吗)",
    ]
    all_text = "\n".join([str(m) for m in msgs])
    all_text_lower = all_text.lower()
    if not fails:
        for pat in identity_patterns:
            if re.search(pat, all_text_lower):
                fails.append(
                    {
                        "id": "assistant_like_response",
                        "reason": "检测到自称AI/助手等身份模板，不符合拟人化要求",
                        "evidence": all_text[:200],
                    }
                )
                break
    if not fails:
        for pat in service_patterns:
            if re.search(pat, all_text_lower):
                fails.append(
                    {
                        "id": "assistant_like_response",
                        "reason": "检测到客服模板句式，不符合拟人化要求",
                        "evidence": all_text[:200],
                    }
                )
                break

    # 7) P0 unsolicited advice/tutorial (message-text-based)
    if not fails:
        user_asks_advice = bool(requirements.get("user_asks_advice", False))
        unsolicited_advice_patterns = [
            r"我建议",
            r"建议你",
            r"你应该",
            r"步骤如下",
            r"(第一|首先).{0,12}(第二|其次|然后)",
            r"总结一下",
            r"给你(几个|三点|几点)建议",
        ]
        if not user_asks_advice:
            for pat in unsolicited_advice_patterns:
                if re.search(pat, all_text, re.IGNORECASE):
                    fails.append(
                        {
                            "id": "unsolicited_advice",
                            "reason": "未被请求却出现建议/教程式口吻（容易变助手）",
                            "evidence": all_text[:220],
                        }
                    )
                    break

    log_computation("Evaluator", "硬门槛检查结果", outputs={"failed_checks": fails, "passed": len(fails) == 0})
    return fails


CHOREO_SCORER_SYSTEM = """你是常识经验丰富的语言学专家，现在担任拟人节奏评审。
你将看到：背景信息、完整对话正文、以及候选将发送的 messages[]。

评分范围：0.0~1.0。
关键要求（必须遵守）：
- assistantiness: 0=像真人朋友，1=像AI助手/客服。若 assistantiness>0.5，则 overall_score 必须 <0.3。
- immersion_break: 0=完全入戏，1=明显出戏。若 immersion_break>0.2，则 overall_score 必须 <0.3。
""".strip()


CHOREO_SCORER_BATCH_SYSTEM = """你是常识经验丰富的语言学专家，现在担任拟人节奏评审。
你将看到：背景信息、完整对话正文、以及多个候选（每个候选包含最终将发送的 messages[]）。

评分范围：0.0~1.0。
关键要求（必须遵守）：
- 对每个候选都必须给出 results 条目（一个不漏，按 idx 对齐）
- assistantiness: 0=像真人朋友，1=像AI助手/客服。若 assistantiness>0.5，则 overall_score 必须 <0.3。
- immersion_break: 0=完全入戏，1=明显出戏。若 immersion_break>0.2，则 overall_score 必须 <0.3。
""".strip()


def soft_score_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    reply_plan: ReplyPlan,
    processor_plan: ProcessorPlan,
    requirements: Dict[str, Any],
) -> Optional[Tuple[float, Dict[str, float], List[str], Dict[str, Any]]]:
    """Soft scorer via LLM. Returns (overall, breakdown, notes, details)."""
    if llm_invoker is None:
        return None

    # NOTE: For evaluation, only allow bot_basic_info + user_basic_info + conversation + candidate messages.
    bot_basic_info = state.get("bot_basic_info") or {}
    user_basic_info = state.get("user_basic_info") or {}

    system_prompt = f"""{CHOREO_SCORER_SYSTEM}

## Background
bot_basic_info: {safe_text(bot_basic_info)}
user_basic_info: {safe_text(user_basic_info)}
""".strip()

    msgs = processor_plan.get("messages") or []
    user_input = safe_text(state.get("external_user_text") or state.get("user_input"))

    task = f"""请对候选将发送的回复进行评分。

用户输入：
{user_input}

最终 messages[]：
{safe_text(msgs)}
""".strip()

    body_messages = get_chat_buffer_body_messages(state, limit=200)
    log_prompt_and_params(
        "Evaluator (LLM Soft Scorer)",
        system_prompt=system_prompt,
        user_prompt=task,
        messages=body_messages,
        params={"messages_count": len(msgs) if isinstance(msgs, list) else 0},
    )

    try:
        data = None
        if hasattr(llm_invoker, "with_structured_output"):
            try:
                structured = llm_invoker.with_structured_output(EvaluatorSoftScore)
                obj = structured.invoke([SystemMessage(content=system_prompt), *body_messages, HumanMessage(content=task)])
                data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
            except Exception:
                data = None
        if data is None:
            resp = llm_invoker.invoke([SystemMessage(content=system_prompt), *body_messages, HumanMessage(content=task)])
            content = getattr(resp, "content", "") or ""
            data = parse_json_from_llm(content)
        if not isinstance(data, dict):
            return None
        log_llm_response("Evaluator (LLM Soft Scorer)", None, parsed_result=data)

        bd_raw = data.get("score_breakdown")
        breakdown = _dimension_scores_to_dict(bd_raw) if not isinstance(bd_raw, dict) else {k: float(v) for k, v in (bd_raw or {}).items() if isinstance(v, (int, float))}
        overall = float(data.get("overall_score", 0.0) or 0.0)

        return _clamp01(overall), breakdown, [], {}
    except Exception as e:
        log_computation("Evaluator", "LLM soft scorer failed", inputs={"error": str(e)[:160]})
        return None


def soft_score_batch_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    candidates: List[Dict[str, Any]],
    requirements: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """
    Batched soft scorer via LLM.
    candidates: list of {"idx":int, "reply_plan":ReplyPlan, "processor_plan":ProcessorPlan}
    Returns: idx -> {"overall":float, "breakdown":dict[str,float], "raw":dict}
    """
    out: Dict[int, Dict[str, Any]] = {}
    if llm_invoker is None:
        return out

    # NOTE: For evaluation, only allow bot_basic_info + user_basic_info + conversation + candidate messages.
    bot_basic_info = state.get("bot_basic_info") or {}
    user_basic_info = state.get("user_basic_info") or {}

    system_prompt = f"""{CHOREO_SCORER_BATCH_SYSTEM}

## Background
bot_basic_info: {safe_text(bot_basic_info)}
user_basic_info: {safe_text(user_basic_info)}
""".strip()

    user_input = safe_text(state.get("external_user_text") or state.get("user_input"))

    blocks: List[str] = []
    expected_idxs: List[int] = []
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        try:
            idx = int(c.get("idx"))
        except Exception:
            continue
        expected_idxs.append(idx)
        proc = c.get("processor_plan") or {}
        msgs = (proc or {}).get("messages") or []
        blocks.append(
            f"""[Candidate idx={idx}]
final messages: {safe_text(msgs)}
""".strip()
        )

    task = f"""请对以下候选逐个评分（results 按 idx 对齐）。

用户输入：
{user_input}

候选列表：
{safe_text(blocks)}
""".strip()

    body_messages = get_chat_buffer_body_messages(state, limit=200)
    log_prompt_and_params(
        "Evaluator (LLM Soft Scorer Batch)",
        system_prompt=system_prompt,
        user_prompt=task,
        messages=body_messages,
        params={"candidates": len(expected_idxs)},
    )

    try:
        data = None
        if hasattr(llm_invoker, "with_structured_output"):
            try:
                structured = llm_invoker.with_structured_output(EvaluatorSoftScoreBatch)
                obj = structured.invoke([SystemMessage(content=system_prompt), *body_messages, HumanMessage(content=task)])
                data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
            except Exception:
                data = None
        if data is None:
            resp = llm_invoker.invoke([SystemMessage(content=system_prompt), *body_messages, HumanMessage(content=task)])
            content = getattr(resp, "content", "") or ""
            data = parse_json_from_llm(content)
        if not isinstance(data, dict):
            return out
        log_llm_response("Evaluator (LLM Soft Scorer Batch)", None, parsed_result={"keys": list(data.keys())})
        results = data.get("results")
        if not isinstance(results, list):
            return out
        for r in results:
            if not isinstance(r, dict):
                continue
            try:
                idx = int(r.get("idx"))
            except Exception:
                continue
            bd_raw = r.get("score_breakdown")
            breakdown = _dimension_scores_to_dict(bd_raw) if not isinstance(bd_raw, dict) else {k: float(v) for k, v in (bd_raw or {}).items() if isinstance(v, (int, float))}
            overall = float(r.get("overall_score", 0.0) or 0.0)
            out[idx] = {
                "overall": _clamp01(overall),
                "breakdown": breakdown,
                "raw": r,
            }
        # Ensure all expected idxs exist (missing => empty)
        for idx in expected_idxs:
            out.setdefault(int(idx), {})
        return out
    except Exception as e:
        log_computation("Evaluator", "LLM soft scorer batch failed", inputs={"error": str(e)[:160]})
        return out


GATE1_CHECK_BATCH_SYSTEM = """你是常识经验丰富的语言学专家，现在担任 Gate1 审核员。
你将看到：bot_basic_info、user_basic_info、完整对话正文、以及多个候选回复（每个候选只包含最终将发送的 messages[]）。

你只检查 3 点（不打分，只回答是否）：
1) assistantiness（助手味）：是否像 AI 助手/客服/教程式口吻。像则不通过。
2) identity（身份是否对）：是否把自己身份说错/搞错自己是谁（例如把自己说成另一个人、另一个 bot、或自称 AI/系统）。错则不通过。
3) immersion（对话/背景是否跳脱）：是否出戏、跳出当前对话与给定背景（例如提“设定/系统/模型/剧本”等，或突然讲和当前对话无关的元信息）。跳脱则不通过。

硬性要求：
- 对每个候选都必须给出 results 条目（一个不漏，按 idx 对齐）。
（输出格式由系统约束。）
""".strip()


def gate1_check_batch_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    candidates: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """
    Gate1 (batched, boolean-only):
    - No scoring.
    - Only uses bot_basic_info + user_basic_info + conversation + candidate messages[].
    Returns: idx -> {"pass": bool, "checks": {...}, "failed": [str]}
    """
    out: Dict[int, Dict[str, Any]] = {}
    if llm_invoker is None:
        return out

    bot_basic_info = state.get("bot_basic_info") or {}
    user_basic_info = state.get("user_basic_info") or {}

    expected_idxs: List[int] = []
    blocks: List[str] = []
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        try:
            idx = int(c.get("idx"))
        except Exception:
            continue
        expected_idxs.append(idx)
        proc = c.get("processor_plan") or {}
        msgs = (proc or {}).get("messages") or []
        blocks.append(
            f"""[Candidate idx={idx}]
final messages: {safe_text(msgs)}
""".strip()
        )

    system_prompt = f"""{GATE1_CHECK_BATCH_SYSTEM}

## Background
bot_basic_info: {safe_text(bot_basic_info)}
user_basic_info: {safe_text(user_basic_info)}
""".strip()

    user_input = safe_text(state.get("external_user_text") or state.get("user_input"))
    user_prompt = f"""请逐个检查候选（results 按 idx 对齐）。

用户输入：
{user_input}

候选列表：
{safe_text(blocks)}
""".strip()

    body_messages = get_chat_buffer_body_messages(state, limit=200)
    log_prompt_and_params(
        "Gate1 (Batch Boolean)",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        messages=body_messages,
        params={"candidates": len(expected_idxs)},
    )

    try:
        data = None
        if hasattr(llm_invoker, "with_structured_output"):
            try:
                structured = llm_invoker.with_structured_output(EvaluatorGate1Batch)
                obj = structured.invoke([SystemMessage(content=system_prompt), *body_messages, HumanMessage(content=user_prompt)])
                data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
            except Exception:
                data = None
        if data is None:
            resp = llm_invoker.invoke([SystemMessage(content=system_prompt), *body_messages, HumanMessage(content=user_prompt)])
            content = getattr(resp, "content", "") or ""
            data = parse_json_from_llm(content)
        if not isinstance(data, dict):
            return out
        results = data.get("results")
        if not isinstance(results, list):
            return out
        for r in results:
            if not isinstance(r, dict):
                continue
            try:
                idx = int(r.get("idx"))
            except Exception:
                continue
            assistantiness_ok = bool(r.get("assistantiness_ok"))
            identity_ok = bool(r.get("identity_ok"))
            immersion_ok = bool(r.get("immersion_ok"))
            checks = {
                "assistantiness_ok": assistantiness_ok,
                "identity_ok": identity_ok,
                "immersion_ok": immersion_ok,
            }
            failed: List[str] = []
            if not assistantiness_ok:
                failed.append("assistantiness")
            if not identity_ok:
                failed.append("identity")
            if not immersion_ok:
                failed.append("immersion")
            out[idx] = {"pass": bool(assistantiness_ok and identity_ok and immersion_ok), "checks": checks, "failed": failed}

        for idx in expected_idxs:
            out.setdefault(int(idx), {"pass": False, "checks": {}, "failed": ["parse_missing_idx"]})
        return out
    except Exception as e:
        log_computation("Gate1", "batch boolean check failed", inputs={"error": str(e)[:160]})
        for idx in expected_idxs:
            out.setdefault(int(idx), {"pass": False, "checks": {}, "failed": ["exception"]})
        return out


def gate_judge_batch_via_soft_scorer(
    state: Dict[str, Any],
    llm_invoker: Any,
    candidates: List[Dict[str, Any]],
    requirements: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """
    Layer-1 Gate (batched):
    candidates: list of {"idx":int, "reply_plan":ReplyPlan, "processor_plan":ProcessorPlan}
    Returns: idx -> gate dict (same shape as gate_judge_via_soft_scorer)
    """
    out: Dict[int, Dict[str, Any]] = {}
    expected_idxs: List[int] = []
    for c in candidates or []:
        if isinstance(c, dict):
            try:
                expected_idxs.append(int(c.get("idx")))
            except Exception:
                continue

    soft_map = soft_score_batch_via_llm(state, llm_invoker, candidates, requirements) if llm_invoker else {}
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        try:
            idx = int(c.get("idx"))
        except Exception:
            continue
        soft = soft_map.get(idx) or {}
        if not soft:
            out[idx] = {
                "pass": False,
                "gate_score": 0.0,
                "breakdown": {"assistantiness": 1.0, "immersion_break": 1.0, "background_fit": 0.0},
            }
            continue

        bd = soft.get("breakdown") if isinstance(soft.get("breakdown"), dict) else {}
        llm_overall = float(soft.get("overall", 0.0) or 0.0)

        assistantiness = float(bd.get("assistantiness", 1.0) or 1.0)
        immersion_break = float(bd.get("immersion_break", 1.0) or 1.0)
        persona = float(bd.get("persona_consistency", 0.0) or 0.0)
        rel_fit = float(bd.get("relationship_fit", 0.0) or 0.0)
        mem_faith = float(bd.get("memory_faithfulness", 0.0) or 0.0)

        background_fit = _clamp01(0.40 * persona + 0.35 * rel_fit + 0.25 * mem_faith)

        passed = True
        reasons: List[str] = []
        if assistantiness > 0.35:
            passed = False
            reasons.append(f"assistantiness_too_high:{assistantiness:.2f}")
        if immersion_break > 0.20:
            passed = False
            reasons.append(f"immersion_break:{immersion_break:.2f}")
        if background_fit < 0.45:
            passed = False
            reasons.append(f"background_fit_low:{background_fit:.2f}")
        gate_score = _clamp01(0.55 * float(llm_overall) + 0.45 * background_fit)
        out[idx] = {
            "pass": bool(passed),
            "gate_score": float(round(gate_score, 4)),
            "breakdown": {
                "assistantiness": float(round(_clamp01(assistantiness), 4)),
                "immersion_break": float(round(_clamp01(immersion_break), 4)),
                "background_fit": float(round(background_fit, 4)),
                "persona_consistency": float(round(_clamp01(persona), 4)),
                "relationship_fit": float(round(_clamp01(rel_fit), 4)),
                "memory_faithfulness": float(round(_clamp01(mem_faith), 4)),
                "soft_overall": float(round(_clamp01(float(llm_overall)), 4)),
            },
        }

    for idx in expected_idxs:
        out.setdefault(int(idx), {})
    return out


def gate_judge_via_soft_scorer(
    state: Dict[str, Any],
    llm_invoker: Any,
    reply_plan: ReplyPlan,
    processor_plan: ProcessorPlan,
    requirements: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Layer-1 Gate:
    - Uses soft scorer evidence.
    - Deterministic pass/fail based on assistantiness / immersion_break / background_fit.
    """
    soft = soft_score_via_llm(state, llm_invoker, reply_plan, processor_plan, requirements) if llm_invoker else None
    if not soft:
        return {
            "pass": False,
            "gate_score": 0.0,
            "breakdown": {"assistantiness": 1.0, "immersion_break": 1.0, "background_fit": 0.0},
        }

    llm_overall, bd, _, _ = soft
    assistantiness = float(bd.get("assistantiness", 1.0) or 1.0)
    immersion_break = float(bd.get("immersion_break", 1.0) or 1.0)
    persona = float(bd.get("persona_consistency", 0.0) or 0.0)
    rel_fit = float(bd.get("relationship_fit", 0.0) or 0.0)
    mem_faith = float(bd.get("memory_faithfulness", 0.0) or 0.0)

    background_fit = _clamp01(0.40 * persona + 0.35 * rel_fit + 0.25 * mem_faith)

    passed = True
    reasons: List[str] = []
    if assistantiness > 0.35:
        passed = False
        reasons.append(f"assistantiness_too_high:{assistantiness:.2f}")
    if immersion_break > 0.20:
        passed = False
        reasons.append(f"immersion_break:{immersion_break:.2f}")
    if background_fit < 0.45:
        passed = False
        reasons.append(f"background_fit_low:{background_fit:.2f}")
    gate_score = _clamp01(0.55 * float(llm_overall) + 0.45 * background_fit)

    return {
        "pass": bool(passed),
        "gate_score": float(round(gate_score, 4)),
        "breakdown": {
            "assistantiness": float(round(_clamp01(assistantiness), 4)),
            "immersion_break": float(round(_clamp01(immersion_break), 4)),
            "background_fit": float(round(background_fit, 4)),
            "persona_consistency": float(round(_clamp01(persona), 4)),
            "relationship_fit": float(round(_clamp01(rel_fit), 4)),
            "memory_faithfulness": float(round(_clamp01(mem_faith), 4)),
            "soft_overall": float(round(_clamp01(float(llm_overall)), 4)),
        },
    }


def _simple_json_judge(
    *,
    label: str,
    system_prompt: str,
    user_prompt: str,
    llm_invoker: Any,
    body_messages: List[Any],
    response_schema: Any = None,
) -> Dict[str, Any]:
    """Helper: invoke judge and parse JSON; returns {} on failure. response_schema: EvaluatorJudgeResult or EvaluatorJudgeBatch for method2."""
    try:
        data = None
        if response_schema is not None and hasattr(llm_invoker, "with_structured_output"):
            try:
                structured = llm_invoker.with_structured_output(response_schema)
                obj = structured.invoke([SystemMessage(content=system_prompt), *body_messages, HumanMessage(content=user_prompt)])
                data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
            except Exception:
                data = None
        if data is None:
            resp = llm_invoker.invoke([SystemMessage(content=system_prompt), *body_messages, HumanMessage(content=user_prompt)])
            content = getattr(resp, "content", "") or ""
            data = parse_json_from_llm(content)
        if isinstance(data, dict):
            # 将 schema 返回的 List[DimensionScore] 转为 dict，供下游使用
            if "sub_scores" in data and isinstance(data.get("sub_scores"), list):
                data = dict(data)
                data["sub_scores"] = _dimension_scores_to_dict(data["sub_scores"])
            if "results" in data and isinstance(data.get("results"), list):
                data = dict(data)
                data["results"] = [
                    dict(r) if isinstance(r, dict) else r
                    for r in data["results"]
                ]
                for r in data["results"]:
                    if isinstance(r, dict) and "sub_scores" in r and isinstance(r.get("sub_scores"), list):
                        r["sub_scores"] = _dimension_scores_to_dict(r["sub_scores"])
            return data
    except Exception as e:
        log_computation("Judge", f"{label} failed", inputs={"error": str(e)[:120]})
    return {}


def judge_dimension_relationship_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    reply_plan: ReplyPlan,
    processor_plan: ProcessorPlan,
    requirements: Dict[str, Any],
) -> Dict[str, Any]:
    """Layer-2 Judge A: 6-dim relationship fit."""
    bot_basic_info = state.get("bot_basic_info") or {}
    user_basic_info = state.get("user_basic_info") or {}
    system_prompt = f"""你是常识经验丰富的语言学专家。你只评估：候选回复是否符合对话中隐含的关系距离感与互动姿态，按 6 个维度给分（closeness/trust/liking/respect/attractiveness/power）。
要求：sub_scores 包含 6 个维度，范围 0.0~1.0（越符合越高）；score 为 sub_scores 平均值（允许 0.01 内浮动）。（输出格式由系统约束。）

## Background
bot_basic_info: {safe_text(bot_basic_info)}
user_basic_info: {safe_text(user_basic_info)}
""".strip()
    body_messages = get_chat_buffer_body_messages(state, limit=200)
    msgs = processor_plan.get("messages") or []
    user_prompt = f"""候选最终 messages：
{safe_text(msgs)}
""".strip()
    return _simple_json_judge(
        label="RelJudge", system_prompt=system_prompt, user_prompt=user_prompt,
        llm_invoker=llm_invoker, body_messages=body_messages, response_schema=EvaluatorJudgeResult
    )


def judge_dimension_stage_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    reply_plan: ReplyPlan,
    processor_plan: ProcessorPlan,
    requirements: Dict[str, Any],
) -> Dict[str, Any]:
    """Layer-2 Judge B: Knapp stage tasks/constraints fit."""
    bot_basic_info = state.get("bot_basic_info") or {}
    user_basic_info = state.get("user_basic_info") or {}
    stage_id = str(state.get("current_stage") or (requirements.get("stage_targets") or {}).get("stage") or "experimenting")
    judge_criteria = _get_stage_judge_criteria(stage_id)
    criteria_block = f"\n\n## 本阶段评判要点（怎么判）\n{judge_criteria}" if judge_criteria else ""
    system_prompt = f"""你是常识经验丰富的语言学专家。你只评估：候选回复是否符合对话中隐含的关系阶段与节奏（别越界：太像客服/太亲密太快/或突然冷淡），并按 4 个维度给分。
要求：sub_scores 包含以上 4 个维度，范围 0.0~1.0（越符合越高）；score 为 sub_scores 平均值（允许 0.01 内浮动）。（输出格式由系统约束。）

## Background
bot_basic_info: {safe_text(bot_basic_info)}
user_basic_info: {safe_text(user_basic_info)}{criteria_block}
""".strip()
    body_messages = get_chat_buffer_body_messages(state, limit=200)
    msgs = processor_plan.get("messages") or []
    user_prompt = f"""候选最终 messages：
{safe_text(msgs)}
""".strip()
    return _simple_json_judge(
        label="StageJudge", system_prompt=system_prompt, user_prompt=user_prompt,
        llm_invoker=llm_invoker, body_messages=body_messages, response_schema=EvaluatorJudgeResult
    )


def judge_dimension_mood_busy_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    reply_plan: ReplyPlan,
    processor_plan: ProcessorPlan,
    requirements: Dict[str, Any],
) -> Dict[str, Any]:
    """Layer-2 Judge C: PAD mood + busyness fit."""
    bot_basic_info = state.get("bot_basic_info") or {}
    user_basic_info = state.get("user_basic_info") or {}
    system_prompt = f"""你是常识经验丰富的语言学专家。你只评估：候选回复是否符合对话中隐含的情绪与忙碌节奏（从用户措辞、上下文推断），并按 4 个维度给分（pleasure/arousal/dominance/busyness）。
要求：sub_scores 包含以上 4 个维度，范围 0.0~1.0（越符合越高）；score 为 sub_scores 平均值（允许 0.01 内浮动）。（输出格式由系统约束。）

## Background
bot_basic_info: {safe_text(bot_basic_info)}
user_basic_info: {safe_text(user_basic_info)}
""".strip()
    body_messages = get_chat_buffer_body_messages(state, limit=200)
    msgs = processor_plan.get("messages") or []
    user_prompt = f"""候选最终 messages：
{safe_text(msgs)}
""".strip()
    return _simple_json_judge(
        label="MoodBusyJudge", system_prompt=system_prompt, user_prompt=user_prompt,
        llm_invoker=llm_invoker, body_messages=body_messages, response_schema=EvaluatorJudgeResult
    )


def _parse_batch_judge_results(data: Dict[str, Any], expected_idxs: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Parse batched judge output:
    {
      "results": [
        {"idx": 0, "score":..., "sub_scores":{...}},
        ...
      ]
    }
    Returns idx->result dict (missing idx => {}).
    """
    out: Dict[int, Dict[str, Any]] = {int(i): {} for i in expected_idxs}
    if not isinstance(data, dict):
        return out
    results = data.get("results")
    if not isinstance(results, list):
        return out
    for r in results:
        if not isinstance(r, dict):
            continue
        try:
            idx = int(r.get("idx"))
        except Exception:
            continue
        if idx in out:
            out[idx] = r
    return out


def judge_dimension_relationship_batch_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    candidates: List[Dict[str, Any]],
    requirements: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """
    Layer-2 Judge A (batched): evaluate 6-dim relationship fit for multiple candidates.
    candidates: list of {"idx":int, "processor_plan":ProcessorPlan, "reply_plan":ReplyPlan}
    Returns: idx -> judge result dict
    """
    if llm_invoker is None:
        return {int(c.get("idx", 0) or 0): {} for c in (candidates or []) if isinstance(c, dict)}

    bot_basic_info = state.get("bot_basic_info") or {}
    user_basic_info = state.get("user_basic_info") or {}
    system_prompt = f"""你是常识经验丰富的语言学专家。你只评估：候选回复是否符合对话中隐含的关系距离感与互动姿态，按 6 个维度给分（closeness/trust/liking/respect/attractiveness/power）。
你将一次性评估多个候选（按 idx）。
要求：results 包含每个候选的 idx（一个不漏）；sub_scores 包含 6 个维度，范围 0.0~1.0；score 为 sub_scores 平均值。（输出格式由系统约束。）

## Background
bot_basic_info: {safe_text(bot_basic_info)}
user_basic_info: {safe_text(user_basic_info)}
""".strip()

    body_messages = get_chat_buffer_body_messages(state, limit=200)

    blocks: List[str] = []
    expected_idxs: List[int] = []
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        try:
            idx = int(c.get("idx"))
        except Exception:
            continue
        expected_idxs.append(idx)
        proc = c.get("processor_plan") or {}
        blocks.append(
            f"""[Candidate idx={idx}]
final messages: {safe_text((proc or {}).get("messages") or [])}
""".strip()
        )

    user_prompt = f"""候选列表（逐个评估）：
{safe_text(blocks)}
""".strip()

    data = _simple_json_judge(
        label="RelJudgeBatch",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        llm_invoker=llm_invoker,
        body_messages=body_messages,
        response_schema=EvaluatorJudgeBatch,
    )
    return _parse_batch_judge_results(data, expected_idxs)


def judge_dimension_stage_batch_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    candidates: List[Dict[str, Any]],
    requirements: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """
    Layer-2 Judge B (batched): Knapp stage tasks/constraints fit for multiple candidates.
    Returns: idx -> judge result dict
    """
    if llm_invoker is None:
        return {int(c.get("idx", 0) or 0): {} for c in (candidates or []) if isinstance(c, dict)}

    bot_basic_info = state.get("bot_basic_info") or {}
    user_basic_info = state.get("user_basic_info") or {}
    stage_id = str(state.get("current_stage") or (requirements.get("stage_targets") or {}).get("stage") or "experimenting")
    judge_criteria = _get_stage_judge_criteria(stage_id)
    criteria_block = f"\n\n## 本阶段评判要点（怎么判）\n{judge_criteria}" if judge_criteria else ""
    system_prompt = f"""你是常识经验丰富的语言学专家。你只评估：候选回复是否符合对话中隐含的关系阶段与节奏（别越界：太像客服/太亲密太快/或突然冷淡），并按 4 个维度给分。
你将一次性评估多个候选（按 idx）。
要求：results 包含每个候选的 idx（一个不漏）；sub_scores 包含以上 4 个维度，范围 0.0~1.0；score 为 sub_scores 平均值。（输出格式由系统约束。）

## Background
bot_basic_info: {safe_text(bot_basic_info)}
user_basic_info: {safe_text(user_basic_info)}{criteria_block}
""".strip()

    body_messages = get_chat_buffer_body_messages(state, limit=200)
    blocks: List[str] = []
    expected_idxs: List[int] = []
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        try:
            idx = int(c.get("idx"))
        except Exception:
            continue
        expected_idxs.append(idx)
        proc = c.get("processor_plan") or {}
        blocks.append(
            f"""[Candidate idx={idx}]
final messages: {safe_text((proc or {}).get("messages") or [])}
""".strip()
        )

    user_prompt = f"""候选列表（逐个评估）：
{safe_text(blocks)}
""".strip()

    data = _simple_json_judge(
        label="StageJudgeBatch",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        llm_invoker=llm_invoker,
        body_messages=body_messages,
        response_schema=EvaluatorJudgeBatch,
    )
    return _parse_batch_judge_results(data, expected_idxs)


def judge_dimension_mood_busy_batch_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    candidates: List[Dict[str, Any]],
    requirements: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """
    Layer-2 Judge C (batched): PAD mood + busyness fit for multiple candidates.
    Returns: idx -> judge result dict
    """
    if llm_invoker is None:
        return {int(c.get("idx", 0) or 0): {} for c in (candidates or []) if isinstance(c, dict)}

    bot_basic_info = state.get("bot_basic_info") or {}
    user_basic_info = state.get("user_basic_info") or {}
    system_prompt = f"""你是常识经验丰富的语言学专家。你只评估：候选回复是否符合对话中隐含的情绪与忙碌节奏（从用户措辞、上下文推断），并按 4 个维度给分（pleasure/arousal/dominance/busyness）。
你将一次性评估多个候选（按 idx）。
要求：results 包含每个候选的 idx（一个不漏）；sub_scores 包含以上 4 个维度，范围 0.0~1.0；score 为 sub_scores 平均值。（输出格式由系统约束。）

## Background
bot_basic_info: {safe_text(bot_basic_info)}
user_basic_info: {safe_text(user_basic_info)}
""".strip()

    body_messages = get_chat_buffer_body_messages(state, limit=200)
    blocks: List[str] = []
    expected_idxs: List[int] = []
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        try:
            idx = int(c.get("idx"))
        except Exception:
            continue
        expected_idxs.append(idx)
        proc = c.get("processor_plan") or {}
        blocks.append(
            f"""[Candidate idx={idx}]
final messages: {safe_text((proc or {}).get("messages") or [])}
""".strip()
        )

    user_prompt = f"""候选列表（逐个评估）：
{safe_text(blocks)}
""".strip()

    data = _simple_json_judge(
        label="MoodBusyJudgeBatch",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        llm_invoker=llm_invoker,
        body_messages=body_messages,
        response_schema=EvaluatorJudgeBatch,
    )
    return _parse_batch_judge_results(data, expected_idxs)


def judge_dimension_task_completion_batch_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    candidates: List[Dict[str, Any]],
    requirements: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """
    Layer-2 Judge D (batched): Task completion assessment for multiple candidates.
    Evaluates how well each candidate reply completes tasks from tasks_for_lats.
    Returns: idx -> judge result dict
    """
    if llm_invoker is None:
        return {int(c.get("idx", 0) or 0): {} for c in (candidates or []) if isinstance(c, dict)}

    tasks_for_lats = requirements.get("tasks_for_lats") or []
    if not isinstance(tasks_for_lats, list) or len(tasks_for_lats) == 0:
        # No tasks to evaluate, return empty results
        return {int(c.get("idx", 0) or 0): {"score": 0.0, "sub_scores": {}} for c in (candidates or []) if isinstance(c, dict)}

    bot_basic_info = state.get("bot_basic_info") or {}
    user_basic_info = state.get("user_basic_info") or {}
    
    # Format tasks list for prompt
    tasks_str_parts = []
    for task in tasks_for_lats:
        if not isinstance(task, dict):
            continue
        task_id = task.get("id", "")
        task_desc = task.get("description", "") or task.get("desc", "")
        is_urgent = task.get("is_urgent", False) or task.get("task_type") == "urgent"
        urgent_marker = "【紧急】" if is_urgent else ""
        tasks_str_parts.append(f"- id={task_id} {urgent_marker}描述={task_desc}")
    tasks_str = "\n".join(tasks_str_parts) if tasks_str_parts else "无任务"

    system_prompt = f"""你是任务完成度评估专家。你需要评估候选回复对「本轮任务列表」的完成情况。
你将一次性评估多个候选（按 idx）。
要求：results 包含每个候选的 idx（一个不漏）；score 范围 0.0~1.0；sub_scores 含 overall_completion、urgent_tasks_completed、normal_tasks_completed。紧急任务未完成时 score 应大幅降低（至少扣 0.3~0.5）。紧急任务必须显式完成。（输出格式由系统约束。）

## Background
bot_basic_info: {safe_text(bot_basic_info)}
user_basic_info: {safe_text(user_basic_info)}

## 本轮任务列表
{tasks_str}
""".strip()

    body_messages = get_chat_buffer_body_messages(state, limit=200)
    blocks: List[str] = []
    expected_idxs: List[int] = []
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        try:
            idx = int(c.get("idx"))
        except Exception:
            continue
        expected_idxs.append(idx)
        proc = c.get("processor_plan") or {}
        blocks.append(
            f"""[Candidate idx={idx}]
final messages: {safe_text((proc or {}).get("messages") or [])}
""".strip()
        )

    user_prompt = f"""候选列表（逐个评估任务完成情况）：
{safe_text(blocks)}
""".strip()

    data = _simple_json_judge(
        label="TaskCompletionJudgeBatch",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        llm_invoker=llm_invoker,
        body_messages=body_messages,
        response_schema=EvaluatorJudgeBatch,
    )
    return _parse_batch_judge_results(data, expected_idxs)


def judge_dimension_strategy_batch_via_llm(
    state: Dict[str, Any],
    llm_invoker: Any,
    candidates: List[Dict[str, Any]],
    requirements: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """
    Layer-2 Judge E (batched): 评估候选回复是否符合当前策略（current_strategy）的意图与硬约束。
    Returns: idx -> judge result dict (score 0~1, sub_scores.strategy_fit)
    """
    if llm_invoker is None:
        return {int(c.get("idx", 0) or 0): {} for c in (candidates or []) if isinstance(c, dict)}

    cur = state.get("current_strategy") or {}
    strategy_name = str(cur.get("name") or "当前策略").strip() or "当前策略"
    strategy_prompt = (cur.get("prompt") or "").strip() or "（无具体策略描述）"
    strategy_trigger = (cur.get("trigger") or "").strip() or "（无触发说明）"

    bot_basic_info = state.get("bot_basic_info") or {}
    user_basic_info = state.get("user_basic_info") or {}

    system_prompt = f"""你是常识经验丰富的语言学专家。你只评估：候选回复是否符合「当前策略」的意图与硬约束（是否按策略要求的口吻、边界、禁忌在说话）。
你将一次性评估多个候选（按 idx）。
要求：results 包含每个候选的 idx；score 与 strategy_fit 范围 0.0~1.0（越符合策略越高）。（输出格式由系统约束。）

## 当前策略
- 策略名称：{safe_text(strategy_name)}
- 触发条件：{safe_text(strategy_trigger)}
- 策略要求与硬约束：{safe_text(strategy_prompt[:1500])}

## Background
bot_basic_info: {safe_text(bot_basic_info)}
user_basic_info: {safe_text(user_basic_info)}
""".strip()

    body_messages = get_chat_buffer_body_messages(state, limit=200)
    blocks: List[str] = []
    expected_idxs: List[int] = []
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        try:
            idx = int(c.get("idx"))
        except Exception:
            continue
        expected_idxs.append(idx)
        proc = c.get("processor_plan") or {}
        blocks.append(f"[Candidate idx={idx}]\nfinal messages: {safe_text((proc or {}).get('messages') or [])}")
    user_prompt = f"候选列表（逐个评估是否符合上述策略）：\n{safe_text(blocks)}".strip()

    data = _simple_json_judge(
        label="StrategyJudgeBatch",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        llm_invoker=llm_invoker,
        body_messages=body_messages,
        response_schema=EvaluatorJudgeBatch,
    )
    return _parse_batch_judge_results(data, expected_idxs)


def evaluate_candidate(
    state: Dict[str, Any],
    reply_plan: ReplyPlan,
    processor_plan: ProcessorPlan,
    requirements: Dict[str, Any],
    *,
    llm_soft_scorer: Any = None,
) -> SimReport:
    """
    Compatibility wrapper (used by nodes/lats_search skip path):
    - Runs hard_gate
    - Runs Layer-1 gate (soft scorer based) when available
    Returns a SimReport-like dict.
    """
    failures = hard_gate(processor_plan, requirements)

    gate = {"pass": False, "checks": {}, "failed": ["no_judge_llm"]}
    if llm_soft_scorer:
        got = gate1_check_batch_via_llm(state, llm_soft_scorer, [{"idx": 0, "reply_plan": reply_plan, "processor_plan": processor_plan}])
        gate = got.get(0) or gate

    passed = (len(failures) == 0) and bool(gate.get("pass"))
    score_breakdown: Dict[str, float] = {
        "gate_pass": 1.0 if passed else 0.0,
    }
    checks = gate.get("checks") if isinstance(gate, dict) else None
    if isinstance(checks, dict):
        for k, v in checks.items():
            score_breakdown[f"gate_{k}"] = 1.0 if bool(v) else 0.0

    return {
        "found_solution": bool(passed),
        "eval_score": 1.0 if bool(passed) else 0.0,
        "failed_checks": failures if not passed else [],
        "score_breakdown": score_breakdown,
        "llm_status": "ok" if llm_soft_scorer else "skipped",
        "llm_details": {},
    }


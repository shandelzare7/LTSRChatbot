from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from app.lats.evaluator import (
    gate1_check_batch_via_llm,
    judge_dimension_mood_busy_batch_via_llm,
    judge_dimension_relationship_batch_via_llm,
    judge_dimension_stage_batch_via_llm,
)
from app.lats.reply_planner import plan_reply_candidates_via_llm, plan_reply_via_llm
from app.state import ProcessorPlan, ReplyPlan, SimReport


def _normalize_text(s: str, max_len: int = 1200) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = " ".join(s.split())
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _judge_cache_key_from_proc(proc: Dict[str, Any]) -> str:
    """Cache key: normalized concatenation of processor messages."""
    try:
        msgs = (proc.get("messages") or []) if isinstance(proc, dict) else []
    except Exception:
        msgs = []
    parts: List[str] = []
    for m in msgs if isinstance(msgs, list) else []:
        if isinstance(m, dict):
            parts.append(str(m.get("content") or ""))
        else:
            parts.append(str(m))
    joined = "\n".join([p.strip() for p in parts if str(p).strip()])
    return _normalize_text(joined)


def _compile_reply_plan_to_text_plan(reply_plan: ReplyPlan, *, max_messages: int) -> ProcessorPlan:
    """
    LATS no longer generates/uses delays/actions. For judging, only keep text messages[].
    Accepts either planner's string messages or dict messages with {content}.
    """
    msgs_raw = (reply_plan or {}).get("messages") or []
    out: List[str] = []
    if isinstance(msgs_raw, list):
        for m in msgs_raw[: int(max_messages)]:
            if isinstance(m, str):
                t = m.strip()
            elif isinstance(m, dict):
                t = str(m.get("content") or "").strip()
            else:
                t = str(m).strip()
            if t:
                out.append(t)
    return {"messages": out}


def lats_search_best_plan(
    state: Dict[str, Any],
    llm_planner: Any,
    *,
    llm_soft_scorer: Any = None,
    rollouts: int = 6,
    expand_k: int = 4,
    max_messages: int = 5,
) -> Tuple[Optional[ReplyPlan], Optional[ProcessorPlan], Optional[SimReport], Dict[str, Any]]:
    """
    LATS V2:
    - Planner: generate K candidates with full context (default K=8).
    - Judge layer-1: gate on background_fit / immersion_break / assistantiness,
      using existing soft scorer as evidence.
    - Judge layer-2: 3 concurrent judges on (relationship / stage / mood+busy) and weighted aggregate.
    - If quality is low, inject reasons into planner prompt and regenerate (max 2 times).

    Note: rollouts/expand_k are kept only for compatibility with existing call sites.
    """

    def _clamp01(x: float) -> float:
        try:
            x = float(x)
        except Exception:
            return 0.0
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    requirements = state.get("requirements") or {}
    requirements = requirements if isinstance(requirements, dict) else {}

    # Config
    try:
        k = int(state.get("lats_candidate_k", 8) or 8)
    except Exception:
        k = 8
    k = max(2, min(16, k))
    try:
        # 0 = disable re-plan/regenerate; only pick best within the first planning round
        max_regens = int(state.get("lats_max_regens", 0) or 0)
    except Exception:
        max_regens = 0
    max_regens = max(0, min(5, max_regens))
    try:
        pass_rate_min = float(state.get("lats_gate_pass_rate_min", 0.5) or 0.5)
    except Exception:
        pass_rate_min = 0.5
    # Default threshold tuned for latency in bot-to-bot / interactive usage.
    # Still overrideable via state["lats_final_score_threshold"].
    try:
        final_threshold = float(state.get("lats_final_score_threshold", 0.6) or 0.6)
    except Exception:
        final_threshold = 0.6

    # weights for 3 judges
    try:
        w_rel = float(state.get("lats_dim_w_relationship", 1.0 / 3.0) or (1.0 / 3.0))
        w_stage = float(state.get("lats_dim_w_stage", 1.0 / 3.0) or (1.0 / 3.0))
        w_mood = float(state.get("lats_dim_w_mood_busy", 1.0 / 3.0) or (1.0 / 3.0))
    except Exception:
        w_rel, w_stage, w_mood = (1.0 / 3.0), (1.0 / 3.0), (1.0 / 3.0)
    s = max(1e-6, w_rel + w_stage + w_mood)
    w_rel, w_stage, w_mood = w_rel / s, w_stage / s, w_mood / s

    print(f"\n[LATS_V2] k={k}, max_regens={max_regens}, pass_rate_min={pass_rate_min:.2f}, threshold={final_threshold:.2f}")

    tree: Dict[str, Any] = {"version": "v2", "rounds": [], "best_id": None}

    # Caches (keyed by normalized proc messages)
    gate_cache: Dict[str, Dict[str, Any]] = {}
    rel_cache: Dict[str, Dict[str, Any]] = {}
    stage_cache: Dict[str, Dict[str, Any]] = {}
    mood_cache: Dict[str, Dict[str, Any]] = {}

    def _regen_hints_from_gate(gates: List[Dict[str, Any]], *, limit: int = 4) -> str:
        """
        Build short, positive regen hints from Gate1 boolean checks.
        We avoid asking the LLM to output long reasons/evidence.
        """
        cnt: Dict[str, int] = {}
        for g in gates:
            if not isinstance(g, dict):
                continue
            ck = g.get("checks") if isinstance(g.get("checks"), dict) else {}
            if not bool(ck.get("assistantiness_ok", True)):
                cnt["减少助手味：更口语、更像朋友聊天，避免建议/教程/客服模板"] = cnt.get(
                    "减少助手味：更口语、更像朋友聊天，避免建议/教程/客服模板", 0
                ) + 1
            if not bool(ck.get("identity_ok", True)):
                cnt["身份别说错：不要自称 AI/系统/模型，也不要把自己说成另一个人"] = cnt.get(
                    "身份别说错：不要自称 AI/系统/模型，也不要把自己说成另一个人", 0
                ) + 1
            if not bool(ck.get("immersion_ok", True)):
                cnt["保持入戏：只围绕当前对话内容与背景表达，别提设定/系统/剧本等元信息"] = cnt.get(
                    "保持入戏：只围绕当前对话内容与背景表达，别提设定/系统/剧本等元信息", 0
                ) + 1
        top = sorted(cnt.items(), key=lambda x: x[1], reverse=True)[:limit]
        return "\n".join([f"- {t}" for t, _ in top]) if top else ""

    def _regen_hints_from_layer2(best_layer2: Dict[str, Any], *, limit: int = 4) -> str:
        """
        Build short hints from numeric sub_scores (no LLM-provided reasons).
        """
        hints: List[str] = []
        rel = best_layer2.get("rel") if isinstance(best_layer2.get("rel"), dict) else {}
        stg = best_layer2.get("stage") if isinstance(best_layer2.get("stage"), dict) else {}
        mood = best_layer2.get("mood") if isinstance(best_layer2.get("mood"), dict) else {}

        def _low_keys(subs: Any, keys: List[str], *, thr: float = 0.55) -> List[str]:
            if not isinstance(subs, dict):
                return []
            out2 = []
            for k2 in keys:
                try:
                    v = float(subs.get(k2))
                except Exception:
                    continue
                if v < thr:
                    out2.append(k2)
            return out2

        rel_low = _low_keys(rel.get("sub_scores"), ["warmth", "trust", "closeness", "respect", "liking", "power"])
        if rel_low:
            hints.append("关系维度更贴合（重点关注：" + ",".join(rel_low[:3]) + "）")

        stg_low = _low_keys(stg.get("sub_scores"), ["stage_goal_alignment", "pacing_notes_followed", "allowed_acts_fit", "forbidden_acts_avoided"])
        if stg_low:
            hints.append("更贴合当前阶段的节奏与任务（重点关注：" + ",".join(stg_low[:2]) + "）")

        mood_low = _low_keys(mood.get("sub_scores"), ["pleasure", "arousal", "dominance", "busyness"])
        if mood_low:
            hints.append("更贴合当前情绪/忙碌表达（重点关注：" + ",".join(mood_low[:2]) + "）")

        return "\n".join([f"- {h}" for h in hints[:limit]]) if hints else ""

    best_item: Optional[Dict[str, Any]] = None
    best_score = -1.0
    extra_constraints_text: Optional[str] = None

    for gen_round in range(max_regens + 1):
        round_report: Dict[str, Any] = {"gen_round": gen_round, "candidates": []}
        tree["rounds"].append(round_report)

        cands = plan_reply_candidates_via_llm(
            state,
            llm_planner,
            k=int(k),
            max_messages=int(max_messages),
            extra_constraints_text=extra_constraints_text,
            global_guidelines=None,
        )

        if not cands:
            rp = plan_reply_via_llm(state, llm_planner, max_messages=int(max_messages))
            if not rp:
                return None, None, None, {"error": "planner_failed"}
            proc = _compile_reply_plan_to_text_plan(rp, max_messages=int(max_messages))
            rep = {
                "found_solution": False,
                "eval_score": 0.0,
                "failed_checks": [{"id": "no_candidates", "reason": "planner 未能生成候选"}],
                "score_breakdown": {},
            }
            return rp, proc, rep, tree

        passed: List[Dict[str, Any]] = []
        failed_gates: List[Dict[str, Any]] = []

        # Gate (batched): at most 1 LLM call for all uncached candidates.
        compiled: List[Dict[str, Any]] = []
        for idx, rp in enumerate(cands):
            proc = _compile_reply_plan_to_text_plan(rp, max_messages=int(max_messages))
            key = _judge_cache_key_from_proc(proc if isinstance(proc, dict) else {})
            compiled.append({"idx": idx, "reply_plan": rp, "processor_plan": proc, "key": key})

        need_gate: List[Dict[str, Any]] = []
        gate_by_idx: Dict[int, Dict[str, Any]] = {}
        for it in compiled:
            idx = int(it.get("idx") or 0)
            key = it.get("key") or ""
            if key and key in gate_cache:
                gate_by_idx[idx] = gate_cache[key]
            else:
                need_gate.append(it)

        if need_gate:
            if llm_soft_scorer:
                got = gate1_check_batch_via_llm(state, llm_soft_scorer, need_gate)
            else:
                got = {int(it.get("idx") or 0): {"pass": False, "checks": {}, "failed": ["no_judge_llm"]} for it in need_gate}
            for it in need_gate:
                idx = int(it.get("idx") or 0)
                key = it.get("key") or ""
                g = got.get(idx) or {"pass": False, "checks": {}, "failed": ["gate1_failed"]}
                gate_by_idx[idx] = g
                if key:
                    gate_cache[key] = g

        for it in compiled:
            idx = int(it.get("idx") or 0)
            rp = it["reply_plan"]
            proc = it["processor_plan"]
            key = it.get("key") or ""
            gate = gate_by_idx.get(idx) or {"pass": False, "checks": {}, "failed": ["gate1_missing"]}

            round_report["candidates"].append({"idx": idx, "gate": gate})
            if bool(gate.get("pass")):
                passed.append({"idx": idx, "reply_plan": rp, "processor_plan": proc, "key": key, "gate": gate})
            else:
                failed_gates.append(gate if isinstance(gate, dict) else {})

        pass_rate = len(passed) / max(1, len(cands))
        round_report["gate_pass_rate"] = round(pass_rate, 4)
        print(f"[LATS_V2] round={gen_round}: gate_pass={len(passed)}/{len(cands)} ({pass_rate:.2f})")
        if pass_rate == 0 and round_report["candidates"]:
            first_gate = (round_report["candidates"][0] or {}).get("gate") or {}
            ck = first_gate.get("checks") or {}
            print(
                f"[LATS_V2] gate_fail_sample idx=0: assistantiness_ok={ck.get('assistantiness_ok')}, identity_ok={ck.get('identity_ok')}, immersion_ok={ck.get('immersion_ok')}"
            )

        if pass_rate < pass_rate_min:
            reasons_txt = _regen_hints_from_gate(failed_gates)
            round_report["regen_reason"] = "gate_pass_rate_low"
            round_report["regen_reasons_summary"] = reasons_txt
            if gen_round < max_regens:
                extra_constraints_text = (reasons_txt or "- 更口语、更像朋友聊天\n- 保持入戏\n- 更贴合人设与关系阶段")[:1800]
                continue
            # return best-by-gate among all
            best_ok = -1
            best_idx = 0
            def _ok_count(g: Dict[str, Any]) -> int:
                ck = g.get("checks") if isinstance(g, dict) else None
                if not isinstance(ck, dict):
                    return 0
                return int(bool(ck.get("assistantiness_ok"))) + int(bool(ck.get("identity_ok"))) + int(bool(ck.get("immersion_ok")))
            for it in round_report["candidates"]:
                g = it.get("gate") if isinstance(it, dict) else None
                if isinstance(g, dict):
                    sc_ok = _ok_count(g)
                    if sc_ok > best_ok:
                        best_ok = sc_ok
                        best_idx = int(it.get("idx", 0) or 0)
            rp = cands[best_idx]
            proc = _compile_reply_plan_to_text_plan(rp, max_messages=int(max_messages))
            rep = {
                "found_solution": False,
                "eval_score": round(float(best_ok) / 3.0, 4),
                "failed_checks": [{"id": "gate_pass_rate_low", "reason": (reasons_txt or "")[:300]}],
                "score_breakdown": {"gate_pass_rate": round(pass_rate, 4), "gate_best_ok_count": float(best_ok), "gate_best_ok_ratio": round(float(best_ok) / 3.0, 4)},
            }
            tree["best_id"] = f"round{gen_round}_idx{best_idx}"
            return rp, proc, rep, tree

        # Layer-2 judges (batched): 3 LLM calls total for this round (rel/stage/mood), executed concurrently.

        def _score_field(d: Dict[str, Any]) -> float:
            # Prefer explicit score; fallback to mean(sub_scores).
            sc = None
            try:
                sc = float(d.get("score")) if d.get("score") is not None else None
            except Exception:
                sc = None
            if sc is not None:
                return _clamp01(sc)
            subs = d.get("sub_scores")
            if isinstance(subs, dict) and subs:
                vals: List[float] = []
                for v in subs.values():
                    try:
                        vals.append(float(v))
                    except Exception:
                        continue
                if vals:
                    return _clamp01(sum(vals) / float(len(vals)))
            return 0.0

        batch = [{"idx": it["idx"], "reply_plan": it["reply_plan"], "processor_plan": it["processor_plan"], "key": it.get("key") or ""} for it in passed]
        expected_idxs = [int(it["idx"]) for it in passed]

        def _run_rel_batch() -> Dict[int, Dict[str, Any]]:
            # Cache by key when available
            need = []
            out: Dict[int, Dict[str, Any]] = {}
            for it in batch:
                k2 = it.get("key") or ""
                idx2 = int(it.get("idx") or 0)
                if k2 and k2 in rel_cache:
                    out[idx2] = rel_cache[k2]
                else:
                    need.append(it)
            if need and llm_soft_scorer:
                got = judge_dimension_relationship_batch_via_llm(state, llm_soft_scorer, need, requirements)
                for it2 in need:
                    idx2 = int(it2.get("idx") or 0)
                    k2 = it2.get("key") or ""
                    r2 = got.get(idx2) or {}
                    out[idx2] = r2
                    if k2:
                        rel_cache[k2] = r2
            return out

        def _run_stage_batch() -> Dict[int, Dict[str, Any]]:
            need = []
            out: Dict[int, Dict[str, Any]] = {}
            for it in batch:
                k2 = it.get("key") or ""
                idx2 = int(it.get("idx") or 0)
                if k2 and k2 in stage_cache:
                    out[idx2] = stage_cache[k2]
                else:
                    need.append(it)
            if need and llm_soft_scorer:
                got = judge_dimension_stage_batch_via_llm(state, llm_soft_scorer, need, requirements)
                for it2 in need:
                    idx2 = int(it2.get("idx") or 0)
                    k2 = it2.get("key") or ""
                    r2 = got.get(idx2) or {}
                    out[idx2] = r2
                    if k2:
                        stage_cache[k2] = r2
            return out

        def _run_mood_batch() -> Dict[int, Dict[str, Any]]:
            need = []
            out: Dict[int, Dict[str, Any]] = {}
            for it in batch:
                k2 = it.get("key") or ""
                idx2 = int(it.get("idx") or 0)
                if k2 and k2 in mood_cache:
                    out[idx2] = mood_cache[k2]
                else:
                    need.append(it)
            if need and llm_soft_scorer:
                got = judge_dimension_mood_busy_batch_via_llm(state, llm_soft_scorer, need, requirements)
                for it2 in need:
                    idx2 = int(it2.get("idx") or 0)
                    k2 = it2.get("key") or ""
                    r2 = got.get(idx2) or {}
                    out[idx2] = r2
                    if k2:
                        mood_cache[k2] = r2
            return out

        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {
                # Propagate current-node context into worker threads so [LLM_CALL] can show node=lats_search.
                ex.submit(contextvars.copy_context().run, _run_rel_batch): "rel",
                ex.submit(contextvars.copy_context().run, _run_stage_batch): "stage",
                ex.submit(contextvars.copy_context().run, _run_mood_batch): "mood",
            }
            batch_res: Dict[str, Dict[int, Dict[str, Any]]] = {}
            for fut in as_completed(futs):
                batch_res[futs[fut]] = fut.result() or {}

        for item in passed:
            idx = int(item.get("idx") or 0)
            res = {
                "rel": (batch_res.get("rel") or {}).get(idx) or {},
                "stage": (batch_res.get("stage") or {}).get(idx) or {},
                "mood": (batch_res.get("mood") or {}).get(idx) or {},
            }
            s_rel = _score_field(res.get("rel") or {})
            s_stage = _score_field(res.get("stage") or {})
            s_mood = _score_field(res.get("mood") or {})
            final_score = _clamp01(w_rel * s_rel + w_stage * s_stage + w_mood * s_mood)

            item["layer2"] = res
            item["final_score"] = final_score

            if final_score > best_score:
                best_score = final_score
                best_item = dict(item)

        round_report["best_final_score"] = round(best_score, 4)
        print(f"[LATS_V2] round={gen_round}: best_final_score={best_score:.3f}")

        if best_item and best_score >= final_threshold:
            rp = best_item["reply_plan"]
            proc = best_item["processor_plan"]
            gate = best_item.get("gate") or {}
            layer2 = best_item.get("layer2") or {}
            rel_sub = ((layer2.get("rel") or {}).get("sub_scores") or {}) if isinstance(layer2.get("rel"), dict) else {}
            stage_sub = ((layer2.get("stage") or {}).get("sub_scores") or {}) if isinstance(layer2.get("stage"), dict) else {}
            mood_sub = ((layer2.get("mood") or {}).get("sub_scores") or {}) if isinstance(layer2.get("mood"), dict) else {}

            def _sub(d: Any, k2: str) -> float:
                try:
                    if isinstance(d, dict) and k2 in d:
                        return _clamp01(float(d.get(k2) or 0.0))
                except Exception:
                    pass
                return 0.0

            rep_ok: Dict[str, Any] = {
                "found_solution": True,
                "eval_score": round(best_score, 4),
                "failed_checks": [],
                "score_breakdown": {
                    "final_score": round(best_score, 4),
                    "gate_pass": 1.0,
                    "gate_assistantiness_ok": 1.0 if bool(((gate.get("checks") or {}).get("assistantiness_ok"))) else 0.0,
                    "gate_identity_ok": 1.0 if bool(((gate.get("checks") or {}).get("identity_ok"))) else 0.0,
                    "gate_immersion_ok": 1.0 if bool(((gate.get("checks") or {}).get("immersion_ok"))) else 0.0,
                    "judge_rel": round(float((layer2.get("rel") or {}).get("score", 0.0) or 0.0), 4),
                    "judge_rel_closeness": round(_sub(rel_sub, "closeness"), 4),
                    "judge_rel_trust": round(_sub(rel_sub, "trust"), 4),
                    "judge_rel_liking": round(_sub(rel_sub, "liking"), 4),
                    "judge_rel_respect": round(_sub(rel_sub, "respect"), 4),
                    "judge_rel_warmth": round(_sub(rel_sub, "warmth"), 4),
                    "judge_rel_power": round(_sub(rel_sub, "power"), 4),
                    "judge_stage": round(float((layer2.get("stage") or {}).get("score", 0.0) or 0.0), 4),
                    "judge_stage_stage_goal_alignment": round(_sub(stage_sub, "stage_goal_alignment"), 4),
                    "judge_stage_pacing_notes_followed": round(_sub(stage_sub, "pacing_notes_followed"), 4),
                    "judge_stage_allowed_acts_fit": round(_sub(stage_sub, "allowed_acts_fit"), 4),
                    "judge_stage_forbidden_acts_avoided": round(_sub(stage_sub, "forbidden_acts_avoided"), 4),
                    "judge_mood_busy": round(float((layer2.get("mood") or {}).get("score", 0.0) or 0.0), 4),
                    "judge_mood_pleasure": round(_sub(mood_sub, "pleasure"), 4),
                    "judge_mood_arousal": round(_sub(mood_sub, "arousal"), 4),
                    "judge_mood_dominance": round(_sub(mood_sub, "dominance"), 4),
                    "judge_mood_busyness": round(_sub(mood_sub, "busyness"), 4),
                    "w_rel": round(w_rel, 4),
                    "w_stage": round(w_stage, 4),
                    "w_mood_busy": round(w_mood, 4),
                },
                "llm_status": "ok",
                "llm_details": {
                    "layer2": layer2,
                },
            }
            tree["best_id"] = f"round{gen_round}_idx{best_item.get('idx')}"
            return rp, proc, rep_ok, tree

        # regen if still low (derive hints from numeric sub_scores of best candidate)
        reasons_txt = _regen_hints_from_layer2((best_item or {}).get("layer2") or {})
        round_report["regen_reason"] = "final_score_low"
        round_report["regen_reasons_summary"] = reasons_txt
        if gen_round < max_regens:
            extra_constraints_text = (reasons_txt or "- 更贴合关系/阶段/情绪忙碌")[:1800]
            continue

    # exhausted
    if best_item:
        rp = best_item["reply_plan"]
        proc = best_item["processor_plan"]
        rep_fail = {
            "found_solution": False,
            "eval_score": round(best_score, 4),
            "failed_checks": [{"id": "no_candidate_meets_threshold", "reason": f"best_final_score={best_score:.3f} < {final_threshold:.2f}"}],
            "score_breakdown": {"final_score": round(best_score, 4), "threshold": float(final_threshold)},
        }
        tree["best_id"] = f"exhausted_idx{best_item.get('idx')}"
        return rp, proc, rep_fail, tree

    return None, None, None, {"error": "no_best_candidate"}


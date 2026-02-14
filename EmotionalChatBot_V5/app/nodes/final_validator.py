from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from app.state import AgentState, HumanizedOutput, ResponseSegment
from utils.tracing import trace_if_enabled


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _hard_gate_segments(
    segments: List[str],
    *,
    max_messages: int = 5,
    min_first_len: int = 8,
    max_message_len: int = 200,
) -> List[Dict[str, str]]:
    fails: List[Dict[str, str]] = []
    if not segments:
        return [{"id": "empty", "reason": "final_segments 为空", "evidence": ""}]

    if len(segments) > max_messages:
        fails.append(
            {
                "id": "too_many_messages",
                "reason": f"消息条数超上限({len(segments)}>{max_messages})",
                "evidence": "",
            }
        )

    first = (segments[0] or "").strip()
    if len(first) < min_first_len:
        fails.append(
            {
                "id": "first_too_short",
                "reason": f"首条过短({len(first)}<{min_first_len})",
                "evidence": first,
            }
        )

    for i, s in enumerate(segments):
        t = (s or "").strip()
        if not t:
            fails.append({"id": "empty_message", "reason": f"第{i+1}条为空", "evidence": ""})
        if len(t) > max_message_len:
            fails.append(
                {
                    "id": "message_too_long",
                    "reason": f"第{i+1}条过长({len(t)}>{max_message_len})",
                    "evidence": t[:120],
                }
            )
    return fails


def _minimal_patch_processor_plan(plan: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
    """一次性最小修补：只做合并/压条数/修首条，不回到 LATS。"""
    msgs = list(plan.get("messages") or [])
    delays = list(plan.get("delays") or [])
    actions = list(plan.get("actions") or [])

    max_messages = int(requirements.get("max_messages", 5) or 5)
    min_first_len = int(requirements.get("min_first_len", 8) or 8)

    # 1) 首条太短：优先与第二条合并
    if len(msgs) >= 2 and len((msgs[0] or "").strip()) < min_first_len:
        msgs[0] = (str(msgs[0]).strip() + " " + str(msgs[1]).strip()).strip()
        del msgs[1]
        # 合并 delay：取 max（更像“等更久后一次发出”），action：取更“离线”的 idle
        if len(delays) >= 2:
            delays[0] = float(max(float(delays[0] or 0), float(delays[1] or 0)))
            del delays[1]
        if len(actions) >= 2:
            actions[0] = "idle" if ("idle" in (actions[0], actions[1])) else "typing"
            del actions[1]

    # 2) 条数超上限：从尾部开始合并
    while len(msgs) > max_messages and len(msgs) >= 2:
        msgs[-2] = (str(msgs[-2]).strip() + " " + str(msgs[-1]).strip()).strip()
        del msgs[-1]
        if len(delays) >= len(msgs) + 1:
            # 被删的是最后一条 delay
            del delays[-1]
        if len(actions) >= len(msgs) + 1:
            del actions[-1]

    # 补齐 delays/actions
    if len(delays) != len(msgs):
        delays = (delays[: len(msgs)] + [0.6] * len(msgs))[: len(msgs)]
    if len(actions) != len(msgs):
        actions = (actions[: len(msgs)] + ["typing"] * len(msgs))[: len(msgs)]

    patched = dict(plan)
    patched["messages"] = msgs
    patched["delays"] = [round(float(_clamp(float(d or 0.6), 0.0, 86400.0)), 2) for d in delays]
    patched["actions"] = [a if a in ("typing", "idle") else "typing" for a in actions]
    meta = patched.get("meta") if isinstance(patched.get("meta"), dict) else {}
    meta = dict(meta)
    meta["minimal_patch_applied"] = True
    patched["meta"] = meta
    return patched


def _build_humanized_from_plan(plan: Dict[str, Any]) -> HumanizedOutput:
    msgs = list(plan.get("messages") or [])
    delays = list(plan.get("delays") or [])
    actions = list(plan.get("actions") or [])
    segments: List[ResponseSegment] = []
    for m, d, a in zip(msgs, delays, actions):
        text = (str(m or "")).strip()
        if not text:
            continue
        try:
            delay_val = float(d)
        except Exception:
            delay_val = 0.6
        segments.append(
            {
                "content": text,
                "delay": round(float(_clamp(delay_val, 0.0, 86400.0)), 2),
                "action": a if a in ("typing", "idle") else "typing",
            }
        )
    total = sum(float(s["delay"]) for s in segments) if segments else 0.0
    return {
        "total_latency_seconds": round(float(total), 2),
        "segments": segments,
        "is_macro_delay": False,
        "total_latency_simulated": round(float(total), 2),
        "latency_breakdown": {"macro_delay": 0.0, "t_read": 0.0, "t_think": 0.0, "macro_reason": 0.0},
    }


def create_final_validator_node() -> Callable[[AgentState], dict]:
    @trace_if_enabled(
        name="Response/FinalValidator",
        run_type="chain",
        tags=["node", "final_validator", "safety", "quality"],
        metadata={"state_outputs": ["final_segments", "final_response", "processor_plan", "humanized_output"]},
    )
    def node(state: AgentState) -> Dict[str, Any]:
        requirements = state.get("requirements") or {}
        max_messages = int(requirements.get("max_messages", 5) or 5)
        min_first_len = int(requirements.get("min_first_len", 8) or 8)
        max_message_len = int(requirements.get("max_message_len", 200) or 200)

        segments = state.get("final_segments") or []
        if not isinstance(segments, list):
            segments = []

        fails = _hard_gate_segments(
            [str(x) for x in segments],
            max_messages=max_messages,
            min_first_len=min_first_len,
            max_message_len=max_message_len,
        )
        if not fails:
            print("[FinalValidator] pass")
            return {}

        # 一次性最小修补：优先修 processor_plan（保证模拟=真实）
        plan = state.get("processor_plan") if isinstance(state.get("processor_plan"), dict) else None
        if not plan:
            print("[FinalValidator] fail-no-plan")
            return {}

        patched = _minimal_patch_processor_plan(plan, requirements)
        patched_segments = list(patched.get("messages") or [])
        patched_text = " ".join([str(x) for x in patched_segments]).strip()
        humanized = _build_humanized_from_plan(patched)

        print("[FinalValidator] patched")
        return {
            "processor_plan": patched,
            "final_segments": patched_segments,
            "final_response": patched_text,
            "humanized_output": humanized,
        }

    return node


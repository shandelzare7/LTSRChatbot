from __future__ import annotations

from typing import Any, Callable, Dict, List

from app.state import AgentState, HumanizedOutput, ResponseSegment
from utils.tracing import trace_if_enabled


def _hard_gate_segments(
    segments: List[str],
    *,
    max_messages: int = 5,
) -> List[Dict[str, str]]:
    """仅校验：空、条数超上限、单条为空。不再校验 min_first_len / max_message_len。"""
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
    for i, s in enumerate(segments):
        t = (s or "").strip()
        if not t:
            fails.append({"id": "empty_message", "reason": f"第{i+1}条为空", "evidence": ""})
    return fails


def _minimal_patch_segments(segments: List[str], max_messages: int = 5) -> List[str]:
    """一次性最小修补：仅压条数（超过 max_messages 则从尾部合并）。"""
    msgs = [str(s or "").strip() for s in segments if str(s or "").strip()]
    while len(msgs) > max_messages and len(msgs) >= 2:
        msgs[-2] = (msgs[-2] + " " + msgs[-1]).strip()
        del msgs[-1]
    return msgs


def _build_humanized_from_segments(segments: List[str]) -> HumanizedOutput:
    """从字符串列表构建 HumanizedOutput，默认每条 delay=0.6。"""
    out: List[ResponseSegment] = []
    for text in segments:
        if not (text or "").strip():
            continue
        out.append({"content": text.strip(), "delay": 0.6, "action": "typing"})
    total = sum(float(s["delay"]) for s in out) if out else 0.0
    return {
        "total_latency_seconds": round(float(total), 2),
        "segments": out,
        "is_macro_delay": False,
        "total_latency_simulated": round(float(total), 2),
        "latency_breakdown": {"macro_delay": 0.0, "t_read": 0.0, "t_think": 0.0, "macro_reason": 0.0},
    }


def create_final_validator_node() -> Callable[[AgentState], dict]:
    @trace_if_enabled(
        name="Response/FinalValidator",
        run_type="chain",
        tags=["node", "final_validator", "safety", "quality"],
        metadata={"state_outputs": ["final_segments", "final_response", "humanized_output"]},
    )
    def node(state: AgentState) -> Dict[str, Any]:
        max_messages = 5
        segments = state.get("final_segments") or []
        if not isinstance(segments, list):
            segments = []

        fails = _hard_gate_segments([str(x) for x in segments], max_messages=max_messages)
        if not fails:
            print("[FinalValidator] pass")
            return {}

        if not segments:
            print("[FinalValidator] fail-no-segments")
            return {}
        patched_segments = _minimal_patch_segments(segments, max_messages=max_messages)
        patched_text = " ".join(patched_segments).strip()
        humanized = _build_humanized_from_segments(patched_segments)
        print("[FinalValidator] patched")
        return {
            "final_segments": patched_segments,
            "final_response": patched_text,
            "humanized_output": humanized,
        }

    return node


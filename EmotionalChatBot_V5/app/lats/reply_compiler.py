from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.state import ProcessorPlan, ReplyPlan
from utils.detailed_logging import log_computation


DELAY_BUCKET_SECONDS: Dict[str, float] = {
    "instant": 0.2,
    "short": 0.6,
    "medium": 1.2,
    "long": 2.5,
    # offline 是“长离线感”，默认 15min（可被 final_validator 降级）
    "offline": 900.0,
}

PAUSE_BONUS_SECONDS: Dict[str, float] = {
    "none": 0.0,
    "beat": 0.3,
    "polite": 0.6,
    "thinking": 1.0,
    "long": 3.0,
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _stage_delay_factor(stage: str) -> float:
    # 复用 processor 的阶段倾向，但保持简单确定（这里不引入随机）
    mapping = {
        "initiating": 1.1,
        "experimenting": 1.0,
        "intensifying": 0.8,
        "integrating": 0.9,
        "bonding": 0.9,
        "differentiating": 1.1,
        "circumscribing": 1.2,
        "stagnating": 1.6,
        "avoiding": 2.0,
        "terminating": 1.8,
    }
    return float(mapping.get(str(stage or "experimenting"), 1.0))


def compile_reply_plan_to_processor_plan(
    reply_plan: ReplyPlan,
    state: Dict[str, Any],
    *,
    max_messages: int = 5,
) -> ProcessorPlan:
    """将 ReplyPlan 编译成可执行 ProcessorPlan（确定性、无副作用）。"""
    stage = str(state.get("current_stage") or "experimenting")
    mood = state.get("mood_state") or {}
    busyness = float(mood.get("busyness", 0.0) or 0.0)

    messages_raw = reply_plan.get("messages") or []
    msgs: List[str] = []
    pause_after: List[str] = []
    delay_bucket: List[str] = []

    for m in messages_raw[:max_messages]:
        # Tolerant parsing:
        # - planner should output dict messages, but in practice LLM may output strings.
        # - Accept string messages to avoid empty-output fallbacks that break bot-to-bot runs.
        if isinstance(m, str):
            c = m.strip()
            if not c:
                continue
            msgs.append(c)
            pause_after.append("none")
            delay_bucket.append("short")
            continue
        if not isinstance(m, dict):
            continue
        c = str(m.get("content") or "").strip()
        if not c:
            continue
        msgs.append(c)
        pause_after.append(str(m.get("pause_after") or "none"))
        delay_bucket.append(str(m.get("delay_bucket") or "short"))

    if not msgs:
        # fallback：至少给一条，避免执行器空输出
        text = (state.get("final_response") or state.get("draft_response") or "").strip()
        if text:
            msgs = [text]
            pause_after = ["none"]
            delay_bucket = ["short"]
        else:
            # 避免“…”占位符进入下一轮（会让对话退化成无意义的省略号循环）
            msgs = ["（刚才生成回复失败了。可能是模型服务不可用/余额不足。你可以稍后再试。）"]
            pause_after = ["none"]
            delay_bucket = ["short"]

    # delay 计算：第一条包含“读+想+打字”的合成近似，其余以 bucket+pause 为主
    user_input = str(state.get("user_input") or "")
    base_first = 0.6 + min(1.8, len(user_input) * 0.03)  # 读
    think = 0.5 + min(2.0, len("".join(msgs)) * 0.01)
    stage_factor = _stage_delay_factor(stage)
    busy_factor = 1.0 + busyness * 1.0
    first_delay = (base_first + think) * stage_factor * busy_factor
    first_delay = _clamp(first_delay, 0.4, 6.0)
    
    # 记录编译过程
    log_computation(
        "ReplyCompiler",
        "延迟计算",
        inputs={
            "reply_plan": {
                "messages_count": len(msgs),
                "pause_after": pause_after,
                "delay_bucket": delay_bucket,
            },
            "state": {
                "stage": stage,
                "busyness": busyness,
                "user_input_len": len(user_input),
            },
        },
        intermediate_steps=[
            {
                "step": "首条延迟计算",
                "base_first": base_first,
                "think": think,
                "stage_factor": stage_factor,
                "busy_factor": busy_factor,
                "first_delay_raw": (base_first + think) * stage_factor * busy_factor,
                "first_delay_clamped": first_delay,
            }
        ],
    )

    delays: List[float] = []
    actions: List[str] = []
    
    delay_calculations = []

    for i, (b, p) in enumerate(zip(delay_bucket, pause_after)):
        bucket_sec = float(DELAY_BUCKET_SECONDS.get(b, 0.6))
        pause_sec = float(PAUSE_BONUS_SECONDS.get(p, 0.0))
        if i == 0:
            d = first_delay + bucket_sec * 0.3  # 首条 bucket 只占小头
            act = "idle" if b == "offline" else "typing"
        else:
            d = (bucket_sec + pause_sec) * stage_factor * busy_factor
            d = _clamp(d, 0.05, 60.0)
            act = "idle" if b == "offline" else "typing"
        delays.append(round(float(d), 2))
        actions.append(act)
        
        delay_calculations.append({
            "index": i,
            "delay_bucket": b,
            "pause_after": p,
            "bucket_seconds": bucket_sec,
            "pause_seconds": pause_sec,
            "final_delay": round(float(d), 2),
            "action": act,
        })
    
    log_computation(
        "ReplyCompiler",
        "延迟计算详情",
        outputs={
            "delays": delays,
            "actions": actions,
            "calculations": delay_calculations,
        },
    )

    meta: Dict[str, Any] = {
        "source": "reply_plan_compiler",
        "stage": stage,
        "busyness": busyness,
        "delay_bucket": delay_bucket,
        "pause_after": pause_after,
        "reply_plan_justification": reply_plan.get("justification"),
        # 便于 evaluator / 日志诊断：保留 planner 的结构化分配信息
        "reply_plan_messages_count": reply_plan.get("messages_count"),
        "reply_plan_must_cover_map": reply_plan.get("must_cover_map"),
        "reply_plan_first_message_role": reply_plan.get("first_message_role"),
    }

    result = {
        "messages": msgs,
        "delays": delays,
        "actions": actions,  # type: ignore[typeddict-item]
        "meta": meta,
    }
    
    log_computation(
        "ReplyCompiler",
        "编译完成",
        outputs={
            "processor_plan": {
                "messages_count": len(msgs),
                "total_delay": sum(delays),
                "meta": meta,
            },
        },
    )
    
    return result


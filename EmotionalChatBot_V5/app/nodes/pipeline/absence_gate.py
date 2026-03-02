"""
absence_gate.py — 宏观缺席门控节点

功能：
- 在 safety 检查通过之后、正式生成之前运行。
- 调用 HumanizationProcessor.calculate_absence() 判断 bot 是否会延迟回复。
- REAL_MODE_ENABLED=false（默认）时完全透明，不影响现有流程。
- REAL_MODE_ENABLED=true 时：
    - 若 calculate_absence() 返回 seconds > 0：将任务写入 DB，并在 state 中
      设置 absence_triggered=True，图路由将直接跳到 END（本次不生成回复）。
    - 若 seconds == 0（online）：透明通过，流程继续。
- 若 state["_scheduled_run"] 为 True（cron runner 调用），始终透明通过，
  避免到期任务再次触发缺席进入死循环。

DB 写入由外部注入的 db_manager（DBManager 实例）完成，为 None 时跳过写入（降级）。
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

from app.nodes.pipeline.processor import HumanizationProcessor
from app.state import AgentState

logger = logging.getLogger(__name__)

_REAL_MODE_KEY = "REAL_MODE_ENABLED"


def _is_real_mode() -> bool:
    return str(os.getenv(_REAL_MODE_KEY, "false")).strip().lower() in ("1", "true", "yes", "on")


async def absence_gate_node(state: AgentState, db_manager: Any = None) -> Dict[str, Any]:
    """
    异步纯代码节点，无 LLM 调用。返回字典：
      {"absence_triggered": False}               — 透明通过
      {"absence_triggered": True,
       "absence_delay_seconds": float,
       "absence_reason": str,
       "absence_sub_reason": str,
       "pending_task_id": str | None}             — 触发缺席
    """
    # ── cron runner 调用时直接透传，避免重复触发 ──────────────────────────────
    if state.get("_scheduled_run"):
        return {"absence_triggered": False}

    if not _is_real_mode():
        return {"absence_triggered": False}

    try:
        processor = HumanizationProcessor(state)
        dyn = processor.calculate_dynamics_modifiers()
        seconds, reason, sub_reason = processor.calculate_absence(dyn)
    except Exception as exc:
        logger.warning("[AbsenceGate] calculate_absence failed, falling through: %s", exc)
        return {"absence_triggered": False}

    if seconds <= 0.0 or reason == "online":
        return {"absence_triggered": False}

    logger.info(
        "[AbsenceGate] Absence triggered: reason=%s sub=%s delay=%.1fs (%.2fh)",
        reason, sub_reason, seconds, seconds / 3600.0,
    )

    # ── 写入 DB ────────────────────────────────────────────────────────────────
    task_id: Optional[str] = None
    if db_manager is not None:
        try:
            deliver_at = datetime.now(timezone.utc) + timedelta(seconds=seconds)
            user_external_id = str(state.get("user_id") or "")
            bot_id = str(state.get("bot_id") or "")
            user_message = str(state.get("user_input") or "")

            task_id = await db_manager.create_pending_response(
                user_external_id=user_external_id,
                bot_id=bot_id,
                user_message=user_message,
                deliver_at=deliver_at,
                absence_reason=reason,
                absence_sub_reason=sub_reason,
                absence_seconds=seconds,
            )
            logger.info("[AbsenceGate] DB pending task created: id=%s deliver_at=%s", task_id, deliver_at.isoformat())
        except Exception as e_db:
            logger.warning("[AbsenceGate] DB write failed: %s", e_db)
    else:
        logger.info("[AbsenceGate] No db_manager provided; skipping DB write.")

    return {
        "absence_triggered": True,
        "absence_delay_seconds": round(seconds, 1),
        "absence_reason": reason,
        "absence_sub_reason": sub_reason,
        "pending_task_id": task_id,
    }


def create_absence_gate_node(db_manager: Any = None) -> Callable[[AgentState], Any]:
    """工厂函数，返回绑定了 db_manager 的异步 absence_gate 节点函数。"""
    async def node(state: AgentState) -> dict:
        return await absence_gate_node(state, db_manager=db_manager)
    return node

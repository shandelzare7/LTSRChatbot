"""
evolver.py

6维关系演化引擎 (Relationship Engine)

动静分离：
- 静态的“信号判断标准”放在 `config/relationship_signals.yaml`
- 动态的 State 由本模块处理

双层处理：
- Node 1 (Analyzer): LLM 接收完整上下文 + YAML 标准，输出 JSON Deltas
- Node 2 (Updater): Python 应用阻尼公式（边际收益递减 + 背叛惩罚）更新 relationship_state

说明：
- 本文件对应 LangGraph 节点名：`evolver`（与文件名保持一致）。
- evolver 节点内部顺序执行：Analyzer -> Updater（强绑定，避免拆成两个节点造成混淆）。
- 注意：用户基础信息/画像抽取已迁移至 memory_manager，本文件不再做 basic_info / inferred_profile 写入。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.state import AgentState
from src.schemas import RelationshipAnalysis
from src.prompts.relationship import build_analyzer_prompt
from utils.llm_json import parse_json_from_llm
from utils.tracing import trace_if_enabled


REL_DIMS = ("closeness", "trust", "liking", "respect", "attractiveness", "power")

# ============================================================================
# 仅用于“6维数值校准”的参数（不引入额外参数/不新增功能）
# ============================================================================
REL_HI_CAP = 0.98  # 避免 1.0 顶格饱和锁死（DB里写了1.0也会被收口）

# Analyzer 输出 -3..+3 的“每级”基础增量（量纲核心旋钮）
# 取 0.06：+1=0.06, +3=0.18（之后还会过阻尼 + stage倍率 + max_step）
# ═══════════════════════════════════════════════════════════════════════════
# 速率控制：只需调整这一个值即可改变所有关系变化速率
# ═══════════════════════════════════════════════════════════════════════════
RATE_MULTIPLIER = 20.0  # 速率倍率（原5.0×4）：1.0=原始速率，越大关系/状态变化越快

# Analyzer 输出 -3..+3 的"每级"基础增量（量纲核心旋钮）
# 基准值：+1=0.006, +3=0.018（之后还会过阻尼 + stage倍率 + max_step）
# 实际值 = 基准值 × RATE_MULTIPLIER
DELTA_UNIT_BASE = 0.006
DELTA_UNIT = DELTA_UNIT_BASE * RATE_MULTIPLIER

# 按阶段对“正向推进”减速：对齐你给的轮次估算（早期快、后期极慢）
_STAGE_UP_MULT = {
    "initiating": 0.50,      # 10–40 轮级别
    "experimenting": 0.22,   # 60–200 轮级别
    "intensifying": 0.13,    # 200–600 轮级别
    "integrating": 0.014,    # 600–2000 轮级别
    "bonding": 0.006,        # bonding 后几乎不再继续“变更亲密”
}

# 负向变化允许更快（背叛/冲突下降通常比上升快）
_STAGE_DOWN_MULT = {
    "initiating": 1.00,
    "experimenting": 1.05,
    "intensifying": 1.10,
    "integrating": 1.15,
    "bonding": 1.20,
}

# 每轮每维最大步长（最终写回前的硬帽，进一步避免“几轮冲顶”）
# 基准值，实际值 = 基准值 × RATE_MULTIPLIER
_MAX_STEP_UP_BASE = {
    "closeness": 0.012,
    "trust": 0.012,
    "liking": 0.014,
    "respect": 0.010,
    "attractiveness": 0.014,
    "power": 0.010,
}
_MAX_STEP_DOWN_BASE = {
    "closeness": 0.030,
    "trust": 0.035,
    "liking": 0.032,
    "respect": 0.030,
    "attractiveness": 0.032,
    "power": 0.020,
}

# 应用速率倍率
_MAX_STEP_UP = {k: v * RATE_MULTIPLIER for k, v in _MAX_STEP_UP_BASE.items()}
_MAX_STEP_DOWN = {k: v * RATE_MULTIPLIER for k, v in _MAX_STEP_DOWN_BASE.items()}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _stage_key(x: Any) -> str:
    s = str(x or "").strip().lower()
    if not s:
        return "experimenting"
    if s == "initiation":
        return "initiating"
    if s == "experimentation":
        return "experimenting"
    if s == "intensification":
        return "intensifying"
    if s == "integration":
        return "integrating"
    if s == "termination":
        return "terminating"
    return s


def _ensure_relationship_defaults(state: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(state)
    rel = dict(s.get("relationship_state") or {})
    rel.setdefault("closeness", 0.3)
    rel.setdefault("trust", 0.3)
    rel.setdefault("liking", 0.3)
    rel.setdefault("respect", 0.3)
    # attractiveness 继承原 warmth（平滑过渡，若数据库已清洗可酌情移除）
    rel.setdefault("attractiveness", rel.get("warmth", 0.3))
    rel.setdefault("power", 0.5)

    # 关键：把 DB 里潜在的 1.0 收口到 REL_HI_CAP，避免饱和锁死
    for k in REL_DIMS:
        try:
            rel[k] = float(rel.get(k, 0.5))
        except Exception:
            rel[k] = 0.5
        rel[k] = round(_clamp(rel[k], 0.0, REL_HI_CAP), 4)

    s["relationship_state"] = rel
    s.setdefault("mood_state", {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0, "busyness": 0.0})
    s.setdefault("current_stage", "experimenting")
    s.setdefault("user_input", s.get("user_input") or "")
    return s


def calculate_damped_delta(current_score: float, raw_delta: float) -> float:
    """
    阻尼公式：实现边际收益递减和背叛惩罚。
    """
    try:
        cs = float(current_score)
    except Exception:
        cs = 0.5
    try:
        rd = float(raw_delta)
    except Exception:
        rd = 0.0

    if rd > 0:
        if cs >= 0.9:
            return rd * 0.1
        if cs >= 0.6:
            return rd * 0.5
        return rd * 1.0

    if rd < 0:
        if cs >= 0.8:
            return rd * 1.5
        return rd * 1.0

    return 0.0


def _normalize_delta(x: Any) -> float:
    """
    统一 delta 量纲：Analyzer 按 schema 输出 -3..+3 整数。
    这里把 -3..+3 映射到 [-DELTA_UNIT*3, +DELTA_UNIT*3]（再经阻尼+stage倍率+max_step）。
    实际值会根据 RATE_MULTIPLIER 自动调整。
    """
    try:
        v = float(x)
    except Exception:
        return 0.0

    # 主路径：-3..+3
    if -3.0 <= v <= 3.0:
        return float(v * DELTA_UNIT)

    # 兼容：模型偶尔输出 [-1,1] 小数 → 视作 [-3,3] 的缩放
    if abs(v) <= 1.0:
        return float(_clamp(v * 3.0, -3.0, 3.0) * DELTA_UNIT)

    # 兼容：百分制/异常值
    if abs(v) <= 100.0:
        return float(_clamp(v / 10.0, -3.0, 3.0) * DELTA_UNIT)

    return float(_clamp(v, -3.0, 3.0) * DELTA_UNIT)


_GREETING_PAT = re.compile(
    r"^\s*(hi|hello|hey|你好|您好|嗨|哈喽|早上好|中午好|晚上好|晚安)"
    r"([\s,，!！。．]*"
    r"(很高兴认识你|认识你很高兴|见到你很高兴|见到你真好|很高兴见到你))?"
    r"[\s,，!！。．]*$",
    re.IGNORECASE,
)


def _is_low_info_greeting(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    if len(t) > 32:
        return False
    return bool(_GREETING_PAT.match(t))


# -------------------------------------------------------------------
# Node 1: Analyzer (LLM) —— 只分析，不改分
# -------------------------------------------------------------------


def create_relationship_analyzer_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    @trace_if_enabled(
        name="Relationship/Analyzer",
        run_type="chain",
        tags=["node", "relationship", "analyzer"],
        metadata={"state_outputs": ["latest_relationship_analysis", "relationship_deltas"]},
    )
    def node(state: AgentState) -> dict:
        safe = _ensure_relationship_defaults(state)

        sys_prompt = build_analyzer_prompt(safe)
        user_msg = safe.get("user_input") or ""

        chat_buffer = safe.get("chat_buffer") or []
        body_messages = list(chat_buffer[-20:])

        raw = ""
        data: Dict[str, Any] | None = None
        analysis = None
        try:
            if hasattr(llm_invoker, "with_structured_output"):
                structured = llm_invoker.with_structured_output(RelationshipAnalysis)
                analysis = structured.invoke(
                    [SystemMessage(content=sys_prompt), *body_messages, HumanMessage(content=user_msg)]
                )
        except Exception:
            analysis = None
        if analysis is None:
            try:
                resp = llm_invoker.invoke(
                    [SystemMessage(content=sys_prompt), *body_messages, HumanMessage(content=user_msg)]
                )
                raw = (getattr(resp, "content", str(resp)) or "").strip()
                raw = str(raw) if raw else ""
                data = parse_json_from_llm(raw) if raw else None
                if data is None and raw:
                    data = json.loads(raw)
            except Exception:
                data = None

            if data is None:
                preview = (raw[:200] + "…") if len(raw) > 200 else raw
                print(f"[Relationship Analyzer] parse error (raw empty or invalid JSON), preview: {preview!r}")
                data = {
                    "thought_process": "Fallback: unable to parse model output; assume neutral.",
                    "detected_signals": [],
                    "deltas": {k: 0 for k in REL_DIMS},
                    "completed_task_ids": [],
                    "attempted_task_ids": [],
                }

            try:
                analysis = RelationshipAnalysis.model_validate(data)
            except Exception:
                analysis = RelationshipAnalysis.parse_obj(data)  # type: ignore[attr-defined]

        analysis_dict = analysis.model_dump() if hasattr(analysis, "model_dump") else analysis.dict()
        deltas_dict = analysis.deltas.model_dump() if hasattr(analysis.deltas, "model_dump") else analysis.deltas.dict()

        return {
            "latest_relationship_analysis": analysis_dict,
            "relationship_deltas": deltas_dict,
        }

    return node


# -------------------------------------------------------------------
# Node 2: Updater (Math) —— 应用阻尼公式，更新 relationship_state
# -------------------------------------------------------------------


def create_relationship_updater_node() -> Callable[[AgentState], dict]:
    @trace_if_enabled(
        name="Relationship/Updater",
        run_type="chain",
        tags=["node", "relationship", "updater"],
        metadata={"state_outputs": ["relationship_state", "relationship_deltas_applied"]},
    )
    def node(state: AgentState) -> dict:
        safe = _ensure_relationship_defaults(state)

        rel: Dict[str, float] = dict(safe.get("relationship_state") or {})
        raw_deltas = safe.get("relationship_deltas") or {}

        stage = _stage_key(safe.get("current_stage"))
        up_mult = float(_STAGE_UP_MULT.get(stage, _STAGE_UP_MULT["experimenting"]))
        down_mult = float(_STAGE_DOWN_MULT.get(stage, 1.05))

        user_text = str(safe.get("user_input") or "").strip()
        try:
            conv_len = len(list(safe.get("chat_buffer") or []))
        except Exception:
            conv_len = 0

        greeting_gate = _is_low_info_greeting(user_text) and conv_len <= 2

        applied: Dict[str, float] = {}
        for dim in REL_DIMS:
            score = float(rel.get(dim, 0.5))
            score = _clamp(score, 0.0, REL_HI_CAP)

            raw_val = raw_deltas.get(dim, 0)
            raw_delta = _normalize_delta(raw_val)

            if greeting_gate:
                # attractiveness 暂不参与门控判断
                if dim in ("liking", "respect") and raw_delta > 0:
                    raw_delta *= 0.25
                # 保留轻微启动意图，但按新量纲缩小
                if dim in ("closeness", "trust") and abs(raw_delta) < 1e-6:
                    raw_delta = 0.003 * RATE_MULTIPLIER

            if raw_delta == 0.0:
                applied[dim] = 0.0
                continue

            real_change = float(calculate_damped_delta(score, raw_delta))

            # 阶段倍率：只依赖 current_stage
            if real_change > 0:
                real_change *= up_mult
                lim = float(_MAX_STEP_UP.get(dim, _MAX_STEP_UP_BASE.get(dim, 0.012) * RATE_MULTIPLIER))
                real_change = _clamp(real_change, -lim, lim)
            else:
                real_change *= down_mult
                lim = float(_MAX_STEP_DOWN.get(dim, _MAX_STEP_DOWN_BASE.get(dim, 0.030) * RATE_MULTIPLIER))
                real_change = _clamp(real_change, -lim, lim)

            if greeting_gate and dim in ("liking", "respect") and real_change > 0:
                real_change = min(real_change, 0.008 * RATE_MULTIPLIER)

            new_score = _clamp(score + real_change, 0.0, REL_HI_CAP)
            rel[dim] = round(new_score, 4)
            applied[dim] = round(real_change, 4)

        if greeting_gate:
            print(f"[Evolver] greeting_gate applied (stage={stage}, conv_len={conv_len}, user_input={user_text!r})")
        return {"relationship_state": rel, "relationship_deltas_applied": applied}

    return node


# -------------------------------------------------------------------
# Relationship Engine (Analyzer -> Updater)
# -------------------------------------------------------------------


def create_relationship_engine_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    analyzer = create_relationship_analyzer_node(llm_invoker)
    updater = create_relationship_updater_node()

    @trace_if_enabled(
        name="Relationship/Engine",
        run_type="chain",
        tags=["node", "relationship", "engine"],
        metadata={
            "state_outputs": [
                "latest_relationship_analysis",
                "relationship_deltas",
                "relationship_deltas_applied",
                "relationship_state",
            ]
        },
    )
    def node(state: AgentState) -> dict:
        out: Dict[str, Any] = {}
        out.update(analyzer(state))
        merged = dict(state)
        merged.update(out)
        out.update(updater(merged))
        return out

    return node


# -------------------------------------------------------------------
# 任务完成检测与会话池更新（原样保留）
# -------------------------------------------------------------------


# 规则兜底：Bot 回复中若包含「问名字」语义，则视为 ask_user_name 已完成（不依赖 Analyzer JSON）
_ASK_NAME_PAT = re.compile(
    r"(你|您)(叫|称呼|贵姓)|怎么(称呼|叫你)|(请问|能问一下)(你|您)(的)?(名字|称呼)|叫什么名字"
)


def _get_task_completion_from_analysis(state: Dict[str, Any]) -> tuple:
    analysis = state.get("latest_relationship_analysis") or {}
    completed = analysis.get("completed_task_ids")
    attempted = analysis.get("attempted_task_ids")

    completed_ids = {str(x) for x in (completed or []) if str(x).strip()}
    attempted_ids = {str(x) for x in (attempted or []) if str(x).strip()}

    # 规则兜底：若本轮 Bot 回复中明确出现「问名字」语义，则视为 ask_user_name 已完成
    bot_text = (state.get("final_response") or state.get("draft_response") or "").strip()
    if not bot_text and isinstance(state.get("final_segments"), list):
        bot_text = " ".join(str(s or "").strip() for s in state.get("final_segments") or []).strip()
    if bot_text and _ASK_NAME_PAT.search(bot_text):
        completed_ids = completed_ids | {"ask_user_name"}
        attempted_ids = attempted_ids | {"ask_user_name"}

    tasks_for_lats = state.get("tasks_for_lats") or []
    if isinstance(tasks_for_lats, list):
        all_ids = {str(t.get("id")) for t in tasks_for_lats if isinstance(t, dict) and t.get("id")}
        if not attempted_ids and all_ids:
            attempted_ids = set(all_ids)

    return (completed_ids, attempted_ids)


def _detect_completed_tasks_and_replenish(state: Dict[str, Any]) -> Dict[str, Any]:
    from app.nodes.task_planner import (
        BACKLOG_SESSION_TARGET,
        CURRENT_SESSION_TASKS_CAP,
        _sample_backlog_excluding,
    )

    current_session_tasks: List[Dict[str, Any]] = list(state.get("current_session_tasks") or [])
    bot_task_list: List[Dict[str, Any]] = list(state.get("bot_task_list") or [])
    tasks_for_lats = state.get("tasks_for_lats") or []

    completed_ids, attempted_ids = _get_task_completion_from_analysis(state)

    current_session_tasks = [t for t in current_session_tasks if str(t.get("id")) not in completed_ids]

    cleaned: List[Dict[str, Any]] = []
    for t in current_session_tasks:
        tt = str(t.get("task_type") or "").strip()
        if tt in ("backlog", "immediate"):
            cleaned.append(t)
    current_session_tasks = cleaned

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bumped = 0
    if attempted_ids:
        for t in current_session_tasks:
            tid = str(t.get("id") or "")
            if not tid or tid in completed_ids or tid not in attempted_ids:
                continue
            try:
                t["attempt_count"] = int(t.get("attempt_count", 0) or 0) + 1
                t["last_attempt_at"] = now_iso
                bumped += 1
            except Exception:
                pass
        for t in bot_task_list:
            tid = str(t.get("id") or "")
            if not tid or tid in completed_ids or tid not in attempted_ids:
                continue
            try:
                t["attempt_count"] = int(t.get("attempt_count", 0) or 0) + 1
                t["last_attempt_at"] = now_iso
            except Exception:
                pass

    def _dec_ttl_and_filter(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for t in tasks:
            tt = str(t.get("task_type") or "").strip()
            if tt != "immediate":
                out.append(t)
                continue
            try:
                ttl = int(t.get("ttl_turns", 0) or 0)
            except Exception:
                ttl = 0
            if ttl <= 0:
                ttl = 1
            ttl -= 1
            if ttl <= 0:
                continue
            nt = dict(t)
            nt["ttl_turns"] = ttl
            out.append(nt)
        return out

    current_session_tasks = _dec_ttl_and_filter(current_session_tasks)

    for tid in completed_ids:
        for t in list(bot_task_list):
            if str(t.get("id")) == tid and str(t.get("task_type") or "").strip() == "backlog":
                bot_task_list.remove(t)
                break

    backlog_items = [t for t in current_session_tasks if str(t.get("task_type") or "").strip() == "backlog"]
    immediate_items = [t for t in current_session_tasks if str(t.get("task_type") or "").strip() == "immediate"]

    backlog_in_pool = len(backlog_items)
    trimmed_backlog = 0
    if backlog_in_pool > int(BACKLOG_SESSION_TARGET):
        keep = backlog_items[: int(BACKLOG_SESSION_TARGET)]
        trimmed_backlog = backlog_in_pool - len(keep)
        current_session_tasks = keep + immediate_items
        backlog_items = keep
        backlog_in_pool = len(backlog_items)

    need_backlog = max(0, int(BACKLOG_SESSION_TARGET) - int(backlog_in_pool))
    existing_ids = {str(t.get("id")) for t in current_session_tasks if t.get("id")}
    backlog_new = _sample_backlog_excluding(bot_task_list, existing_ids, need_backlog) if need_backlog > 0 else []
    added_backlog = 0

    def _norm(t: Dict[str, Any], prefix: str, i: int) -> Dict[str, Any]:
        return {
            "id": t.get("id") or f"{prefix}_{i}",
            "description": str(t.get("description") or t.get("id") or "").strip() or "（无描述）",
            "task_type": str(t.get("task_type") or "other"),
        }

    for i, t in enumerate(backlog_new):
        current_session_tasks.append(_norm(t, "backlog", len(current_session_tasks)))
        added_backlog += 1

    carry_max = int(CURRENT_SESSION_TASKS_CAP)
    if len(current_session_tasks) > carry_max:
        scored: List[tuple] = []
        for idx, t in enumerate(current_session_tasks):
            tt = str(t.get("task_type") or "").strip()
            is_immediate = 1 if tt == "immediate" else 0
            try:
                imp = float(t.get("importance", 0.5) or 0.5)
            except Exception:
                imp = 0.5
            try:
                ac = int(t.get("attempt_count", 0) or 0)
            except Exception:
                ac = 0
            score = (is_immediate, min(10, ac), imp, idx)
            scored.append((score, str(t.get("id") or ""), idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        keep_ids = {tid for _, tid, _ in scored[:carry_max] if tid}
        current_session_tasks = [t for t in current_session_tasks if str(t.get("id") or "") in keep_ids]

    if len(current_session_tasks) > CURRENT_SESSION_TASKS_CAP:
        current_session_tasks = current_session_tasks[-CURRENT_SESSION_TASKS_CAP:]

    try:
        if completed_ids or attempted_ids or added_backlog or trimmed_backlog or bumped:
            print(
                f"[Evolver] completed={len(completed_ids)} attempted={len(attempted_ids)} bumped={bumped} "
                f"backlog_pool={backlog_in_pool}->{sum(1 for t in current_session_tasks if str(t.get('task_type') or '').strip() == 'backlog')}, "
                f"trimmed_backlog={trimmed_backlog}, added_backlog={added_backlog}, session_tasks={len(current_session_tasks)}"
            )
    except Exception:
        pass

    try:
        urgent_in_lats = [
            t
            for t in (tasks_for_lats or [])
            if isinstance(t, dict) and (t.get("task_type") == "urgent" or t.get("is_urgent"))
        ]
        if urgent_in_lats:
            urgent_ids = {str(t.get("id") or "") for t in urgent_in_lats}
            urgent_completed = urgent_ids & completed_ids
            urgent_attempted = urgent_ids & attempted_ids
            urgent_not_done = urgent_ids - completed_ids
            print(
                f"[URGENT TASK REPORT] ========================================\n"
                f"[URGENT TASK REPORT]  Total urgent tasks: {len(urgent_in_lats)}\n"
                f"[URGENT TASK REPORT]  Completed: {sorted(urgent_completed) if urgent_completed else '(none)'}\n"
                f"[URGENT TASK REPORT]  Attempted but not completed: {sorted(urgent_attempted - urgent_completed) if (urgent_attempted - urgent_completed) else '(none)'}\n"
                f"[URGENT TASK REPORT]  Not attempted: {sorted(urgent_not_done - urgent_attempted) if (urgent_not_done - urgent_attempted) else '(none)'}\n"
                f"[URGENT TASK REPORT]  Descriptions: {[str(t.get('description', '') or '')[:60] for t in urgent_in_lats]}\n"
                f"[URGENT TASK REPORT] ========================================"
            )
    except Exception:
        pass

    return {
        "current_session_tasks": current_session_tasks,
        "bot_task_list": bot_task_list,
        "completed_task_ids": list(completed_ids),
        "attempted_task_ids": list(attempted_ids),
    }


def create_evolver_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    base_engine = create_relationship_engine_node(llm_invoker)

    @trace_if_enabled(
        name="Evolver",
        run_type="chain",
        tags=["node", "evolver", "relationship", "task_pool"],
        metadata={
            "state_outputs": [
                "latest_relationship_analysis",
                "relationship_deltas",
                "relationship_deltas_applied",
                "relationship_state",
                "current_session_tasks",
                "bot_task_list",
                "completed_task_ids",
                "attempted_task_ids",
            ]
        },
    )
    def node(state: AgentState) -> dict:
        out = base_engine(state)
        merged = dict(state)
        merged.update(out)
        out.update(_detect_completed_tasks_and_replenish(merged))
        return out

    return node
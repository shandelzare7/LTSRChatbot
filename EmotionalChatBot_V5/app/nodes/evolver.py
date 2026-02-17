"""
6维关系演化引擎 (Relationship Engine)

动静分离：
- 静态的"信号判断标准"放在 `config/relationship_signals.yaml`
- 动态的 State 由本模块处理

双层处理：
- Node 1 (Analyzer): LLM 接收完整上下文 + YAML 标准，输出 JSON Deltas
- Node 2 (Updater): Python 应用阻尼公式（边际收益递减 + 背叛惩罚）更新 relationship_state

说明：
- 本文件对应 LangGraph 节点名：`evolver`（与文件名保持一致）。
- evolver 节点内部顺序执行：Analyzer -> Updater（强绑定，避免拆成两个节点造成混淆）。
"""

import json
import random
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.state import AgentState
from src.schemas import RelationshipAnalysis
from src.prompts.relationship import build_analyzer_prompt
from utils.tracing import trace_if_enabled
from utils.llm_json import parse_json_from_llm


REL_DIMS = ("closeness", "trust", "liking", "respect", "warmth", "power")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _ensure_relationship_defaults(state: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(state)
    rel = dict(s.get("relationship_state") or {})
    rel.setdefault("closeness", 0.3)  # 默认值 0.3（而非 0.0）
    rel.setdefault("trust", 0.3)
    rel.setdefault("liking", 0.3)
    rel.setdefault("respect", 0.3)
    rel.setdefault("warmth", 0.3)
    rel.setdefault("power", 0.5)
    s["relationship_state"] = rel

    s.setdefault("mood_state", {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0, "busyness": 0.0})
    s.setdefault("current_stage", "experimenting")
    s.setdefault("user_input", s.get("user_input") or "")
    return s


def calculate_damped_delta(current_score: float, raw_delta: float) -> float:
    """
    阻尼公式：实现边际收益递减和背叛惩罚。
    - 正向：越高越难涨
    - 负向：高信任/高亲密被破坏时更痛（背叛惩罚）
    
    注意：current_score 和返回值都是 0-1 范围。
    raw_delta 也是 0-1 范围的增量（例如 0.1 表示 +10%）。
    """
    try:
        cs = float(current_score)
    except Exception:
        cs = 0.5  # 0-1 范围的中性值
    try:
        rd = float(raw_delta)
    except Exception:
        rd = 0.0

    if rd > 0:
        if cs >= 0.9:  # 0-1 范围的高值
            return rd * 0.1
        if cs >= 0.6:
            return rd * 0.5
        return rd * 1.0

    if rd < 0:
        if cs >= 0.8:  # 0-1 范围的高值
            return rd * 1.5
        return rd * 1.0

    return 0.0


def _normalize_delta(x: Any) -> float:
    """
    统一 delta 量纲：relationship_state 为 0-1，增量也需在合理步长内。
    Analyzer 按 prompt/schema 输出 -3..+3 整数（0 无变化，±1 轻微，±2 中等，±3 强烈），
    此处统一映射到 ±0.3：先按 -3..3 归一化，再兼容其它量纲。
    """
    try:
        v = float(x)
    except Exception:
        return 0.0
    # 先处理 Analyzer 标准输出：-3..+3 整档 → 步长 -0.3..+0.3（与 prompt/schema 一致）
    if -3.0 <= v <= 3.0:
        return v / 10.0
    # 已是 0-1 或 ±1 内的小数（其它来源）则原样
    if abs(v) <= 1.0:
        return v
    if abs(v) <= 100.0:
        return v / 100.0
    return float(_clamp(v, -1.0, 1.0))


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
    # 过长通常不是纯寒暄
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
        sys_prompt = build_analyzer_prompt(safe)  # 已含 summary + retrieved 记忆
        user_msg = safe.get("user_input") or ""
        # chat_buffer 分条放正文，非 system
        chat_buffer = safe.get("chat_buffer") or []
        body_messages = list(chat_buffer[-20:])

        # LLM 输出：严格 JSON，先用 parse_json_from_llm 抽取，失败时打日志并用 fallback
        raw = ""
        data = None
        try:
            resp = llm_invoker.invoke(
                [SystemMessage(content=sys_prompt), *body_messages, HumanMessage(content=user_msg)]
            )
            raw = (getattr(resp, "content", str(resp)) or "").strip()
            raw = str(raw) if raw else ""
            data = parse_json_from_llm(raw) if raw else None
            if data is None and raw:
                data = json.loads(raw)
        except Exception as e:
            pass
        if data is None:
            preview = (raw[:200] + "…") if len(raw) > 200 else raw
            print(f"[Relationship Analyzer] parse error (raw empty or invalid JSON), preview: {preview!r}")
            data = {
                "thought_process": "Fallback: unable to parse model output; assume neutral.",
                "detected_signals": [],
                "deltas": {k: 0 for k in REL_DIMS},
            }

        # Pydantic 校验
        try:
            analysis = RelationshipAnalysis.model_validate(data)  # pydantic v2
        except Exception:
            # pydantic v1 兼容
            analysis = RelationshipAnalysis.parse_obj(data)  # type: ignore[attr-defined]

        analysis_dict = analysis.model_dump() if hasattr(analysis, "model_dump") else analysis.dict()
        deltas_dict = analysis.deltas.model_dump() if hasattr(analysis.deltas, "model_dump") else analysis.deltas.dict()

        result: Dict[str, Any] = {
            "latest_relationship_analysis": analysis_dict,
            "relationship_deltas": deltas_dict,  # raw deltas（-3..3）
        }

        # --- User Profiling: merge basic_info_updates & new_inferred_entries ---
        basic_updates = analysis_dict.get("basic_info_updates") or {}
        inferred_updates = analysis_dict.get("new_inferred_entries") or {}
        if basic_updates and isinstance(basic_updates, dict):
            existing_basic = dict(safe.get("user_basic_info") or {})
            for k in ("name", "age", "gender", "occupation", "location"):
                new_val = basic_updates.get(k)
                if new_val is not None and str(new_val).strip():
                    old_val = existing_basic.get(k)
                    if old_val is None or (isinstance(old_val, str) and not old_val.strip()):
                        existing_basic[k] = new_val
            result["user_basic_info"] = existing_basic
        if inferred_updates and isinstance(inferred_updates, dict):
            existing_profile = dict(safe.get("user_inferred_profile") or {})
            for k, v in inferred_updates.items():
                if isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip():
                    existing_profile[k.strip()] = v.strip()
            result["user_inferred_profile"] = existing_profile
        # 可观测：本轮是否写入了画像
        if basic_updates or inferred_updates:
            b_keys = list(basic_updates.keys()) if isinstance(basic_updates, dict) else []
            i_keys = list(inferred_updates.keys()) if isinstance(inferred_updates, dict) else []
            print(f"[Evolver] user profiling: basic_info_updates={b_keys}, new_inferred_entries={i_keys}")

        return result

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
        user_text = str(safe.get("user_input") or "").strip()
        try:
            conv_len = len(list(safe.get("chat_buffer") or []))
        except Exception:
            conv_len = 0

        # 低信息寒暄闸门：避免把礼貌性问候当成强积极信号，导致 liking/warmth 一上来顶格跳变
        greeting_gate = _is_low_info_greeting(user_text) and conv_len <= 2

        applied: Dict[str, float] = {}
        for dim in REL_DIMS:
            score = float(rel.get(dim, 0.5))  # 0-1 范围，默认 0.5
            raw_val = raw_deltas.get(dim, 0)
            raw_delta = _normalize_delta(raw_val)

            if greeting_gate:
                # 对"喜欢/温暖/尊重"的正向增量降权（礼貌寒暄 ≠ 强烈欣赏）
                if dim in ("liking", "warmth", "respect") and raw_delta > 0:
                    raw_delta *= 0.35
                # 对"熟悉/信任"给极小稳定增量（更符合现实：先熟一点，再慢慢喜欢）
                if dim in ("closeness", "trust") and abs(raw_delta) < 1e-6:
                    raw_delta = 0.02

            if raw_delta == 0.0:
                applied[dim] = 0.0
                continue

            real_change = float(calculate_damped_delta(score, raw_delta))
            # 进一步保护：寒暄阶段不允许 liking/warmth 发生"大跃迁"
            if greeting_gate and dim in ("liking", "warmth", "respect") and real_change > 0:
                real_change = min(real_change, 0.06)
            new_score = _clamp(score + real_change, 0.0, 1.0)  # 0-1 范围
            rel[dim] = round(new_score, 4)  # 保留更多小数位
            applied[dim] = round(real_change, 4)

        if greeting_gate:
            print(f"[Evolver] greeting_gate applied (conv_len={conv_len}, user_input={user_text!r})")
        print("[Evolver] done")
        return {
            "relationship_state": rel,
            "relationship_deltas_applied": applied,
        }

    return node


# -------------------------------------------------------------------
# Relationship Engine (Analyzer -> Updater) —— 对外只暴露一个节点，避免混淆
# 位置建议：processor 之后（更新下一轮关系分数，不影响当前轮生成）
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
        out = {}
        out.update(analyzer(state))
        # Updater 需要读取 analyzer 写入的 deltas / analysis，所以把 state+out 合并后再算
        merged = dict(state)
        merged.update(out)
        out.update(updater(merged))
        return out

    return node


# -------------------------------------------------------------------
# 任务完成检测与会话池更新
# 纯 Python 逻辑：依赖 LATS/ReplyPlanner 回写的 completed_task_ids / attempted_task_ids，
# 不额外调用 LLM。
# -------------------------------------------------------------------

def _detect_completed_tasks_and_replenish(
    state: Dict[str, Any],
    llm_invoker: Any,
) -> Dict[str, Any]:
    """
    根据 LATS/ReplyPlanner 回写的 completed_task_ids / attempted_task_ids 结算任务；
    移除已完成的，对 backlog 类型的从 bot_task_list 也移除；
    递补：再补 backlog 进 current_session_tasks（daily 不进入持久化池）。
    """
    from app.nodes.task_planner import (
        BACKLOG_SESSION_TARGET,
        CURRENT_SESSION_TASKS_CAP,
        _sample_backlog_excluding,
    )

    current_session_tasks: List[Dict[str, Any]] = list(state.get("current_session_tasks") or [])
    bot_task_list: List[Dict[str, Any]] = list(state.get("bot_task_list") or [])
    tasks_for_lats = state.get("tasks_for_lats") or []
    final_response = (state.get("final_response") or state.get("draft_response") or "").strip()
    user_input = (state.get("user_input") or "").strip()

    # 使用 LATS/ReplyPlanner 的结构化回写；无结构化字段时保守处理（不额外调 LLM）
    completed_ids: set = set()
    attempted_ids: set = set()
    try:
        c = state.get("completed_task_ids")
        a = state.get("attempted_task_ids")
        if isinstance(c, list):
            completed_ids = {str(x) for x in c if str(x).strip()}
        if isinstance(a, list):
            attempted_ids = {str(x) for x in a if str(x).strip()}
    except Exception:
        pass

    # attempted_ids fallback：没有显式 attempted，就用 tasks_for_lats 的 id 当作"尝试过"
    if not attempted_ids and isinstance(tasks_for_lats, list):
        for t in tasks_for_lats:
            if isinstance(t, dict) and t.get("id"):
                attempted_ids.add(str(t.get("id")))

    # 从 current_session_tasks 中移除已完成的
    current_session_tasks = [t for t in current_session_tasks if str(t.get("id")) not in completed_ids]

    # 清理遗留/非法类型：daily 不应进入持久化池；仅允许 backlog/immediate 常驻
    cleaned: List[Dict[str, Any]] = []
    for t in current_session_tasks:
        tt = str(t.get("task_type") or "").strip()
        if tt in ("backlog", "immediate"):
            cleaned.append(t)
    current_session_tasks = cleaned

    # 对"尝试过但未完成"的任务加急：attempt_count += 1，last_attempt_at=now（写回 DB 的 bot_task_list）
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

    # immediate 任务 TTL：每轮结算时递减；到期则移除（immediate 允许跨轮累加，但不应无限膨胀）
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
                # 没有 ttl 的 immediate，默认给 1 轮寿命（避免永久堆积）
                ttl = 1
            ttl -= 1
            if ttl <= 0:
                continue
            nt = dict(t)
            nt["ttl_turns"] = ttl
            out.append(nt)
        return out

    current_session_tasks = _dec_ttl_and_filter(current_session_tasks)

    # 已完成的 backlog 从 bot_task_list 中移除（未完成的会随 save_turn 写回 DB）
    backlog_task_type = ("backlog",)
    for tid in completed_ids:
        for t in list(bot_task_list):
            if str(t.get("id")) == tid and (t.get("task_type") or "").strip() in backlog_task_type:
                bot_task_list.remove(t)
                break

    # backlog 池容量控制：历史 bug 可能导致 backlog 在 current_session_tasks 中膨胀；
    # 这里强制收敛到固定目标数，只在"数量不足"时补齐。
    backlog_items = [t for t in current_session_tasks if str(t.get("task_type") or "").strip() == "backlog"]
    immediate_items = [t for t in current_session_tasks if str(t.get("task_type") or "").strip() == "immediate"]
    backlog_in_pool = len(backlog_items)
    trimmed_backlog = 0
    if backlog_in_pool > int(BACKLOG_SESSION_TARGET):
        # 保持稳定：保留最早进入池子的 backlog（列表前面的），其余丢弃
        keep = backlog_items[: int(BACKLOG_SESSION_TARGET)]
        trimmed_backlog = backlog_in_pool - len(keep)
        current_session_tasks = keep + immediate_items
        backlog_items = keep
        backlog_in_pool = len(backlog_items)

    need_backlog = max(0, int(BACKLOG_SESSION_TARGET) - int(backlog_in_pool))
    added_backlog = 0
    existing_ids = {str(t.get("id")) for t in current_session_tasks if t.get("id")}
    backlog_new = _sample_backlog_excluding(bot_task_list, existing_ids, need_backlog) if need_backlog > 0 else []

    def _norm(t: Dict[str, Any], prefix: str, i: int) -> Dict[str, Any]:
        return {
            "id": t.get("id") or f"{prefix}_{i}",
            "description": str(t.get("description") or t.get("id") or "").strip() or "（无描述）",
            "task_type": str(t.get("task_type") or "other"),
        }

    for i, t in enumerate(backlog_new):
        current_session_tasks.append(_norm(t, "backlog", len(current_session_tasks)))
        added_backlog += 1

    # carry 硬上限：超过则按 priority + recency 裁剪尾部
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
            # 越大越保留：immediate 优先，其次 attempt_count/importance；recency 用 idx（越靠后越新）
            score = (is_immediate, min(10, ac), imp, idx)
            scored.append((score, str(t.get("id") or ""), idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        keep_ids = set()
        # 先选够 carry_max
        for _, tid, _ in scored[:carry_max]:
            if tid:
                keep_ids.add(tid)
        # 保持原顺序
        current_session_tasks = [t for t in current_session_tasks if str(t.get("id") or "") in keep_ids]

    if len(current_session_tasks) > CURRENT_SESSION_TASKS_CAP:
        current_session_tasks = current_session_tasks[-CURRENT_SESSION_TASKS_CAP:]

    # Debug breadcrumb for logs: task settlement and replenishment
    try:
        if completed_ids or attempted_ids or added_backlog or trimmed_backlog or bumped:
            print(
                f"[Evolver] completed={len(completed_ids)} attempted={len(attempted_ids)} bumped={bumped} "
                f"backlog_pool={backlog_in_pool}->{sum(1 for t in current_session_tasks if str(t.get('task_type') or '').strip() == 'backlog')}, "
                f"trimmed_backlog={trimmed_backlog}, added_backlog={added_backlog}, session_tasks={len(current_session_tasks)}"
            )
    except Exception:
        pass

    # 紧急任务报告：高亮输出紧急任务完成情况
    try:
        urgent_in_lats = [
            t for t in (tasks_for_lats or [])
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
                f"[URGENT TASK REPORT]  Descriptions: {[t.get('description', '')[:60] for t in urgent_in_lats]}\n"
                f"[URGENT TASK REPORT] ========================================"
            )
    except Exception:
        pass

    return {
        "current_session_tasks": current_session_tasks,
        "bot_task_list": bot_task_list,
    }


def create_evolver_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """
    Graph 入口（与文件名一致）：
    evolver = RelationshipEngine = Analyzer -> Updater，再执行任务完成检测与 current_session_tasks/bot_task_list 更新。
    """
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
            ]
        },
    )
    def node(state: AgentState) -> dict:
        out = base_engine(state)
        merged = dict(state)
        merged.update(out)
        task_updates = _detect_completed_tasks_and_replenish(merged, llm_invoker)
        out.update(task_updates)
        return out

    return node

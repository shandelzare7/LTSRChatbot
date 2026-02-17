"""
TaskPlanner 节点：在 LATS 之前运行。
- 先组装候选任务列表（backlog + immediate + daily），再一次 LLM 调用同时完成：
  1) 预算规划：word_budget (0-60)、task_budget_max (0-2)
  2) 任务选择：从候选列表中选出最相关的 2 个 + 1 个随机任务（返回索引）
- 紧急任务（urgent）绕过 LLM 直接注入。
- 输出 tasks_for_lats / task_budget_max / word_budget / completion_temperature 供 LATS 使用。
"""
from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.state import AgentState
from utils.llm_json import parse_json_from_llm
from utils.tracing import trace_if_enabled
from utils.yaml_loader import get_project_root, load_yaml
from utils.prompt_helpers import format_stage_for_llm

# 过滤"系统性/助手味"任务（避免把 LATS 拉回"助手味"）
try:
    from app.core.bot_creation_llm import _is_systemic_backlog_task as _is_systemic_task_desc  # type: ignore
except Exception:
    def _is_systemic_task_desc(desc: str) -> bool:  # type: ignore
        return False

# 日常任务库：优先从 config/daily_tasks.yaml 加载，失败时用内置兜底
def _load_daily_pool() -> List[Dict[str, Any]]:
    try:
        root = get_project_root()
        path = root / "config" / "daily_tasks.yaml"
        if path.exists():
            data = load_yaml(path)
            raw = (data or {}).get("tasks") or []
            if isinstance(raw, list) and raw:
                out = []
                for i, t in enumerate(raw):
                    if not isinstance(t, dict):
                        continue
                    desc = str(t.get("description") or "").strip()
                    if not desc:
                        continue
                    out.append({
                        "id": str(t.get("id") or f"daily_{i}"),
                        "description": desc,
                        "task_type": "daily",
                    })
                if out:
                    return out
    except Exception:
        pass
    return [
        {"id": "daily_echo", "description": "对对方刚说的点做一点共鸣或接话（可用问句也可不用）", "task_type": "daily"},
        {"id": "daily_close", "description": "用一句话结束本轮并留一个小钩子（不是问题也行）", "task_type": "daily"},
    ]


DAILY_POOL: List[Dict[str, Any]] = _load_daily_pool()

# 理解类 task_type（最多保留 1 个）
UNDERSTANDING_TYPES = {"clarify", "ask_scope", "ask_example", "confirm_gap"}


def _parse_dt(s: Any) -> Optional[datetime]:
    if not s:
        return None
    try:
        t = str(s).strip()
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        return datetime.fromisoformat(t)
    except Exception:
        return None


def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
    except Exception:
        return default
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# 单次 LLM 调用：预算规划 + 任务选择
# ---------------------------------------------------------------------------

def _plan_and_select_with_llm(
    state: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    llm_invoker: Any,
) -> Tuple[int, int, List[int]]:
    """
    单次 LLM 调用同时完成预算规划 + 任务选择：
    返回 (word_budget, task_budget_max, selected_indices)
    - word_budget: 0..60
    - task_budget_max: 0..2
    - selected_indices: 最多 3 个候选索引（前 2 个最相关 + 1 个随机）
    """
    if llm_invoker is None or not candidates:
        fallback_indices = list(range(min(3, len(candidates))))
        return 60, 2, fallback_indices

    user_text = str(state.get("external_user_text") or state.get("user_input") or "").strip()[:500]
    rel = state.get("relationship_state") or {}
    mood = state.get("mood_state") or {}
    stage_id = str(state.get("current_stage") or "initiating")
    stage_desc = format_stage_for_llm(stage_id, include_judge_hints=True)
    scores = state.get("detection_scores") or {}
    stage_judge = state.get("detection_stage_judge") or {}
    direction = str((stage_judge or {}).get("direction") or "none")
    inner_monologue = str(state.get("inner_monologue") or "").strip()[:800]

    lines = []
    for i, t in enumerate(candidates):
        desc = t.get("description") or t.get("id") or ""
        lines.append(f"  {i}: {desc}")
    task_list_str = "\n".join(lines)

    sys = f"""你是日常生活语言沟通专家，深谙人际交往中的分寸与节奏。请凭借你对日常沟通的敏锐判断，一次完成以下两件事：

## A. 预算规划
为下游回复生成系统产出两个预算值：
- word_budget：整数 0-60（0 = 本轮不回复）
- task_budget_max：整数 0-2（本轮最多完成的任务数）

决策建议（非硬规则）：
- 用户明显敌意/越界/敷衍：倾向降低 word_budget，必要时给 0。
- 关系早期且越界推进：降低 word_budget 与 task_budget_max。
- 用户提出明确问题且关系/情绪尚可：保持正常预算（40-60）。
- 若有紧急任务，即使保守也别把 word_budget 设为 0（除非极端）。

## B. 任务选择
下方有一组编号候选任务（0 ~ {len(candidates) - 1}）。
- 选出你认为**最相关**的 2 个任务索引（top2_indices）。
- 再从剩余任务中**随机**选 1 个索引（random_index）。
- 如果候选不足 3 个，有多少选多少即可。

## 输出格式（严格 JSON，不要其他文字）
{{{{
  "word_budget": 60,
  "task_budget_max": 2,
  "top2_indices": [0, 3],
  "random_index": 5
}}}}"""

    user_body = f"""【当轮用户消息】
{user_text or "(空)"}

【关系/情绪】
relationship_state={rel}
mood_state={mood}

【阶段信息】
{stage_desc}

【Detection 感知分数】（0-1）
{scores}

【Detection 阶段方向】
direction={direction}

【Inner Monologue】（可选参考）
{inner_monologue or "(无)"}

【候选任务列表】
{task_list_str}"""

    try:
        resp = llm_invoker.invoke([SystemMessage(content=sys), HumanMessage(content=user_body)])
        raw = (getattr(resp, "content", "") or str(resp)).strip()
        data = parse_json_from_llm(raw)
        if isinstance(data, dict):
            wb = _clamp_int(data.get("word_budget"), 0, 60, 60)
            tb = _clamp_int(data.get("task_budget_max"), 0, 2, 2)
            max_idx = len(candidates) - 1
            top2_raw = data.get("top2_indices") or []
            top2 = [int(x) for x in top2_raw if isinstance(x, (int, float)) and 0 <= int(x) <= max_idx][:2]
            rand_raw = data.get("random_index")
            rand_idx: List[int] = []
            if rand_raw is not None:
                try:
                    ri = int(rand_raw)
                    if 0 <= ri <= max_idx and ri not in top2:
                        rand_idx = [ri]
                except Exception:
                    pass
            selected = list(dict.fromkeys(top2 + rand_idx))  # dedupe, preserve order
            return wb, tb, selected
    except Exception:
        pass
    fallback_indices = list(range(min(3, len(candidates))))
    return 60, 2, fallback_indices


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _backlog_weight(task: Dict[str, Any]) -> float:
    """importance × (越新越高) × (尝试越多越低)。"""
    imp = float(task.get("importance", 0.5) or 0.5)
    imp = max(0.01, min(1.0, imp))
    last = _parse_dt(task.get("last_attempt_at"))
    try:
        attempts = int(task.get("attempt_count", 0) or 0)
    except Exception:
        attempts = 0
    attempts = max(0, min(10, attempts))
    now = datetime.now(timezone.utc)

    if last is None:
        age_days = 0.0
    else:
        age_days = max(0.0, (now - last).total_seconds() / 86400.0)
    recency_factor = 0.5 ** (age_days / 3.0) if age_days > 0 else 1.0
    attempt_factor = 0.75 ** float(attempts)

    w = imp * recency_factor * attempt_factor
    return max(1e-4, float(w))


def _sample_backlog(bot_task_list: List[Dict], k: int = 3) -> List[Dict[str, Any]]:
    """从未完成任务中按权重加权抽样 k 个（不重复）。"""
    return _sample_backlog_excluding(bot_task_list, set(), k)


def _sample_backlog_excluding(
    bot_task_list: List[Dict],
    exclude_ids: set,
    k: int = 3,
) -> List[Dict[str, Any]]:
    """从 bot_task_list 中排除 exclude_ids 后按权重抽样 k 个（不重复）。"""
    exclude_ids = set(str(x) for x in exclude_ids)
    available = [
        t
        for t in (bot_task_list or [])
        if str(t.get("id") or "") not in exclude_ids
        and not _is_systemic_task_desc(str(t.get("description") or ""))
    ]
    if not available or k <= 0:
        return []
    weights = [_backlog_weight(t) for t in available]
    total = sum(weights)
    if total <= 0:
        return random.sample([dict(t) for t in available], min(k, len(available)))
    chosen: List[Dict[str, Any]] = []
    indices = list(range(len(available)))
    for _ in range(min(k, len(available))):
        if not indices:
            break
        total_w = sum(weights[i] for i in indices)
        r = random.uniform(0, total_w)
        for i in list(indices):
            w = weights[i]
            if r <= w:
                chosen.append(dict(available[i]))
                indices.remove(i)
                break
            r -= w
    return chosen


# 会话任务池上限（避免膨胀）
CURRENT_SESSION_TASKS_CAP = 20

# 持久化会话池里保留的 backlog 任务数目标
BACKLOG_SESSION_TARGET = 3


def _immediate_from_detection(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 state 的 detection_immediate_tasks 读取当轮任务（Detection 节点已产出，无数量限制）。"""
    tasks = state.get("detection_immediate_tasks") or []
    if not isinstance(tasks, list):
        return []
    out: List[Dict[str, Any]] = []
    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            continue
        desc = str(t.get("description") or "").strip()
        if not desc:
            continue
        out.append({
            "id": t.get("id") or f"detection_immediate_{i}",
            "description": desc,
            "task_type": str(t.get("task_type") or "immediate"),
            "importance": float(t.get("importance", 0.5) or 0.5),
            "ttl_turns": int(t.get("ttl_turns", 4) or 4),
            "source": str(t.get("source") or "detection"),
        })
    return out


def _urgent_tasks_from_state(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """合并 DB 级别 + Detection 级别的紧急任务，标记 is_urgent=True。"""
    out: List[Dict[str, Any]] = []
    idx = 0
    for source_key in ("db_urgent_tasks", "detection_urgent_tasks"):
        tasks = state.get(source_key) or []
        if not isinstance(tasks, list):
            continue
        for t in tasks:
            if not isinstance(t, dict):
                continue
            desc = str(t.get("description") or "").strip()
            if not desc:
                continue
            out.append({
                "id": t.get("id") or f"urgent_{idx}",
                "description": desc,
                "task_type": "urgent",
                "is_urgent": True,
                "importance": float(t.get("importance", 0.9) or 0.9),
                "source": str(t.get("source") or "unknown"),
                "_level": str(t.get("_level") or "detection"),
            })
            idx += 1
    return out


_BASIC_INFO_FIELDS: List[Tuple[str, str, str]] = [
    ("name",       "ask_user_name",       "在合适的时机自然地询问对方的姓名或称呼"),
    ("age",        "ask_user_age",        "在合适的时机自然地了解对方的年龄"),
    ("gender",     "ask_user_gender",     "在合适的时机自然地了解对方的性别"),
    ("occupation", "ask_user_occupation", "在合适的时机自然地了解对方的职业"),
    ("location",   "ask_user_location",   "在合适的时机自然地了解对方是哪里人"),
]


def _basic_info_urgent_task(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """检查 user_basic_info 是否完善；若有缺失，按优先级返回一条紧急任务。"""
    info = state.get("user_basic_info") or {}
    for field, task_id, desc in _BASIC_INFO_FIELDS:
        val = info.get(field)
        if val is None or (isinstance(val, str) and not val.strip()):
            return {
                "id": task_id,
                "description": desc,
                "task_type": "urgent",
                "is_urgent": True,
                "importance": 0.85,
                "source": "basic_info_check",
                "_level": "system",
            }
    return None


def _dedupe_understanding(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """理解类最多保留 1 个（保留第一个）。"""
    understanding = [t for t in tasks if (t.get("task_type") or "").strip() in UNDERSTANDING_TYPES]
    others = [t for t in tasks if (t.get("task_type") or "").strip() not in UNDERSTANDING_TYPES]
    if len(understanding) <= 1:
        return tasks
    return others + [understanding[0]]


# ---------------------------------------------------------------------------
# 节点入口
# ---------------------------------------------------------------------------

def create_task_planner_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """TaskPlanner 节点：产出 tasks_for_lats / task_budget_max / word_budget / completion_temperature。"""

    @trace_if_enabled(
        name="TaskPlanner",
        run_type="chain",
        tags=["node", "task_planner", "task_budget"],
        metadata={"state_outputs": ["tasks_for_lats", "task_budget_max", "word_budget", "completion_temperature", "no_reply", "current_session_tasks"]},
    )
    def task_planner_node(state: AgentState) -> dict:
        completion_temperature = float(state.get("completion_temperature", 1.0) or 1.0)

        # ── 1. 紧急任务：合并 DB + Detection + basic_info 来源，绕过 LLM 直接注入 ──
        urgent_tasks = _urgent_tasks_from_state(state)
        profile_urgent = _basic_info_urgent_task(state)
        if profile_urgent:
            urgent_tasks.append(profile_urgent)

        has_urgent = len(urgent_tasks) > 0
        has_db_urgent = any(t.get("_level") in ("bot", "user") for t in urgent_tasks)

        # ── 2. 组装候选列表（backlog + immediate + daily）──
        bot_task_list = state.get("bot_task_list") or []
        pool: List[Dict[str, Any]] = list(state.get("current_session_tasks") or [])
        pool = [
            t for t in pool
            if isinstance(t, dict) and not _is_systemic_task_desc(str(t.get("description") or ""))
        ]
        existing_ids = {str(t.get("id")) for t in pool if t.get("id")}

        def _norm(t: Dict[str, Any], prefix: str, i: int) -> Dict[str, Any]:
            return {
                "id": t.get("id") or f"{prefix}_{i}",
                "description": str(t.get("description") or t.get("id") or "").strip() or "（无描述）",
                "task_type": str(t.get("task_type") or "other"),
            }

        carry_n = len(pool)
        seeded_backlog = 0
        added_immediate = 0

        backlog_in_pool = sum(1 for t in pool if str(t.get("task_type") or "").strip() == "backlog")
        if backlog_in_pool <= 0 and bot_task_list:
            backlog_seed = _sample_backlog_excluding(bot_task_list, existing_ids, BACKLOG_SESSION_TARGET)
            for i, t in enumerate(backlog_seed):
                nt = _norm(t, "backlog", len(pool))
                tid = str(nt.get("id") or "")
                if tid and tid not in existing_ids:
                    pool.append(nt)
                    existing_ids.add(tid)
                    seeded_backlog += 1

        immediate = _immediate_from_detection(state)
        for i, t in enumerate(immediate):
            nt = _norm(t, "immediate", len(pool))
            if "ttl_turns" in t:
                try:
                    nt["ttl_turns"] = int(t.get("ttl_turns") or 0)
                except Exception:
                    nt["ttl_turns"] = 4
            if "importance" in t:
                try:
                    nt["importance"] = float(t.get("importance") or 0.5)
                except Exception:
                    pass
            if "source" in t:
                nt["source"] = str(t.get("source") or "")
            tid = str(nt.get("id") or "")
            if tid and tid not in existing_ids:
                pool.append(nt)
                existing_ids.add(tid)
                added_immediate += 1

        if len(pool) > CURRENT_SESSION_TASKS_CAP:
            pool = pool[-CURRENT_SESSION_TASKS_CAP:]
        current_session_tasks = pool

        daily2 = random.sample(DAILY_POOL, min(2, len(DAILY_POOL))) if DAILY_POOL else []
        daily_candidates = [_norm(t, "daily", i) for i, t in enumerate(daily2)]

        candidates = list(current_session_tasks) + list(daily_candidates)

        # ── 3. 单次 LLM 调用：预算规划 + 任务选择 ──
        word_budget, task_budget_max, selected_indices = _plan_and_select_with_llm(
            state, candidates, llm_invoker,
        )

        # word_budget=0 且无紧急任务 → NO_REPLY
        if word_budget == 0 and not has_urgent:
            return {
                "tasks_for_lats": [],
                "task_budget_max": 0,
                "word_budget": 0,
                "completion_temperature": completion_temperature,
                "no_reply": True,
                "detection_category": "NO_REPLY",
                "detection_result": "NO_REPLY",
                "current_session_tasks": current_session_tasks,
                "_urgent_tasks_consumed": False,
            }
        if word_budget == 0 and has_urgent:
            word_budget = 60
            print("[TaskPlanner] word_budget was 0 but urgent tasks present, overriding to 60")

        # ── 4. 按 LLM 返回的索引拣选普通任务 ──
        urgent_for_lats: List[Dict[str, Any]] = [
            {"id": t["id"], "description": t["description"], "task_type": "urgent", "is_urgent": True}
            for t in urgent_tasks
        ]
        normal_slots = max(0, 3 - len(urgent_for_lats))

        selected_normal: List[Dict[str, Any]] = []
        if candidates and normal_slots > 0:
            for idx in selected_indices:
                if len(selected_normal) >= normal_slots:
                    break
                if 0 <= idx < len(candidates):
                    selected_normal.append(candidates[idx])
            selected_normal = _dedupe_understanding(selected_normal)

        tasks_for_lats = urgent_for_lats + [
            {"id": t["id"], "description": t["description"], "task_type": t.get("task_type")}
            for t in selected_normal
        ]

        # Debug breadcrumb
        try:
            sel_types = [str(t.get("task_type") or "other") for t in tasks_for_lats]
            sel_daily = sum(1 for x in sel_types if x == "daily")
            sel_backlog = sum(1 for x in sel_types if x == "backlog")
            sel_urgent = sum(1 for x in sel_types if x == "urgent")
            if sel_urgent > 0:
                print(
                    f"[TaskPlanner] ========================================\n"
                    f"[TaskPlanner]  URGENT: {sel_urgent} urgent task(s) injected directly into LATS\n"
                    f"[TaskPlanner]  Descriptions: {[t['description'][:60] for t in urgent_for_lats]}\n"
                    f"[TaskPlanner] ========================================"
                )
            print(
                f"[TaskPlanner] pool(carry={carry_n}, seed_backlog={seeded_backlog}, +immediate={added_immediate}, daily_sampled={len(daily_candidates)}) "
                f"candidates={len(candidates)} selected_indices={selected_indices} selected(urgent={sel_urgent}, backlog={sel_backlog}, daily={sel_daily})"
            )
        except Exception:
            pass

        return {
            "tasks_for_lats": tasks_for_lats,
            "task_budget_max": task_budget_max,
            "word_budget": word_budget,
            "completion_temperature": completion_temperature,
            "no_reply": False,
            "detection_category": "NORMAL",
            "detection_result": "NORMAL",
            "current_session_tasks": current_session_tasks,
            "_urgent_tasks_consumed": has_db_urgent,
        }
    return task_planner_node

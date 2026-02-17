"""
TaskPlanner 节点：在 LATS 之前运行。
- 根据上下文/情绪/关系规划：回复字数上限 word_budget (0-60)、任务完成上限 task_budget_max (0-2)。
- 候选集 C = backlog(3) + daily(2) + immediate(K)；用 gpt4o-mini 打分，固定选 2 个最高分，第 3 个按分数加权随机。
- 理解类任务去重（最多 1 个）。
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

# 过滤“系统性/助手味”任务（避免把 LATS 拉回“助手味”）
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


def _word_budget_from_state(state: Dict[str, Any]) -> int:
    """优先使用 inner_monologue 写入的 word_budget；缺省时回退 60。"""
    wb = state.get("word_budget")
    if wb is not None and isinstance(wb, int):
        return max(0, min(60, wb))
    return 60


def _task_budget_max_from_state(state: Dict[str, Any]) -> int:
    """优先使用 inner_monologue 写入的 task_budget_max；缺省时 2。"""
    tb = state.get("task_budget_max")
    if tb is not None and isinstance(tb, int):
        return max(0, min(2, tb))
    return 2


def _backlog_weight(task: Dict[str, Any]) -> float:
    """importance × (越新越高) × (尝试越多越低)。"""
    imp = float(task.get("importance", 0.5) or 0.5)
    imp = max(0.01, min(1.0, imp))
    last = _parse_dt(task.get("last_attempt_at"))
    # attempt_count：尝试越多，权重越低（避免反复抽同一个任务）
    try:
        attempts = int(task.get("attempt_count", 0) or 0)
    except Exception:
        attempts = 0
    attempts = max(0, min(10, attempts))
    now = datetime.now(timezone.utc)

    # 时间衰减：离上次尝试越久，权重越低；越新越容易被补进池子（符合“保持话题新鲜”策略）
    if last is None:
        age_days = 0.0
    else:
        age_days = max(0.0, (now - last).total_seconds() / 86400.0)
    # 每过 ~3 天权重约减半（可调）
    recency_factor = 0.5 ** (age_days / 3.0) if age_days > 0 else 1.0

    # 尝试衰减：每多尝试一次，权重乘 0.75（可调）
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


def _dedupe_understanding(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """理解类最多保留 1 个（保留分数最高的）。"""
    understanding = [t for t in tasks if (t.get("task_type") or "").strip() in UNDERSTANDING_TYPES]
    others = [t for t in tasks if (t.get("task_type") or "").strip() not in UNDERSTANDING_TYPES]
    if len(understanding) <= 1:
        return tasks
    return others + [understanding[0]]


def _score_tasks_with_llm(
    state: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    llm_invoker: Any,
) -> List[Tuple[Dict[str, Any], float]]:
    """用 gpt4o-mini 给每个候选任务打 0-10 分，返回 (task, score) 列表。"""
    if not candidates or llm_invoker is None:
        return [(t, 0.5) for t in candidates]

    user_text = (state.get("user_input") or state.get("external_user_text") or "")[:300]
    rel = state.get("relationship_state") or {}
    mood = state.get("mood_state") or {}
    inner_monologue = (state.get("inner_monologue") or "").strip()[:800]

    lines = []
    for i, t in enumerate(candidates):
        desc = t.get("description") or t.get("id") or ""
        lines.append(f"{i}: {desc}")
    task_list_str = "\n".join(lines)

    sys = """你根据「当前用户消息」「关系/情绪」以及「内心独白（本轮的动机与策略）」对下列任务打相关性分数。
输出严格 JSON：{"scores": [0.0, 0.0, ...]}，与上面 0~N 顺序一一对应，范围 0.0-10.0，越高越适合在本轮完成。"""
    monologue_block = f"\n内心独白（动机与策略，供打分参考）：\n{inner_monologue}\n" if inner_monologue else ""
    user = f"""当前用户消息：{user_text}
关系/情绪：{rel}; {mood}{monologue_block}
任务列表（按序号）：
{task_list_str}

只输出 JSON，不要其他文字。"""

    try:
        resp = llm_invoker.invoke([SystemMessage(content=sys), HumanMessage(content=user)])
        raw = (getattr(resp, "content", "") or str(resp)).strip()
        data = parse_json_from_llm(raw)
        if isinstance(data, dict) and isinstance(data.get("scores"), list):
            scores = [float(x) for x in data["scores"]]
            out = []
            for i, t in enumerate(candidates):
                s = scores[i] if i < len(scores) else 0.5
                out.append((dict(t), max(0.0, min(10.0, s))))
            return out
    except Exception:
        pass
    return [(dict(t), 0.5) for t in candidates]


def _weighted_random_choice(items: List[Tuple[Dict[str, Any], float]], temperature: float = 1.0) -> Optional[Dict[str, Any]]:
    """按分数加权随机选一个；temperature 调高更随机。"""
    if not items:
        return None
    weights = [max(1e-6, (s ** (1.0 / max(0.1, temperature)))) for _, s in items]
    total = sum(weights)
    if total <= 0:
        return random.choice(items)[0]
    r = random.uniform(0, total)
    for (t, _), w in zip(items, weights):
        if r <= w:
            return t
        r -= w
    return items[-1][0]


def create_task_planner_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """TaskPlanner 节点：产出 tasks_for_lats / task_budget_max / word_budget / completion_temperature。"""

    @trace_if_enabled(
        name="TaskPlanner",
        run_type="chain",
        tags=["node", "task_planner", "task_budget"],
        metadata={"state_outputs": ["tasks_for_lats", "task_budget_max", "word_budget", "completion_temperature", "no_reply", "current_session_tasks"]},
    )
    def task_planner_node(state: AgentState) -> dict:
        word_budget = _word_budget_from_state(state)
        task_budget_max = _task_budget_max_from_state(state)
        completion_temperature = float(state.get("completion_temperature", 1.0) or 1.0)

        # word_budget=0 时直接 NO_REPLY，不调用 LATS；本轮任务不结算完成，current_session_tasks 保持上轮
        if word_budget == 0:
            return {
                "tasks_for_lats": [],
                "task_budget_max": 0,
                "word_budget": 0,
                "completion_temperature": completion_temperature,
                "no_reply": True,
                "detection_category": "NO_REPLY",
                "detection_result": "NO_REPLY",
                "current_session_tasks": state.get("current_session_tasks") or [],
            }

        bot_task_list = state.get("bot_task_list") or []
        # 当前会话任务池（持久化）：由 evolver 结算/补充；planner 只允许 immediate 叠加（daily 永不持久化）
        pool: List[Dict[str, Any]] = list(state.get("current_session_tasks") or [])
        # 清理历史遗留的“系统性/助手味”任务，避免继续 carry/入选
        pool = [
            t
            for t in pool
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

        # 仅在会话池中完全没有 backlog 时做一次性“种子”（避免池子随每轮膨胀）。
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
            # immediate 允许累加：保留 ttl_turns/importance/source 等字段，供 evolver 做过期/结算
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

        # daily 任务：每轮临时抽样，仅参与本轮打分，不进入持久化任务池
        daily2 = random.sample(DAILY_POOL, min(2, len(DAILY_POOL))) if DAILY_POOL else []
        daily_candidates = [_norm(t, "daily", i) for i, t in enumerate(daily2)]

        candidates = list(current_session_tasks) + list(daily_candidates)

        if not candidates:
            return {
                "tasks_for_lats": [],
                "task_budget_max": task_budget_max,
                "word_budget": word_budget,
                "completion_temperature": completion_temperature,
                "current_session_tasks": current_session_tasks,
            }

        scored = _score_tasks_with_llm(state, candidates, llm_invoker)
        scored.sort(key=lambda x: x[1], reverse=True)

        top2 = [t for t, _ in scored[:2]]
        rest = scored[2:]
        third = _weighted_random_choice(rest, completion_temperature) if rest else None
        selected = top2 + ([third] if third else [])
        selected = selected[:3]
        selected = _dedupe_understanding(selected)

        tasks_for_lats = [{"id": t["id"], "description": t["description"], "task_type": t.get("task_type")} for t in selected]

        # Debug breadcrumb for logs: persistent pool composition & selected task types
        try:
            sel_types = [str(t.get("task_type") or "other") for t in tasks_for_lats]
            sel_daily = sum(1 for x in sel_types if x == "daily")
            sel_backlog = sum(1 for x in sel_types if x == "backlog")
            print(
                f"[TaskPlanner] pool(carry={carry_n}, seed_backlog={seeded_backlog}, +immediate={added_immediate}, daily_sampled={len(daily_candidates)}) "
                f"candidates={len(candidates)} selected(backlog={sel_backlog}, daily={sel_daily})"
            )
        except Exception:
            pass

        return {
            "tasks_for_lats": tasks_for_lats[:3],
            "task_budget_max": task_budget_max,
            "word_budget": word_budget,
            "completion_temperature": completion_temperature,
            "no_reply": False,
            "detection_category": "NORMAL",
            "detection_result": "NORMAL",
            "current_session_tasks": current_session_tasks,
        }
    return task_planner_node

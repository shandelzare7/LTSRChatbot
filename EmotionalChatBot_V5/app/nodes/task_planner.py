from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.state import AgentState
from src.schemas import TaskPlannerOutput
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
    detection = state.get("detection") or {}
    hostility = int(detection.get("hostility_level") or 0)
    engagement = int(detection.get("engagement_level") or 5)
    topic_appeal = int(detection.get("topic_appeal") or 5)
    stage_pacing = str(detection.get("stage_pacing") or "正常").strip()
    inner_monologue = str(state.get("inner_monologue") or "").strip()[:800]
    momentum = state.get("conversation_momentum", 1.0)
    momentum = float(momentum) if momentum is not None else 1.0

    # 候选任务列表（保持你原来的“只给描述”口径，不丢你现有实现信息）
    lines: List[str] = []
    for i, t in enumerate(candidates):
        desc = t.get("description") or t.get("id") or ""
        lines.append(f"{i}: {desc}")
    task_list_str = "\n".join(lines)

    # ✅ 自然语言 System Prompt（保留原文开头 + 全部规则点）
    sys = f"""你是日常生活语言沟通专家，深谙人际交往中的分寸与节奏。请凭借你对日常沟通的敏锐判断，一次完成两件事：预算规划 + 任务选择。

你需要产出两个预算值：
- word_budget：整数 0–60（0 = 本轮不回复）
- task_budget_max：整数 0–2（本轮最多完成的任务数）

预算决策建议（必须参考 Detection 输出，0-10 分）：
Detection 输出含义：
- hostility_level: 敌意/攻击性（≥5→降低 word_budget，≥7→可设为0）
- engagement_level: 用户投入度/信息量（高→可正常预算，低→保守）
- topic_appeal: 话题对 Bot 的吸引力（高→可保持预算）
- stage_pacing: 关系节奏（正常/过分亲密/过分生疏）

预算决策规则：
- 用户明显敌意（hostility_level ≥ 5）：倾向降低 word_budget，≥7 时可给 0。
- 用户投入低（engagement_level ≤ 3）：保守预算。
- 投入高（engagement_level ≥ 6）且敌意低（hostility_level ≤ 3）：保持正常预算（40–60）。
- 若有 DB 紧急任务：即使保守也尽量别把 word_budget 设为 0（除非极端情况）。

另外，你必须参考“对话冲量 conversation_momentum = m”做调节：
- m=1.0 表示刚开始精力充沛；越低表示聊得越久越倦怠。
- 当 m < 0.5：
  - word_budget 通常不超过 30（除非话题非常重要）
  - 减少追问类任务（clarify/ask_scope/ask_example/confirm_gap），除非理解置信度很低
  - 不主动开新话题
- 当 m < 0.3：
  - word_budget 通常不超过 20
  - 只执行必要的回应，不附加任何额外内容

当前 m={momentum:.2f}

任务选择规则：
我会给你一组编号候选任务（编号从 0 到 {len(candidates) - 1}）。
- 选出你认为最相关的 2 个任务索引（top2_indices）。
- 再从剩余任务中随机选 1 个索引（random_index），不要与 top2 重复。
- 如果候选不足 3 个，有多少选多少即可。

（输出格式由系统约束。）"""

    # ✅ 自然语言 User Body（信息不丢，但去掉冗余标题/解释）
    user_body = f"""当轮用户消息：
{user_text or "(空)"}

关系/情绪：
relationship_state={rel}
mood_state={mood}（PAD 为 [-1,1]，0 为中性；busyness 为 [0,1]。）

阶段信息：
stage_id={stage_id}
stage_desc={stage_desc}

Detection（0-10）：
hostility_level={hostility}, engagement_level={engagement}, topic_appeal={topic_appeal}, stage_pacing={stage_pacing}

Inner Monologue（可选参考）：
{inner_monologue or "(无)"}

对话冲量：
conversation_momentum={momentum:.2f}

候选任务（编号: 描述）：
{task_list_str}"""

    try:
        data = None
        if hasattr(llm_invoker, "with_structured_output"):
            try:
                structured = llm_invoker.with_structured_output(TaskPlannerOutput)
                obj = structured.invoke([SystemMessage(content=sys), HumanMessage(content=user_body)])
                data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
            except Exception:
                data = None
        if data is None:
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
    """Detection 已简化，不再产出当轮任务，返回空列表。"""
    return []


def _urgent_tasks_from_state(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 state 读取 DB 紧急任务，标记 is_urgent=True。（Detection 已简化，不再产出紧急任务。）"""
    out: List[Dict[str, Any]] = []
    idx = 0
    for source_key in ("db_urgent_tasks",):
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
    ("name",       "ask_user_name",       "本轮或近期回复中务必明确询问对方的姓名或称呼"),
    ("age",        "ask_user_age",        "本轮或近期回复中务必明确询问对方的年龄"),
    ("occupation", "ask_user_occupation", "本轮或近期回复中务必明确询问对方的职业"),
    ("location",   "ask_user_location",   "本轮或近期回复中明确询问对方所在城市/地区"),
]
# 性别不设问性别任务，仅靠 memory_manager 根据对话推断并写回 basic_info


def get_session_basic_info_pending_task_ids(user_basic_info: Dict[str, Any]) -> List[str]:
    """根据 user_basic_info 缺失项，返回本 session 待办的紧急任务 id 列表（问名字/年龄/职业/地区）。会话开始时调用一次，用于初始化 session_basic_info_pending_task_ids。"""
    info = user_basic_info or {}
    out: List[str] = []
    for field, task_id, _desc in _BASIC_INFO_FIELDS:
        val = info.get(field)
        if val is None or (isinstance(val, str) and not val.strip()):
            out.append(task_id)
    return out


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
    """TaskPlanner 节点：产出 tasks_for_lats / task_budget_max / no_reply。"""

    @trace_if_enabled(
        name="TaskPlanner",
        run_type="chain",
        tags=["node", "task_planner", "task_budget"],
        metadata={"state_outputs": ["tasks_for_lats", "task_budget_max", "no_reply", "current_session_tasks"]},
    )
    def task_planner_node(state: AgentState) -> dict:
        # ── 1. 紧急任务：本 session 仅从待办列表取一条，触发后即从列表移除（每 session 每类最多一次）──
        urgent_tasks: List[Dict[str, Any]] = []
        assets = state.get("relationship_assets") or {}
        pending_ids: List[str] = list(assets.get("session_basic_info_pending_task_ids") or [])
        if not isinstance(pending_ids, list):
            pending_ids = []
        updated_assets: Optional[Dict[str, Any]] = None
        if pending_ids:
            task_id = pending_ids[0]
            desc = ""
            for _f, tid, d in _BASIC_INFO_FIELDS:
                if tid == task_id:
                    desc = d
                    break
            urgent_tasks.append({
                "id": task_id,
                "description": desc or "询问用户基本信息",
                "task_type": "urgent",
                "is_urgent": True,
                "importance": 0.85,
                "source": "basic_info_check",
                "_level": "system",
            })
            new_pending = pending_ids[1:]
            updated_assets = {**assets, "session_basic_info_pending_task_ids": new_pending}

        has_urgent = len(urgent_tasks) > 0
        has_db_urgent = any(t.get("_level") in ("bot", "user") for t in urgent_tasks)

        # ── 2. 以下整块暂注释：候选池组装（backlog + immediate + daily）──
        # bot_task_list = state.get("bot_task_list") or []
        # pool: List[Dict[str, Any]] = list(state.get("current_session_tasks") or [])
        # pool = [
        #     t for t in pool
        #     if isinstance(t, dict) and not _is_systemic_task_desc(str(t.get("description") or ""))
        # ]
        # existing_ids = {str(t.get("id")) for t in pool if t.get("id")}
        #
        # def _norm(t: Dict[str, Any], prefix: str, i: int) -> Dict[str, Any]:
        #     return {
        #         "id": t.get("id") or f"{prefix}_{i}",
        #         "description": str(t.get("description") or t.get("id") or "").strip() or "（无描述）",
        #         "task_type": str(t.get("task_type") or "other"),
        #     }
        #
        # carry_n = len(pool)
        # seeded_backlog = 0
        # added_immediate = 0
        #
        # backlog_in_pool = sum(1 for t in pool if str(t.get("task_type") or "").strip() == "backlog")
        # if backlog_in_pool <= 0 and bot_task_list:
        #     backlog_seed = _sample_backlog_excluding(bot_task_list, existing_ids, BACKLOG_SESSION_TARGET)
        #     for i, t in enumerate(backlog_seed):
        #         nt = _norm(t, "backlog", len(pool))
        #         tid = str(nt.get("id") or "")
        #         if tid and tid not in existing_ids:
        #             pool.append(nt)
        #             existing_ids.add(tid)
        #             seeded_backlog += 1
        #
        # immediate = _immediate_from_detection(state)
        # for i, t in enumerate(immediate):
        #     nt = _norm(t, "immediate", len(pool))
        #     if "ttl_turns" in t:
        #         try:
        #             nt["ttl_turns"] = int(t.get("ttl_turns") or 0)
        #         except Exception:
        #             nt["ttl_turns"] = 4
        #     if "importance" in t:
        #         try:
        #             nt["importance"] = float(t.get("importance") or 0.5)
        #         except Exception:
        #             pass
        #     if "source" in t:
        #         nt["source"] = str(t.get("source") or "")
        #     tid = str(nt.get("id") or "")
        #     if tid and tid not in existing_ids:
        #         pool.append(nt)
        #         existing_ids.add(tid)
        #         added_immediate += 1
        #
        # if len(pool) > CURRENT_SESSION_TASKS_CAP:
        #     pool = pool[-CURRENT_SESSION_TASKS_CAP:]
        # current_session_tasks = pool
        #
        # daily2 = random.sample(DAILY_POOL, min(2, len(DAILY_POOL))) if DAILY_POOL else []
        # daily_candidates = [_norm(t, "daily", i) for i, t in enumerate(daily2)]
        #
        # candidates = list(current_session_tasks) + list(daily_candidates)
        current_session_tasks = list(state.get("current_session_tasks") or [])

        # ── 3. 以下暂注释：LLM 预算规划 + 任务选择 ──
        # word_budget, task_budget_max, selected_indices = _plan_and_select_with_llm(
        #     state, candidates, llm_invoker,
        # )
        # if word_budget == 0 and not has_urgent:
        #     return {
        #         "tasks_for_lats": [],
        #         "task_budget_max": 0,
        #         "word_budget": 0,
        #         "no_reply": True,
        #         "detection_category": "NO_REPLY",
        #         "detection_result": "NO_REPLY",
        #         "current_session_tasks": current_session_tasks,
        #         "_urgent_tasks_consumed": False,
        #     }
        # if word_budget == 0 and has_urgent:
        #     word_budget = 60
        #     print("[TaskPlanner] word_budget was 0 but urgent tasks present, overriding to 60")
        task_budget_max = min(1, len(urgent_tasks)) if has_urgent else 0

        # ── 4. 仅保留紧急任务注入；普通任务拣选暂注释 ──
        urgent_for_lats = [
            {"id": t["id"], "description": t["description"], "task_type": "urgent", "is_urgent": True}
            for t in urgent_tasks
        ]
        # normal_slots = max(0, 3 - len(urgent_for_lats))
        # selected_normal = []
        # if candidates and normal_slots > 0:
        #     for idx in selected_indices:
        #         if len(selected_normal) >= normal_slots:
        #             break
        #         if 0 <= idx < len(candidates):
        #             selected_normal.append(candidates[idx])
        #     selected_normal = _dedupe_understanding(selected_normal)
        # tasks_for_lats = urgent_for_lats + [
        #     {"id": t["id"], "description": t["description"], "task_type": t.get("task_type")}
        #     for t in selected_normal
        # ]
        tasks_for_lats = urgent_for_lats

        # if sel_urgent > 0: print(...); print(...)
        if has_urgent:
            print(f"[TaskPlanner] 基本信息紧急任务: {[t['description'][:50] for t in urgent_for_lats]}")

        result: Dict[str, Any] = {
            "tasks_for_lats": tasks_for_lats,
            "task_budget_max": task_budget_max,
            "no_reply": False,
            "detection_category": "NORMAL",
            "detection_result": "NORMAL",
            "current_session_tasks": current_session_tasks,
            "_urgent_tasks_consumed": has_db_urgent,
        }
        if updated_assets is not None:
            result["relationship_assets"] = updated_assets
        return result

    return task_planner_node

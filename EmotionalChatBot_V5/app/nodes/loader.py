"""入口加载节点：优先从 DB 读取状态（Load Early），无 DB 时落本地文件（再无则回退内存）。"""
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import math
import os
import asyncio
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from app.state import AgentState

from app.lats.prompt_utils import sanitize_memory_text, filter_retrieved_memories
from utils.external_text import sanitize_external_text, detect_internal_leak
from utils.prompt_helpers import knapp_baseline_momentum
from utils.busy_schedule import get_busy_fallback_from_schedule
from utils.yaml_loader import load_momentum_formula_config

# 每轮开始 conversation_momentum 的下限（与 config/momentum_formula.yaml 一致）
MOMENTUM_FLOOR = float(load_momentum_formula_config().get("momentum_floor", 0.4))


def _load_daily_context(bot_id: str = "") -> dict:
    """从 config/daily_topics.yaml 加载今日话题和 bot 生活事件，日期不匹配时返回空。
    返回 {"topics": List[str], "bot_recent": List[str]}。
    若 daily_topics.yaml 含 bot_recent_by_bot_id 字段，则按 bot_id 精确匹配；否则降级用 bot_recent。
    """
    try:
        from utils.yaml_loader import get_project_root
        import yaml
        path = get_project_root() / "config" / "daily_topics.yaml"
        if not path.exists():
            return {"topics": [], "bot_recent": []}
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        file_date = str(data.get("date") or "").strip()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if file_date != today:
            return {"topics": [], "bot_recent": []}
        topics = data.get("topics") or []
        # 优先按 bot_id 查 bot_recent_by_bot_id，无则降级用 bot_recent
        bot_recent_default = data.get("bot_recent") or []
        bot_recent_map = data.get("bot_recent_by_bot_id") or {}
        if bot_id and str(bot_id) in bot_recent_map:
            bot_recent = bot_recent_map[str(bot_id)] or []
        else:
            bot_recent = bot_recent_default
        return {
            "topics": [str(t).strip() for t in topics if str(t).strip()][:8],
            "bot_recent": [str(t).strip() for t in bot_recent if str(t).strip()][:8],
        }
    except Exception:
        return {"topics": [], "bot_recent": []}


def _load_daily_topics() -> List[str]:
    """向后兼容包装：仅返回 topics 列表。"""
    return _load_daily_context()["topics"]

# basic_info 字段 → 对应问询任务 id（从 task_planner 迁移）
_BASIC_INFO_FIELDS = [
    ("name",       "ask_user_name",       "本轮或近期回复中务必明确询问对方的姓名或称呼"),
    ("age",        "ask_user_age",        "本轮或近期回复中务必明确询问对方的年龄"),
    ("occupation", "ask_user_occupation", "本轮或近期回复中务必明确询问对方的职业"),
    ("location",   "ask_user_location",   "本轮或近期回复中明确询问对方所在城市/地区"),
]


def get_session_basic_info_pending_task_ids(user_basic_info: Dict[str, Any]) -> List[str]:
    """根据 user_basic_info 缺失项，返回本 session 待办的问询任务 id 列表。"""
    info = user_basic_info or {}
    return [
        task_id
        for field, task_id, _ in _BASIC_INFO_FIELDS
        if not (info.get(field) or "").strip()
    ]

# 距上次消息超过此时长视为新 Session，按 Knapp 阶段重新初始化 conversation_momentum
COLD_START_THRESHOLD_SEC = 4 * 3600  # 4 小时
# 短期离线残留：指数衰减系数 λ，M_init = M_last * exp(-λ * Δt)，Δt 单位为小时
MOMENTUM_DECAY_LAMBDA = 0.5


def _parse_timestamp_to_seconds(ts_str: Optional[str]) -> Optional[float]:
    """解析 ISO 时间戳，返回相对当前时间的秒数（正数表示过去）。解析失败返回 None。"""
    if not ts_str or not isinstance(ts_str, str):
        return None
    try:
        s = ts_str.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (now - dt).total_seconds())
    except Exception:
        return None


def _is_human_message(m: BaseMessage) -> bool:
    t = getattr(m, "type", "") or ""
    return "human" in t.lower() or "user" in t.lower()


def _seconds_since_last_message(
    buf: List[BaseMessage],
    current_user_input: Optional[str] = None,
) -> Optional[float]:
    """
    从「上一轮最后一条消息」的 timestamp 计算距现在的秒数，用于冷启动/冲量重置。
    若 buffer 最后一条是当前用户刚发的消息（与 current_user_input 一致），则用倒数第二条的时间，
    否则用最后一条的时间，避免“当前句”导致间隔≈0、冲量从不重置。
    """
    if not buf:
        return None
    # 最后一条是当前用户消息时，用倒数第二条的时间算间隔
    last = buf[-1]
    if current_user_input is not None and _is_human_message(last):
        last_content = (getattr(last, "content", "") or "").strip()
        if last_content and last_content == (current_user_input or "").strip():
            if len(buf) >= 2:
                last = buf[-2]
            # 若只有一条（就是当前句），无“上一轮”，视为冷启动
            else:
                return None
    kwargs = getattr(last, "additional_kwargs", None) or {}
    return _parse_timestamp_to_seconds(kwargs.get("timestamp"))


def _ensure_rel_scale(rel: Dict[str, Any]) -> Dict[str, Any]:
    """保证 relationship_state 带 rel_scale，缺省 0_1（与 PAD 的 pad_scale 同理）。"""
    d = dict(rel)
    d.setdefault("rel_scale", "0_1")
    return d


def _resolve_conversation_momentum(
    seconds_since_last: Optional[float],
    current_stage: Any,
    stored_momentum: Any,
    default: float = 0.5,
) -> float:
    """
    1) 冷启动：距上次消息 >= COLD_START_THRESHOLD_SEC 或无时间戳 → 按 Knapp 阶段基线。
    2) 短期离线残留：间隔 < 4h → M_init = M_last * exp(-λ * Δt)，Δt 为小时，λ = MOMENTUM_DECAY_LAMBDA。
    无存储值时用 default=0.5，避免未持久化时退化为 1.0 导致 momentum 锁死。
    """
    cold_start = (
        seconds_since_last is None
        or seconds_since_last >= COLD_START_THRESHOLD_SEC
    )
    if cold_start:
        m = knapp_baseline_momentum(current_stage)
        return max(MOMENTUM_FLOOR, min(1.0, m))
    # 短期离线：物理时间衰减 M_init = M_last * e^(-λ * Δt)，Δt 单位小时
    try:
        m_last = float(stored_momentum)
    except (TypeError, ValueError):
        m_last = default
    m_last = max(0.0, min(1.0, m_last))
    delta_t_hours = (seconds_since_last or 0.0) / 3600.0
    m_init = m_last * math.exp(-MOMENTUM_DECAY_LAMBDA * delta_t_hours)
    return max(MOMENTUM_FLOOR, min(1.0, m_init))


# busy 上限：日程表算出后 clamp 到此值，避免过忙导致风格/动量过度抑制
BUSYNESS_CAP = 0.3


def _apply_busy_fallback_to_output(out: Dict[str, Any], state: Dict[str, Any]) -> None:
    """每轮会话开始：按当前时间给 bot 的 busy 赋兜底值，写入 out['mood_state']['busyness']；上限为 BUSYNESS_CAP。"""
    dt = None
    ct = state.get("current_time")
    if isinstance(ct, str) and ct.strip():
        try:
            s = ct.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    busy = get_busy_fallback_from_schedule(dt, use_utc=False)
    busy = min(float(busy), BUSYNESS_CAP)
    mood = dict(out.get("mood_state") or {})
    mood["busyness"] = busy
    mood.setdefault("pad_scale", "m1_1")
    out["mood_state"] = mood


def _ensure_messages_have_timestamp(messages: List[BaseMessage], default_ts: Optional[str] = None) -> List[BaseMessage]:
    """为没有 timestamp 的消息补上 default_ts（ISO 字符串），便于各处展示。"""
    ts = default_ts or datetime.now().isoformat()
    out: List[BaseMessage] = []
    for msg in messages:
        kwargs = getattr(msg, "additional_kwargs", None) or {}
        if kwargs.get("timestamp"):
            out.append(msg)
            continue
        new_kwargs = {**kwargs, "timestamp": ts}
        if isinstance(msg, HumanMessage):
            out.append(HumanMessage(content=msg.content, additional_kwargs=new_kwargs))
        elif isinstance(msg, AIMessage):
            out.append(AIMessage(content=msg.content, additional_kwargs=new_kwargs))
        else:
            out.append(SystemMessage(content=msg.content, additional_kwargs=new_kwargs))
    return out


def _merge_and_dedup_buffers(history: List[BaseMessage], live: List[BaseMessage]) -> List[BaseMessage]:
    """
    合并 history + live，并去重。
    不能用 `msg in history` 这种对象比较：DB 读取出来的消息是新对象，会导致重复拼接。
    这里按 (role/type, content) 做去重键（时间戳缺失时也能去重）。
    """

    def _key(m: BaseMessage) -> tuple:
        t = getattr(m, "type", "") or ""
        c = getattr(m, "content", str(m)) or ""
        return (str(t), str(c))

    out: List[BaseMessage] = []
    seen: set[tuple] = set()
    for m in list(history or []) + list(live or []):
        k = _key(m)
        if k in seen:
            continue
        seen.add(k)
        out.append(m)
    return out

if TYPE_CHECKING:
    from app.services.memory.base import MemoryBase


_DB_MANAGER = None


def _get_db_manager():
    """
    懒加载 DBManager。
    - 未配置 DATABASE_URL 时返回 None
    - 仅在 loader/writer 触发，符合“早读晚写”
    """
    global _DB_MANAGER
    if _DB_MANAGER is not None:
        return _DB_MANAGER
    if not os.getenv("DATABASE_URL"):
        return None
    try:
        from app.core.database import DBManager

        _DB_MANAGER = DBManager.from_env()
        return _DB_MANAGER
    except Exception:
        return None


def _run_async(coro):
    """在同步节点里执行 async DB 调用。若运行在已有 event loop 中，请改用异步入口。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Detected running event loop; please use an async graph entry (ainvoke) for DB operations.")


def create_loader_node(memory_service: "MemoryBase") -> Callable[[AgentState], dict]:
    """创建 loader 节点。返回的节点为 async，需用 app.ainvoke() 多轮对话以避免事件循环冲突。"""

    async def loader_node(state: AgentState) -> dict:
        user_id = state.get("user_id") or "default_user"
        messages = state.get("messages", []) or []
        last = messages[-1] if messages else None
        user_input_raw = getattr(last, "content", "") if last else state.get("user_input", "")
        # external 通道保护：避免 internal prompt / debug dump 混入 user_input
        try:
            user_input = sanitize_external_text(str(user_input_raw or ""))
        except Exception as e:
            leak, reasons = detect_internal_leak(str(user_input_raw or ""))
            print(f"[Loader] ⚠ external_user_text 被污染，已拦截（reasons={reasons[:2]}）：{e}")
            user_input = ""
        chat_buffer = state.get("chat_buffer") or messages

        def _role(msg) -> str:
            if hasattr(msg, "type"):
                t = getattr(msg, "type", "") or ""
                return "user" if "human" in t.lower() or "user" in t.lower() else "bot"
            return "user" if "user" in str(type(msg)).lower() else "bot"

        def _ts(msg) -> str:
            kwargs = getattr(msg, "additional_kwargs", None) or {}
            t = kwargs.get("timestamp") or ""
            return f" [{t}]" if t else ""

        def _format_chat(buf: List[BaseMessage], limit: int = 30) -> str:
            lines = [
                f"{'User' if _role(m) == 'user' else 'Bot'}{_ts(m)}: {getattr(m, 'content', str(m))}"
                for m in (buf or [])[-limit:]
            ]
            return "\n".join(lines)

        def _build_memory_context(summary: str, retrieved: List[str], buf: List[BaseMessage]) -> str:
            parts: List[str] = []
            parts.append("【近期压缩摘要】")
            parts.append(summary.strip() if summary else "（无）")
            parts.append("")
            parts.append("【全量长期记忆召回片段】")
            if retrieved:
                parts.extend([f"- {x}" for x in retrieved])
            else:
                parts.append("（无）")
            parts.append("")
            parts.append("【当前会话原文窗口】")
            parts.append(_format_chat(buf) if buf else "（无）")
            return "\n".join(parts).strip()

        db = _get_db_manager()
        if db:
            bot_id = state.get("bot_id") or (state.get("bot_basic_info") or {}).get("name") or "default_bot"
            db_data: Dict[str, Any] = await db.load_state(str(user_id), str(bot_id))
            bot_name = (db_data.get("bot_basic_info") or {}).get("name") or "?"
            user_name = (db_data.get("user_basic_info") or {}).get("name") or "?"
            print(f"[Loader] 从 DB 加载 user_id={user_id}, bot_id={bot_id}, bot_name={bot_name}, user_name={user_name}")
            history = db_data.get("chat_buffer") or []
            merged_buffer = _merge_and_dedup_buffers(list(history), list(chat_buffer))
            merged_buffer = _ensure_messages_have_timestamp(merged_buffer, state.get("current_time"))
            summary = db_data.get("conversation_summary") or state.get("conversation_summary") or ""

            # 上下文化 query：用户基本信息 + 当前问题 + 近期摘要（多维度召回）
            user_basic = db_data.get("user_basic_info") or {}
            user_profile_hints = []
            if user_basic.get("name"):
                user_profile_hints.append(f"名字：{user_basic['name']}")
            if user_basic.get("occupation"):
                user_profile_hints.append(f"职业：{user_basic['occupation']}")
            if user_basic.get("age"):
                user_profile_hints.append(f"年龄：{user_basic['age']}")
            if user_basic.get("location"):
                user_profile_hints.append(f"地区：{user_basic['location']}")

            ctx_query_parts = [
                "[用户基本信息]",
                " | ".join(user_profile_hints) if user_profile_hints else "（暂无基本信息）",
                "[当前消息]",
                user_input,
                "[最近对话摘要]",
                summary if summary else "（暂无摘要）",
            ]
            ctx_query = "\n".join(ctx_query_parts).strip()

            retrieved: List[str] = []
            try:
                rel_id = str(db_data.get("relationship_id") or "")
                notes = await db.search_notes(relationship_id=rel_id, query=ctx_query, limit=6)
                trans = await db.search_transcripts(relationship_id=rel_id, query=ctx_query, limit=6)
                merged_items = list(notes) + list(trans)
                seen: set[str] = set()
                for it in merged_items:
                    if it.get("store") == "B":
                        line = f"[B/{it.get('note_type') or 'note'}] {it.get('content') or ''} (src={it.get('source_pointer') or ''})"
                    else:
                        ctx = it.get("short_context") or it.get("topic") or ""
                        u = str(it.get("user_text") or "")[:60]
                        b = str(it.get("bot_text") or "")[:60]
                        line = f"[A/{it.get('created_at')}] {ctx} U:{u} B:{b} (id=transcript:{it.get('id')})"
                    line = line.strip()
                    if not line or line in seen:
                        continue
                    seen.add(line)
                    retrieved.append(line)
                    if len(retrieved) >= 8:
                        break
            except Exception as e:
                print(f"[Loader] 召回失败（DB，忽略继续）: {e}")

            memory_context = _build_memory_context(summary, retrieved, merged_buffer)

            # 记忆卫生：过滤“自称助手/AI”的模板片段，避免污染下游（inner_monologue/reasoner/planner）
            summary_clean = sanitize_memory_text(summary)
            retrieved_clean = filter_retrieved_memories(retrieved)

            seconds_since_last = _seconds_since_last_message(merged_buffer, user_input)
            current_stage = db_data.get("current_stage") or state.get("current_stage") or "initiating"
            conversation_momentum = _resolve_conversation_momentum(
                seconds_since_last,
                current_stage,
                db_data.get("conversation_momentum") or state.get("conversation_momentum"),
            )
            # 消息间隔 >= 4 小时视为新 session，轮次从头计 0
            new_session = seconds_since_last is None or seconds_since_last >= COLD_START_THRESHOLD_SEC
            turn_count = 0 if new_session else int(db_data.get("turn_count_in_session") or 0)

            out = {
                "bot_id": str(bot_id),
                "relationship_id": str(db_data.get("relationship_id") or ""),
                "relationship_state": _ensure_rel_scale(db_data.get("relationship_state") or {}),  # 6D + rel_scale 缺省 0_1
                "reply_duration_seconds_list": db_data.get("reply_duration_seconds_list") or [],
                "mood_state": db_data.get("mood_state") or {},
                "current_stage": current_stage,
                "bot_basic_info": db_data.get("bot_basic_info") or state.get("bot_basic_info") or {},
                "bot_big_five": db_data.get("bot_big_five") or state.get("bot_big_five") or {},
                "bot_persona": db_data.get("bot_persona") or state.get("bot_persona") or {},
                "user_basic_info": db_data.get("user_basic_info") or state.get("user_basic_info") or {},
                "user_inferred_profile": db_data.get("user_inferred_profile") or state.get("user_inferred_profile") or {},
                "relationship_assets": db_data.get("relationship_assets") or state.get("relationship_assets") or {},
                "spt_info": db_data.get("spt_info") or state.get("spt_info") or {},
                "conversation_summary": summary_clean,
                "retrieved_memories": retrieved_clean,
                "memory_context": _build_memory_context(summary_clean, retrieved_clean, merged_buffer),
                # 任务池（供 planner/evolver 使用）：必须从 DB 透传回来，否则会一直是空
                "bot_task_list": db_data.get("bot_task_list") or state.get("bot_task_list") or [],
                "current_session_tasks": db_data.get("current_session_tasks") or state.get("current_session_tasks") or [],
                "external_user_text": user_input,
                "user_input": user_input,  # 向后兼容：下游请优先使用 external_user_text
                "chat_buffer": merged_buffer,
                "seconds_since_last_message": seconds_since_last,
                "conversation_momentum": conversation_momentum,
                "user_profile": db_data.get("user_inferred_profile") or {},
                "memories": db_data.get("conversation_summary") or "",
                "turn_count_in_session": turn_count,
                "daily_topics": _load_daily_topics(),
            }
            _ctx = _load_daily_context(bot_id=str(bot_id))
            out["daily_topics"] = _ctx["topics"]
            out["bot_recent_activities"] = _ctx["bot_recent"]
            if new_session:
                ra = dict(out.get("relationship_assets") or {})
                ra["session_basic_info_pending_task_ids"] = get_session_basic_info_pending_task_ids(
                    out.get("user_basic_info") or {}
                )
                out["relationship_assets"] = ra
            _apply_busy_fallback_to_output(out, state)
            return out

        try:
            from app.core.local_store import LocalStoreManager
            store = LocalStoreManager()
            bot_id = state.get("bot_id") or (state.get("bot_basic_info") or {}).get("name") or "default_bot"
            local_data: Dict[str, Any] = store.load_state(str(user_id), str(bot_id))
            history = local_data.get("chat_buffer") or []
            merged_buffer = _merge_and_dedup_buffers(list(history), list(chat_buffer))
            merged_buffer = _ensure_messages_have_timestamp(merged_buffer, state.get("current_time"))
            summary = local_data.get("conversation_summary") or state.get("conversation_summary") or ""

            # 上下文化 query：用户基本信息 + 当前问题 + 近期摘要（多维度召回）
            user_basic = local_data.get("user_basic_info") or {}
            user_profile_hints = []
            if user_basic.get("name"):
                user_profile_hints.append(f"名字：{user_basic['name']}")
            if user_basic.get("occupation"):
                user_profile_hints.append(f"职业：{user_basic['occupation']}")
            if user_basic.get("age"):
                user_profile_hints.append(f"年龄：{user_basic['age']}")
            if user_basic.get("location"):
                user_profile_hints.append(f"地区：{user_basic['location']}")

            ctx_query_parts = [
                "[用户基本信息]",
                " | ".join(user_profile_hints) if user_profile_hints else "（暂无基本信息）",
                "[当前消息]",
                user_input,
                "[最近对话摘要]",
                summary if summary else "（暂无摘要）",
            ]
            ctx_query = "\n".join(ctx_query_parts).strip()
            retrieved: List[str] = []
            try:
                notes = store.search_notes(str(user_id), str(bot_id), ctx_query, limit=6)
                trans = store.search_transcripts(str(user_id), str(bot_id), ctx_query, limit=6)
                merged_items = list(notes) + list(trans)
                seen: set[str] = set()
                for it in merged_items:
                    if it.get("store") == "B":
                        line = f"[B/{it.get('note_type') or 'note'}] {it.get('content') or ''} (src={it.get('source_pointer') or ''})"
                    else:
                        ctx = it.get("short_context") or it.get("topic") or ""
                        u = str(it.get("user_text") or "")[:60]
                        b = str(it.get("bot_text") or "")[:60]
                        line = f"[A/{it.get('created_at')}] {ctx} U:{u} B:{b} (id=transcript:{it.get('id')})"
                    line = line.strip()
                    if not line or line in seen:
                        continue
                    seen.add(line)
                    retrieved.append(line)
                    if len(retrieved) >= 8:
                        break
            except Exception as e:
                print(f"[Loader] 召回失败（LocalStore，忽略继续）: {e}")

            memory_context = _build_memory_context(summary, retrieved, merged_buffer)

            summary_clean = sanitize_memory_text(summary)
            retrieved_clean = filter_retrieved_memories(retrieved)

            seconds_since_last = _seconds_since_last_message(merged_buffer, user_input)
            current_stage = local_data.get("current_stage") or state.get("current_stage") or "initiating"
            conversation_momentum = _resolve_conversation_momentum(
                seconds_since_last,
                current_stage,
                local_data.get("conversation_momentum") or state.get("conversation_momentum"),
            )
            # 消息间隔 >= 4 小时视为新 session，轮次从头计 0
            new_session = seconds_since_last is None or seconds_since_last >= COLD_START_THRESHOLD_SEC
            turn_count = 0 if new_session else int(local_data.get("turn_count_in_session") or 0)

            out = {
                "bot_id": str(bot_id),
                "relationship_state": _ensure_rel_scale(local_data.get("relationship_state") or {}),  # 6D + rel_scale 缺省 0_1
                "reply_duration_seconds_list": local_data.get("reply_duration_seconds_list") or [],
                "mood_state": local_data.get("mood_state") or {},
                "current_stage": current_stage,
                "bot_basic_info": local_data.get("bot_basic_info") or state.get("bot_basic_info") or {},
                "bot_big_five": local_data.get("bot_big_five") or state.get("bot_big_five") or {},
                "bot_persona": local_data.get("bot_persona") or state.get("bot_persona") or {},
                "user_basic_info": local_data.get("user_basic_info") or state.get("user_basic_info") or {},
                "user_inferred_profile": local_data.get("user_inferred_profile") or state.get("user_inferred_profile") or {},
                "relationship_assets": local_data.get("relationship_assets") or state.get("relationship_assets") or {},
                "bot_task_list": (local_data.get("relationship_assets") or {}).get("bot_task_list") or [],
                "current_session_tasks": (local_data.get("relationship_assets") or {}).get("current_session_tasks") or [],
                "spt_info": local_data.get("spt_info") or state.get("spt_info") or {},
                "conversation_summary": summary_clean,
                "retrieved_memories": retrieved_clean,
                "memory_context": _build_memory_context(summary_clean, retrieved_clean, merged_buffer),
                "external_user_text": user_input,
                "user_input": user_input,
                "chat_buffer": merged_buffer,
                "seconds_since_last_message": seconds_since_last,
                "conversation_momentum": conversation_momentum,
                "user_profile": local_data.get("user_inferred_profile") or {},
                "memories": local_data.get("conversation_summary") or "",
                "turn_count_in_session": turn_count,
                "daily_topics": _load_daily_topics(),
            }
            _ctx2 = _load_daily_context(bot_id=str(bot_id))
            out["daily_topics"] = _ctx2["topics"]
            out["bot_recent_activities"] = _ctx2["bot_recent"]
            if new_session:
                ra = dict(out.get("relationship_assets") or {})
                ra["session_basic_info_pending_task_ids"] = get_session_basic_info_pending_task_ids(
                    out.get("user_basic_info") or {}
                )
                out["relationship_assets"] = ra
            _apply_busy_fallback_to_output(out, state)
            return out
        except Exception:
            pass

        profile = memory_service.get_profile(user_id)
        memories = memory_service.get_memories(user_id, limit=10)
        memory_context = _build_memory_context("", [], chat_buffer)
        seconds_since_last = _seconds_since_last_message(chat_buffer, user_input)
        current_stage = state.get("current_stage") or "initiating"
        conversation_momentum = _resolve_conversation_momentum(
            seconds_since_last,
            current_stage,
            state.get("conversation_momentum"),
        )
        # 消息间隔 >= 4 小时视为新 session，轮次从头计 0
        new_session = seconds_since_last is None or seconds_since_last >= COLD_START_THRESHOLD_SEC
        turn_count = 0 if new_session else int(state.get("turn_count_in_session") or 0)
        return {
            "user_profile": profile,
            "memories": memories,
            "external_user_text": user_input,
            "user_input": user_input,
            "chat_buffer": chat_buffer,
            "memory_context": memory_context,
            "seconds_since_last_message": seconds_since_last,
            "conversation_momentum": conversation_momentum,
            "turn_count_in_session": turn_count,
        }

    return loader_node

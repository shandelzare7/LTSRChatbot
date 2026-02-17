"""入口加载节点：优先从 DB 读取状态（Load Early），无 DB 时落本地文件（再无则回退内存）。"""
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import os
import asyncio
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from app.state import AgentState

from app.lats.prompt_utils import sanitize_memory_text, filter_retrieved_memories
from utils.external_text import sanitize_external_text, detect_internal_leak


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

        def _format_chat(buf: List[BaseMessage], limit: int = 15) -> str:
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

            # 上下文化 query：当前问题 + 近期摘要槽位（强稳定召回）
            ctx_query = f"{user_input}\n\n[近期摘要]\n{summary}".strip()

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

            return {
                "bot_id": str(bot_id),
                "relationship_id": str(db_data.get("relationship_id") or ""),
                "relationship_state": db_data.get("relationship_state") or {},
                "mood_state": db_data.get("mood_state") or {},
                "current_stage": db_data.get("current_stage") or state.get("current_stage") or "initiating",
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
                "user_profile": db_data.get("user_inferred_profile") or {},
                "memories": db_data.get("conversation_summary") or "",
            }

        try:
            from app.core.local_store import LocalStoreManager
            store = LocalStoreManager()
            bot_id = state.get("bot_id") or (state.get("bot_basic_info") or {}).get("name") or "default_bot"
            local_data: Dict[str, Any] = store.load_state(str(user_id), str(bot_id))
            history = local_data.get("chat_buffer") or []
            merged_buffer = _merge_and_dedup_buffers(list(history), list(chat_buffer))
            merged_buffer = _ensure_messages_have_timestamp(merged_buffer, state.get("current_time"))
            summary = local_data.get("conversation_summary") or state.get("conversation_summary") or ""
            ctx_query = f"{user_input}\n\n[近期摘要]\n{summary}".strip()
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

            return {
                "bot_id": str(bot_id),
                "relationship_state": local_data.get("relationship_state") or {},
                "mood_state": local_data.get("mood_state") or {},
                "current_stage": local_data.get("current_stage") or state.get("current_stage") or "initiating",
                "bot_basic_info": local_data.get("bot_basic_info") or state.get("bot_basic_info") or {},
                "bot_big_five": local_data.get("bot_big_five") or state.get("bot_big_five") or {},
                "bot_persona": local_data.get("bot_persona") or state.get("bot_persona") or {},
                "user_basic_info": local_data.get("user_basic_info") or state.get("user_basic_info") or {},
                "user_inferred_profile": local_data.get("user_inferred_profile") or state.get("user_inferred_profile") or {},
                "relationship_assets": local_data.get("relationship_assets") or state.get("relationship_assets") or {},
                "spt_info": local_data.get("spt_info") or state.get("spt_info") or {},
                "conversation_summary": summary_clean,
                "retrieved_memories": retrieved_clean,
                "memory_context": _build_memory_context(summary_clean, retrieved_clean, merged_buffer),
                "external_user_text": user_input,
                "user_input": user_input,
                "chat_buffer": merged_buffer,
                "user_profile": local_data.get("user_inferred_profile") or {},
                "memories": local_data.get("conversation_summary") or "",
            }
        except Exception:
            pass

        profile = memory_service.get_profile(user_id)
        memories = memory_service.get_memories(user_id, limit=10)
        memory_context = _build_memory_context("", [], chat_buffer)
        return {
            "user_profile": profile,
            "memories": memories,
            "external_user_text": user_input,
            "user_input": user_input,
            "chat_buffer": chat_buffer,
            "memory_context": memory_context,
        }

    return loader_node

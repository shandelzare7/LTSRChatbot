"""
memory_manager.py

实现记忆系统的「每轮更新」部分：
- 每轮更新 conversation_summary（近期压缩摘要）
- 写入 Memory Store A：Raw Transcript Store（全文 + 元数据）
- 写入 Memory Store B：Derived Notes Store（稳定事实/偏好/决策等，带 source_pointer 溯源）

该节点放在 stage_manager -> memory_writer 之间：
stage_manager 结束后，已经有 user_input + final_response/draft_response。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import os
import uuid

from app.state import AgentState
from utils.llm_json import parse_json_from_llm


_DB_MANAGER = None


def _get_db_manager():
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


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def create_memory_manager_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """
    记忆管理节点（每轮更新）：
    - 用一次 LLM 调用同时做：
      1) 更新 conversation_summary（running summary）
      2) 抽取 derived notes（稳定事实/偏好/决策/正在做什么）
      3) 生成 transcript 元数据（entities/topic/importance/short_context）
    - 落盘：优先 DB（transcripts/derived_notes），无 DB 时用 LocalStore jsonl。
    """

    async def node(state: AgentState) -> Dict[str, Any]:
        user_id = str(state.get("user_id") or "default_user")
        bot_id = str(state.get("bot_id") or "default_bot")
        relationship_id = state.get("relationship_id")

        now = str(state.get("current_time") or "")
        user_input = str(state.get("user_input") or "")
        bot_text = str(state.get("final_response") or state.get("draft_response") or "").strip()
        prev_summary = str(state.get("conversation_summary") or "").strip()

        session_id = state.get("session_id")
        thread_id = state.get("thread_id")
        turn_index = state.get("turn_index")

        # 防御：没有内容就不更新
        if not user_input and not bot_text:
            return {}

        prompt = f"""你是“记忆系统”的写入器。你的任务是：基于【旧摘要】+【本轮对话】，输出一个严格的 JSON，用于更新摘要并沉淀稳定记忆。

要求（重要，影响稳定性）：
1) 摘要要“可持续更新”：在旧摘要基础上增量更新，不要推翻重写；保持精炼、客观、可复用。
2) 只写“稳定事实/偏好/正在做什么/已决策/关键约束”，不要猜测、不要心理分析。
3) Derived notes 要少而精（0~5 条），每条必须是可检验/可复用的信息。
4) entities/topic/short_context 也要保守，不确定就留空或给更泛化的表述。
5) importance 建议 0~1（越接近 1 越重要）。

【旧摘要】
{prev_summary if prev_summary else "（空）"}

【本轮对话】
- time: {now}
- user: {user_input}
- bot: {bot_text}

输出 JSON schema（必须完全符合）：
{{
  "new_summary": "string（建议 80~220 字）",
  "transcript_meta": {{
    "entities": ["string", "..."],
    "topic": "string",
    "importance": 0.0,
    "short_context": "string（<=40字）"
  }},
  "notes": [
    {{
      "note_type": "fact|preference|activity|decision|other",
      "content": "string",
      "importance": 0.0
    }}
  ]
}}
"""

        # 1) 调 LLM 抽取
        try:
            msg = llm_invoker.invoke(prompt)
            content = getattr(msg, "content", str(msg)) or ""
            data = parse_json_from_llm(content) or {}
        except Exception as e:
            data = {}

        new_summary = str(data.get("new_summary") or prev_summary or "").strip()
        meta = data.get("transcript_meta") or {}
        entities = meta.get("entities") or []
        if not isinstance(entities, list):
            entities = []
        topic = meta.get("topic")
        short_context = meta.get("short_context")
        importance = _safe_float(meta.get("importance"))
        notes = data.get("notes") or []
        if not isinstance(notes, list):
            notes = []

        # 2) Store A/B 落盘
        db = _get_db_manager()
        if db and relationship_id:
            # DB 模式：写 transcripts + derived_notes
            try:
                transcript_id = await db.append_transcript(
                    relationship_id=str(relationship_id),
                    user_text=user_input,
                    bot_text=bot_text,
                    session_id=str(session_id) if session_id else None,
                    thread_id=str(thread_id) if thread_id else None,
                    turn_index=int(turn_index) if isinstance(turn_index, int) else None,
                    entities={"entities": entities},
                    topic=str(topic) if topic else None,
                    importance=importance,
                    short_context=str(short_context) if short_context else None,
                )
                # notes 写入（带 source_pointer）
                for n in notes:
                    n.setdefault("source_pointer", f"transcript:{transcript_id}")
                await db.append_notes(
                    relationship_id=str(relationship_id),
                    transcript_id=str(transcript_id),
                    notes=notes,  # type: ignore[arg-type]
                )
            except Exception as e:
                print(f"[MemoryManager] DB 写入失败: {e}")
        else:
            # local store 模式
            try:
                from app.core.local_store import LocalStoreManager

                store = LocalStoreManager()
                transcript_id = str(uuid.uuid4())
                store.append_transcript(
                    user_id,
                    bot_id,
                    {
                        "id": transcript_id,
                        "created_at": now,
                        "session_id": session_id,
                        "thread_id": thread_id,
                        "turn_index": turn_index,
                        "user_text": user_input,
                        "bot_text": bot_text,
                        "entities": {"entities": entities},
                        "topic": topic,
                        "importance": importance,
                        "short_context": short_context,
                    },
                )
                for n in notes:
                    n.setdefault("source_pointer", f"transcript:{transcript_id}")
                    n.setdefault("transcript_id", transcript_id)
                store.append_derived_notes(user_id, bot_id, notes)  # type: ignore[arg-type]
            except Exception as e:
                print(f"[MemoryManager] LocalStore 写入失败: {e}")

        print("[MemoryManager] done")
        return {
            "conversation_summary": new_summary,
        }

    return node


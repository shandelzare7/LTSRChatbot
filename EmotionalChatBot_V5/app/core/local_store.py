from __future__ import annotations

"""
local_store.py

本地文件持久化（用于开发期/无 Supabase 时）：
- 遵循 Load Early, Commit Late：只在 loader / memory_writer 节点读写
- 数据落盘到 EmotionalChatBot_V5/local_data/（默认）

结构（每个 bot_id + user_id 一个目录）：
  local_data/
    rel__{bot_id}__{user_id}/
      relationship.json
      messages.jsonl
      memories.jsonl
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from utils.yaml_loader import get_project_root


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_name(x: str) -> str:
    # 文件夹名安全化
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(x))


def _msg_to_row(msg: BaseMessage) -> Dict[str, Any]:
    t = getattr(msg, "type", "")
    if t == "human":
        role = "user"
    elif t == "ai":
        role = "ai"
    else:
        role = "system"
    return {"role": role, "content": getattr(msg, "content", str(msg)), "created_at": _now_iso(), "metadata": {}}


def _row_to_msg(role: str, content: str) -> BaseMessage:
    if role == "user":
        return HumanMessage(content=content)
    if role == "ai":
        return AIMessage(content=content)
    return SystemMessage(content=content)


@dataclass
class LocalStorePaths:
    root: Path
    rel_dir: Path
    relationship_json: Path
    messages_jsonl: Path
    memories_jsonl: Path


class LocalStoreManager:
    def __init__(self, root_dir: Optional[str] = None) -> None:
        if root_dir is None:
            # 默认落在 EmotionalChatBot_V5/local_data
            root_dir = str(Path(get_project_root()) / "local_data")
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, user_id: str, bot_id: str) -> LocalStorePaths:
        rel_name = f"rel__{_safe_name(bot_id)}__{_safe_name(user_id)}"
        rel_dir = self.root / rel_name
        rel_dir.mkdir(parents=True, exist_ok=True)
        return LocalStorePaths(
            root=self.root,
            rel_dir=rel_dir,
            relationship_json=rel_dir / "relationship.json",
            messages_jsonl=rel_dir / "messages.jsonl",
            memories_jsonl=rel_dir / "memories.jsonl",
        )

    def load_state(self, user_id: str, bot_id: str) -> Dict[str, Any]:
        p = self._paths(user_id, bot_id)

        # relationship.json
        if p.relationship_json.exists():
            rel = json.loads(p.relationship_json.read_text(encoding="utf-8") or "{}")
        else:
            rel = {
                "current_stage": "initiating",
                "relationship_state": {"closeness": 0, "trust": 0, "liking": 0, "respect": 0, "warmth": 0, "power": 50},
                "mood_state": {"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0},
                "user_inferred_profile": {},
                "relationship_assets": {"topic_history": [], "breadth_score": 0, "max_spt_depth": 1},
                "spt_info": {},
                "conversation_summary": "",
                "bot_basic_info": {},
                "bot_big_five": {},
                "bot_persona": {},
                "user_basic_info": {},
            }
            p.relationship_json.write_text(json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")

        # last 20 messages (old -> new)
        chat_rows: List[Dict[str, Any]] = []
        if p.messages_jsonl.exists():
            lines = p.messages_jsonl.read_text(encoding="utf-8").splitlines()
            for ln in lines[-20:]:
                try:
                    chat_rows.append(json.loads(ln))
                except Exception:
                    continue
        chat_buffer = [_row_to_msg(str(r.get("role", "user")), str(r.get("content", ""))) for r in chat_rows]

        return {
            "relationship_state": rel.get("relationship_state") or {},
            "mood_state": rel.get("mood_state") or {},
            "current_stage": rel.get("current_stage") or "initiating",
            "user_inferred_profile": rel.get("user_inferred_profile") or {},
            "relationship_assets": rel.get("relationship_assets") or {},
            "spt_info": rel.get("spt_info") or {},
            "conversation_summary": rel.get("conversation_summary") or "",
            "bot_basic_info": rel.get("bot_basic_info") or {},
            "bot_big_five": rel.get("bot_big_five") or {},
            "bot_persona": rel.get("bot_persona") or {},
            "user_basic_info": rel.get("user_basic_info") or {},
            "chat_buffer": chat_buffer,
        }

    def save_turn(self, user_id: str, bot_id: str, state: Dict[str, Any], new_memory: Optional[str] = None) -> None:
        p = self._paths(user_id, bot_id)

        # append messages.jsonl
        user_input = str(state.get("user_input") or "")
        final_response = str(state.get("final_response") or state.get("draft_response") or "")

        detection_category = state.get("detection_category") or state.get("detection_result")
        latency = (state.get("humanized_output") or {}).get("total_latency_seconds")

        def _append_row(role: str, content: str, metadata: Dict[str, Any]):
            row = {
                "role": role,
                "content": content,
                "created_at": _now_iso(),
                "metadata": metadata,
            }
            with open(p.messages_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if user_input:
            _append_row("user", user_input, {"source": "turn"})
        if final_response:
            _append_row(
                "ai",
                final_response,
                {"source": "turn", "detection_category": detection_category, "latency": latency},
            )

        # update relationship.json (commit late)
        rel = {}
        if p.relationship_json.exists():
            try:
                rel = json.loads(p.relationship_json.read_text(encoding="utf-8") or "{}")
            except Exception:
                rel = {}

        rel.update(
            {
                "current_stage": state.get("current_stage") or rel.get("current_stage") or "initiating",
                "relationship_state": state.get("relationship_state") or rel.get("relationship_state") or {},
                "mood_state": state.get("mood_state") or rel.get("mood_state") or {},
                "user_inferred_profile": state.get("user_inferred_profile") or rel.get("user_inferred_profile") or {},
                "relationship_assets": state.get("relationship_assets") or rel.get("relationship_assets") or {},
                "spt_info": state.get("spt_info") or rel.get("spt_info") or {},
                "conversation_summary": state.get("conversation_summary") or rel.get("conversation_summary") or "",
                "bot_basic_info": state.get("bot_basic_info") or rel.get("bot_basic_info") or {},
                "bot_big_five": state.get("bot_big_five") or rel.get("bot_big_five") or {},
                "bot_persona": state.get("bot_persona") or rel.get("bot_persona") or {},
                "user_basic_info": state.get("user_basic_info") or rel.get("user_basic_info") or {},
                "updated_at": _now_iso(),
            }
        )
        p.relationship_json.write_text(json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")

        # memories.jsonl (optional)
        if new_memory is None:
            new_memory = state.get("generated_new_memory_text") or state.get("new_memory_content")
        if new_memory:
            row = {"content": str(new_memory), "created_at": _now_iso()}
            with open(p.memories_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from utils.yaml_loader import get_project_root
from app.core.profile_factory import generate_bot_profile, generate_user_profile
from app.core.relationship_templates import get_random_relationship_template


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


def _row_to_msg(role: str, content: str, created_at: Optional[str] = None) -> BaseMessage:
    kwargs = {"timestamp": created_at} if created_at else {}
    if role == "user":
        return HumanMessage(content=content, additional_kwargs=kwargs)
    if role == "ai":
        return AIMessage(content=content, additional_kwargs=kwargs)
    return SystemMessage(content=content, additional_kwargs=kwargs)


def _merge_and_sanitize_relationship(prev: Dict[str, Any], inc: Dict[str, Any]) -> Dict[str, float]:
    """
    合并关系维度写入：
    - 避免部分字段覆盖导致其它维度丢失（如 respect 突然变 0）
    - 兼容旧 points(0-100) 写入
    - 截断单轮跳变，避免 0.22 -> 1.00 这种爆炸
    """
    prev = dict(prev or {})
    inc = dict(inc or {})

    def _norm01(v: Any) -> float:
        try:
            x = float(v)
        except Exception:
            return 0.0
        if x > 1.0:
            if x <= 100.0:
                x = x / 100.0
            else:
                x = 1.0
        return float(max(0.0, min(1.0, x)))

    merged: Dict[str, Any] = dict(prev)
    merged.update(inc)
    for k, default in (
        ("closeness", 0.3),
        ("trust", 0.3),
        ("liking", 0.3),
        ("respect", 0.3),
        ("warmth", 0.3),
        ("power", 0.5),
    ):
        if k not in merged:
            merged[k] = prev.get(k, default)

    max_step = 0.20
    out: Dict[str, float] = {}
    for k in ("closeness", "trust", "liking", "respect", "warmth", "power"):
        old = _norm01(prev.get(k, merged.get(k)))
        new = _norm01(merged.get(k))
        d = new - old
        if abs(d) > max_step:
            new = old + (max_step if d > 0 else -max_step)
        out[k] = round(new, 4)

    return out


@dataclass
class LocalStorePaths:
    root: Path
    rel_dir: Path
    relationship_json: Path
    messages_jsonl: Path
    memories_jsonl: Path
    transcripts_jsonl: Path
    derived_notes_jsonl: Path


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
            transcripts_jsonl=rel_dir / "transcripts.jsonl",
            derived_notes_jsonl=rel_dir / "derived_notes.jsonl",
        )

    @staticmethod
    def _tokenize_query(query: str) -> List[str]:
        seps = [" ", "\n", "\t", ",", "，", ".", "。", "?", "？", "!", "！", ";", "；", ":", "：", "、", "（", "）", "(", ")", "[", "]"]
        s = str(query or "")
        for sep in seps:
            s = s.replace(sep, " ")
        toks = [t.strip() for t in s.split(" ") if t.strip()]
        out: List[str] = []
        for t in toks:
            if len(t) < 2:
                continue
            if t not in out:
                out.append(t)
        return out[:12]

    @staticmethod
    def _score_text(text: str, terms: List[str]) -> float:
        if not text:
            return 0.0
        t = text.lower()
        score = 0.0
        for w in terms:
            ww = w.lower()
            c = t.count(ww)
            if c:
                score += min(3, c)
        return float(score)

    def append_transcript(self, user_id: str, bot_id: str, record: Dict[str, Any]) -> None:
        """Store A（Raw Transcript）落盘：追加一行 JSON 到 transcripts.jsonl。"""
        p = self._paths(user_id, bot_id)
        row = dict(record or {})
        row.setdefault("created_at", _now_iso())
        with open(p.transcripts_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def append_derived_notes(self, user_id: str, bot_id: str, notes: List[Dict[str, Any]]) -> int:
        """Store B（Derived Notes）落盘：追加多行 JSON 到 derived_notes.jsonl。"""
        p = self._paths(user_id, bot_id)
        n = 0
        with open(p.derived_notes_jsonl, "a", encoding="utf-8") as f:
            for row in notes or []:
                content = str(row.get("content") or "").strip()
                if not content:
                    continue
                out = dict(row)
                out.setdefault("created_at", _now_iso())
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                n += 1
        return n

    def search_transcripts(self, user_id: str, bot_id: str, query: str, *, limit: int = 6, scan_limit: int = 200) -> List[Dict[str, Any]]:
        """本地 Store A 召回：扫描近期行并做 term-match 排序。"""
        p = self._paths(user_id, bot_id)
        if not p.transcripts_jsonl.exists():
            return []
        terms = self._tokenize_query(query)
        if not terms:
            return []
        lines = p.transcripts_jsonl.read_text(encoding="utf-8").splitlines()
        rows: List[Dict[str, Any]] = []
        for ln in lines[-int(scan_limit):]:
            try:
                rows.append(json.loads(ln))
            except Exception:
                continue
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for r in rows:
            text = " ".join(
                [
                    str(r.get("topic") or ""),
                    str(r.get("short_context") or ""),
                    str(r.get("user_text") or ""),
                    str(r.get("bot_text") or ""),
                ]
            )
            s = self._score_text(text, terms)
            if s <= 0:
                continue
            imp = float(r.get("importance") or 0.0)
            scored.append(
                (
                    s + imp,
                    {
                        "store": "A",
                        **r,
                    },
                )
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[: int(limit)]]

    def search_notes(self, user_id: str, bot_id: str, query: str, *, limit: int = 6, scan_limit: int = 400) -> List[Dict[str, Any]]:
        """本地 Store B 召回：扫描近期行并做 term-match 排序。"""
        p = self._paths(user_id, bot_id)
        if not p.derived_notes_jsonl.exists():
            return []
        terms = self._tokenize_query(query)
        if not terms:
            return []
        lines = p.derived_notes_jsonl.read_text(encoding="utf-8").splitlines()
        rows: List[Dict[str, Any]] = []
        for ln in lines[-int(scan_limit):]:
            try:
                rows.append(json.loads(ln))
            except Exception:
                continue
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for r in rows:
            s = self._score_text(str(r.get("content") or ""), terms)
            if s <= 0:
                continue
            imp = float(r.get("importance") or 0.0)
            scored.append((s + imp + 0.5, {"store": "B", **r}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[: int(limit)]]

    def load_state(self, user_id: str, bot_id: str) -> Dict[str, Any]:
        p = self._paths(user_id, bot_id)

        # relationship.json
        if p.relationship_json.exists():
            rel = json.loads(p.relationship_json.read_text(encoding="utf-8") or "{}")
        else:
            bot_basic_info, bot_big_five, bot_persona = generate_bot_profile(bot_id)
            user_basic_info, user_inferred_profile = generate_user_profile(user_id)
            # 随机选择一个关系维度模板
            relationship_template = get_random_relationship_template()
            rel = {
                "current_stage": "initiating",
                "relationship_state": relationship_template,
                "mood_state": {"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0},
                "user_inferred_profile": user_inferred_profile,
                "relationship_assets": {"topic_history": [], "breadth_score": 0, "max_spt_depth": 1},
                "spt_info": {},
                "conversation_summary": "",
                "bot_basic_info": bot_basic_info,
                "bot_big_five": bot_big_five,
                "bot_persona": bot_persona,
                "user_basic_info": user_basic_info,
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
        chat_buffer = [
            _row_to_msg(str(r.get("role", "user")), str(r.get("content", "")), r.get("created_at"))
            for r in chat_rows
        ]

        return {
            "relationship_state": _merge_and_sanitize_relationship(rel.get("relationship_state") or {}, {}),
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

    def clear_relationship(self, user_id: str, bot_id: str) -> bool:
        """
        危险操作：删除该 (bot_id, user_id) 的本地持久化目录（relationship/messages/memories/transcripts/notes 全清空）。
        用于“把记忆都清理干净后重新测试”。
        """
        p = self._paths(user_id, bot_id)
        if p.rel_dir.exists():
            shutil.rmtree(p.rel_dir, ignore_errors=True)
            return True
        return False

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
                # 关系维度：合并写入，避免部分字段把其它维度抹掉；统一到 0-1 并截断单轮跳变
                "relationship_state": (lambda prev, inc: _merge_and_sanitize_relationship(prev, inc))(
                    rel.get("relationship_state") or {},
                    state.get("relationship_state") or {},
                ),
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


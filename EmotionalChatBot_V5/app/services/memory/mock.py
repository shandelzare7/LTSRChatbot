"""记忆服务 Mock 实现：内存字典，不落库。"""
from typing import Any, Dict, List, Optional

from app.services.memory.base import MemoryBase


class MockMemory(MemoryBase):
    """内存版记忆，仅用于当前进程。"""

    def __init__(self) -> None:
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._memories: Dict[str, List[Dict[str, Any]]] = {}

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        return self._profiles.get(user_id, {"user_id": user_id, "intimacy": 0})

    def get_memories(self, user_id: str, limit: int = 10) -> str:
        items = self._memories.get(user_id, [])[-limit:]
        return "\n".join([x.get("content", "") for x in items]) if items else "（暂无记忆）"

    def append_memory(self, user_id: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self._memories.setdefault(user_id, []).append({"content": content, "meta": meta or {}})

"""记忆服务抽象基类。"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryBase(ABC):
    """记忆接口：按 user_id 加载档案与记忆片段。"""

    @abstractmethod
    def get_profile(self, user_id: str) -> Dict[str, Any]:
        """获取用户画像/档案。"""
        pass

    @abstractmethod
    def get_memories(self, user_id: str, limit: int = 10) -> str:
        """获取近期记忆文本，供上下文使用。返回拼接后的字符串。"""
        pass

    @abstractmethod
    def append_memory(self, user_id: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """写入一条记忆（可选落库）。"""
        pass

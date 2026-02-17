"""
会话管理模块
使用内存存储会话（生产环境建议使用 Redis）
"""
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional
import hashlib

# 会话存储（生产环境应使用 Redis）
_sessions: Dict[str, Dict] = {}


def generate_session_id() -> str:
    """生成唯一的 session_id"""
    return str(uuid.uuid4())


def generate_user_id_from_request(request) -> str:
    """
    基于请求生成或获取 user_id
    旧实现用 IP + User-Agent，IP 变化会导致“用户变了、历史像丢了”。
    现在改为生成随机匿名 ID，由 web_app 写入持久 cookie（web_user_id）后复用。
    """
    return f"web_user_{uuid.uuid4().hex[:16]}"


def create_session(user_id: str, bot_id: str) -> str:
    """创建新会话"""
    session_id = generate_session_id()
    _sessions[session_id] = {
        "user_id": user_id,
        "bot_id": bot_id,
        "created_at": datetime.now(),
        "last_active": datetime.now(),
    }
    return session_id


def get_session(session_id: str) -> Optional[Dict]:
    """获取会话信息"""
    if session_id not in _sessions:
        return None
    
    session = _sessions[session_id]
    
    # 检查是否过期（7天）
    if datetime.now() - session["last_active"] > timedelta(days=7):
        del _sessions[session_id]
        return None
    
    # 更新最后活跃时间
    session["last_active"] = datetime.now()
    return session


def delete_session(session_id: str) -> bool:
    """删除会话"""
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


def update_session_active(session_id: str) -> bool:
    """更新会话活跃时间"""
    if session_id in _sessions:
        _sessions[session_id]["last_active"] = datetime.now()
        return True
    return False

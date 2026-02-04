"""
模型定义：将 ChromaDBWrapper 的 personality、persona、extended_impressions 等
映射为 SQL 表中的 JSON 字段，intimacy / trust 单独作为索引字段便于查询。
"""
import os
from datetime import datetime

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    JSON,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import sessionmaker, declarative_base

# --- 1. 数据库连接设置 ---
# 默认使用本地 SQLite，生产环境可改为 Supabase 的 postgresql://...
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./emotional_bot.db")

connect_args = (
    {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    """初始化数据库表结构"""
    Base.metadata.create_all(bind=engine)


# --- 2. 核心数据模型 ---


class BotModel(Base):
    """
    机器人表
    对应原 Wrapper 的: basic_info, personality, persona, emotion_state
    """
    __tablename__ = "bots"

    id = Column(String, primary_key=True)  # 例如 "bot2"
    name = Column(String, default="Unknown")

    # --- 静态设定 ---
    # 存储 {name, gender, age, region, occupation...}
    basic_info = Column(JSON, default=dict)

    # 存储 {openness, conscientiousness, ...} (大五人格)
    personality = Column(JSON, default=dict)

    # 存储 {hobbies: [], background: ...} (详细人设)
    persona = Column(JSON, default=dict)

    # --- 动态状态 ---
    # 存储 {valence, arousal, dominance, busyness, current_activity}
    emotion_state = Column(JSON, default=dict)

    # 存储机器人的自身记忆 (非 RAG，而是关键设定记忆)，对应原 memories_json
    memories = Column(JSON, default=list)


class UserModel(Base):
    """
    用户表
    对应原 Wrapper 的: basic_info, facts (部分), persona (部分)
    """
    __tablename__ = "users"

    id = Column(String, primary_key=True)  # 例如 "user01"
    name = Column(String, default="User")

    # 存储 {gender, age}
    basic_info = Column(JSON, default=dict)

    # 用户画像摘要 (LLM 总结生成的纯文本，用于快速理解用户)
    profile_summary = Column(Text, default="")

    # 用户的个人设定/偏好，对应原 persona_json
    persona = Column(JSON, default=dict)


class RelationshipModel(Base):
    """
    关系表 (核心替代向量库部分)
    对应原 Wrapper 的: facts, extended_impressions, inner_monologue
    """
    __tablename__ = "relationships"

    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    bot_id = Column(String, ForeignKey("bots.id"), primary_key=True)

    # --- 核心数值 (提出来方便 SQL 排序/查询) ---
    intimacy = Column(Float, default=0.0)  # 对应 extended_impressions['relationship_closeness']
    trust = Column(Float, default=0.0)  # 对应 extended_impressions['trust']

    # --- 复杂印象 (JSON) ---
    # 存储 {respect, liking, warmth, power_distance...} 等其余 10+ 个维度
    extended_impressions = Column(JSON, default=dict)

    # --- 显式记忆 (替代 Vector RAG) ---
    # 存储事实列表 ["用户叫小榴莲", "用户住在北京"]，对应原 facts_json
    facts_list = Column(JSON, default=list)

    # --- 心理侧写 ---
    # 对应原 inner_monologue_json，记录 Bot 对该用户的最近一次内心独白
    last_inner_monologue = Column(JSON, default=dict)


class ChatLogModel(Base):
    """
    聊天记录表 (用于 Context Window)
    """
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)  # f"{bot_id}_{user_id}"

    role = Column(String)  # "user" or "assistant"
    content = Column(Text)

    # 记录当时的上下文，如 {"mode": "stress", "timestamp": ...}
    meta_data = Column(JSON, default=dict)

    timestamp = Column(DateTime, default=datetime.utcnow)

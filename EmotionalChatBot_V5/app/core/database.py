from __future__ import annotations

"""
database.py (Async SQLAlchemy)

遵循 "Load Early, Commit Late"：
- 流程头：Loader 只读（load_state）
- 流程尾：Writer 才写（save_turn，事务）

注意：
- 需要 PostgreSQL（Supabase 亦可）
- 连接串使用 env: DATABASE_URL（形如 postgresql+asyncpg://...）
"""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# -----------------------------
# ORM Base
# -----------------------------


class Base(DeclarativeBase):
    pass


# -----------------------------
# ORM Models
# -----------------------------


class Bot(Base):
    __tablename__ = "bots"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    basic_info: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    big_five: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    persona: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id: Mapped[Optional[str]] = mapped_column(Text, unique=True, nullable=True)
    basic_info: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Relationship(Base):
    __tablename__ = "relationships"
    __table_args__ = (UniqueConstraint("bot_id", "user_id", name="uq_relationship_bot_user"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bot_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"))
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))

    current_stage: Mapped[str] = mapped_column(Text, nullable=False, default="initiating")
    dimensions: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=lambda: {
            "closeness": 0,
            "trust": 0,
            "liking": 0,
            "respect": 0,
            "warmth": 0,
            "power": 50,
        },
    )
    mood_state: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=lambda: {"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0},
    )
    inferred_profile: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    assets: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    spt_info: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    conversation_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    bot: Mapped[Bot] = relationship("Bot")
    user: Mapped[User] = relationship("User")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("role IN ('user','ai','system')", name="ck_messages_role"),
        Index("idx_messages_rel_time", "relationship_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    relationship_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("relationships.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # SQLAlchemy Declarative API 中 metadata 为保留名，这里用 meta 映射到列名 metadata
    meta: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    relationship_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("relationships.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


Index("idx_relationships_lookup", Relationship.bot_id, Relationship.user_id)


# -----------------------------
# DBManager
# -----------------------------


def _to_uuid_or_none(x: str) -> Optional[uuid.UUID]:
    try:
        return uuid.UUID(str(x))
    except Exception:
        return None


def _to_langchain_messages(rows: List[Tuple[str, str]]) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for role, content in rows:
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "ai":
            out.append(AIMessage(content=content))
        else:
            out.append(SystemMessage(content=content))
    return out


class DBManager:
    """
    Async DB access helper.

    推荐用法：
      db = DBManager.from_env()
      state = await db.load_state(user_external_id, bot_id)
      await db.save_turn(user_external_id, bot_id, state)
    """

    def __init__(self, engine: AsyncEngine):
        self.engine = engine
        self.Session: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self.engine, expire_on_commit=False
        )

    @classmethod
    def from_env(cls) -> "DBManager":
        url = os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL 未设置，无法初始化 DBManager")
        engine = create_async_engine(url, echo=False, future=True)
        return cls(engine)

    async def _get_or_create_user(self, session: AsyncSession, external_id: str) -> User:
        q = select(User).where(User.external_id == external_id)
        user = (await session.execute(q)).scalars().first()
        if user:
            return user
        user = User(external_id=external_id, basic_info={})
        session.add(user)
        await session.flush()
        return user

    async def _get_or_create_bot(self, session: AsyncSession, bot_id: str) -> Bot:
        # 支持两种输入：UUID（推荐）或 bot name（fallback）
        bot_uuid = _to_uuid_or_none(bot_id)
        if bot_uuid:
            q = select(Bot).where(Bot.id == bot_uuid)
            bot = (await session.execute(q)).scalars().first()
            if bot:
                return bot
            bot = Bot(id=bot_uuid, name="default_bot", basic_info={}, big_five={}, persona={})
            session.add(bot)
            await session.flush()
            return bot

        # fallback: treat as name
        q = select(Bot).where(Bot.name == bot_id)
        bot = (await session.execute(q)).scalars().first()
        if bot:
            return bot
        bot = Bot(name=bot_id, basic_info={}, big_five={}, persona={})
        session.add(bot)
        await session.flush()
        return bot

    async def _get_or_create_relationship(self, session: AsyncSession, bot: Bot, user: User) -> Relationship:
        q = select(Relationship).where(Relationship.bot_id == bot.id, Relationship.user_id == user.id)
        rel = (await session.execute(q)).scalars().first()
        if rel:
            return rel
        rel = Relationship(
            bot_id=bot.id,
            user_id=user.id,
            current_stage="initiating",
            dimensions={"closeness": 0, "trust": 0, "liking": 0, "respect": 0, "warmth": 0, "power": 50},
            mood_state={"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0},
            inferred_profile={},
            assets={},
            spt_info={},
            conversation_summary="",
        )
        session.add(rel)
        await session.flush()
        return rel

    async def load_state(self, user_id: str, bot_id: str) -> Dict[str, Any]:
        """
        Loader：只读（必要时会创建缺失的 user/bot/relationship 行，属于“初始化”）。
        返回：用于灌入 AgentState 的字典片段（增量更新）。
        """
        async with self.Session() as session:
            async with session.begin():
                user = await self._get_or_create_user(session, user_id)
                bot = await self._get_or_create_bot(session, bot_id)
                rel = await self._get_or_create_relationship(session, bot, user)

                # 最近 20 条消息（新->旧），再翻转成 旧->新
                q = (
                    select(Message.role, Message.content)
                    .where(Message.relationship_id == rel.id)
                    .order_by(Message.created_at.desc())
                    .limit(20)
                )
                rows = list((await session.execute(q)).all())
                rows.reverse()
                chat_buffer = _to_langchain_messages([(r, c) for (r, c) in rows])

                return {
                    "relationship_id": str(rel.id),
                    "relationship_state": rel.dimensions or {},
                    "mood_state": rel.mood_state or {},
                    "current_stage": rel.current_stage,
                    "user_inferred_profile": rel.inferred_profile or {},
                    "relationship_assets": rel.assets or {},
                    "spt_info": rel.spt_info or {},
                    "conversation_summary": rel.conversation_summary or "",
                    "bot_basic_info": bot.basic_info or {},
                    "bot_big_five": bot.big_five or {},
                    "bot_persona": bot.persona or {},
                    "user_basic_info": user.basic_info or {},
                    "chat_buffer": chat_buffer,
                }

    async def save_turn(self, user_id: str, bot_id: str, state: Dict[str, Any], new_memory: Optional[str] = None) -> None:
        """
        Writer：事务写入一轮对话。
        - 写入 user/ai messages
        - 更新 relationships（dimensions/mood/current_stage/inferred_profile/assets/spt_info/summary）
        - 可选写入 memories
        """
        async with self.Session() as session:
            async with session.begin():
                user = await self._get_or_create_user(session, user_id)
                bot = await self._get_or_create_bot(session, bot_id)
                rel = await self._get_or_create_relationship(session, bot, user)

                user_input = str(state.get("user_input") or "")
                final_response = str(state.get("final_response") or state.get("draft_response") or "")

                # metadata：只存轻量可观测字段
                meta_user = {
                    "source": "turn",
                    "current_stage": state.get("current_stage"),
                }
                meta_ai = {
                    "source": "turn",
                    "current_stage": state.get("current_stage"),
                    "detection_category": state.get("detection_category") or state.get("detection_result"),
                    "latency": (state.get("humanized_output") or {}).get("total_latency_seconds"),
                }

                if user_input:
                    session.add(
                        Message(
                            relationship_id=rel.id,
                            role="user",
                            content=user_input,
                            meta=meta_user,
                        )
                    )
                if final_response:
                    session.add(
                        Message(
                            relationship_id=rel.id,
                            role="ai",
                            content=final_response,
                            meta=meta_ai,
                        )
                    )

                # 更新关系状态
                rel.current_stage = str(state.get("current_stage") or rel.current_stage or "initiating")
                rel.dimensions = dict(state.get("relationship_state") or rel.dimensions or {})
                rel.mood_state = dict(state.get("mood_state") or rel.mood_state or {})
                rel.inferred_profile = dict(state.get("user_inferred_profile") or rel.inferred_profile or {})
                rel.assets = dict(state.get("relationship_assets") or rel.assets or {})
                rel.spt_info = dict(state.get("spt_info") or rel.spt_info or {})
                rel.conversation_summary = state.get("conversation_summary") or rel.conversation_summary
                rel.updated_at = func.now()  # type: ignore[assignment]

                if new_memory is None:
                    new_memory = state.get("new_memory_content")  # optional convention
                if new_memory:
                    session.add(Memory(relationship_id=rel.id, content=str(new_memory)))


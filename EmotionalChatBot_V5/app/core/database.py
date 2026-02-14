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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    case,
    String,
    Text,
    UniqueConstraint,
    delete,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID, ENUM
from sqlalchemy.engine.url import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.profile_factory import generate_bot_profile, generate_user_profile
from app.core.relationship_templates import get_random_relationship_template


def _create_async_engine_from_database_url(database_url: str) -> AsyncEngine:
    """
    Render 等平台常提供形如 `postgres://...` 或 `postgresql://...` 的 URL，
    但本项目使用 asyncpg，需要 `postgresql+asyncpg://...`。

    同时某些托管库会附带 `?sslmode=require`（libpq 参数），asyncpg 不识别。
    我们做一次 best-effort 归一化：
    - scheme: postgres/postgresql -> postgresql+asyncpg
    - sslmode=require/verify-* -> connect_args={"ssl": True}
    - 从 URL query 中移除 sslmode
    """
    u = make_url(database_url.strip())
    driver = (u.drivername or "").strip()
    if driver.startswith("postgres://"):
        # make_url 通常不会返回这个，但保守处理
        driver = "postgres"
    if driver in {"postgres", "postgresql"}:
        u = u.set(drivername="postgresql+asyncpg")
    elif driver == "postgresql+asyncpg":
        pass
    else:
        # 其它 drivername（例如已带 +asyncpg 或用户自定义）直接使用
        # 但若是 postgres:// 这类别名，也可能被解析为 postgresql
        pass

    query = dict(u.query or {})
    sslmode = (query.pop("sslmode", "") or "").lower()
    connect_args: dict[str, Any] = {}
    if sslmode in {"require", "verify-full", "verify-ca"}:
        # asyncpg: ssl=True 会创建默认 SSLContext（best-effort）
        connect_args["ssl"] = True

    u = u.set(query=query)
    # IMPORTANT: `str(URL)` hides password by default ("***"), which will break auth.
    # Use the full rendered URL string for actual connections.
    try:
        url_str = u.render_as_string(hide_password=False)  # type: ignore[attr-defined]
    except Exception:
        # Fallback: best-effort; should still work for simple URLs
        url_str = str(u)
    return create_async_engine(url_str, echo=False, future=True, connect_args=connect_args)

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
    """用户表：挂在 bot 下，每个 user 绑定一个 bot。(bot_id, external_id) 唯一。bot_name 为明文存 bot 名称便于查看。"""
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("bot_id", "external_id", name="uq_users_bot_external"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bot_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"), nullable=False)
    bot_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    external_id: Mapped[str] = mapped_column(Text, nullable=False)
    basic_info: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    current_stage: Mapped[str] = mapped_column(
        ENUM(
            "initiating", "experimenting", "intensifying", "integrating", "bonding",
            "differentiating", "circumscribing", "stagnating", "avoiding", "terminating",
            name="knapp_stage", create_type=False,
        ),
        nullable=False,
        default="initiating",
    )
    dimensions: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False,
        default=lambda: get_random_relationship_template(),
    )
    mood_state: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False,
        default=lambda: {"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0},
    )
    inferred_profile: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    assets: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    spt_info: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    conversation_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    bot: Mapped[Bot] = relationship("Bot")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("role IN ('user','ai','system')", name="ck_messages_role"),
        Index("idx_messages_user_time", "user_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Transcript(Base):
    """
    Memory Store A: Raw Transcript Store，挂在 user 下。
    """
    __tablename__ = "transcripts"
    __table_args__ = (Index("idx_transcripts_user_time", "user_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    # 可选：用于把一次“会话运行”（例如一次 console session）串起来
    session_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 可选：线程/会话标识（未来接入 LangGraph thread_id 或前端会话 id）
    thread_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 可选：该会话中的 turn 序号（从 1 开始）
    turn_index: Mapped[Optional[int]] = mapped_column(nullable=True)

    user_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    bot_text: Mapped[str] = mapped_column(Text, nullable=False, default="")

    entities: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    topic: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    importance: Mapped[Optional[float]] = mapped_column(nullable=True)
    short_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DerivedNote(Base):
    """
    Memory Store B: Derived Notes Store，挂在 user 下。
    """
    __tablename__ = "derived_notes"
    __table_args__ = (
        Index("idx_notes_user_time", "user_id", "created_at"),
        Index("idx_notes_transcript", "transcript_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    transcript_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("transcripts.id", ondelete="CASCADE"), nullable=False
    )

    note_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # fact/preference/activity/decision/other
    content: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[Optional[float]] = mapped_column(nullable=True)

    # 可选：冗余存一份可读 source_pointer（例如 "transcript:{id}"）
    source_pointer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# -----------------------------
# DBManager
# -----------------------------


def _to_uuid_or_none(x: str) -> Optional[uuid.UUID]:
    try:
        return uuid.UUID(str(x))
    except Exception:
        return None


def _to_langchain_messages(rows: List[Tuple[str, str, Any]]) -> List[BaseMessage]:
    """(role, content, created_at) -> BaseMessage，created_at 放入 additional_kwargs['timestamp']（ISO 字符串）。"""
    out: List[BaseMessage] = []
    for role, content, created_at in rows:
        ts = None
        if created_at is not None:
            ts = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
        kwargs = {"timestamp": ts} if ts else {}
        if role == "user":
            out.append(HumanMessage(content=content, additional_kwargs=kwargs))
        elif role == "ai":
            out.append(AIMessage(content=content, additional_kwargs=kwargs))
        else:
            out.append(SystemMessage(content=content, additional_kwargs=kwargs))
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
        self._memory_schema_ready: bool = False

    async def ensure_memory_schema(self) -> None:
        """
        确保 Memory Store A/B 的表存在（不做 destructive migration）。
        只创建本文件新增的 transcripts / derived_notes 两张表。
        """
        if self._memory_schema_ready:
            return
        async with self.engine.begin() as conn:
            await conn.run_sync(lambda sync_conn: Transcript.__table__.create(sync_conn, checkfirst=True))
            await conn.run_sync(lambda sync_conn: DerivedNote.__table__.create(sync_conn, checkfirst=True))
        self._memory_schema_ready = True

    @staticmethod
    def _tokenize_query(query: str) -> List[str]:
        """
        非依赖分词器的稳定 tokenization：
        - 以空白/常见标点切分
        - 去掉太短的 token
        """
        seps = [" ", "\n", "\t", ",", "，", ".", "。", "?", "？", "!", "！", ";", "；", ":", "：", "、", "（", "）", "(", ")", "[", "]"]
        s = str(query or "")
        for sep in seps:
            s = s.replace(sep, " ")
        toks = [t.strip() for t in s.split(" ") if t.strip()]
        # 中文/英文统一：长度>=2 更稳，且避免全是单字噪声
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
            if not ww:
                continue
            # count occurrences (bounded) for stability
            c = t.count(ww)
            if c:
                score += min(3, c)
        return float(score)

    async def append_transcript(
        self,
        *,
        relationship_id: str,
        user_text: str,
        bot_text: str,
        session_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        turn_index: Optional[int] = None,
        entities: Optional[Dict[str, Any]] = None,
        topic: Optional[str] = None,
        importance: Optional[float] = None,
        short_context: Optional[str] = None,
    ) -> str:
        """写入 Store A（Raw Transcript），返回 transcript_id（uuid str）。"""
        await self.ensure_memory_schema()
        async with self.Session() as session:
            async with session.begin():
                uid = _to_uuid_or_none(relationship_id)
                if not uid:
                    raise ValueError("relationship_id (user_id) must be a valid UUID")
                tr = Transcript(
                    user_id=uid,
                    session_id=session_id,
                    thread_id=thread_id,
                    turn_index=turn_index,
                    user_text=str(user_text or ""),
                    bot_text=str(bot_text or ""),
                    entities=entities or {},
                    topic=topic,
                    importance=importance,
                    short_context=short_context,
                )
                session.add(tr)
                await session.flush()
                return str(tr.id)

    async def append_notes(
        self,
        *,
        relationship_id: str,
        transcript_id: str,
        notes: List[Dict[str, Any]],
    ) -> int:
        """写入 Store B（Derived Notes），返回写入条数。"""
        await self.ensure_memory_schema()
        tid = _to_uuid_or_none(transcript_id)
        rid = _to_uuid_or_none(relationship_id)
        if not tid or not rid:
            return 0
        async with self.Session() as session:
            async with session.begin():
                n = 0
                for row in notes or []:
                    content = str(row.get("content") or "").strip()
                    if not content:
                        continue
                    note = DerivedNote(
                        user_id=rid,
                        transcript_id=tid,
                        note_type=(row.get("note_type") or row.get("type")),
                        content=content,
                        importance=row.get("importance"),
                        source_pointer=str(row.get("source_pointer") or f"transcript:{transcript_id}"),
                    )
                    session.add(note)
                    n += 1
                return n

    async def search_transcripts(
        self,
        *,
        relationship_id: str,
        query: str,
        limit: int = 6,
        scan_limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        从 Store A 召回：
        - 为稳定性，优先扫描最近 scan_limit 条，再用简单 term-match + importance/recency 加权排序
        - 返回结构化结果，供上层 merge/rerank
        """
        await self.ensure_memory_schema()
        rid = _to_uuid_or_none(relationship_id)
        if not rid:
            return []
        terms = self._tokenize_query(query)
        if not terms:
            return []

        async with self.Session() as session:
            q = (
                select(
                    Transcript.id,
                    Transcript.created_at,
                    Transcript.session_id,
                    Transcript.turn_index,
                    Transcript.user_text,
                    Transcript.bot_text,
                    Transcript.topic,
                    Transcript.importance,
                    Transcript.short_context,
                )
                .where(Transcript.user_id == rid)
                .order_by(Transcript.created_at.desc())
                .limit(int(scan_limit))
            )
            rows = list((await session.execute(q)).all())

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for (tid, created_at, session_id, turn_index, user_text, bot_text, topic, importance, short_context) in rows:
            text = " ".join([str(topic or ""), str(short_context or ""), str(user_text or ""), str(bot_text or "")])
            s = self._score_text(text, terms)
            if s <= 0:
                continue
            imp = float(importance) if importance is not None else 0.0
            # 轻量加权：importance 0-1 直接加，recency 用 rank position 模拟（越新越高）
            scored.append(
                (
                    s + imp,
                    {
                        "store": "A",
                        "id": str(tid),
                        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
                        "session_id": session_id,
                        "turn_index": turn_index,
                        "topic": topic,
                        "importance": imp,
                        "short_context": short_context,
                        "user_text": user_text,
                        "bot_text": bot_text,
                    },
                )
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[: int(limit)]]

    async def search_notes(
        self,
        *,
        relationship_id: str,
        query: str,
        limit: int = 6,
        scan_limit: int = 400,
    ) -> List[Dict[str, Any]]:
        """从 Store B 召回（同上：扫描近期 + term-match + importance 加权）。"""
        await self.ensure_memory_schema()
        rid = _to_uuid_or_none(relationship_id)
        if not rid:
            return []
        terms = self._tokenize_query(query)
        if not terms:
            return []

        async with self.Session() as session:
            q = (
                select(
                    DerivedNote.id,
                    DerivedNote.created_at,
                    DerivedNote.note_type,
                    DerivedNote.content,
                    DerivedNote.importance,
                    DerivedNote.source_pointer,
                    DerivedNote.transcript_id,
                )
                .where(DerivedNote.user_id == rid)
                .order_by(DerivedNote.created_at.desc())
                .limit(int(scan_limit))
            )
            rows = list((await session.execute(q)).all())

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for (nid, created_at, note_type, content, importance, source_pointer, transcript_id) in rows:
            s = self._score_text(str(content or ""), terms)
            if s <= 0:
                continue
            imp = float(importance) if importance is not None else 0.0
            scored.append(
                (
                    # notes 给一点稳定性加成
                    s + imp + 0.5,
                    {
                        "store": "B",
                        "id": str(nid),
                        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
                        "note_type": note_type,
                        "content": content,
                        "importance": imp,
                        "source_pointer": source_pointer,
                        "transcript_id": str(transcript_id),
                    },
                )
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[: int(limit)]]

    @classmethod
    def from_env(cls) -> "DBManager":
        url = os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL 未设置，无法初始化 DBManager")
        engine = _create_async_engine_from_database_url(url)
        return cls(engine)

    async def _get_or_create_user(self, session: AsyncSession, bot_id: str, external_id: str) -> User:
        """按 (bot_id, external_id) 获取或创建用户（user 挂在 bot 下）。"""
        bot = await self._get_or_create_bot(session, bot_id)
        q = select(User).where(User.bot_id == bot.id, User.external_id == external_id)
        user = (await session.execute(q)).scalars().first()
        if user:
            return user
        user_basic_info, user_inferred = generate_user_profile(external_id)
        # 随机选择一个关系维度模板
        relationship_template = get_random_relationship_template()
        user = User(
            bot_id=bot.id,
            bot_name=bot.name,
            external_id=external_id,
            basic_info=user_basic_info,
            current_stage="initiating",
            dimensions=relationship_template,
            mood_state={"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0},
            inferred_profile=user_inferred,
            assets={"topic_history": [], "breadth_score": 0, "max_spt_depth": 1},
            spt_info={},
            conversation_summary="",
        )
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
            bot_basic_info, bot_big_five, bot_persona = generate_bot_profile(str(bot_uuid))
            bot = Bot(
                id=bot_uuid,
                name=str(bot_basic_info.get("name") or "default_bot"),
                basic_info=bot_basic_info,
                big_five=bot_big_five,
                persona=bot_persona,
            )
            session.add(bot)
            await session.flush()
            return bot

        # fallback: treat as name
        q = select(Bot).where(Bot.name == bot_id)
        bot = (await session.execute(q)).scalars().first()
        if bot:
            return bot
        bot_basic_info, bot_big_five, bot_persona = generate_bot_profile(bot_id)
        bot = Bot(
            name=bot_id,
            basic_info=bot_basic_info,
            big_five=bot_big_five,
            persona=bot_persona,
        )
        session.add(bot)
        await session.flush()
        return bot

    async def load_state(self, user_id: str, bot_id: str) -> Dict[str, Any]:
        """
        Loader：只读（必要时会创建缺失的 user/bot/relationship 行，属于“初始化”）。
        返回：用于灌入 AgentState 的字典片段（增量更新）。
        """
        async with self.Session() as session:
            async with session.begin():
                user = await self._get_or_create_user(session, bot_id, user_id)
                bot = await self._get_or_create_bot(session, bot_id)

                # Stable ordering:
                # In the same DB transaction, Message.created_at (server_default=now()) can be identical for
                # both user+ai inserts. Order by role to ensure user appears before ai in the reconstructed history.
                role_order = case(
                    (Message.role == "user", 0),
                    (Message.role == "ai", 1),
                    else_=2,
                )
                q = (
                    select(Message.role, Message.content, Message.created_at)
                    .where(Message.user_id == user.id)
                    .order_by(Message.created_at.desc(), role_order.desc(), Message.id.desc())
                    .limit(20)
                )
                rows = list((await session.execute(q)).all())
                rows.reverse()
                chat_buffer = _to_langchain_messages([(r, c, t) for (r, c, t) in rows])

                user_basic = dict(user.basic_info or {})
                if user_basic.get("name") is None:
                    user_basic["name"] = user_basic.get("nickname") or "User"

                # 关系维度：统一 0-1 量纲并补齐缺失 key（缺失不应默认为 0）
                dims = dict(user.dimensions or {})

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

                normalized_dims: Dict[str, float] = {}
                for k, default in (
                    ("closeness", 0.3),
                    ("trust", 0.3),
                    ("liking", 0.3),
                    ("respect", 0.3),
                    ("warmth", 0.3),
                    ("power", 0.5),
                ):
                    normalized_dims[k] = round(_norm01(dims.get(k, default)), 4)
                return {
                    "relationship_id": str(user.id),
                    "relationship_state": normalized_dims,
                    "mood_state": user.mood_state or {},
                    "current_stage": user.current_stage,
                    "user_inferred_profile": user.inferred_profile or {},
                    "relationship_assets": user.assets or {},
                    "spt_info": user.spt_info or {},
                    "conversation_summary": user.conversation_summary or "",
                    "bot_basic_info": bot.basic_info or {},
                    "bot_big_five": bot.big_five or {},
                    "bot_persona": bot.persona or {},
                    "user_basic_info": user_basic,
                    "chat_buffer": chat_buffer,
                }

    async def clear_messages_for(self, user_id: str, bot_id: str) -> int:
        """删除该 bot 下该用户的所有消息，返回删除条数。"""
        async with self.Session() as session:
            async with session.begin():
                user = await self._get_or_create_user(session, bot_id, user_id)
                result = await session.execute(delete(Message).where(Message.user_id == user.id))
                return result.rowcount or 0

    async def clear_all_memory_for(self, user_id: str, bot_id: str, *, reset_profile: bool = True) -> Dict[str, int]:
        """
        危险操作：彻底清空该 bot 下该用户的所有对话与记忆（messages/memories/transcripts/derived_notes + summary）。
        用于“清理干净后重新测试拟人化”。
        """
        await self.ensure_memory_schema()
        counts = {"messages": 0, "memories": 0, "transcripts": 0, "derived_notes": 0}
        async with self.Session() as session:
            async with session.begin():
                user = await self._get_or_create_user(session, bot_id, user_id)

                # 先删 notes，再删 transcripts（FK 安全）
                res = await session.execute(delete(DerivedNote).where(DerivedNote.user_id == user.id))
                counts["derived_notes"] = res.rowcount or 0

                res = await session.execute(delete(Transcript).where(Transcript.user_id == user.id))
                counts["transcripts"] = res.rowcount or 0

                res = await session.execute(delete(Memory).where(Memory.user_id == user.id))
                counts["memories"] = res.rowcount or 0

                res = await session.execute(delete(Message).where(Message.user_id == user.id))
                counts["messages"] = res.rowcount or 0

                # 清空 summary，并可选重置关系状态/阶段（避免“terminating”这种旧阶段继续污染）
                user.conversation_summary = ""
                if reset_profile:
                    user.current_stage = "initiating"
                    # 重置时也随机选择一个关系维度模板
                    relationship_template = get_random_relationship_template()
                    user.dimensions = relationship_template
                    user.mood_state = {"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0}
                    user.assets = {"topic_history": [], "breadth_score": 0, "max_spt_depth": 1}
                    user.spt_info = {}
                user.updated_at = func.now()  # type: ignore[assignment]

        return counts

    async def save_turn(self, user_id: str, bot_id: str, state: Dict[str, Any], new_memory: Optional[str] = None) -> None:
        """
        Writer：事务写入一轮对话。user 挂在 bot 下；写入 messages 并更新 user 的关系状态与可选 memories。
        """

        def _parse_dt(v: Any) -> Optional[datetime]:
            if not v:
                return None
            try:
                s = str(v).strip()
                if not s:
                    return None
                # allow Z suffix
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                return datetime.fromisoformat(s)
            except Exception:
                return None

        # 业务语义时间戳：
        # - user_created_at: 收到用户消息、进入流程前
        # - ai_created_at: 生成完成、准备返回给用户前
        user_created_at = _parse_dt(state.get("user_received_at")) or _parse_dt(state.get("current_time"))
        ai_created_at = _parse_dt(state.get("ai_sent_at")) or datetime.now(timezone.utc)

        async with self.Session() as session:
            async with session.begin():
                user = await self._get_or_create_user(session, bot_id, user_id)

                user_input = str(state.get("user_input") or "")
                final_response = str(state.get("final_response") or state.get("draft_response") or "")

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
                            user_id=user.id,
                            role="user",
                            content=user_input,
                            meta=meta_user,
                            created_at=user_created_at,
                        )
                    )
                if final_response:
                    session.add(
                        Message(
                            user_id=user.id,
                            role="ai",
                            content=final_response,
                            meta=meta_ai,
                            created_at=ai_created_at,
                        )
                    )

                user.current_stage = str(state.get("current_stage") or user.current_stage or "initiating")
                # 关系维度：避免“部分字段写入”把其它维度抹掉；并统一到 0-1 量纲 + 单轮跳变截断
                prev_dims = dict(user.dimensions or {})
                incoming_dims = dict(state.get("relationship_state") or {})

                def _norm01(v: Any) -> float:
                    try:
                        x = float(v)
                    except Exception:
                        return 0.0
                    # 兼容旧 points(0-100)
                    if x > 1.0:
                        if x <= 100.0:
                            x = x / 100.0
                        else:
                            x = 1.0
                    if x < 0.0:
                        x = 0.0
                    if x > 1.0:
                        x = 1.0
                    return float(x)

                merged_dims = dict(prev_dims)
                merged_dims.update(incoming_dims)

                # 补齐关键维度（缺失不应回退到 0）
                for k, default in (
                    ("closeness", 0.3),
                    ("trust", 0.3),
                    ("liking", 0.3),
                    ("respect", 0.3),
                    ("warmth", 0.3),
                    ("power", 0.5),
                ):
                    if k not in merged_dims:
                        merged_dims[k] = prev_dims.get(k, default)

                # 单轮跳变审计与截断（避免 0.22 -> 1.00 这种爆炸）
                max_step = 0.20
                audited: Dict[str, Any] = {}
                for k in ("closeness", "trust", "liking", "respect", "warmth", "power"):
                    old = _norm01(prev_dims.get(k, merged_dims.get(k)))
                    new = _norm01(merged_dims.get(k))
                    delta = new - old
                    if abs(delta) > max_step:
                        new = old + (max_step if delta > 0 else -max_step)
                        delta = new - old
                    merged_dims[k] = round(new, 4)
                    audited[k] = {"old": round(old, 4), "new": round(new, 4), "delta": round(delta, 4)}

                # 打印一行审计日志，方便追查谁把值写炸
                print(f"[Relationship] audit dims: {audited}")
                user.dimensions = merged_dims
                user.mood_state = dict(state.get("mood_state") or user.mood_state or {})
                user.inferred_profile = dict(state.get("user_inferred_profile") or user.inferred_profile or {})
                user.assets = dict(state.get("relationship_assets") or user.assets or {})
                user.spt_info = dict(state.get("spt_info") or user.spt_info or {})
                user.conversation_summary = state.get("conversation_summary") or user.conversation_summary
                user.updated_at = func.now()  # type: ignore[assignment]

                if new_memory is None:
                    new_memory = state.get("new_memory_content")
                if new_memory:
                    session.add(Memory(user_id=user.id, content=str(new_memory)))


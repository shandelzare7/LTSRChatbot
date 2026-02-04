"""
数据服务层：基于 SQLAlchemy 的 CRUD，保留原 Wrapper 的 Schema 常量，
ensure_exists 时按 Schema 填充默认值，供 LangGraph 使用。
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.database import (
    BotModel,
    ChatLogModel,
    RelationshipModel,
    SessionLocal,
    UserModel,
    init_db,
)


class DatabaseService:
    # --- 从原 Wrapper 移植来的 Schema 常量 ---
    BOT_BASIC_INFO_KEYS = [
        "name",
        "gender",
        "age",
        "region",
        "occupation",
        "education",
        "native_language",
    ]
    BOT_PERSONALITY_KEYS = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]
    USER_BASIC_INFO_KEYS = ["name", "gender", "age"]
    # 扩展印象：relationship_closeness 和 trust 已被提升为 SQL 字段，这里保留其他的
    USER_IMPRESSION_KEYS = [
        "respect",
        "liking",
        "warmth",
        "power_distance",
        "cooperativeness",
        "supportiveness",
        "comfort_level",
        "shared_background",
        "perceived_authority",
        "empathy_towards_other",
        "expectations",
        "admiration",
        "anxiety_towards",
    ]

    def __init__(self):
        init_db()
        self.db: Session = SessionLocal()

    def close(self):
        self.db.close()

    # ==========================================
    # 1. 核心上下文获取 (LLM Context Loader)
    # ==========================================

    def get_context_data(self, user_id: str, bot_id: str) -> Dict[str, Any]:
        """
        一次性拉取所有数据，组装成 V5.0 架构需要的 Prompt Context
        """
        self._ensure_user_bot_exist(user_id, bot_id)

        user = self.db.query(UserModel).filter_by(id=user_id).first()
        bot = self.db.query(BotModel).filter_by(id=bot_id).first()
        rel = self.db.query(RelationshipModel).filter_by(
            user_id=user_id, bot_id=bot_id
        ).first()

        logs = (
            self.db.query(ChatLogModel)
            .filter_by(session_id=f"{bot_id}_{user_id}")
            .order_by(ChatLogModel.timestamp.desc())
            .limit(20)
            .all()
        )
        history = [{"role": l.role, "content": l.content} for l in reversed(logs)]

        return {
            "bot_profile": {
                "basic": bot.basic_info,
                "personality": bot.personality,
                "persona": bot.persona,
                "emotion": bot.emotion_state,
            },
            "user_profile": {
                "basic": user.basic_info,
                "summary": user.profile_summary,
                "persona": user.persona,
            },
            "relationship": {
                "intimacy": rel.intimacy,
                "trust": rel.trust,
                "impressions": rel.extended_impressions,
                "facts": rel.facts_list,
                "last_monologue": rel.last_inner_monologue,
            },
            "history": history,
        }

    # ==========================================
    # 2. 状态更新 (Update Logic)
    # ==========================================

    def update_bot_state(
        self,
        bot_id: str,
        emotion_update: Optional[Dict] = None,
        memory_text: Optional[str] = None,
    ):
        """更新 Bot 的情绪或自身记忆"""
        bot = self.db.query(BotModel).filter_by(id=bot_id).first()
        if not bot:
            return

        if emotion_update:
            current = dict(bot.emotion_state or {})
            current.update(emotion_update)
            bot.emotion_state = current

        if memory_text:
            mems = list(bot.memories or [])
            mems.append({"text": memory_text, "timestamp": str(datetime.now())})
            bot.memories = mems

        self.db.commit()

    def update_relationship(
        self,
        user_id: str,
        bot_id: str,
        intimacy_delta: float = 0,
        trust_delta: float = 0,
        new_facts: Optional[List[str]] = None,
        impression_updates: Optional[Dict] = None,
        monologue: Optional[Dict] = None,
    ):
        """
        更新人际关系：包含数值、事实列表、印象标签
        """
        rel = (
            self.db.query(RelationshipModel)
            .filter_by(user_id=user_id, bot_id=bot_id)
            .first()
        )
        if not rel:
            return

        if intimacy_delta:
            rel.intimacy += intimacy_delta
        if trust_delta:
            rel.trust += trust_delta

        if new_facts:
            current_facts = set(rel.facts_list or [])
            for f in new_facts:
                current_facts.add(f)
            rel.facts_list = list(current_facts)

        if impression_updates:
            curr_imp = dict(rel.extended_impressions or {})
            curr_imp.update(impression_updates)
            rel.extended_impressions = curr_imp

        if monologue:
            rel.last_inner_monologue = monologue

        self.db.commit()

    def add_chat_log(
        self,
        user_id: str,
        bot_id: str,
        role: str,
        content: str,
        meta: Optional[Dict] = None,
    ):
        """写入流水账"""
        log = ChatLogModel(
            session_id=f"{bot_id}_{user_id}",
            role=role,
            content=content,
            meta_data=meta or {},
        )
        self.db.add(log)
        self.db.commit()

    # ==========================================
    # 3. 初始化辅助 (Data Migration Helper)
    # ==========================================

    def _ensure_user_bot_exist(self, user_id: str, bot_id: str):
        """
        惰性初始化：如果不存在，则按照 Schema 填入默认值 (unknown/0.0)
        """
        bot = self.db.query(BotModel).filter_by(id=bot_id).first()
        if not bot:
            default_personality = {k: "unknown" for k in self.BOT_PERSONALITY_KEYS}
            default_basic = {k: "unknown" for k in self.BOT_BASIC_INFO_KEYS}

            new_bot = BotModel(
                id=bot_id,
                name="Unknown Bot",
                basic_info=default_basic,
                personality=default_personality,
                emotion_state={"valence": 0.5, "arousal": 0.5},
                memories=[],
            )
            self.db.add(new_bot)

        user = self.db.query(UserModel).filter_by(id=user_id).first()
        if not user:
            default_basic = {k: "unknown" for k in self.USER_BASIC_INFO_KEYS}
            new_user = UserModel(
                id=user_id,
                name="User",
                basic_info=default_basic,
                profile_summary="新用户",
                persona={},
            )
            self.db.add(new_user)

        rel = (
            self.db.query(RelationshipModel)
            .filter_by(user_id=user_id, bot_id=bot_id)
            .first()
        )
        if not rel:
            default_impressions = {
                k: "unknown" for k in self.USER_IMPRESSION_KEYS
            }
            new_rel = RelationshipModel(
                user_id=user_id,
                bot_id=bot_id,
                intimacy=0.0,
                trust=0.0,
                extended_impressions=default_impressions,
                facts_list=[],
                last_inner_monologue={},
            )
            self.db.add(new_rel)

        self.db.commit()

"""
seed_three_bots_via_llm.py

1. 清空当前 DATABASE_URL 指向的库中的示例数据（保留表结构）
2. 使用 LLM 或本地规则生成 3 个 Bot 人设 + 每个 Bot 对应的 1 个默认 User
3. 写入 bots 与 users 表（及所需默认关系状态）

前置：.env 中 DATABASE_URL 已指向目标库（如 ltsrchatbot_v5）

运行：
  cd EmotionalChatBot_V5
  python devtools/seed_three_bots_via_llm.py
  # 默认用本地规则生成（稳定、快速）。若要用 LLM 生成人设：
  SEED_USE_LLM=1 python devtools/seed_three_bots_via_llm.py
  # 不删现有数据，仅用 LLM 再创建 3 个 Bot 并追加：
  SEED_ADD_VIA_LLM=1 python devtools/seed_three_bots_via_llm.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from sqlalchemy import text
from langchain_core.messages import HumanMessage

from app.core.database import Bot, DBManager, User
from app.services.llm import get_llm
from utils.llm_json import parse_json_from_llm


async def truncate_all_data(db: DBManager) -> None:
    """清空所有业务表数据，保留表结构。"""
    async with db.engine.connect() as conn:
        ac = await conn.execution_options(isolation_level="AUTOCOMMIT")
        await ac.execute(
            text(
                "TRUNCATE TABLE derived_notes, transcripts, messages, memories, users, bots RESTART IDENTITY CASCADE;"
            )
        )
    print("已清空新库中的示例数据。")


def _ensure_bot_persona(data: dict, fallback_name: str) -> tuple[dict, dict, dict]:
    """从 LLM 返回的 data 中取出 basic_info, big_five, persona 并补全默认值。"""
    from app.core.profile_factory import generate_bot_profile
    basic_info = data.get("basic_info") or {}
    big_five = data.get("big_five") or {}
    persona = data.get("persona") or {}
    if not basic_info.get("name"):
        basic_info["name"] = fallback_name
    if not basic_info.get("native_language"):
        basic_info["native_language"] = "zh"
    
    # 验证和修正年龄（必须在18-35之间）
    age = basic_info.get("age")
    if age is not None:
        try:
            age = int(age)
            if age < 18 or age > 35:
                age = 22  # 默认值
            basic_info["age"] = age
        except (ValueError, TypeError):
            basic_info["age"] = 22
    else:
        basic_info["age"] = 22
    
    for key in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
        if key not in big_five:
            big_five[key] = 0.0
        else:
            try:
                big_five[key] = max(-1.0, min(1.0, float(big_five[key])))
            except (ValueError, TypeError):
                big_five[key] = 0.0
    if "attributes" not in persona:
        persona["attributes"] = {}
    if "collections" not in persona:
        persona["collections"] = {}
    if "lore" not in persona:
        persona["lore"] = {}
    return basic_info, big_five, persona


def _ensure_user_basic_info(data: dict, fallback_name: str) -> dict:
    """从 LLM 返回的 user 中取出 basic_info。"""
    basic_info = data.get("basic_info") or data
    if not isinstance(basic_info, dict):
        basic_info = {}
    if not basic_info.get("name") and not basic_info.get("nickname"):
        basic_info["name"] = fallback_name
    return basic_info


async def create_three_bots_and_users_via_llm(db: DBManager, add_only: bool = False) -> None:
    """用 LLM（可选）或本地 profile_factory 生成 3 个 bot + 3 个 user，写入 DB。add_only 时强制用 LLM，不删已有数据。"""
    from app.core.profile_factory import generate_bot_profile, generate_user_profile

    use_llm = add_only or os.getenv("SEED_USE_LLM", "").strip().lower() in ("1", "true", "yes")
    polarized = add_only or os.getenv("SEED_POLARIZED", "").strip().lower() in ("1", "true", "yes", "on")
    llm_timeout = 120.0 if add_only else 60.0
    llm = None
    data = None
    if use_llm:
        try:
            llm = get_llm()
            if polarized:
                prompt = """请为 3 个聊天机器人（Bot）各生成一份「差异极大、性格极化」的人设，并为每个 Bot 生成一个默认聊天用户（user）的 basic_info。\n\n要求：\n- 3 个 Bot 必须在：说话风格、情绪表达、互动姿态、边界感、节奏上明显不同（不要只是换名字/职业）。\n- big_five 给出 5 维：openness/conscientiousness/extraversion/agreeableness/neuroticism，取值范围 -1.0..+1.0，尽量极化（例如 +0.85、-0.85），并符合人设。\n- persona 要像真人：\n  - attributes.catchphrase（口头禅 2~6 字）\n  - collections.hobbies（3 个）\n  - collections.quirks（2 个，小毛病/小特长）\n  - lore.origin / lore.secret（各 1 句）\n- basic_info: name(中文全名，姓+名)、gender(男/女)、age(18~35)、region(如 CN-北京)、occupation、education、native_language=zh、speaking_style(一句话描述)\n\n三种极化方向（每个 Bot 选一个，不要混）：\n1) 超外向+高主导+爱开玩笑：直球、短句、多反问、节奏快。\n2) 超内向+高尽责+低情绪外露：克制、条理、少表情、节奏慢。\n3) 高神经质+低信任+敏感防御：容易犹豫/反复确认、语气飘忽、碎片化更强。\n\n输出仅一个 JSON（严格遵循结构）：\n{\n  \"bots\": [\n    {\"basic_info\": {...}, \"big_five\": {...}, \"persona\": {...}},\n    {\"basic_info\": {...}, \"big_five\": {...}, \"persona\": {...}},\n    {\"basic_info\": {...}, \"big_five\": {...}, \"persona\": {...}}\n  ],\n  \"users\": [\n    {\"basic_info\": {...}},\n    {\"basic_info\": {...}},\n    {\"basic_info\": {...}}\n  ]\n}\n\nbots 与 users 一一对应。只输出 JSON，不要任何额外文字。"""
            else:
                prompt = """请为 3 个聊天机器人（Bot）各生成一份人设，并为每个 Bot 生成一个默认聊天用户（user）的 basic_info。输出仅一个 JSON：{"bots":[{"basic_info":{...},"big_five":{...},"persona":{...}}, ...共3个], "users":[{"basic_info":{...}}, ...共3个]}。bots 与 users 一一对应。"""
            print("正在用 LLM 生成 3 个 Bot + 3 个 User 人设…")
            resp = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, lambda: llm.invoke([HumanMessage(content=prompt)])),
                timeout=llm_timeout,
            )
            content = getattr(resp, "content", "") or ""
            data = parse_json_from_llm(content)
        except (asyncio.TimeoutError, Exception) as e:
            print(f"LLM 超时/出错 ({e})，改用本地规则生成。")
            data = None
            if add_only:
                print("追加模式下 LLM 未成功，仍用本地规则生成 3 组并追加。")
    else:
        print("使用本地规则生成 3 组 Bot + User（设置 SEED_USE_LLM=1 可改为 LLM 生成）。")

    if not isinstance(data, dict) or len(data.get("bots") or []) < 3:
        # 本地回退：3 组 (bot, user)，追加时用不同序号避免名字重复
        base = 4 if add_only else 1
        bots_data = []
        users_data = []
        for i in range(3):
            bid = str(uuid.uuid4())
            uid = f"default_user_{base + i}"
            bot_bi, bot_bf, bot_p = generate_bot_profile(bid)
            user_bi, user_inf = generate_user_profile(uid)
            bots_data.append({"basic_info": bot_bi, "big_five": bot_bf, "persona": bot_p})
            users_data.append({"basic_info": user_bi})
        data = {"bots": bots_data, "users": users_data}

    bots_data = data.get("bots") or []
    users_data = data.get("users") or []
    while len(users_data) < 3:
        users_data.append({"basic_info": {"name": f"用户{len(users_data)+1}", "nickname": f"小{len(users_data)+1}"}})

    async with db.Session() as session:
        async with session.begin():
            base = 4 if add_only else 1
            bots_created = []
            for i in range(3):
                b = bots_data[i]
                u = users_data[i] if i < len(users_data) else {}
                basic_info, big_five, persona = _ensure_bot_persona(b, f"Bot{base+i}")
                bot_id = uuid.uuid4()
                bot = Bot(
                    id=bot_id,
                    name=str(basic_info.get("name") or f"Bot{i+1}"),
                    basic_info=basic_info,
                    big_five=big_five,
                    persona=persona,
                )
                session.add(bot)
                await session.flush()
                bots_created.append(bot)

                user_basic = _ensure_user_basic_info(u, basic_info.get("name") or f"User{base+i}")
                external_id = f"default_user_{base + i}"
                from app.core.relationship_templates import get_random_relationship_template
                relationship_template = get_random_relationship_template()
                user = User(
                    bot_id=bot_id,
                    bot_name=bot.name,
                    external_id=external_id,
                    basic_info=user_basic,
                    current_stage="initiating",
                    dimensions=relationship_template,
                    mood_state={"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0},
                    inferred_profile={},
                    assets={"topic_history": [], "breadth_score": 0, "max_spt_depth": 1},
                    spt_info={},
                    conversation_summary="",
                )
                session.add(user)
                print(f"  已写入 Bot {i+1}: {bot.name}，对应 User: {user_basic.get('name') or external_id}")

            # 当使用 LLM 生成人设时，为每个 Bot 生成人物侧写与个性任务库（B1–B6）
            if llm and bots_created:
                from app.core.bot_creation_llm import generate_sidewrite_and_backlog
                for idx, bot in enumerate(bots_created):
                    try:
                        print(f"  生成 Bot {idx+1} 人物侧写与个性任务库…")
                        sidewrite, backlog = await generate_sidewrite_and_backlog(
                            llm, bot.basic_info or {}, bot.big_five or {}, bot.persona or {}
                        )
                        bot.character_sidewrite = sidewrite or None
                        bot.backlog_tasks = backlog if backlog else None
                        if backlog:
                            print(f"    ✓ {bot.name} 个性任务库 {len(backlog)} 条")
                    except Exception as e:
                        print(f"    ⚠ 侧写/任务库生成失败: {e}")

    print("3 个 Bot 与 3 个 User 已写入新库。")


async def main() -> None:
    if not os.getenv("DATABASE_URL"):
        print("DATABASE_URL 未设置，请在 .env 中配置目标库（如 ltsrchatbot_v5）。")
        sys.exit(1)
    db = DBManager.from_env()
    add_only = os.getenv("SEED_ADD_VIA_LLM", "").strip().lower() in ("1", "true", "yes")
    if add_only:
        print("SEED_ADD_VIA_LLM=1：不清空现有数据，仅用 LLM 再创建 3 个 Bot 并追加。\n")
        await create_three_bots_and_users_via_llm(db, add_only=True)
        print("\n✅ 完成：已用 LLM 追加 3 个 Bot 及其对应用户。")
    else:
        await truncate_all_data(db)
        await create_three_bots_and_users_via_llm(db)
        print("\n✅ 完成：新库已清空并写入 3 个 Bot 及其对应用户。")


if __name__ == "__main__":
    asyncio.run(main())

"""
bot_to_bot_chat.py

用途：
- 创建两个 Bot（Bot A 和 Bot B）
- 在各自 Bot 下创建对应的 User（Bot A 下 User B，Bot B 下 User A）
- 实现对话循环：两个 bot 轮流发送消息，进行 5 轮对话
- 记录对话内容和日志

前置：
1) 启动本地 Postgres
2) 执行 init_schema.sql 初始化表结构
3) 在 EmotionalChatBot_V5/.env 设置 DATABASE_URL（postgresql+asyncpg://...）

运行：
  cd EmotionalChatBot_V5
  python3 devtools/bot_to_bot_chat.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select

# allow running from devtools/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# load .env (same behavior as main.py)
try:
    from utils.env_loader import load_project_env

    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from app.core.database import Bot, DBManager, User
from app.graph import build_graph
from app.services.llm import get_llm
from main import _make_initial_state
from utils.llm_json import parse_json_from_llm


def _age_to_age_group(age: int | None) -> str:
    if age is None:
        return "20s"
    try:
        a = int(age)
    except Exception:
        return "20s"
    if a < 20:
        return "teen"
    if a < 30:
        return "20s"
    if a < 40:
        return "30s"
    return "40s"


def _region_to_location(region: str | None) -> str:
    r = str(region or "").strip()
    if not r:
        return "CN"
    # e.g. "CN-上海" -> "CN"
    if "-" in r:
        return r.split("-", 1)[0].strip() or "CN"
    return r


def _user_profiles_from_bot(bot_basic_info: dict, bot_persona: dict, bot_big_five: dict) -> tuple[dict, dict]:
    """
    bot-to-bot 压测：把“对方是谁”的 User 画像直接绑定到对方 Bot 的人设（避免随机人类画像污染）。
    Returns: (user_basic_info, user_inferred_profile)
    """
    basic = dict(bot_basic_info or {})
    persona = dict(bot_persona or {})
    big5 = dict(bot_big_five or {})

    name = str(basic.get("name") or "对方").strip() or "对方"
    age = basic.get("age")
    age_group = _age_to_age_group(age if isinstance(age, (int, float, str)) else None)
    location = _region_to_location(basic.get("region"))

    hobbies = []
    try:
        hobbies = list((((persona.get("collections") or {}).get("hobbies")) or []))
    except Exception:
        hobbies = []
    hobbies = [str(x).strip() for x in hobbies if str(x).strip()][:6]

    speaking_style = str(basic.get("speaking_style") or "").strip()
    comm_style = "casual, short, emotive"
    if speaking_style:
        # 简单把 speaking_style 作为沟通风格补充（不让“推断画像”反客为主）
        comm_style = f"casual; {speaking_style}"

    extraversion = None
    try:
        extraversion = float(big5.get("extraversion"))
    except Exception:
        extraversion = None
    expressiveness = "medium"
    if isinstance(extraversion, float):
        expressiveness = "high" if extraversion >= 0.66 else ("low" if extraversion <= 0.33 else "medium")

    user_basic_info = {
        "name": name,
        "nickname": name,
        "gender": basic.get("gender"),
        "age_group": age_group,
        "location": location,
        "occupation": basic.get("occupation"),
        # 标记：该 user 是 bot-to-bot 中的“对方 bot 代理画像”
        "bot_proxy": True,
    }

    user_inferred_profile = {
        # 关键：inner_monologue / reasoner 主要读取 inferred_profile 来“塑形对方是谁”
        "communication_style": comm_style,
        "expressiveness_baseline": expressiveness,
        "interests": hobbies,
        "sensitive_topics": ["违法行为", "隐私泄露", "露骨性内容", "金钱诈骗"],
        "bot_proxy": True,
    }
    return user_basic_info, user_inferred_profile


def _split_sql_statements(sql: str) -> list[str]:
    """Very small SQL splitter: splits by ';' and drops empty chunks."""
    parts = []
    for chunk in sql.split(";"):
        stmt = chunk.strip()
        if stmt:
            parts.append(stmt)
    return parts


async def _ensure_schema(db: DBManager) -> None:
    """使用 SQLAlchemy 直接执行 init_schema.sql（不依赖 psql）。"""
    from sqlalchemy import text

    schema_path = Path(__file__).resolve().parents[1] / "init_schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    statements = _split_sql_statements(sql)
    async with db.engine.connect() as conn:
        ac = await conn.execution_options(isolation_level="AUTOCOMMIT")
        for stmt in statements:
            try:
                await ac.execute(text(stmt))
            except Exception as e:
                msg = str(e).lower()
                if "already exists" in msg or "duplicate" in msg:
                    continue
                if "create extension" in stmt.lower():
                    continue
                raise


async def create_bot_via_llm(
    llm,
    bot_name: str,
    bot_description: str,
    log_line_func,
) -> tuple[dict, dict, dict]:
    """
    使用 LLM 创建 bot 人设。
    返回: (bot_basic_info, bot_big_five, bot_persona)
    """
    prompt = f"""请为一个名为"{bot_name}"的聊天机器人创建完整的人设档案。

Bot 描述：{bot_description}

请生成以下三个部分：

1. **basic_info** (基本信息):
   - name: 名字（中文）
   - gender: 性别（"男" 或 "女"）
   - age: 年龄（20-30之间的整数）
   - region: 地区（如 "CN-北京", "CN-上海"）
   - occupation: 职业（如 "学生", "设计师", "程序员"）
   - education: 教育程度（如 "本科", "硕士"）
   - native_language: "zh"
   - speaking_style: 说话风格描述（如 "说话爱用短句、偶尔带语气词"）

2. **big_five** (大五人格，范围 0.0 到 1.0，必须严格在区间内；若超界请你自己修正后再输出):
   - openness: 开放性（脑洞 vs 现实）
   - conscientiousness: 尽责性（严谨 vs 随性）
   - extraversion: 外向性（热情 vs 内向）
   - agreeableness: 宜人性（配合 vs 毒舌）
   - neuroticism: 神经质（情绪波动率）

3. **persona** (动态人设):
   - attributes: {{"catchphrase": "常用口头禅"}}
   - collections: {{"hobbies": ["爱好1", "爱好2", "爱好3"], "quirks": ["小特点1", "小特点2"]}}
   - lore: {{"origin": "背景故事", "secret": "小秘密"}}

请以 JSON 格式输出，格式如下：
{{
  "basic_info": {{...}},
  "big_five": {{...}},
  "persona": {{...}}
}}
"""

    try:
        log_line_func(f"  正在使用 LLM 生成 {bot_name} 的人设...")
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = getattr(resp, "content", "") or ""
        data = parse_json_from_llm(content)
        
        if not isinstance(data, dict):
            log_line_func(f"  ⚠ LLM 返回格式错误，使用默认人设")
            from app.core.profile_factory import generate_bot_profile
            return generate_bot_profile(bot_name)
        
        basic_info = data.get("basic_info", {})
        big_five = data.get("big_five", {})
        persona = data.get("persona", {})
        
        # 确保必要字段存在
        if not basic_info.get("name"):
            basic_info["name"] = bot_name
        if not basic_info.get("native_language"):
            basic_info["native_language"] = "zh"
        
        # 验证和修正年龄（必须在18-35之间）
        age = basic_info.get("age")
        if age is not None:
            try:
                age = int(age)
                if age < 18 or age > 35:
                    log_line_func(f"  ⚠ 年龄 {age} 超出范围，修正为 22")
                    age = 22
                basic_info["age"] = age
            except (ValueError, TypeError):
                log_line_func(f"  ⚠ 年龄格式错误，设置为默认值 22")
                basic_info["age"] = 22
        else:
            basic_info["age"] = 22
        
        # 确保 big_five 所有字段都是 float 且在 0..1（系统其余模块按 0..1 使用）
        for key in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            if key not in big_five:
                big_five[key] = 0.5
            else:
                try:
                    big_five[key] = float(big_five[key])
                    # 限制在 0.0 到 1.0 之间
                    big_five[key] = max(0.0, min(1.0, big_five[key]))
                except (ValueError, TypeError):
                    big_five[key] = 0.5
        
        # 确保 persona 结构正确
        if not isinstance(persona, dict):
            persona = {}
        if "attributes" not in persona:
            persona["attributes"] = {}
        if "collections" not in persona:
            persona["collections"] = {}
        if "lore" not in persona:
            persona["lore"] = {}
        
        log_line_func(f"  ✓ {bot_name} 人设生成成功")
        log_line_func(f"    名字: {basic_info.get('name')}, 年龄: {basic_info.get('age')}, 职业: {basic_info.get('occupation')}")
        
        return basic_info, big_five, persona
        
    except Exception as e:
        log_line_func(f"  ⚠ LLM 生成失败 ({e})，使用默认人设")
        from app.core.profile_factory import generate_bot_profile
        return generate_bot_profile(bot_name)


async def run_one_turn(
    app,
    user_id: str,
    bot_id: str,
    message: str,
    log_file,
    original_stdout,
) -> tuple[str, dict]:
    """运行一轮对话，返回 bot 的回复。"""
    from main import FileOnlyWriter
    from utils.external_text import sanitize_external_text

    state = _make_initial_state(user_id, bot_id)
    # bot-to-bot 压测：更偏“探索拟人化”而非“根计划过线就早退”
    state["lats_rollouts"] = int(os.getenv("BOT2BOT_LATS_ROLLOUTS", "4"))
    state["lats_expand_k"] = int(os.getenv("BOT2BOT_LATS_EXPAND_K", "4"))
    state["lats_early_exit_root_score"] = float(os.getenv("BOT2BOT_EARLY_EXIT_SCORE", "0.82"))
    state["lats_early_exit_plan_alignment_min"] = float(os.getenv("BOT2BOT_EARLY_EXIT_PLAN_MIN", "0.75"))
    state["lats_early_exit_assistantiness_max"] = float(os.getenv("BOT2BOT_EARLY_EXIT_ASSIST_MAX", "0.22"))
    state["lats_early_exit_mode_fit_min"] = float(os.getenv("BOT2BOT_EARLY_EXIT_MODE_MIN", "0.60"))
    state["lats_disable_early_exit"] = (str(os.getenv("BOT2BOT_DISABLE_EARLY_EXIT", "1")).lower() not in ("0", "false", "no", "off"))
    state["lats_skip_low_risk"] = (str(os.getenv("BOT2BOT_SKIP_LATS_LOW_RISK", "0")).lower() in ("1", "true", "yes", "on"))

    # 注意：LATS_Search 节点优先读取 mode.lats_budget（若存在）而不是 state.lats_rollouts/lats_expand_k。
    # 所以 bot-to-bot 压测要同步覆盖 mode 的预算，否则你设了 state 也不生效。
    try:
        cm = state.get("current_mode")
        if cm is not None and hasattr(cm, "lats_budget"):
            lb = getattr(cm, "lats_budget", None)
            if lb is not None:
                if hasattr(lb, "rollouts"):
                    setattr(lb, "rollouts", int(state["lats_rollouts"]))
                if hasattr(lb, "expand_k"):
                    setattr(lb, "expand_k", int(state["lats_expand_k"]))
    except Exception:
        pass
    # external 通道净化：任何 internal prompt/debug 泄漏都不允许进入压测对话
    clean_message = sanitize_external_text(str(message or ""))

    now_iso = datetime.now().isoformat()
    state["user_input"] = clean_message
    state["external_user_text"] = clean_message
    state["messages"] = [HumanMessage(content=clean_message, additional_kwargs={"timestamp": now_iso})]
    state["current_time"] = now_iso

    # graph 内部所有 print 只写日志文件，不输出到控制台
    sys.stdout = FileOnlyWriter(log_file)
    try:
        timeout_s = float(os.getenv("BOT2BOT_TURN_TIMEOUT_S", "180") or 180)
        task = asyncio.create_task(app.ainvoke(state, config={"recursion_limit": 50}))
        try:
            result = await asyncio.wait_for(task, timeout=timeout_s)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except Exception:
                pass
            raise TimeoutError(f"turn timeout after {os.getenv('BOT2BOT_TURN_TIMEOUT_S','180')}s")
    except asyncio.TimeoutError:
        sys.stdout = original_stdout
        raise TimeoutError(f"turn timeout after {os.getenv('BOT2BOT_TURN_TIMEOUT_S','180')}s")
    finally:
        sys.stdout = original_stdout

    reply = result.get("final_response") or ""
    if not reply and result.get("final_segments"):
        reply = " ".join(result["final_segments"])
    if not reply:
        reply = result.get("draft_response") or "（无回复）"

    reply_clean = sanitize_external_text(str(reply or ""))
    return reply_clean, (result if isinstance(result, dict) else {})


async def main() -> None:
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL 未设置：请在 .env 里配置本地 PostgreSQL 连接串。")

    # 创建日志文件（提前创建：即使 DB/schema 卡住也能看到进度）
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"bot_to_bot_chat_{ts}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    chat_log_path = log_dir / f"bot_to_bot_chat_{ts}.txt"
    chat_log_file = open(chat_log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout

    def log_line(msg: str):
        """写一行到日志文件、对话记录文件并打印到控制台。"""
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
        chat_log_file.write(msg + "\n")
        chat_log_file.flush()

    db = DBManager.from_env()
    # schema 初始化：偶发情况下 DDL 可能等待锁；bot-to-bot 压测允许跳过/超时继续（表通常已存在）
    if str(os.getenv("BOT2BOT_SKIP_SCHEMA", "0")).lower() not in ("1", "true", "yes", "on"):
        log_line("=" * 60)
        log_line("确保数据库 schema（init_schema.sql）")
        log_line("=" * 60)
        try:
            await asyncio.wait_for(_ensure_schema(db), timeout=float(os.getenv("BOT2BOT_SCHEMA_TIMEOUT_S", "20")))
            log_line("✓ schema 已就绪")
        except asyncio.TimeoutError:
            log_line("⚠ schema 初始化超时（继续执行；若后续报表不存在，请先手动 init_schema.sql）")
        except Exception as e:
            log_line(f"⚠ schema 初始化失败（继续执行；若后续报表不存在，请先手动 init_schema.sql）: {e}")

    log_line("=" * 60)
    log_line("查找或创建两个 Bot")
    log_line("=" * 60)

    # 尝试查找已存在的 bot（通过名称匹配）
    bot_a_id = None
    bot_b_id = None
    bot_a = None
    bot_b = None
    
    async with db.Session() as session:
        # 查找名为"小A"或包含"Bot A"的 bot
        result_a = await session.execute(select(Bot).where(Bot.name.in_(["小A", "Bot A"])))
        bot_a = result_a.scalars().first()
        if bot_a:
            bot_a_id = str(bot_a.id)
            log_line(f"✓ 找到已存在的 Bot A: {bot_a.name} (ID: {bot_a_id})")
        
        # 查找名为"博特·比"/"小智"或包含"Bot B"的 bot
        result_b = await session.execute(select(Bot).where(Bot.name.in_(["博特·比", "小智", "Bot B"])))
        bot_b = result_b.scalars().first()
        if bot_b:
            bot_b_id = str(bot_b.id)
            log_line(f"✓ 找到已存在的 Bot B: {bot_b.name} (ID: {bot_b_id})")
    
    # 如果 bot 不存在，则创建新的
    if not bot_a or not bot_b:
        log_line("\n" + "=" * 60)
        log_line("使用 LLM 创建新的 Bot")
        log_line("=" * 60)
        
        # 获取 LLM 实例
        llm = get_llm()
        log_line(f"LLM 模型: {getattr(llm, 'model_name', 'unknown')}")
        log_line("")

        if not bot_a:
            # 创建两个 Bot ID（UUID 字符串）
            bot_a_id = str(uuid.uuid4())
            # 使用 LLM 创建 Bot A 的人设
            log_line("创建 Bot A...")
            bot_a_basic_info, bot_a_big_five, bot_a_persona = await create_bot_via_llm(
                llm,
                "Bot A",
                "一个性格开朗、喜欢交流的聊天机器人，对新鲜事物充满好奇",
                log_line,
            )
        else:
            bot_a_basic_info = bot_a.basic_info
            bot_a_big_five = bot_a.big_five
            bot_a_persona = bot_a.persona

        if not bot_b:
            bot_b_id = str(uuid.uuid4())
            # 使用 LLM 创建 Bot B 的人设
            log_line("\n创建 Bot B...")
            bot_b_basic_info, bot_b_big_five, bot_b_persona = await create_bot_via_llm(
                llm,
                "Bot B",
                "一个性格温和、善于倾听的聊天机器人，喜欢深入思考问题",
                log_line,
            )
        else:
            bot_b_basic_info = bot_b.basic_info
            bot_b_big_five = bot_b.big_five
            bot_b_persona = bot_b.persona

        log_line("\n" + "=" * 60)
        log_line("将 Bot 写入数据库")
        log_line("=" * 60)

        # 手动创建 Bot 记录（如果不存在）
        async with db.Session() as session:
            async with session.begin():
                if not bot_a:
                    # 创建 Bot A
                    bot_a_uuid = uuid.UUID(bot_a_id)
                    bot_a = Bot(
                        id=bot_a_uuid,
                        name=str(bot_a_basic_info.get("name") or "Bot A"),
                        basic_info=bot_a_basic_info,
                        big_five=bot_a_big_five,
                        persona=bot_a_persona,
                    )
                    session.add(bot_a)
                    await session.flush()
                    log_line(f"✓ Bot A 已创建: {bot_a.name} (ID: {bot_a_id})")

                if not bot_b:
                    # 创建 Bot B
                    bot_b_uuid = uuid.UUID(bot_b_id)
                    bot_b = Bot(
                        id=bot_b_uuid,
                        name=str(bot_b_basic_info.get("name") or "Bot B"),
                        basic_info=bot_b_basic_info,
                        big_five=bot_b_big_five,
                        persona=bot_b_persona,
                    )
                    session.add(bot_b)
                    await session.flush()
                    log_line(f"✓ Bot B 已创建: {bot_b.name} (ID: {bot_b_id})")

    # 为每个 Bot 创建对应的 User 记录（external_id 使用 bot_id）
    # Bot A 作为 User A，Bot B 作为 User B
    user_a_external_id = f"bot_user_{bot_a_id}"
    user_b_external_id = f"bot_user_{bot_b_id}"

    log_line("\n" + "=" * 60)
    log_line("在各自 Bot 下创建 User（get-or-create）")
    log_line("=" * 60)

    # Bot A 下创建/获取 User B；Bot B 下创建/获取 User A
    log_line(f"\nBot A 下 User B: load_state({user_b_external_id!r}, {bot_a_id[:8]}...)")
    _ = await db.load_state(user_b_external_id, bot_a_id)

    log_line(f"Bot B 下 User A: load_state({user_a_external_id!r}, {bot_b_id[:8]}...)")
    _ = await db.load_state(user_a_external_id, bot_b_id)

    # bot-to-bot 关键修复：把 user 画像绑定到“对方 bot 的 persona/basic_info”，避免随机人类画像污染
    try:
        async with db.Session() as session:
            async with session.begin():
                # 重新拉一遍 bot，确保拿到 DB 中的完整字段
                bot_a_db = (await session.execute(select(Bot).where(Bot.id == uuid.UUID(bot_a_id)))).scalar_one()
                bot_b_db = (await session.execute(select(Bot).where(Bot.id == uuid.UUID(bot_b_id)))).scalar_one()

                # Bot A 视角：user_b_external_id 代表“Bot B 这个人”
                u_ab = (
                    (await session.execute(
                        select(User).where(User.bot_id == uuid.UUID(bot_a_id), User.external_id == user_b_external_id)
                    ))
                    .scalars()
                    .first()
                )
                if u_ab:
                    user_basic, user_inferred = _user_profiles_from_bot(
                        bot_b_db.basic_info or {}, bot_b_db.persona or {}, bot_b_db.big_five or {}
                    )
                    u_ab.basic_info = user_basic
                    u_ab.inferred_profile = user_inferred

                # Bot B 视角：user_a_external_id 代表“Bot A 这个人”
                u_ba = (
                    (await session.execute(
                        select(User).where(User.bot_id == uuid.UUID(bot_b_id), User.external_id == user_a_external_id)
                    ))
                    .scalars()
                    .first()
                )
                if u_ba:
                    user_basic, user_inferred = _user_profiles_from_bot(
                        bot_a_db.basic_info or {}, bot_a_db.persona or {}, bot_a_db.big_five or {}
                    )
                    u_ba.basic_info = user_basic
                    u_ba.inferred_profile = user_inferred
        log_line("✓ bot-to-bot: 已将 User 画像绑定为“对方 Bot 人设”")
    except Exception as e:
        log_line(f"⚠ bot-to-bot: 绑定对方画像失败（将继续使用默认画像）: {e}")

    # 可选：每次压测前清空两边关系（避免旧对话/高 liking 把“寒暄增量”掩盖掉）
    if str(os.getenv("BOT2BOT_CLEAR_BEFORE_RUN", "0")).lower() in ("1", "true", "yes", "on"):
        try:
            log_line("\n" + "=" * 60)
            log_line("bot-to-bot: 清空两边关系与记忆（可选）")
            log_line("=" * 60)
            _ = await db.clear_all_memory_for(user_b_external_id, bot_a_id, reset_profile=True)
            _ = await db.clear_all_memory_for(user_a_external_id, bot_b_id, reset_profile=True)
            log_line("✓ 已清空完成")
        except Exception as e:
            log_line(f"⚠ 清空失败（继续执行）: {e}")

    log_line("\n✓ User 初始化完成\n")

    log_line("=" * 60)
    log_line("Bot to Bot 对话开始")
    log_line(f"日志文件: {log_path}")
    log_line(f"对话记录: {chat_log_path}")
    log_line("=" * 60)
    log_line("")

    # 构建 graph
    app = build_graph()

    # 初始消息：Bot A 先说话
    current_message = "你好，很高兴认识你！"
    current_speaker = "Bot A"
    current_user_id = user_b_external_id  # Bot A 把 Bot B 当作 user
    current_bot_id = bot_a_id

    log_line(f"[第 1 轮] {current_speaker} 说: {current_message}")
    log_line("")

    aborted_reason = ""
    # 进行 5 轮对话
    for turn in range(1, 6):
        log_line(f"\n{'=' * 60}")
        log_line(f"第 {turn} 轮对话")
        log_line(f"{'=' * 60}")

        # 当前说话者发送消息
        log_line(f"\n[{current_speaker}] 发送: {current_message}")
        log_line(f"   (user_id={current_user_id}, bot_id={current_bot_id})")
        log_line("")

        # 运行对话
        try:
            # 记录当前日志文件位置（用于定位详细日志）
            log_file_pos_before = log_file.tell() if hasattr(log_file, 'tell') else None
            
            reply, result_state = await run_one_turn(
                app,
                current_user_id,
                current_bot_id,
                current_message,
                log_file,
                original_stdout,
            )
            
            log_file_pos_after = log_file.tell() if hasattr(log_file, 'tell') else None
            log_size_info = ""
            if log_file_pos_before is not None and log_file_pos_after is not None:
                bytes_written = log_file_pos_after - log_file_pos_before
                log_size_info = f" (本轮详细日志: {bytes_written // 1024}KB)"
            
            log_line(f"[{current_speaker} 的 Bot] 回复: {reply}{log_size_info}")
            log_line("")

            # 保存这一轮对话到数据库：
            # - 必须用 graph 真实产出的 state（含 evolver/stage_manager 后的 relationship_state/current_stage）
            # - 仅覆盖必要字段，避免把默认空 dict 写回去导致“关系/阶段不演化”
            state_after = dict(result_state or {})
            state_after.update(
                {
                    "user_id": current_user_id,
                    "bot_id": current_bot_id,
                    "current_time": datetime.now().isoformat(),
                    "user_input": current_message,
                    "final_response": reply,
                }
            )
            await db.save_turn(current_user_id, current_bot_id, state_after)

        except Exception as e:
            log_line(f"[错误] {current_speaker} 的 Bot 回复失败: {e}")
            # 关键：不要把“失败占位文本”继续喂给下一轮（会污染 user_input，造成连锁退化）
            if isinstance(e, TimeoutError):
                aborted_reason = str(e)
                log_line(f"[中止] 本次 bot-to-bot 对话因超时中止：{aborted_reason}")
                break
            aborted_reason = str(e)
            break

        # 切换到另一个 bot
        if current_speaker == "Bot A":
            current_speaker = "Bot B"
            current_user_id = user_a_external_id  # Bot B 把 Bot A 当作 user
            current_bot_id = bot_b_id
        else:
            current_speaker = "Bot A"
            current_user_id = user_b_external_id  # Bot A 把 Bot B 当作 user
            current_bot_id = bot_a_id

        # 下一轮的消息就是当前 bot 的回复
        current_message = reply

        log_line(f"\n{'=' * 60}")
        log_line(f"第 {turn} 轮对话完成")
        log_line(f"{'=' * 60}\n")

    log_line("\n" + "=" * 60)
    if aborted_reason:
        log_line(f"Bot to Bot 对话结束（提前中止，原因: {aborted_reason}）")
    else:
        log_line("Bot to Bot 对话结束（5 轮完成）")
    log_line("=" * 60)
    log_line(f"\n日志文件: {log_path}")
    log_line(f"对话记录: {chat_log_path}")
    
    # 统计日志文件大小
    try:
        log_size = log_path.stat().st_size
        log_size_mb = log_size / (1024 * 1024)
        log_line(f"\n详细日志文件大小: {log_size_mb:.2f}MB ({log_size:,} 字节)")
        log_line("详细日志包含:")
        log_line("  - Detection 节点（用户信号检测）")
        log_line("  - Inner Monologue（内心独白生成）")
        log_line("  - Reasoner（对话策略规划）")
        log_line("  - LATS 搜索（ReplyPlan 生成与搜索过程）")
        log_line("  - Evaluator（硬门槛检查与软评分，含 LLM 详细评分）")
        log_line("  - Processor（消息编译与延迟规划）")
        log_line("  - Evolver（关系状态更新）")
        log_line("  - 所有节点的完整 prompt 和 LLM 响应")
    except Exception:
        pass

    log_file.close()
    chat_log_file.close()

    print(f"\n✅ 完成！对话已保存到:")
    print(f"   日志文件: {log_path}")
    print(f"   对话记录: {chat_log_path}")


if __name__ == "__main__":
    asyncio.run(main())

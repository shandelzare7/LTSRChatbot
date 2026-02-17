"""
bot_to_bot_chat.py

用途：
- 创建两个 Bot（Bot A 和 Bot B），在各自 Bot 下创建对应的 User（互相当对方用户）
- 两 bot 互聊：默认 3 次会话 × 每次 5 轮（可用环境变量 BOT2BOT_NUM_RUNS / BOT2BOT_ROUNDS_PER_RUN 覆盖），首句从池中随机
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
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# 首句池：两 bot 互聊时每次会话的首句随机（避免都是“你好”式打招呼）
FIRST_MESSAGE_POOL = [
    "今天天气好怪啊，一会儿晴一会儿阴的。",
    "你最近有看什么剧或书吗？我剧荒了。",
    "刚想到一个冷笑话，要听吗？",
    "你觉得周末最适合干嘛？睡觉还是出门？",
    "我昨天梦到一件特别离谱的事。",
    "如果只能选一种零食吃一辈子你选啥？",
    "你平时会自己做饭吗？",
    "有没有什么你一直想学但没学的东西？",
    "你更喜欢早起还是熬夜？",
    "假如明天开始不用上班/上学，你第一件事会做啥？",
]

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
from app.services.llm import get_llm, get_llm_stats, reset_llm_stats
from main import _make_initial_state
from utils.llm_json import parse_json_from_llm


def _age_to_age_group(age: int | None) -> str:
    if age is None:
        return "20s"
    try:
        a = int(age)
    except Exception:
        return "20s"
    # bot-to-bot：bot basic_info 偶尔会有脏数据（例如 age=5）。
    # 这里把不合理年龄归一化，避免对方画像被映射成 teen，影响语境与沉浸感。
    if a < 18 or a > 35:
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


async def _ensure_migration_sidewrite_backlog(db: DBManager) -> None:
    """执行 bots 表迁移：增加 character_sidewrite、backlog_tasks 列（若不存在）。"""
    from sqlalchemy import text

    migration_path = Path(__file__).resolve().parent / "migrate_add_bot_sidewrite_backlog.sql"
    if not migration_path.exists():
        return
    sql = migration_path.read_text(encoding="utf-8")
    # 按分号拆分，只丢弃纯注释段（整段 strip 后全是注释或空）
    statements = []
    for s in sql.split(";"):
        stmt = s.strip()
        if not stmt:
            continue
        # 去掉段内首尾的注释行，保留非注释行组成的语句
        lines = [line for line in stmt.splitlines() if line.strip() and not line.strip().startswith("--")]
        stmt = " ".join(lines).strip()
        if stmt:
            statements.append(stmt)
    async with db.engine.connect() as conn:
        ac = await conn.execution_options(isolation_level="AUTOCOMMIT")
        for stmt in statements:
            await ac.execute(text(stmt + ";"))
    # 验证：若列仍不存在则说明 ALTER 未生效（例如连到别的库）
    async with db.engine.connect() as conn:
        try:
            await conn.execute(text("SELECT character_sidewrite FROM bots LIMIT 1"))
        except Exception as e:
            raise RuntimeError(
                "迁移后 bots.character_sidewrite 仍不存在，请检查 DATABASE_URL 是否指向目标库，并手动执行: "
                "devtools/migrate_add_bot_sidewrite_backlog.sql"
            ) from e


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
) -> tuple[str, dict, float]:
    """运行一轮对话，返回 (bot 的回复, result_state, 本轮耗时秒数)。"""
    from main import FileOnlyWriter
    from utils.external_text import sanitize_external_text

    state = _make_initial_state(user_id, bot_id)
    # bot-to-bot 压测：更偏“探索拟人化”而非“根计划过线就早退”
    state["lats_rollouts"] = int(os.getenv("BOT2BOT_LATS_ROLLOUTS", "4"))
    # 默认 expand_k=2：与线上“平衡版”一致（避免变体生成与 soft scorer 调用爆炸）
    state["lats_expand_k"] = int(os.getenv("BOT2BOT_LATS_EXPAND_K", "2"))
    state["lats_early_exit_root_score"] = float(os.getenv("BOT2BOT_EARLY_EXIT_SCORE", "0.82"))
    state["lats_early_exit_plan_alignment_min"] = float(os.getenv("BOT2BOT_EARLY_EXIT_PLAN_MIN", "0.75"))
    state["lats_early_exit_assistantiness_max"] = float(os.getenv("BOT2BOT_EARLY_EXIT_ASSIST_MAX", "0.22"))
    state["lats_early_exit_mode_fit_min"] = float(os.getenv("BOT2BOT_EARLY_EXIT_MODE_MIN", "0.60"))
    state["lats_disable_early_exit"] = (str(os.getenv("BOT2BOT_DISABLE_EARLY_EXIT", "1")).lower() not in ("0", "false", "no", "off"))
    state["lats_skip_low_risk"] = (str(os.getenv("BOT2BOT_SKIP_LATS_LOW_RISK", "0")).lower() in ("1", "true", "yes", "on"))
    # soft scorer 仍启用，但只评 Top1，且并发=1（更稳更省）
    try:
        state["lats_llm_soft_top_n"] = int(os.getenv("BOT2BOT_LLM_SOFT_TOP_N", "1") or 1)
    except Exception:
        state["lats_llm_soft_top_n"] = 1
    try:
        state["lats_llm_soft_max_concurrency"] = int(os.getenv("BOT2BOT_LLM_SOFT_MAX_CONCURRENCY", "1") or 1)
    except Exception:
        state["lats_llm_soft_max_concurrency"] = 1
    try:
        state["lats_assistant_check_top_n"] = int(os.getenv("BOT2BOT_ASSISTANT_CHECK_TOP_N", "0") or 0)
    except Exception:
        state["lats_assistant_check_top_n"] = 0

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
    t0 = time.perf_counter()
    try:
        # Reset LLM stats for this turn (best-effort; only active when LTSR_LLM_STATS/LTSR_PROFILE_STEPS is enabled).
        try:
            reset_llm_stats()
        except Exception:
            pass
        timeout_s = float(os.getenv("BOT2BOT_TURN_TIMEOUT_S", "180") or 180)
        task = asyncio.create_task(app.ainvoke(state, config={"recursion_limit": 50}))
        try:
            result = await asyncio.wait_for(task, timeout=timeout_s)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            raise TimeoutError(f"turn timeout after {os.getenv('BOT2BOT_TURN_TIMEOUT_S','180')}s")
    except asyncio.TimeoutError:
        sys.stdout = original_stdout
        raise TimeoutError(f"turn timeout after {os.getenv('BOT2BOT_TURN_TIMEOUT_S','180')}s")
    finally:
        sys.stdout = original_stdout

    elapsed = time.perf_counter() - t0  # 仅成功完成时计算
    reply = result.get("final_response") or ""
    if not reply and result.get("final_segments"):
        reply = " ".join(result["final_segments"])
    if not reply:
        reply = result.get("draft_response") or "（无回复）"

    reply_clean = sanitize_external_text(str(reply or ""))
    out_state = (result if isinstance(result, dict) else {})
    try:
        out_state["_llm_stats"] = get_llm_stats()
    except Exception:
        pass
    return reply_clean, out_state, elapsed


async def main() -> None:
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL 未设置：请在 .env 里配置本地 PostgreSQL 连接串。")

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    original_stdout = sys.stdout

    log_file = None  # 整次运行共用一个 .log 文件，供 log_line 与 run_one_turn 写入
    def log_line(msg: str):
        """写一行到当前日志文件并打印到控制台。"""
        print(msg)
        if log_file is not None:
            log_file.write(msg + "\n")
            log_file.flush()

    # 整次运行只写一个文件：启动信息 + 所有会话/轮次都追加到同一 .log
    single_log_path = log_dir / f"bot_to_bot_chat_{ts}.log"
    log_file = open(single_log_path, "w", encoding="utf-8")

    db = DBManager.from_env()
    # schema 初始化：偶发情况下 DDL 可能等待锁；bot-to-bot 压测允许跳过/超时继续（表通常已存在）
    if str(os.getenv("BOT2BOT_SKIP_SCHEMA", "0")).lower() not in ("1", "true", "yes", "on"):
        log_line("=" * 60)
        log_line("确保数据库 schema（init_schema.sql）")
        log_line("=" * 60)
        try:
            await asyncio.wait_for(_ensure_schema(db), timeout=float(os.getenv("BOT2BOT_SCHEMA_TIMEOUT_S", "20")))
            log_line("执行 migration: bots 表增加 character_sidewrite / backlog_tasks")
            await _ensure_migration_sidewrite_backlog(db)
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
    
    # 仅使用新生成的两个 Bot 做 bot-to-bot（支持 LLM 生成的全名，如李阳/林静怡 或 李浩然/苏雨桐）
    BOT_A_NAMES = ["李阳", "李浩然"]
    BOT_B_NAMES = ["林静怡", "苏雨桐"]

    async with db.Session() as session:
        result_a = await session.execute(select(Bot).where(Bot.name.in_(BOT_A_NAMES)))
        bot_a = result_a.scalars().first()
        if bot_a:
            bot_a_id = str(bot_a.id)
            log_line(f"✓ 找到 Bot A: {bot_a.name} (ID: {bot_a_id})")
        
        result_b = await session.execute(select(Bot).where(Bot.name.in_(BOT_B_NAMES)))
        bot_b = result_b.scalars().first()
        if bot_b:
            bot_b_id = str(bot_b.id)
            log_line(f"✓ 找到 Bot B: {bot_b.name} (ID: {bot_b_id})")
    
    if not bot_a or not bot_b:
        log_line("")
        log_line("未找到新 Bot（李阳、林静怡）。请先执行：")
        log_line("  1) 删除旧 Bot: python -m devtools.delete_old_bots_keep_new")
        log_line("  2) 创建新 Bot: python -m devtools.create_two_bots_for_render")
        log_line("然后再运行本脚本。")
        sys.exit(1)

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

    # 可选：仅在“第一次”压测前清空（BOT2BOT_CLEAR_BEFORE_RUN=1）
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

    # 构建 graph
    app = build_graph()

    aborted_reason = ""
    # Allow overriding run counts for profiling / quick tests
    try:
        num_runs = int(os.getenv("BOT2BOT_NUM_RUNS", "3") or 3)
    except Exception:
        num_runs = 3
    try:
        rounds_per_run = int(os.getenv("BOT2BOT_ROUNDS_PER_RUN", "5") or 5)
    except Exception:
        rounds_per_run = 5
    turn_times: list[float] = []  # 每轮回复耗时（秒），用于算平均
    
    log_line("=" * 60)
    log_line(f"Bot to Bot 对话开始（{num_runs} 次会话 × 每次 {rounds_per_run} 轮，首句随机）")
    log_line(f"本次运行全部写入: {single_log_path.name}")
    log_line("=" * 60)
    log_line("")

    for run in range(1, num_runs + 1):
        # 每次会话前清空，使多次会话互不干扰；首句随机
        if run > 1:
            try:
                await db.clear_all_memory_for(user_b_external_id, bot_a_id, reset_profile=True)
                await db.clear_all_memory_for(user_a_external_id, bot_b_id, reset_profile=True)
            except Exception:
                pass
        current_message = random.choice(FIRST_MESSAGE_POOL)
        current_speaker = "Bot A"
        current_user_id = user_b_external_id
        current_bot_id = bot_a_id

        log_line("\n" + "=" * 60)
        log_line(f"第 {run}/{num_runs} 次会话（首句随机）")
        log_line("=" * 60)
        log_line(f"[会话 {run}] 首句: {current_message}")
        log_line("")

        for turn in range(1, rounds_per_run + 1):
            log_line(f"\n--- 第 {run} 次会话 / 第 {turn} 轮 ---")
            log_line(f"[{current_speaker}] 发送: {current_message}")
            log_line(f"   (user_id={current_user_id}, bot_id={current_bot_id})")
            log_line("")

            try:
                log_file_pos_before = log_file.tell() if hasattr(log_file, "tell") else None
                reply, result_state, elapsed = await run_one_turn(
                    app,
                    current_user_id,
                    current_bot_id,
                    current_message,
                    log_file,
                    original_stdout,
                )
                turn_times.append(elapsed)
                log_file_pos_after = log_file.tell() if hasattr(log_file, "tell") else None
                log_size_info = ""
                if log_file_pos_before is not None and log_file_pos_after is not None:
                    log_size_info = f" (本轮详细日志: {(log_file_pos_after - log_file_pos_before) // 1024}KB)"
                log_line(f"[{current_speaker} 的 Bot] 回复: {reply} [耗时 {elapsed:.2f}s]{log_size_info}")

                # Optional: step-by-step profiling report (requires LTSR_PROFILE_STEPS=1 / LTSR_LLM_STATS=1)
                prof = (result_state or {}).get("_profile") if isinstance(result_state, dict) else None
                llm_stats = (result_state or {}).get("_llm_stats") if isinstance(result_state, dict) else None
                if isinstance(prof, dict) and isinstance(prof.get("nodes"), list):
                    log_line("  [PROFILE] 节点耗时与 LLM 调用增量：")
                    for item in prof.get("nodes") or []:
                        name = str(item.get("name") or "")
                        dt_ms = float(item.get("dt_ms", 0.0) or 0.0)
                        delta = item.get("llm_delta") if isinstance(item.get("llm_delta"), dict) else {}
                        # Summarize delta calls
                        delta_calls = sum(int(v.get("calls", 0) or 0) for v in delta.values()) if isinstance(delta, dict) else 0
                        log_line(f"    - {name}: {dt_ms:.2f}ms, llm_calls_delta={delta_calls}")
                if isinstance(llm_stats, dict) and llm_stats:
                    log_line("  [PROFILE] 本轮各模型/角色 API 调用统计：")
                    # Sort by calls desc
                    rows = []
                    for k, v in llm_stats.items():
                        try:
                            calls = int(v.get("calls", 0) or 0)
                            total_ms = float(v.get("total_ms", 0.0) or 0.0)
                        except Exception:
                            continue
                        rows.append((calls, total_ms, str(k)))
                    rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    for calls, total_ms, k in rows[:20]:
                        avg_ms = (total_ms / calls) if calls else 0.0
                        log_line(f"    - {k}: calls={calls}, total_ms={total_ms:.1f}, avg_ms={avg_ms:.1f}")
                log_line("")

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
                if isinstance(e, TimeoutError):
                    aborted_reason = str(e)
                    log_line(f"[中止] 因超时中止：{aborted_reason}")
                else:
                    aborted_reason = str(e)
                break

            if current_speaker == "Bot A":
                current_speaker = "Bot B"
                current_user_id = user_a_external_id
                current_bot_id = bot_b_id
            else:
                current_speaker = "Bot A"
                current_user_id = user_b_external_id
                current_bot_id = bot_a_id
            current_message = reply

        if aborted_reason:
            break
        log_line(f"\n第 {run}/{num_runs} 次会话（{rounds_per_run} 轮）完成\n")

    if log_file is not None:
        log_file.close()
        log_file = None

    # 总结只打控制台
    print("\n" + "=" * 60)
    if aborted_reason:
        print(f"Bot to Bot 对话结束（提前中止，原因: {aborted_reason}）")
    else:
        print(f"Bot to Bot 对话结束（{num_runs} 次会话 × {rounds_per_run} 轮完成）")
    print("=" * 60)
    print(f"日志文件: {single_log_path}")
    try:
        if single_log_path.exists():
            print(f"文件大小: {single_log_path.stat().st_size / (1024 * 1024):.2f} MB")
    except Exception:
        pass
    if turn_times:
        avg_time = sum(turn_times) / len(turn_times)
        print(f"\n📊 回复耗时统计: 共 {len(turn_times)} 轮, 平均回复时间 = {avg_time:.2f} 秒")
    print("\n✅ 完成！本次运行所有内容已写入同一日志文件。")


if __name__ == "__main__":
    asyncio.run(main())

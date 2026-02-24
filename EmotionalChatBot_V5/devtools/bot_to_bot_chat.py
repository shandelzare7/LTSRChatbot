"""
bot_to_bot_chat.py

用途：
- 创建两个 Bot（Bot A 和 Bot B），在各自 Bot 下创建对应的 User（互相当对方用户）
- 两 bot 互聊：默认 1 次会话 × 10 轮（可用环境变量 BOT2BOT_NUM_RUNS / BOT2BOT_ROUNDS_PER_RUN 覆盖），首句从池中随机
- 默认开启 BOT2BOT_FULL_LOGS=1：完整记录生成与评审的 prompt 与输出（含 LATS 27 候选生成、单模型评估）
- 默认不走 fast 路由（force_fast_route=0），走 LATS 生成+评审
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
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Big Five 与 PADB 随机化，增加 Bot 多样性（避免每次两 bot 人格/情绪相同）
BIG_FIVE_KEYS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
PADB_KEYS = ["pleasure", "arousal", "dominance", "busyness"]


def _random_big_five() -> Dict[str, float]:
    """大五人格各维 [0, 1] 随机。"""
    return {k: round(random.uniform(0.0, 1.0), 4) for k in BIG_FIVE_KEYS}


def _random_padb() -> Dict[str, float]:
    """PAD 情绪：pleasure/arousal/dominance 在 [-1, 1]，busyness 在 [0, 1]。"""
    out = {k: round(random.uniform(-1.0, 1.0), 4) for k in ["pleasure", "arousal", "dominance"]}
    out["busyness"] = round(random.uniform(0.0, 1.0), 4)
    return out


# Bot 描述随机池
_PERSONALITY_POOL_MALE = [
    "性格开朗、喜欢交流，对新鲜事物充满好奇",
    "话少但幽默感强，喜欢冷幽默和自嘲",
    "有点社恐但内心热情，熟了之后话很多",
    "大大咧咧不拘小节，情绪表达直接",
    "文艺青年气质，说话慢条斯理但有深度",
    "理性派，喜欢讲逻辑但也会开玩笑",
    "热心肠，爱操心朋友的事，有时话多",
]
_PERSONALITY_POOL_FEMALE = [
    "性格温和、善于倾听，喜欢深入思考",
    "活泼外向，反应快，说话带点俏皮",
    "慢热但真诚，跟熟人聊天很放松",
    "独立有主见，说话简洁利落",
    "感性细腻，容易被小事打动，表达丰富",
    "有点毒舌但心软，嘴硬心软型",
    "安静内敛，偶尔冒出金句让人惊艳",
]
_OCCUPATION_POOL_MALE = [
    "产品经理", "设计师", "教师", "插画师", "自由撰稿人",
    "咖啡师", "建筑师", "纪录片导演", "翻译", "乐队吉他手",
    "体育教练", "书店老板", "广告创意总监", "厨师", "旅行博主",
]
_OCCUPATION_POOL_FEMALE = [
    "编辑", "运营", "心理咨询师", "策展人", "摄影师",
    "花艺师", "独立音乐人", "插画师", "甜品师", "瑜伽教练",
    "播客主播", "图书馆员", "手工皮具匠人", "建筑师", "调酒师",
]


def _random_bot_description(gender: str) -> str:
    """为 bot 创建 prompt 随机生成描述。"""
    if gender == "男":
        personality = random.choice(_PERSONALITY_POOL_MALE)
        occupations = random.sample(_OCCUPATION_POOL_MALE, 3)
        avoid = "避免李明、张伟、王强等常见名"
    else:
        personality = random.choice(_PERSONALITY_POOL_FEMALE)
        occupations = random.sample(_OCCUPATION_POOL_FEMALE, 3)
        avoid = "避免李静怡、王芳、张敏等常见名"
    occ_str = "、".join(occupations)
    return (
        f"请为人设起一个少见、有记忆点的中文全名（姓+名），{avoid}；"
        f"{gender}性。{personality}。"
        f"职业不要程序员，请从以下任选其一：{occ_str}。"
    )


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
from app.core.relationship_templates import get_relationship_template_by_name
from app.graph import build_graph
from app.services.llm import get_llm, get_llm_stats, reset_llm_stats, set_current_node, reset_current_node
from main import _make_initial_state
from utils.llm_json import parse_json_from_llm

# 随机种子：BOT2BOT_SEED 环境变量，不设则每次不同
_seed_env = os.getenv("BOT2BOT_SEED", "").strip()
if _seed_env:
    random.seed(int(_seed_env))
    print(f"[bot_to_bot] 使用固定随机种子: {_seed_env}")
else:
    _auto_seed = random.randint(0, 2**31)
    random.seed(_auto_seed)
    print(f"[bot_to_bot] 自动随机种子: {_auto_seed}（可用 BOT2BOT_SEED={_auto_seed} 复现）")


# ==========================================
# 辅助函数：指标追踪
# ==========================================


def check_punctuation_removal(original_text: str, processed_segments: List[Any]) -> Dict[str, Any]:
    """检查标点符号去除情况"""
    unwanted_punct = ['：', '～', '——', '（', '）', '(', ')', '：', '；']
    original_count = sum(original_text.count(p) for p in unwanted_punct)
    
    # 处理segments：可能是字符串列表或字典列表
    processed_text_parts = []
    if isinstance(processed_segments, list):
        for seg in processed_segments:
            if isinstance(seg, dict):
                content = seg.get("content", "")
                processed_text_parts.append(str(content))
            else:
                processed_text_parts.append(str(seg))
    else:
        processed_text_parts.append(str(processed_segments))
    
    processed_text = ' '.join(processed_text_parts)
    processed_count = sum(processed_text.count(p) for p in unwanted_punct)
    removed = original_count - processed_count
    success_rate = (removed / original_count) if original_count > 0 else 1.0
    return {
        'original_count': original_count,
        'processed_count': processed_count,
        'removed': removed,
        'success_rate': success_rate,
        'original_text': original_text,
        'processed_text': processed_text,
    }


async def get_user_basic_info(db: DBManager, user_id: str, bot_id: str) -> Dict[str, Any]:
    """从数据库读取user的basic_info"""
    try:
        async with db.Session() as session:
            result = await session.execute(
                select(User).where(User.bot_id == uuid.UUID(bot_id), User.external_id == user_id)
            )
            user_obj = result.scalars().first()
            if user_obj:
                return dict(user_obj.basic_info) if user_obj.basic_info else {}
    except Exception as e:
        print(f"[WARN] Failed to read user basic_info: {e}")
    return {}


async def get_user_inferred_profile(db: DBManager, user_id: str, bot_id: str) -> Dict[str, Any]:
    """从数据库读取 user 的 inferred_profile（memory_writer 写入的画像）"""
    try:
        async with db.Session() as session:
            result = await session.execute(
                select(User).where(User.bot_id == uuid.UUID(bot_id), User.external_id == user_id)
            )
            user_obj = result.scalars().first()
            if user_obj:
                return dict(user_obj.inferred_profile) if user_obj.inferred_profile else {}
    except Exception as e:
        print(f"[WARN] Failed to read user inferred_profile: {e}")
    return {}


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
    try:
        age = int(basic.get("age")) if basic.get("age") is not None else 22
    except (TypeError, ValueError):
        age = 22
    if age < 18 or age > 40:
        age = 22
    location = _region_to_location(basic.get("region"))

    user_basic_info = {
        "name": name,
        "gender": basic.get("gender"),
        "age": age,
        "location": location,
        "occupation": basic.get("occupation"),
        # 标记：该 user 是 bot-to-bot 中的“对方 bot 代理画像”
        "bot_proxy": True,
    }

    # inferred_profile 无固定字段；bot-to-bot 仅标记代理来源
    user_inferred_profile = {"bot_proxy": True}
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
   - lore: {{"origin": "简短来历", "secret": "小秘密"}}
   - story: "一段话的背景故事（可选，如成长经历、为何做现在的事、对关系的态度等）"

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
        # 如果遇到负值（旧数据），自动转换为绝对值
        for key in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            if key not in big_five:
                big_five[key] = 0.5
            else:
                try:
                    val = float(big_five[key])
                    # 如果值是负数（旧数据），转换为绝对值
                    if val < 0.0:
                        val = abs(val)
                    # 限制在 0.0 到 1.0 之间
                    big_five[key] = max(0.0, min(1.0, val))
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
        if "story" not in persona or not isinstance(persona.get("story"), str):
            persona["story"] = ""
        
        log_line_func(f"  ✓ {bot_name} 人设生成成功")
        log_line_func(f"    名字: {basic_info.get('name')}, 年龄: {basic_info.get('age')}, 职业: {basic_info.get('occupation')}")
        
        return basic_info, big_five, persona
        
    except Exception as e:
        log_line_func(f"  ⚠ LLM 生成失败 ({e})，使用默认人设")
        from app.core.profile_factory import generate_bot_profile
        return generate_bot_profile(bot_name)


class _TeeStderr:
    """将 stderr 同时写入 log 与原始 stderr，使 [LLM_ELAPSED] 既实时显示又进入日志供报告解析。"""

    def __init__(self, log_file, original_stderr):
        self._log = log_file
        self._err = original_stderr

    def write(self, s: str) -> None:
        self._log.write(s)
        self._log.flush()
        self._err.write(s)
        self._err.flush()

    def flush(self) -> None:
        self._log.flush()
        self._err.flush()


async def run_one_turn(
    app,
    user_id: str,
    bot_id: str,
    message: str,
    log_file,
    original_stdout,
    original_stderr,
) -> tuple[str, dict, float]:
    """运行一轮对话，返回 (bot 的回复, result_state, 本轮耗时秒数)。"""
    from main import FileOnlyWriter
    from utils.external_text import sanitize_external_text

    state = _make_initial_state(user_id, bot_id)
    # 仅本脚本运行时强制不走 fast 路由（不依赖环境变量 FAST_ROUTE_WHEN_QUICK_REPLY_ENABLED）
    state["bot2bot_disable_fast_route"] = True
    state["force_fast_route"] = False  # 明确不走 fast_reply，走 LATS 生成+评审
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
    # reply_planner 重跑次数上限（LATS 内 planner 质量不达标时最多再生成几轮候选）
    state["lats_max_regens"] = int(os.getenv("BOT2BOT_LATS_MAX_REGENS", "2") or 2)
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

    # external 通道净化：任何 internal prompt/debug 泄漏都不允许进入压测对话
    clean_message = sanitize_external_text(str(message or ""))

    now_iso = datetime.now().isoformat()
    state["user_input"] = clean_message
    state["external_user_text"] = clean_message
    state["messages"] = [HumanMessage(content=clean_message, additional_kwargs={"timestamp": now_iso})]
    state["current_time"] = now_iso

    # graph 内部所有 print 只写日志文件；stderr tee 到 log+控制台，便于 [LLM_ELAPSED] 实时监控且进日志
    sys.stdout = FileOnlyWriter(log_file)
    sys.stderr = _TeeStderr(log_file, original_stderr)
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
        sys.stderr = original_stderr
        raise TimeoutError(f"turn timeout after {os.getenv('BOT2BOT_TURN_TIMEOUT_S','180')}s")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

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

    # 强制关闭「前两条回复总用时走 fast」的开关，确保 30 轮全走 LATS 生成+评审（与 state 里 force_fast_route=False 一致）
    os.environ["FAST_ROUTE_WHEN_QUICK_REPLY_ENABLED"] = "0"

    # 完整记录生成与评审的 prompt 与输出（含 LATS 27 候选生成、单模型评估）
    os.environ.setdefault("BOT2BOT_FULL_LOGS", "1")

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    original_stdout = sys.stdout
    original_stderr = sys.stderr

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
    log_line(f"FAST_ROUTE_WHEN_QUICK_REPLY_ENABLED={os.getenv('FAST_ROUTE_WHEN_QUICK_REPLY_ENABLED', '')}（脚本内已强制 0，走 LATS）")

    bot_a_id = None
    bot_b_id = None
    bot_a = None
    bot_b = None
    # 本脚本用于「两个新 Bot + 互为空 User + 30 轮」：不使用已有 Bot，始终创建两个新 Bot
    create_new_bots = True

    if create_new_bots:
        # 新建 User 关系维度参考：app/core/relationship_templates.py（RELATIONSHIP_TEMPLATES）
        log_line("创建两个新 Bot；空 User 的关系维度参考 app/core/relationship_templates.py")
        from app.core.bot_creation_llm import generate_sidewrite_and_backlog
        llm = get_llm(role="fast")  # 创建 bot 等脚本统一用 gpt-4o-mini，不用 gpt-4o
        bot_a_id = str(uuid.uuid4())
        bot_b_id = str(uuid.uuid4())
        # 默认 PADB（异常时 fallback）
        b1_padb = b2_padb = {"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0}
        tok = set_current_node("bot_creation")
        try:
            desc_a = _random_bot_description("男")
            log_line(f"创建 Bot A（男）… 描述: {desc_a[:60]}…")
            b1_basic, b1_big_five, b1_persona = await create_bot_via_llm(
                llm, "Bot A", desc_a, log_line,
            )
            b1_sidewrite, b1_backlog = None, None
            try:
                b1_sidewrite, b1_backlog = await generate_sidewrite_and_backlog(llm, b1_basic, b1_big_five, b1_persona)
            except Exception as e:
                log_line(f"  ⚠ 侧写/任务库生成失败: {e}")
            desc_b = _random_bot_description("女")
            log_line(f"创建 Bot B（女）… 描述: {desc_b[:60]}…")
            b2_basic, b2_big_five, b2_persona = await create_bot_via_llm(
                llm, "Bot B", desc_b, log_line,
            )
            b2_sidewrite, b2_backlog = None, None
            try:
                b2_sidewrite, b2_backlog = await generate_sidewrite_and_backlog(llm, b2_basic, b2_big_five, b2_persona)
            except Exception as e:
                log_line(f"  ⚠ 侧写/任务库生成失败: {e}")
            # 用随机 Big Five 与 PADB 覆盖 LLM 输出，增加两 Bot 多样性
            b1_big_five = _random_big_five()
            b2_big_five = _random_big_five()
            b1_padb = _random_padb()
            b2_padb = _random_padb()
            log_line(f"  Bot A 随机 big_five={b1_big_five}  mood_state(PADB)={b1_padb}")
            log_line(f"  Bot B 随机 big_five={b2_big_five}  mood_state(PADB)={b2_padb}")
        finally:
            reset_current_node(tok)
        async with db.Session() as session:
            async with session.begin():
                bot1 = Bot(
                    id=uuid.UUID(bot_a_id),
                    name=str(b1_basic.get("name") or "Bot A"),
                    basic_info=b1_basic,
                    big_five=b1_big_five,
                    persona=b1_persona,
                    character_sidewrite=b1_sidewrite,
                    backlog_tasks=b1_backlog or [],
                    mood_state=b1_padb,
                )
                bot2 = Bot(
                    id=uuid.UUID(bot_b_id),
                    name=str(b2_basic.get("name") or "Bot B"),
                    basic_info=b2_basic,
                    big_five=b2_big_five,
                    persona=b2_persona,
                    character_sidewrite=b2_sidewrite,
                    backlog_tasks=b2_backlog or [],
                    mood_state=b2_padb,
                )
                session.add(bot1)
                session.add(bot2)
                await session.flush()
                bot_a, bot_b = bot1, bot2
        log_line(f"✓ Bot A: {bot_a.name} (ID: {bot_a_id[:8]}...)")
        log_line(f"✓ Bot B: {bot_b.name} (ID: {bot_b_id[:8]}...)")

    # 为每个 Bot 创建对应的 User 记录（external_id 使用 bot_id）
    # Bot A 作为 User A，Bot B 作为 User B
    user_a_external_id = f"bot_user_{bot_a_id}"
    user_b_external_id = f"bot_user_{bot_b_id}"

    # 【身份约定，勿反】run_one_turn(user_id, bot_id, message) 语义：
    # - user_id = 本轮回话中「发消息的人」在 DB 里的 external_id（即对方 Bot 的 proxy）
    # - bot_id = 本轮回话中「回复的 Bot」的 id
    # - 首句：Bot A 发 → (user_a_external_id, bot_b_id)，即 Bot B 视角下「用户=Bot A」收首句并回复
    # - 轮换：本轮是 bot_id 回复后，下一轮 sender=该 bot，replier=对方 → (对方 proxy 的 user_id, 对方 bot_id)

    log_line("\n" + "=" * 60)
    log_line("在各自 Bot 下创建 User（get-or-create）")
    log_line("=" * 60)

    # Bot A 下创建/获取 User B；Bot B 下创建/获取 User A
    log_line(f"\nBot A 下 User B: load_state({user_b_external_id!r}, {bot_a_id[:8]}...)")
    _ = await db.load_state(user_b_external_id, bot_a_id)

    log_line(f"Bot B 下 User A: load_state({user_a_external_id!r}, {bot_b_id[:8]}...)")
    _ = await db.load_state(user_a_external_id, bot_b_id)

    # 强制设置为空人设，且关系维度使用参考值（app/core/relationship_templates.py）
    template_name = (os.getenv("BOT2BOT_USER_DIMENSIONS_TEMPLATE") or "friendly_icebreaker").strip()
    if template_name not in ("neutral_stranger", "friendly_icebreaker", "moderate_acquaintance"):
        template_name = "friendly_icebreaker"
    ref_dims = get_relationship_template_by_name(template_name)  # type: ignore[arg-type]
    log_line("\n" + "=" * 60)
    log_line("强制设置 User：basic_info/inferred_profile 为空，dimensions 使用参考值")
    log_line(f"  参考文档: app/core/relationship_templates.py  模板: {template_name}")
    log_line(f"  参考值: {ref_dims}")
    log_line("=" * 60)
    try:
        async with db.Session() as session:
            async with session.begin():
                u_ab = (
                    (await session.execute(
                        select(User).where(User.bot_id == uuid.UUID(bot_a_id), User.external_id == user_b_external_id)
                    ))
                    .scalars()
                    .first()
                )
                if u_ab:
                    u_ab.basic_info = {}
                    u_ab.inferred_profile = {}
                    u_ab.dimensions = dict(ref_dims)
                    log_line(f"✓ Bot A 下的 User B: 空人设 + dimensions={template_name}")
                u_ba = (
                    (await session.execute(
                        select(User).where(User.bot_id == uuid.UUID(bot_b_id), User.external_id == user_a_external_id)
                    ))
                    .scalars()
                    .first()
                )
                if u_ba:
                    u_ba.basic_info = {}
                    u_ba.inferred_profile = {}
                    u_ba.dimensions = dict(ref_dims)
                    log_line(f"✓ Bot B 下的 User A: 空人设 + dimensions={template_name}")
        log_line("✓ 空人设与参考值设置完成")
    except Exception as e:
        log_line(f"⚠ 空人设设置失败: {e}")
    
    # 保留原有逻辑作为fallback（如果不需要空人设）
    use_empty_profile = str(os.getenv("BOT2BOT_EMPTY_USER_PROFILE", "1")).lower() in ("1", "true", "yes", "on")
    if not use_empty_profile:
        # bot-to-bot 关键修复：把 user 画像绑定到“对方 bot 的 persona/basic_info”，避免随机人类画像污染
        try:
            async with db.Session() as session:
                async with session.begin():
                    bot_a_db = (await session.execute(select(Bot).where(Bot.id == uuid.UUID(bot_a_id)))).scalar_one()
                    bot_b_db = (await session.execute(select(Bot).where(Bot.id == uuid.UUID(bot_b_id)))).scalar_one()

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
    # 30轮测试：默认单次会话30轮
    try:
        num_runs = int(os.getenv("BOT2BOT_NUM_RUNS", "1") or 1)
    except Exception:
        num_runs = 1
    try:
        rounds_per_run = int(os.getenv("BOT2BOT_ROUNDS_PER_RUN", "10") or 10)
    except Exception:
        rounds_per_run = 10
    turn_times: list[float] = []  # 每轮回复耗时（秒），用于算平均
    time_to_reply_ms_list: list[float] = []  # 每轮「截止到生成回复」的毫秒数（需 LTSR_PROFILE_STEPS=1）
    
    # 30轮测试追踪数据
    conversation_log: List[Dict[str, Any]] = []  # 完整聊天记录
    momentum_history: List[float] = []  # 冲量历史（扁平列表，用于汇总 max/min/avg）
    # 上一轮动量，按 (user_id, bot_id) 存，使 delta = 当前会话的 x(t)-x(t-1)，不混对方轮次
    prev_momentum_by_key: Dict[Tuple[str, str], float] = {}
    # 上一轮 6 维关系状态，用于计算每轮变化；(user_id, bot_id) -> {dim: value}
    prev_relationship_state: Dict[Tuple[str, str], Dict[str, float]] = {}
    DIM_KEYS = ("closeness", "trust", "liking", "respect", "attractiveness", "power")
    basic_info_task_triggered: Dict[str, int] = {}  # 基础信息任务触发次数
    basic_info_task_executed: Dict[str, int] = {}  # 基础信息任务执行次数（attempted）
    basic_info_task_completed: Dict[str, int] = {}  # 基础信息任务完成次数
    basic_info_written: Dict[str, int] = {}  # 流程写入 DB 的基础信息次数（真实流程）
    punctuation_removal_stats: List[Dict[str, any]] = []  # 标点去除统计
    
    log_line("=" * 60)
    log_line(f"Bot to Bot 对话开始（{num_runs} 次会话 × 每次 {rounds_per_run} 轮，首句随机）")
    log_line(f"本次运行全部写入: {single_log_path.name}")
    log_line("完整 prompt/输出日志: BOT2BOT_FULL_LOGS=1 已开启")
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
            prev_momentum_by_key.clear()  # 新会话首轮 delta 按 N/A，不沿用上一会话
        current_message = random.choice(FIRST_MESSAGE_POOL)
        # 首句是 Bot A 说的 → 语义应为「Bot A 发、Bot B 回」，即 user=Bot A 的 proxy，bot=Bot B
        current_speaker = "Bot A"
        current_user_id = user_a_external_id
        current_bot_id = bot_b_id

        log_line("\n" + "=" * 60)
        log_line(f"第 {run}/{num_runs} 次会话（首句随机）")
        log_line("=" * 60)
        log_line(f"[会话 {run}] 首句: {current_message}")
        log_line("")

        for turn in range(1, rounds_per_run + 1):
            log_line(f"\n{'='*60}")
            log_line(f"--- 第 {run} 次会话 / 第 {turn} 轮 ---")
            log_line(f"{'='*60}")
            log_line(f"[{current_speaker}] 发送: {current_message}")
            log_line(f"   (user_id={current_user_id}, bot_id={current_bot_id})")
            log_line("")

            # 记录本轮开始前的basic_info状态（用于检测写入）
            basic_info_before = await get_user_basic_info(db, current_user_id, current_bot_id)

            try:
                log_file_pos_before = log_file.tell() if hasattr(log_file, "tell") else None
                reply, result_state, elapsed = await run_one_turn(
                    app,
                    current_user_id,
                    current_bot_id,
                    current_message,
                    log_file,
                    original_stdout,
                    original_stderr,
                )
                turn_times.append(elapsed)
                log_file_pos_after = log_file.tell() if hasattr(log_file, "tell") else None
                log_size_info = ""
                if log_file_pos_before is not None and log_file_pos_after is not None:
                    log_size_info = f" (本轮详细日志: {(log_file_pos_after - log_file_pos_before) // 1024}KB)"
                log_line(f"[{current_speaker} 的 Bot] 回复: {reply} [耗时 {elapsed:.2f}s]{log_size_info}")

                # ==========================================
                # 1. 每轮回复原文输出（包括断句）
                # ==========================================
                final_response = (result_state or {}).get("final_response") or ""
                final_segments = (result_state or {}).get("final_segments") or []
                humanized_output = (result_state or {}).get("humanized_output") or {}
                segments = humanized_output.get("segments") or final_segments
                
                log_line("\n" + "="*60)
                log_line(f"[ROUND {turn} REPLY]")
                log_line("="*60)
                log_line("原始回复（final_response）:")
                log_line(f"  {final_response}")
                log_line("")
                log_line("断句后的segments:")
                if segments:
                    for i, seg in enumerate(segments, 1):
                        if isinstance(seg, dict):
                            content = seg.get("content", "")
                            delay = seg.get("delay", 0)
                            action = seg.get("action", "typing")
                            log_line(f"  [{i}] {content} [delay={delay}s, action={action}]")
                        else:
                            log_line(f"  [{i}] {seg}")
                else:
                    log_line("  (无segments)")
                log_line("="*60)
                
                # ==========================================
                # 2. 追踪指标
                # ==========================================
                
                # 2.1 每轮总用时（截止到processor结束）
                prof = (result_state or {}).get("_profile") if isinstance(result_state, dict) else None
                time_to_processor_ms = None
                if isinstance(prof, dict) and isinstance(prof.get("nodes"), list):
                    nodes_list = prof.get("nodes") or []
                    seen = set()
                    time_to_processor_ms = 0.0
                    for item in nodes_list:
                        name = str(item.get("name") or "")
                        dt_ms = float(item.get("dt_ms", 0) or 0)
                        # 累加所有节点的时间（每个节点只计算一次，避免并行分支重复计入）
                        if name not in seen:
                            seen.add(name)
                            time_to_processor_ms += dt_ms
                        # 如果遇到 processor，停止累加（processor 是最后一个节点）
                        if name == "processor":
                            break
                
                # 2.2 冲量变化追踪（delta 按当前 (user_id, bot_id) 的上一轮动量计算，保证 delta = x(t)-x(t-1)）
                momentum = result_state.get("conversation_momentum") if isinstance(result_state, dict) else None
                momentum_f = float(momentum) if momentum is not None else None
                if momentum_f is not None:
                    key_m = (current_user_id, current_bot_id)
                    prev_momentum = prev_momentum_by_key.get(key_m)  # 仅本会话上一轮，非对方
                    momentum_delta = (momentum_f - prev_momentum) if prev_momentum is not None else None
                    prev_momentum_by_key[key_m] = momentum_f
                    momentum_history.append(momentum_f)
                else:
                    momentum_delta = None
                
                # 2.2.1 轮次计数追踪
                turn_count = result_state.get("turn_count_in_session") if isinstance(result_state, dict) else None
                turn_count_int = int(turn_count) if turn_count is not None else None
                
                # 2.2.2 六维关系状态追踪及每轮变化
                relationship_state = result_state.get("relationship_state") if isinstance(result_state, dict) else {}
                rel_current: Dict[str, Optional[float]] = {}
                for dim in DIM_KEYS:
                    v = relationship_state.get(dim) if isinstance(relationship_state, dict) else None
                    try:
                        rel_current[dim] = float(v) if v is not None else None
                    except (TypeError, ValueError):
                        rel_current[dim] = None
                prev_rel = prev_relationship_state.get((current_user_id, current_bot_id)) or {}
                rel_deltas: Dict[str, Optional[float]] = {}
                for dim in DIM_KEYS:
                    cur = rel_current.get(dim)
                    prev_val = prev_rel.get(dim)
                    if cur is not None and prev_val is not None:
                        rel_deltas[dim] = round(cur - prev_val, 4)
                    elif cur is not None:
                        rel_deltas[dim] = None  # 首轮无 delta
                    else:
                        rel_deltas[dim] = None
                # 更新上一轮状态供下一轮用
                prev_relationship_state[(current_user_id, current_bot_id)] = {
                    k: v for k, v in rel_current.items() if v is not None
                }
                # 兼容旧变量名（供下方 N/A 判断）
                rel_closeness = rel_current.get("closeness")
                rel_liking = rel_current.get("liking")
                rel_attractiveness = rel_current.get("attractiveness", rel_current.get("warmth"))
                rel_trust = rel_current.get("trust")
                
                # 2.3 基础信息任务触发追踪
                basic_info_task_ids = {"ask_user_name", "ask_user_age", "ask_user_occupation", "ask_user_location"}
                tasks_for_lats = (result_state or {}).get("tasks_for_lats") or []
                triggered_this_round = {}
                for task_list, label in [(tasks_for_lats, "tasks_for_lats")]:
                    if not isinstance(task_list, list):
                        continue
                    for t in task_list:
                        tid = str((t or {}).get("id") or "")
                        if tid in basic_info_task_ids:
                            triggered_this_round[tid] = triggered_this_round.get(tid, 0) + 1
                            basic_info_task_triggered[tid] = basic_info_task_triggered.get(tid, 0) + 1
                
                # 2.4 基础信息任务执行追踪
                completed_task_ids = set((result_state or {}).get("completed_task_ids") or [])
                attempted_task_ids = set((result_state or {}).get("attempted_task_ids") or [])
                
                executed_this_round = {}
                completed_this_round = {}
                for tid in basic_info_task_ids:
                    if tid in attempted_task_ids:
                        executed_this_round[tid] = executed_this_round.get(tid, 0) + 1
                        basic_info_task_executed[tid] = basic_info_task_executed.get(tid, 0) + 1
                    if tid in completed_task_ids:
                        completed_this_round[tid] = completed_this_round.get(tid, 0) + 1
                        basic_info_task_completed[tid] = basic_info_task_completed.get(tid, 0) + 1
                
                # 2.5 数据库写入验证（真实流程：memory_manager 抽取 + save_turn 写库）
                basic_info_after = await get_user_basic_info(db, current_user_id, current_bot_id)
                written_this_round = {}
                for key in ["name", "age", "gender", "occupation", "location"]:
                    before_val = basic_info_before.get(key)
                    after_val = basic_info_after.get(key)
                    if (before_val is None or (isinstance(before_val, str) and not before_val.strip())) and \
                       (after_val is not None and (not isinstance(after_val, str) or after_val.strip())):
                        written_this_round[key] = after_val
                        basic_info_written[key] = basic_info_written.get(key, 0) + 1
                # 2.6 本轮回写后 DB 中的 inferred_profile（memory_writer 写入）
                inferred_profile_after = await get_user_inferred_profile(db, current_user_id, current_bot_id)

                # 2.7 标点符号去除验证
                punct_check = check_punctuation_removal(final_response, segments)
                punctuation_removal_stats.append(punct_check)
                
                # ==========================================
                # 3. 输出详细指标报告
                # ==========================================
                log_line("\n" + "="*60)
                log_line(f"[ROUND {turn} METRICS]")
                log_line("="*60)
                log_line(f"1. Total time (to processor end): {time_to_processor_ms/1000:.2f}s" if time_to_processor_ms is not None else "1. Total time (to processor end): N/A")
                log_line(f"2. Turn count: {turn_count_int}" if turn_count_int is not None else "2. Turn count: N/A")
                log_line(f"3. Momentum: {momentum_f:.2f}" + (f" (delta: {momentum_delta:+.2f})" if momentum_delta is not None else "") if momentum_f is not None else "3. Momentum: N/A")
                # 4. [Evolver] 本轮回写 relationship_state（六维 + 相对上轮变化）
                if any(rel_current.get(d) is not None for d in DIM_KEYS):
                    parts = []
                    for dim in DIM_KEYS:
                        v = rel_current.get(dim)
                        d = rel_deltas.get(dim)
                        if v is not None:
                            delta_str = f" (Δ{d:+.3f})" if d is not None else ""
                            parts.append(f"{dim}={v:.3f}{delta_str}")
                    log_line("4. [Evolver] relationship_state 本轮回写: " + ", ".join(parts))
                else:
                    log_line("4. [Evolver] relationship_state: N/A")
                # 5. Basic info：触发=本轮 tasks_for_lats 中出现次数（仅参考）；完成=仅以「写入 DB」为准（attempted 已废弃）
                log_line(f"5. Basic info 触发(tasks_for_lats 出现次数，仅参考): {sum(triggered_this_round.values())} ({', '.join(f'{k}:{v}' for k, v in triggered_this_round.items())})")
                log_line(f"6. Basic info 完成(仅写入 DB 为准，state.completed_task_ids): {sum(completed_this_round.values())} ({', '.join(f'{k}:{v}' for k, v in completed_this_round.items())})")
                log_line(f"7. Basic info 写入 DB (本轮回写): {written_this_round}")
                # inferred_profile 是否被 DB 写入（memory_writer 写 user.inferred_profile）
                inf_keys = list(inferred_profile_after.keys()) if inferred_profile_after else []
                log_line(f"8. inferred_profile 写入 DB: keys={inf_keys}" + (f" (内容摘要: {str(inferred_profile_after)[:200]}...)" if len(str(inferred_profile_after)) > 200 else f" (内容: {inferred_profile_after})"))
                log_line(f"9. Punctuation removal:")
                log_line(f"   Original text: {punct_check.get('original_text', 'N/A')[:100]}{'...' if len(punct_check.get('original_text', '')) > 100 else ''}")
                log_line(f"   Processed text: {punct_check.get('processed_text', 'N/A')[:100]}{'...' if len(punct_check.get('processed_text', '')) > 100 else ''}")
                log_line(f"   Stats: original={punct_check['original_count']}, processed={punct_check['processed_count']}, removed={punct_check['removed']}, success_rate={punct_check['success_rate']:.2%}")
                log_line("="*60 + "\n")
                
                # ==========================================
                # 4. 保存到聊天记录（含 6 维关系及变化）
                # ==========================================
                conversation_log.append({
                    "round": turn,
                    "run": run,
                    "speaker": current_speaker,
                    "user_message": current_message,
                    "bot_reply": final_response,
                    "segments": segments,
                    "momentum": momentum_f,
                    "time_to_processor_ms": time_to_processor_ms,
                    "relationship_state": {k: v for k, v in rel_current.items() if v is not None},
                    "relationship_deltas": {k: v for k, v in rel_deltas.items() if v is not None},
                })

                # Optional: step-by-step profiling report (requires LTSR_PROFILE_STEPS=1 / LTSR_LLM_STATS=1)
                llm_stats = (result_state or {}).get("_llm_stats") if isinstance(result_state, dict) else None
                if isinstance(prof, dict) and isinstance(prof.get("nodes"), list):
                    log_line("\n  [PROFILE] 节点耗时与 LLM 调用增量：")
                    nodes_list = prof.get("nodes") or []
                    # 按节点名聚合，避免并行分支合并导致的重复条目刷屏（每节点只打一行）
                    agg: dict = {}
                    for item in nodes_list:
                        name = str(item.get("name") or "")
                        dt_ms = float(item.get("dt_ms", 0.0) or 0.0)
                        delta = item.get("llm_delta") if isinstance(item.get("llm_delta"), dict) else {}
                        delta_calls = sum(int(v.get("calls", 0) or 0) for v in delta.values()) if isinstance(delta, dict) else 0
                        if name not in agg:
                            agg[name] = {"dt_ms": 0.0, "calls": 0}
                        agg[name]["dt_ms"] += dt_ms
                        agg[name]["calls"] += delta_calls
                    for name, v in agg.items():
                        log_line(f"    - {name}: {v['dt_ms']:.2f}ms, llm_calls_delta={v['calls']}")
                    # 截止到生成回复：按节点首次出现顺序求和，避免并行重复计入
                    seen = set()
                    time_to_reply_ms = 0.0
                    for item in nodes_list:
                        name = str(item.get("name") or "")
                        dt_ms = float(item.get("dt_ms", 0) or 0)
                        # 累加所有节点的时间（每个节点只计算一次，避免并行分支重复计入）
                        if name not in seen:
                            seen.add(name)
                            time_to_reply_ms += dt_ms
                        # 如果遇到 processor，停止累加（processor 是最后一个节点）
                        if name == "processor":
                            break
                    log_line(f"  [PROFILE] 截止到生成回复(含 processor): {time_to_reply_ms:.0f}ms ({time_to_reply_ms/1000:.2f}s)")
                    time_to_reply_ms_list.append(time_to_reply_ms)
                if isinstance(llm_stats, dict) and llm_stats:
                    log_line("\n  [PROFILE] 本轮各模型/角色 API 调用统计：")
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
                # 确保第二轮 save_turn 带上「对方名字」等 user_basic_info，避免 graph 返回的 state 未含 memory_manager 的更新时把 DB 覆盖成空
                # 优先用 result_state 的 user_basic_info；若为空则用本轮回写后 DB 中的 basic_info（与 Loader 下一轮读取的链路一致）
                from_result = (result_state or {}).get("user_basic_info")
                if from_result and isinstance(from_result, dict) and any(str(from_result.get(k) or "").strip() for k in ("name", "age", "gender", "occupation", "location")):
                    state_after["user_basic_info"] = dict(from_result)
                else:
                    state_after["user_basic_info"] = dict(basic_info_after) if basic_info_after else {}
                log_line(f"  [save_turn] user_basic_info 来源: {'result_state' if from_result and state_after.get('user_basic_info') else 'DB_after'}, name={state_after.get('user_basic_info', {}).get('name') or '?'}")
                await db.save_turn(current_user_id, current_bot_id, state_after)

            except Exception as e:
                log_line(f"[错误] {current_speaker} 的 Bot 回复失败: {e}")
                if isinstance(e, TimeoutError):
                    aborted_reason = str(e)
                    log_line(f"[中止] 因超时中止：{aborted_reason}")
                else:
                    aborted_reason = str(e)
                break

            # 下一轮：sender = 本轮刚回复的 bot (current_bot_id)，replier = 对方；交换为 (对方 proxy user_id, 对方 bot_id)
            current_message = reply
            current_speaker = "Bot A" if current_bot_id == bot_a_id else "Bot B"
            if current_speaker == "Bot A":
                current_user_id = user_a_external_id
                current_bot_id = bot_b_id
            else:
                current_user_id = user_b_external_id
                current_bot_id = bot_a_id

        if aborted_reason:
            break
        log_line(f"\n第 {run}/{num_runs} 次会话（{rounds_per_run} 轮）完成\n")

    # ==========================================
    # 最终统计报告和完整聊天记录
    # ==========================================
    
    # 输出完整聊天记录
    log_line("\n" + "=" * 80)
    log_line("[COMPLETE CONVERSATION LOG]")
    log_line("=" * 80)
    for entry in conversation_log:
        log_line(f"\nRound {entry.get('round')} (Run {entry.get('run', 1)}):")
        log_line(f"  [{entry['speaker']}] {entry['user_message']}")
        log_line(f"  [Bot Reply] {entry['bot_reply']}")
        rs = entry.get("relationship_state") or {}
        rd = entry.get("relationship_deltas") or {}
        if rs or rd:
            dim_str = ", ".join(f"{d}={rs.get(d, 0):.3f}" + (f" (Δ{rd.get(d):+.3f})" if rd.get(d) is not None else "") for d in DIM_KEYS)
            log_line(f"  [6-dim] {dim_str}")
        if entry.get('segments'):
            log_line("  [Segments]:")
            for i, seg in enumerate(entry['segments'], 1):
                if isinstance(seg, dict):
                    content = seg.get("content", "")
                    delay = seg.get("delay", 0)
                    action = seg.get("action", "typing")
                    log_line(f"    [{i}] {content} [delay={delay}s, action={action}]")
                else:
                    log_line(f"    [{i}] {seg}")
    log_line("\n" + "=" * 80)
    
    # 输出汇总统计报告
    log_line("\n" + "=" * 80)
    log_line(f"[{rounds_per_run * num_runs} ROUND SUMMARY REPORT]")
    log_line("=" * 80)
    
    # 1. 平均每轮用时
    if turn_times:
        avg_time = sum(turn_times) / len(turn_times)
        log_line(f"\n1. 平均每轮用时（全链路）: {avg_time:.2f}秒")
    if time_to_reply_ms_list:
        avg_to_reply_ms = sum(time_to_reply_ms_list) / len(time_to_reply_ms_list)
        log_line(f"   平均每轮用时（到processor结束）: {avg_to_reply_ms/1000:.2f}秒")
    
    # 2. 轮次计数统计
    log_line(f"\n2. 轮次计数统计:")
    log_line(f"   总轮次: {num_runs * rounds_per_run} 轮")
    log_line(f"   实际完成轮次: {len(momentum_history)} 轮")
    
    # 3. 冲量变化趋势
    if momentum_history:
        max_momentum = max(momentum_history)
        min_momentum = min(momentum_history)
        avg_momentum = sum(momentum_history) / len(momentum_history)
        if len(momentum_history) > 1:
            decay_rate = (momentum_history[0] - momentum_history[-1]) / len(momentum_history)
        else:
            decay_rate = 0.0
        log_line(f"\n3. 冲量变化趋势:")
        log_line(f"   最高值: {max_momentum:.2f}")
        log_line(f"   最低值: {min_momentum:.2f}")
        log_line(f"   平均值: {avg_momentum:.2f}")
        log_line(f"   平均衰减率: {decay_rate:.4f}/轮")
        log_line(f"   冲量历史: {[f'{m:.2f}' for m in momentum_history]}")
    else:
        log_line(f"\n3. 冲量变化趋势: 无数据")
    
    # 4. 基础信息任务触发总次数（仅参考：= planner 放入 tasks_for_lats 的次数，不代表执行或完成）
    log_line(f"\n4. Basic info 触发总次数（tasks_for_lats 出现次数，仅参考）:")
    for tid in basic_info_task_ids:
        count = basic_info_task_triggered.get(tid, 0)
        log_line(f"   {tid}: {count}次")
    log_line("   说明: 完成仅以「写入 DB」为准；attempted 已废弃，不再统计。")

    # 5. 数据库写入总次数（流程 memory_manager + save_turn 实际写入，即真实「完成」）
    log_line(f"\n5. 数据库写入总次数（按字段，= 真实完成）:")
    for key in ["name", "age", "gender", "occupation", "location"]:
        count = basic_info_written.get(key, 0)
        log_line(f"   {key}: {count}次")
    
    # 6. 六维关系指标每轮监控（汇总表）
    log_line(f"\n6. 六维关系指标每轮变化（closeness, trust, liking, respect, attractiveness, power）:")
    if conversation_log:
        log_line("   Round | closeness | trust | liking | respect | attractiveness | power | deltas(cl,tr,li,re,at,po)")
        for entry in conversation_log:
            r = entry.get("round", 0)
            rs = entry.get("relationship_state") or {}
            rd = entry.get("relationship_deltas") or {}
            vals = [f"{rs.get(d, 0):.3f}" for d in DIM_KEYS]
            deltas = [f"{rd.get(d, 0):+.3f}" if rd.get(d) is not None else "-" for d in DIM_KEYS]
            log_line(f"   {r:5} | " + " | ".join(vals) + " | " + ",".join(deltas))
    else:
        log_line("   无数据")
    
    # 7. 标点符号去除成功率
    if punctuation_removal_stats:
        total_original = sum(s['original_count'] for s in punctuation_removal_stats)
        total_processed = sum(s['processed_count'] for s in punctuation_removal_stats)
        total_removed = sum(s['removed'] for s in punctuation_removal_stats)
        overall_success_rate = (total_removed / total_original) if total_original > 0 else 1.0
        log_line(f"\n8. 标点符号去除效果:")
        log_line(f"   原始标点总数: {total_original}")
        log_line(f"   处理后标点数: {total_processed}")
        log_line(f"   去除数量: {total_removed}")
        log_line(f"   总体成功率: {overall_success_rate:.2%}")
        # 显示最后一轮的去标点结果作为示例
        if punctuation_removal_stats:
            last_check = punctuation_removal_stats[-1]
            log_line(f"   最后一轮示例:")
            log_line(f"     原始文本: {last_check.get('original_text', 'N/A')[:150]}{'...' if len(last_check.get('original_text', '')) > 150 else ''}")
            log_line(f"     处理后文本: {last_check.get('processed_text', 'N/A')[:150]}{'...' if len(last_check.get('processed_text', '')) > 150 else ''}")
    
    log_line("\n" + "=" * 80)

    # LLM 用时分析（解析 [LLM_ELAPSED] 并写入日志 + 控制台）
    llm_elapsed_entries: List[Dict[str, Any]] = []
    if log_file is not None:
        log_file.flush()
        try:
            text = single_log_path.read_text(encoding="utf-8", errors="replace")
            llm_elapsed_re = re.compile(r"\[LLM_ELAPSED\]\s+node=(\S+)\s+model=([^\s]+)\s+dt_ms=([\d.]+)")
            for line in text.splitlines():
                mo = llm_elapsed_re.search(line)
                if mo:
                    llm_elapsed_entries.append({"node": mo.group(1), "model": mo.group(2), "dt_ms": float(mo.group(3))})
        except Exception as e:
            log_line(f"\n[WARN] 解析 LLM_ELAPSED 失败: {e}")
        if llm_elapsed_entries:
            total_ms = sum(e["dt_ms"] for e in llm_elapsed_entries)
            by_node: Dict[str, List[float]] = {}
            by_model: Dict[str, List[float]] = {}
            for e in llm_elapsed_entries:
                by_node.setdefault(e["node"], []).append(e["dt_ms"])
                by_model.setdefault(e["model"], []).append(e["dt_ms"])
            log_line("\n" + "=" * 80)
            log_line("8. LLM 用时分析（LTSR_LLM_ELAPSED_LOG=1 时记录）")
            log_line("=" * 80)
            log_line(f"   总 LLM 调用次数: {len(llm_elapsed_entries)}")
            log_line(f"   总耗时(ms): {total_ms:.1f}  总耗时(秒): {total_ms/1000:.2f}")
            log_line("   按节点:")
            for node in sorted(by_node.keys()):
                vals = by_node[node]
                log_line(f"     {node}: 次数={len(vals)} 总ms={sum(vals):.1f} 平均ms={sum(vals)/len(vals):.1f}")
            log_line("   按模型:")
            for model in sorted(by_model.keys()):
                vals = by_model[model]
                log_line(f"     {model}: 次数={len(vals)} 总ms={sum(vals):.1f} 平均ms={sum(vals)/len(vals):.1f}")
            log_line("=" * 80)
        else:
            log_line("\n9. LLM 用时分析: 未解析到 [LLM_ELAPSED]。运行前请设置 LTSR_LLM_ELAPSED_LOG=1")
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
        print(f"\n📊 回复耗时统计: 共 {len(turn_times)} 轮, 平均回复时间(全链路) = {avg_time:.2f} 秒")
    if time_to_reply_ms_list:
        avg_to_reply_ms = sum(time_to_reply_ms_list) / len(time_to_reply_ms_list)
        print(f"📊 截止到生成回复(含 processor): 共 {len(time_to_reply_ms_list)} 轮, 平均 = {avg_to_reply_ms:.0f}ms ({avg_to_reply_ms/1000:.2f}s)")
    if llm_elapsed_entries:
        total_ms = sum(e["dt_ms"] for e in llm_elapsed_entries)
        print(f"\n📊 LLM 用时: 总调用 {len(llm_elapsed_entries)} 次, 总耗时 {total_ms/1000:.2f}s, 详见日志「9. LLM 用时分析」")
    print("\n✅ 完成！本次运行所有内容已写入同一日志文件。")
    print("   详细指标报告和完整聊天记录请查看日志文件。")


if __name__ == "__main__":
    asyncio.run(main())

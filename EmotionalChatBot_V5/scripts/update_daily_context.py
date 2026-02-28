#!/usr/bin/env python3
"""
每日聊天素材更新脚本
====================
功能：
  1. 从 RSS 抓取今日资讯（娱乐/生活/科技），写入 daily_topics.yaml 的 topics 字段
  2. 用 LLM 基于 bot 人设生成 bot 最近的生活事件，写入 bot_recent 字段
  3. 兴趣相关话题由 LLM 结合 bot_persona 与 interests 自动推断

cron 安装（每天早 7:00 运行）：
  crontab -e
  0 7 * * * cd /path/to/LTSRChatbot/EmotionalChatBot_V5 && python scripts/update_daily_context.py --bot-id <bot_id> >> logs/daily_context.log 2>&1

手动运行：
  python scripts/update_daily_context.py --bot-id default_bot
  python scripts/update_daily_context.py --bot-id default_bot --date 2026-02-27   # 测试指定日期
  python scripts/update_daily_context.py --dry-run                                 # 只打印不写文件
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

_TZ_CST = timezone(timedelta(hours=8))  # UTC+8 中国标准时间
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 路径初始化（无论从哪里调用都能找到项目根）
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# 加载 .env（.env 可能在 _PROJECT_ROOT 或其父目录）
try:
    from utils.env_loader import load_project_env  # type: ignore
    load_project_env(_PROJECT_ROOT)
    load_project_env(_PROJECT_ROOT.parent)
except Exception:
    try:
        from dotenv import load_dotenv
        _env_path = _PROJECT_ROOT / ".env"
        if not _env_path.exists():
            _env_path = _PROJECT_ROOT.parent / ".env"
        if _env_path.exists():
            load_dotenv(str(_env_path), override=False)
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("update_daily_context")

# ---------------------------------------------------------------------------
# RSS 源列表（娱乐/生活/科技类，无需 API key）
# ---------------------------------------------------------------------------
RSS_SOURCES = [
    # 少数派（科技/效率/生活）
    "https://sspai.com/feed",
    # 36氪（科技商业）
    "https://36kr.com/feed",
    # 豆瓣（文化/电影/书籍）- 如可访问
    "https://www.douban.com/feed/review/movie",
]

# 过滤掉不适合聊天的关键词（政治/灾难/负面）
_FILTER_KEYWORDS = [
    "死亡", "暴力", "战争", "地震", "爆炸", "恐怖", "崩溃",
    "坠机", "离世", "去世", "逝世", "习近平", "政府", "制裁",
    "反腐", "事故", "火灾", "洪水",
]

# ---------------------------------------------------------------------------
# RSS 抓取
# ---------------------------------------------------------------------------

def _fetch_url(url: str, timeout: int = 8) -> Optional[str]:
    """简单 HTTP GET，返回响应文本；失败返回 None。"""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DailyContextBot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = "utf-8"
            ct = resp.headers.get("Content-Type", "")
            if "charset=" in ct:
                charset = ct.split("charset=")[-1].strip().split(";")[0].strip()
            return resp.read().decode(charset, errors="replace")
    except Exception as e:
        logger.warning("抓取失败 %s: %s", url, e)
        return None


def _parse_rss_titles(xml: str) -> List[str]:
    """从 RSS XML 中提取 <title> 标签内容（简单正则，无需 lxml）。"""
    import re
    # 跳过 channel 级别的第一个 title
    titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>|<title>(.*?)</title>", xml, re.S)
    results: List[str] = []
    for t1, t2 in titles:
        t = (t1 or t2).strip()
        if t and len(t) > 3:
            results.append(t)
    # 第一条通常是 feed 名称，去掉
    return results[1:] if len(results) > 1 else results


def _is_ok_topic(title: str) -> bool:
    """过滤掉不适合聊天的话题。"""
    for kw in _FILTER_KEYWORDS:
        if kw in title:
            return False
    return True


def fetch_news_topics(max_items: int = 6) -> List[str]:
    """从 RSS 源抓取并过滤新闻标题，返回适合聊天的条目。"""
    collected: List[str] = []
    for url in RSS_SOURCES:
        if len(collected) >= max_items:
            break
        xml = _fetch_url(url)
        if not xml:
            continue
        titles = _parse_rss_titles(xml)
        for t in titles:
            if len(collected) >= max_items:
                break
            t_clean = t[:60].strip()
            if _is_ok_topic(t_clean) and t_clean not in collected:
                collected.append(t_clean)
        logger.info("  RSS %s → %d 条标题", url, len(titles))
    logger.info("抓取完成，共 %d 条有效资讯", len(collected))
    return collected


# ---------------------------------------------------------------------------
# LLM 调用（复用项目已有的 LLM 配置）
# ---------------------------------------------------------------------------

def _get_llm():
    """复用项目 LLM 配置，返回一个可调用的 LLM 对象（fast 角色）。"""
    try:
        from app.services.llm import get_llm  # type: ignore
        return get_llm(role="fast")
    except Exception:
        pass
    try:
        from utils.llm_factory import get_llm  # type: ignore
        return get_llm(role="fast")
    except Exception:
        pass
    # 备用：直接用环境变量构建 ChatOpenAI-compatible 客户端
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("FAST_MODEL", "qwen-plus"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            temperature=0.8,
            max_tokens=800,
        )
    except Exception as e:
        logger.error("LLM 初始化失败: %s", e)
        return None


def generate_bot_recent(
    bot_persona: str,
    bot_interests: List[str],
    date_str: str,
    n: int = 5,
) -> List[str]:
    """用 LLM 生成 bot 最近几天的生活事件（第一人称，自然口语）。"""
    llm = _get_llm()
    if llm is None:
        logger.warning("LLM 不可用，使用默认生活事件")
        return [
            "最近在追一部剧，看得有点停不下来",
            "今天天气挺好的，午休出去转了一圈",
            "昨晚做饭失手了，焦糊味还没散",
        ]

    interests_text = "、".join(bot_interests[:6]) if bot_interests else "日常生活"
    prompt = f"""你要扮演以下角色，为这个角色生成"{date_str}前后几天"内发生的真实生活事件。

【角色描述】
{bot_persona[:400]}

【已知兴趣/爱好】
{interests_text}

要求：
- 生成 {n} 条生活事件，第一人称，口语化，像是角色日记片段
- 每条 20-50 字，不超过 60 字
- 内容要多样：可以是日常小事、正在做的事、情绪状态、想做但还没做的事
- 风格要符合角色性格，不要过于正能量或鸡汤
- 不要包含具体日期或"今天/昨天"等时间词（直接描述事件）
- 只输出事件列表，每行一条，不要编号或前缀

示例格式（不要照抄，只是格式参考）：
最近迷上了一款手游，睡前老是忍不住多玩一会儿
上周末和朋友聚了一次，聊了很久感觉充了电
最近一直想整理房间但一直拖着没动"""

    try:
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = (getattr(resp, "content", "") or str(resp)).strip()
        lines = [ln.strip().lstrip("•-·*0123456789. ") for ln in content.splitlines()]
        lines = [ln for ln in lines if 10 <= len(ln) <= 80]
        logger.info("LLM 生成 bot_recent %d 条", len(lines))
        return lines[:n]
    except Exception as e:
        logger.error("LLM 生成 bot_recent 失败: %s", e)
        return [
            "最近在追一部剧，看得有点停不下来",
            "今天天气挺好的，出去转了一圈",
        ]


def generate_interest_topics(
    bot_persona: str,
    bot_interests: List[str],
    n: int = 3,
) -> List[str]:
    """用 LLM 生成与 bot 兴趣相关的话题（作为 topics 补充，不依赖 RSS）。"""
    llm = _get_llm()
    if llm is None:
        return []

    interests_text = "、".join(bot_interests[:6]) if bot_interests else "日常生活"
    prompt = f"""根据以下角色和兴趣，生成 {n} 条适合在聊天中自然提起的话题。

【兴趣/爱好】{interests_text}
【角色描述】{bot_persona[:200]}

要求：
- 每条 20-50 字，口语化，像是角色说"最近发现/听说/感觉..."这种语气
- 话题要有具体内容，不要只说"最近很忙"这种空话
- 只输出话题列表，每行一条，不要编号"""

    try:
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = (getattr(resp, "content", "") or str(resp)).strip()
        lines = [ln.strip().lstrip("•-·*0123456789. ") for ln in content.splitlines()]
        lines = [ln for ln in lines if 10 <= len(ln) <= 80]
        return lines[:n]
    except Exception as e:
        logger.error("LLM 生成 interest_topics 失败: %s", e)
        return []


# ---------------------------------------------------------------------------
# Bot 信息读取
# ---------------------------------------------------------------------------

async def _load_bot_profile_from_db(bot_id: str, db_url: str) -> Optional[Dict[str, Any]]:
    """从 DB bots 表读取 bot 人设（优先按 name 匹配，再按 UUID）。"""
    try:
        import uuid as _uuid
        from sqlalchemy.ext.asyncio import async_sessionmaker
        from sqlalchemy import select, or_
        from app.core.database import _create_async_engine_from_database_url, Bot

        engine = _create_async_engine_from_database_url(db_url)
        Session = async_sessionmaker(engine, expire_on_commit=False)
        row = None
        async with Session() as s:
            # 先按 name 精确匹配
            r = await s.execute(select(Bot).where(Bot.name == bot_id))
            row = r.scalars().first()
            if row is None:
                # 再试 UUID
                try:
                    uid = _uuid.UUID(bot_id)
                    r2 = await s.execute(select(Bot).where(Bot.id == uid))
                    row = r2.scalars().first()
                except ValueError:
                    pass
        await engine.dispose()

        if row is None:
            return None

        persona_raw = row.persona or {}
        persona_parts: List[str] = []
        if isinstance(persona_raw, dict):
            for section in ("lore", "attributes"):
                sec = persona_raw.get(section) or {}
                if isinstance(sec, dict):
                    for v in sec.values():
                        persona_parts.append(str(v)[:120])
            colls = persona_raw.get("collections") or {}
            if isinstance(colls, dict):
                for v in colls.values():
                    if isinstance(v, list):
                        persona_parts.append("、".join(str(x) for x in v[:5]))
        persona_str = "；".join(persona_parts)[:600] or json.dumps(persona_raw, ensure_ascii=False)[:400]

        big_five = row.big_five or {}
        interests: List[str] = []
        if isinstance(big_five, dict):
            interests = list(big_five.get("interests") or [])
        # basic_info 补充兴趣线索
        basic = row.basic_info or {}
        if isinstance(basic, dict) and basic.get("occupation"):
            interests.insert(0, str(basic["occupation"]))

        logger.info("DB 读取 bot 人设: name=%s, persona_len=%d, interests=%s", row.name, len(persona_str), interests[:4])
        return {"persona": persona_str, "name": row.name, "interests": interests[:8]}
    except Exception as e:
        logger.warning("DB 读取 bot 人设失败(%s): %s", bot_id, e)
        return None


def load_bot_profile(bot_id: str, user_id: str = "default_user", db_url: str = "") -> Dict[str, Any]:
    """从 DB（优先）/ LocalStore / bot_config.yaml 加载 bot 人设信息。"""
    # 优先：DB（Render 或本地 PostgreSQL）
    if db_url:
        try:
            result = asyncio.run(_load_bot_profile_from_db(bot_id, db_url))
            if result:
                return result
        except Exception as e:
            logger.warning("DB 路径失败，降级 LocalStore: %s", e)

    # 次优：LocalStore
    try:
        from app.core.local_store import LocalStoreManager
        store = LocalStoreManager()
        data = store.load_state(user_id, bot_id)
        if data:
            return {
                "persona": str(data.get("bot_persona") or ""),
                "name": str((data.get("bot_basic_info") or {}).get("name") or bot_id),
                "interests": list((data.get("bot_big_five") or {}).get("interests") or []),
            }
    except Exception as e:
        logger.debug("LocalStore 读取失败: %s", e)

    # 备用：config/bot_config.yaml
    bot_config_path = _PROJECT_ROOT / "config" / "bot_config.yaml"
    if bot_config_path.exists():
        try:
            import yaml
            with open(bot_config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            bots = cfg.get("bots") or {}
            bot = bots.get(bot_id) or bots.get("default") or {}
            return {
                "persona": str(bot.get("persona") or ""),
                "name": str(bot.get("name") or bot_id),
                "interests": list(bot.get("interests") or []),
            }
        except Exception as e:
            logger.debug("bot_config.yaml 读取失败: %s", e)

    # 最终兜底
    logger.warning("无法加载 bot 人设，使用空白配置")
    return {"persona": "", "name": bot_id, "interests": []}


# ---------------------------------------------------------------------------
# YAML 写入
# ---------------------------------------------------------------------------

def write_daily_context(
    date_str: str,
    topics: List[str],
    bot_recent: List[str],
    bot_recent_by_bot_id: Optional[Dict[str, List[str]]] = None,
    dry_run: bool = False,
) -> None:
    """将结果写入 config/daily_topics.yaml。支持多 bot 的 bot_recent_by_bot_id 块。"""
    out_path = _PROJECT_ROOT / "config" / "daily_topics.yaml"

    def _esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    # 手工构建 YAML（保留注释头部）
    lines = [
        "# 每日聊天素材注入 — 由 scripts/update_daily_context.py 自动生成",
        f'# 生成时间: {datetime.now(_TZ_CST).strftime("%Y-%m-%d %H:%M CST")}',
        "",
        f'date: "{date_str}"',
        "",
        "topics:",
    ]
    for t in topics:
        lines.append(f'  - "{_esc(t)}"')
    lines.append("")
    lines.append("bot_recent:  # 单 bot 兼容（取第一个 bot 或 default）")
    for t in bot_recent:
        lines.append(f'  - "{_esc(t)}"')
    lines.append("")

    if bot_recent_by_bot_id:
        lines.append("bot_recent_by_bot_id:  # 多 bot：loader 优先按 bot_id 匹配")
        for bid, items in bot_recent_by_bot_id.items():
            lines.append(f'  "{_esc(str(bid))}":')
            for t in items:
                lines.append(f'    - "{_esc(t)}"')
        lines.append("")

    content = "\n".join(lines)

    if dry_run:
        print("=== DRY RUN: 以下内容将写入 config/daily_topics.yaml ===")
        print(content)
        return

    out_path.write_text(content, encoding="utf-8")
    logger.info("已写入 %s", out_path)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="每日聊天素材更新脚本（兼容 cron）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--bot-id", default="",
        help="单个 Bot ID（兼容旧用法；优先使用 --bot-ids）",
    )
    parser.add_argument(
        "--bot-ids", nargs="+", default=[],
        help="一次处理多个 bot id（UUID 或 name），空格分隔",
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("RENDER_DATABASE_URL", "") or os.getenv("DATABASE_URL", ""),
        help="DB 连接 URL，用于读取 bot 人设（默认读 RENDER_DATABASE_URL 或 DATABASE_URL 环境变量）",
    )
    parser.add_argument(
        "--user-id", default="default_user",
        help="User ID，仅用于 LocalStore 路径（默认: default_user）",
    )
    parser.add_argument(
        "--date",
        default=datetime.now(_TZ_CST).strftime("%Y-%m-%d"),
        help="目标日期 YYYY-MM-DD（默认: 今天 UTC+8）",
    )
    parser.add_argument(
        "--no-news", action="store_true",
        help="跳过 RSS 抓取（网络受限时使用）",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="跳过 LLM 生成（只抓新闻，不生成 bot_recent）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印结果，不写文件",
    )
    args = parser.parse_args()

    # 合并 --bot-id 和 --bot-ids，去重保序
    bot_ids: List[str] = []
    seen: set = set()
    for bid in (args.bot_ids or []) + ([args.bot_id] if args.bot_id else []):
        if bid and bid not in seen:
            bot_ids.append(bid)
            seen.add(bid)
    if not bot_ids:
        bot_ids = ["default_bot"]

    logger.info("===== update_daily_context 开始 =====")
    logger.info("bot_ids=%s  date=%s  db_url_set=%s  no_news=%s  no_llm=%s",
                bot_ids, args.date, bool(args.db_url), args.no_news, args.no_llm)

    # 1. 抓取共享资讯（只抓一次）
    topics: List[str] = []
    if not args.no_news:
        logger.info("-- 抓取 RSS 资讯 --")
        topics = fetch_news_topics(max_items=5)
    else:
        logger.info("-- 跳过 RSS 抓取 --")

    # 2. 逐 bot 生成 bot_recent
    bot_recent_by_bot_id: Dict[str, List[str]] = {}
    first_bot_recent: List[str] = []

    for bot_id in bot_ids:
        logger.info("-- 处理 bot: %s --", bot_id)
        profile = load_bot_profile(bot_id, args.user_id, db_url=args.db_url)
        logger.info("  name=%s  persona_len=%d  interests=%s",
                    profile["name"], len(profile["persona"]), profile["interests"][:4])

        # 补充兴趣话题（若资讯不足）
        if not args.no_llm and len(topics) < 5:
            extra = generate_interest_topics(
                profile["persona"], profile["interests"], n=5 - len(topics)
            )
            topics.extend(extra)

        # 生成该 bot 的生活事件
        if not args.no_llm:
            bot_recent = generate_bot_recent(
                profile["persona"], profile["interests"], args.date, n=5
            )
        else:
            bot_recent = []

        bot_recent_by_bot_id[bot_id] = bot_recent[:6]
        if not first_bot_recent:
            first_bot_recent = bot_recent[:6]

        logger.info("  bot_recent %d 条: %s", len(bot_recent), bot_recent[:2])

    # 3. 写入 YAML
    logger.info("-- 写入 daily_topics.yaml --")
    write_daily_context(
        args.date,
        topics[:6],
        first_bot_recent,
        bot_recent_by_bot_id=bot_recent_by_bot_id if len(bot_ids) > 1 or bot_ids[0] != "default_bot" else None,
        dry_run=args.dry_run,
    )

    logger.info("===== update_daily_context 完成 =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())

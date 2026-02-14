"""
create_new_database.py

创建新的 PostgreSQL 数据库、执行 init_schema.sql、并写入一条示例 bot+user+消息。

前置：
- 本地已安装并启动 PostgreSQL
- .env 中已配置 DATABASE_URL（用于连接 postgres 库以执行 CREATE DATABASE）

用法：
  cd EmotionalChatBot_V5
  # 创建新库并播种（默认库名 ltsrchatbot_v5，可通过环境变量覆盖）
  python devtools/create_new_database.py

  # 指定新库名
  NEW_DB_NAME=my_chatbot python devtools/create_new_database.py

执行后会将 .env 中的 DATABASE_URL 更新为指向新库（可选，见下方 --update-env）。
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass


def _parse_db_url(url: str) -> tuple[str, str, str, int, str]:
    """解析 postgresql+asyncpg://user:pass@host:port/dbname 返回 (user, password, host, port, dbname)。"""
    # 兼容 postgresql+asyncpg:// 或 postgresql://
    m = re.match(
        r"^(?:postgresql(?:\+asyncpg)?)?://([^:]+):([^@]+)@([^:/]+):?(\d+)?/([^/?]+)",
        url.strip(),
    )
    if m:
        user, password, host, port, dbname = m.groups()
        return user, password, host, int(port or "5432"), dbname
    m = re.match(r"^(?:postgresql(?:\+asyncpg)?)?://([^@]+)@([^:/]+):?(\d+)?/([^/?]+)", url.strip())
    if m:
        user, host, port, dbname = m.groups()
        return user, "", host, int(port or "5432"), dbname
    raise ValueError("无法解析 DATABASE_URL，需要形如 postgresql+asyncpg://user:pass@host:port/dbname")


def _build_url(user: str, password: str, host: str, port: int, dbname: str) -> str:
    if password:
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{dbname}"
    return f"postgresql+asyncpg://{user}@{host}:{port}/{dbname}"


async def _create_database_if_not_exists(new_db_name: str) -> str:
    """连接当前 DATABASE_URL 的「系统库」执行 CREATE DATABASE；返回指向新库的 URL。"""
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL 未设置，请在 .env 中配置（可先指向 postgres 或任意已存在库）")
    user, password, host, port, _ = _parse_db_url(url)
    # 连接 postgres 或 template1 以执行 CREATE DATABASE
    sys_url = _build_url(user, password, host, port, "postgres")
    try:
        import asyncpg
    except ImportError:
        raise RuntimeError("需要 asyncpg：pip install asyncpg")

    conn = await asyncpg.connect(sys_url.replace("postgresql+asyncpg://", "postgresql://"))
    try:
        r = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            new_db_name,
        )
        if r is None:
            await conn.execute(f'CREATE DATABASE "{new_db_name}"')
            print(f"已创建数据库: {new_db_name}")
        else:
            print(f"数据库已存在，将复用: {new_db_name}")
    finally:
        await conn.close()

    return _build_url(user, password, host, port, new_db_name)


def _split_sql_statements(sql: str) -> list[str]:
    parts = []
    for chunk in sql.split(";"):
        stmt = chunk.strip()
        if stmt:
            parts.append(stmt)
    return parts


async def _ensure_schema(db) -> None:
    from sqlalchemy import text
    schema_path = PROJECT_ROOT / "init_schema.sql"
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


async def main() -> None:
    update_env = "--update-env" in sys.argv or "-u" in sys.argv
    new_db_name = os.getenv("NEW_DB_NAME", "ltsrchatbot_v5")

    if not os.getenv("DATABASE_URL"):
        print("DATABASE_URL 未设置。请先在 .env 中配置（可指向 postgres 或现有库）。")
        print("示例: DATABASE_URL=postgresql+asyncpg://postgres:密码@localhost:5432/postgres")
        sys.exit(1)

    print("== 创建新库并初始化 ==")
    print("新库名:", new_db_name)

    new_url = await _create_database_if_not_exists(new_db_name)

    # 使用新 URL 初始化 schema 并播种
    old_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = new_url

    try:
        from app.core.database import Bot, DBManager, Memory, Message, User
        from sqlalchemy import select

        db = DBManager.from_env()
        await _ensure_schema(db)
        print("已执行 init_schema.sql")

        bot_id = str(uuid.uuid4())
        user_external_id = f"local_user_{uuid.uuid4().hex[:8]}"
        _ = await db.load_state(user_external_id, bot_id)
        now = datetime.now().replace(microsecond=0).isoformat()
        state = {
            "user_id": user_external_id,
            "bot_id": bot_id,
            "current_time": now,
            "user_input": "你好，这是新库的首次对话。",
            "final_response": "你好！新库已就绪。",
            "detection_category": "NORMAL",
            "humanized_output": {"total_latency_seconds": 0.5},
            "current_stage": "initiating",
            "relationship_state": {"closeness": 0, "trust": 0, "liking": 0, "respect": 0, "warmth": 0, "power": 50},
            "mood_state": {"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0},
            "new_memory_content": "新库创建并完成首次对话。",
        }
        await db.save_turn(user_external_id, bot_id, state)
        print("已写入示例 bot + user + 消息")
        print("bot_id:", bot_id)
        print("user_external_id:", user_external_id)
    finally:
        if old_url is not None:
            os.environ["DATABASE_URL"] = old_url

    if update_env:
        env_path = PROJECT_ROOT / ".env"
        lines = []
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("DATABASE_URL="):
                    lines.append(f"DATABASE_URL={new_url}")
                    continue
                lines.append(line)
        if not any(l.strip().startswith("DATABASE_URL=") for l in lines):
            lines.append(f"DATABASE_URL={new_url}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("已更新 .env 中的 DATABASE_URL 为新库")
    else:
        print("\n当前 DATABASE_URL 未修改。若要让项目默认使用新库，请：")
        print("  1) 在 .env 中设置: DATABASE_URL=" + new_url[:50] + "...")
        print("  2) 或重新运行: python devtools/create_new_database.py --update-env")

    print("\n✅ 新库创建并播种完成。")


if __name__ == "__main__":
    asyncio.run(main())

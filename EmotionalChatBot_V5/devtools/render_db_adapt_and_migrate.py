"""
Render 数据库适配 + Knapp stage 重置 + liking→attractiveness 迁移。

仅做增量更新，不删表、不 TRUNCATE、不整体覆盖 dimensions/assets。
- Schema：执行 init_schema.sql（忽略 already exists）、sidewrite/backlog、mood_state/urgent_tasks
- 数据：把 dimensions 中的 liking 赋给 attractiveness；可选 warmth→attractiveness 补缺
- Stage：仅 UPDATE users SET current_stage = 'initiating'

使用：
  RENDER_DATABASE_URL=postgresql+asyncpg://... python -m devtools.render_db_adapt_and_migrate
  python -m devtools.render_db_adapt_and_migrate --dry-run  # 仅打印步骤，不执行
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
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
from app.core.database import _create_async_engine_from_database_url


def _split_sql_statements(sql: str) -> list[str]:
    """按分号拆分 SQL，去掉空段。"""
    parts = []
    for chunk in sql.split(";"):
        stmt = chunk.strip()
        if stmt:
            parts.append(stmt)
    return parts


# 与 run_migrate_bots_users_urgent_mood 一致
ALTER_MOOD_URGENT = """
ALTER TABLE bots ADD COLUMN IF NOT EXISTS mood_state JSONB DEFAULT '{"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0}'::jsonb;
ALTER TABLE bots ADD COLUMN IF NOT EXISTS urgent_tasks JSONB DEFAULT '[]'::jsonb;
ALTER TABLE users ADD COLUMN IF NOT EXISTS urgent_tasks JSONB DEFAULT '[]'::jsonb;
"""


async def _run_init_schema(conn) -> None:
    schema_path = PROJECT_ROOT / "init_schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    statements = _split_sql_statements(sql)
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


async def _run_sidewrite_backlog(conn) -> None:
    migration_path = Path(__file__).resolve().parent / "migrate_add_bot_sidewrite_backlog.sql"
    if not migration_path.exists():
        return
    raw = migration_path.read_text(encoding="utf-8")
    statements = []
    for s in raw.split(";"):
        stmt = s.strip()
        if not stmt:
            continue
        lines = [line for line in stmt.splitlines() if line.strip() and not line.strip().startswith("--")]
        stmt = " ".join(lines).strip()
        if stmt:
            statements.append(stmt)
    ac = await conn.execution_options(isolation_level="AUTOCOMMIT")
    for stmt in statements:
        await ac.execute(text(stmt + ";"))


async def _run_alter_mood_urgent(conn) -> None:
    ac = await conn.execution_options(isolation_level="AUTOCOMMIT")
    for line in ALTER_MOOD_URGENT.strip().split(";"):
        line = line.strip()
        if not line:
            continue
        await ac.execute(text(line))


async def _print_user_counts(conn) -> None:
    r = await conn.execute(text("""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE dimensions IS NOT NULL AND dimensions ? 'liking') AS with_liking,
            COUNT(*) FILTER (WHERE dimensions IS NOT NULL AND dimensions ? 'attractiveness') AS with_attractiveness,
            COUNT(*) FILTER (WHERE dimensions IS NOT NULL AND dimensions ? 'warmth') AS with_warmth
        FROM users
    """))
    row = r.one()
    print(f"[users] total={row.total}, with_liking={row.with_liking}, with_attractiveness={row.with_attractiveness}, with_warmth={row.with_warmth}")


async def _migrate_liking_to_attractiveness(conn) -> None:
    # 把 dimensions.liking 的值赋给 dimensions.attractiveness（存在 liking 的行）
    await conn.execute(text("""
        UPDATE users
        SET dimensions = jsonb_set(COALESCE(dimensions, '{}'::jsonb), '{attractiveness}', dimensions->'liking')
        WHERE dimensions IS NOT NULL AND dimensions ? 'liking'
    """))


async def _migrate_warmth_to_attractiveness_if_missing(conn) -> None:
    # 仅当 attractiveness 仍缺失时，用 warmth 填充（老数据兼容）
    await conn.execute(text("""
        UPDATE users
        SET dimensions = jsonb_set(COALESCE(dimensions, '{}'::jsonb), '{attractiveness}', dimensions->'warmth')
        WHERE dimensions IS NOT NULL AND dimensions ? 'warmth'
          AND (NOT (dimensions ? 'attractiveness') OR dimensions->'attractiveness' IS NULL)
    """))


async def _reset_current_stage(conn) -> None:
    await conn.execute(text("UPDATE users SET current_stage = 'initiating'"))


async def run(dry_run: bool) -> int:
    url = os.getenv("RENDER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: 请设置 RENDER_DATABASE_URL 或 DATABASE_URL")
        return 1

    steps = [
        "1. 执行 init_schema.sql（忽略 already exists）",
        "2. 执行 migrate_add_bot_sidewrite_backlog.sql",
        "3. 执行 bots/users mood_state、urgent_tasks 补齐",
        "4. 打印 users 计数（total, with_liking, with_attractiveness, with_warmth）",
        "5. UPDATE dimensions: liking → attractiveness",
        "6. UPDATE dimensions: warmth → attractiveness（仅当 attractiveness 仍缺失）",
        "7. UPDATE users SET current_stage = 'initiating'",
    ]
    if dry_run:
        print("--dry-run: 将执行以下步骤（不实际连接与写库）：")
        for s in steps:
            print(" ", s)
        return 0

    engine = _create_async_engine_from_database_url(url)
    try:
        async with engine.connect() as conn:
            print("执行 init_schema...")
            await _run_init_schema(conn)
            await conn.rollback()
            print("执行 sidewrite/backlog 迁移...")
            await _run_sidewrite_backlog(conn)
            await conn.rollback()
            print("执行 mood_state/urgent_tasks 迁移...")
            await _run_alter_mood_urgent(conn)
            await conn.rollback()

            print("迁移前 users 计数:")
            await _print_user_counts(conn)
            await conn.rollback()

            async with conn.begin():
                print("liking → attractiveness...")
                await _migrate_liking_to_attractiveness(conn)
                print("warmth → attractiveness（补缺）...")
                await _migrate_warmth_to_attractiveness_if_missing(conn)
                print("current_stage → initiating...")
                await _reset_current_stage(conn)

            print("迁移后 users 计数:")
            await _print_user_counts(conn)
        print("Render 数据库适配与 liking→attractiveness 迁移已完成。")
    except Exception as e:
        print(f"执行失败: {e}")
        return 1
    finally:
        await engine.dispose()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Render 数据库适配 + stage 重置 + liking→attractiveness 迁移")
    p.add_argument("--dry-run", action="store_true", help="仅打印步骤，不连接库、不执行")
    args = p.parse_args()
    return asyncio.run(run(args.dry_run))


if __name__ == "__main__":
    sys.exit(main())

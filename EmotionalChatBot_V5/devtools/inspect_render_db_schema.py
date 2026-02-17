"""
检查 Render（或 DATABASE_URL）上的数据库结构是否与 init_schema.sql + 迁移一致。

使用：
  RENDER_DATABASE_URL=postgresql+asyncpg://... python -m devtools.inspect_render_db_schema
  或依赖 .env 中的 RENDER_DATABASE_URL / DATABASE_URL
"""
from __future__ import annotations

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


# 期望的表结构（init_schema.sql + migrate_add_bot_sidewrite_backlog.sql）
EXPECTED_TABLES = {
    "bots": [
        "id", "name", "basic_info", "big_five", "persona", "character_sidewrite", "backlog_tasks",
        "mood_state", "urgent_tasks", "created_at",
    ],
    "users": [
        "id", "bot_id", "bot_name", "external_id", "basic_info", "current_stage", "dimensions",
        "inferred_profile", "assets", "spt_info", "conversation_summary", "urgent_tasks",
        "created_at", "updated_at",
    ],
    "messages": ["id", "user_id", "role", "content", "metadata", "created_at"],
    "memories": ["id", "user_id", "content", "created_at"],
    "transcripts": [
        "id", "user_id", "session_id", "thread_id", "turn_index", "user_text", "bot_text",
        "entities", "topic", "importance", "short_context", "created_at",
    ],
    "derived_notes": [
        "id", "user_id", "transcript_id", "note_type", "content", "importance",
        "source_pointer", "created_at",
    ],
    "bot_tasks": [
        "id", "user_id", "bot_id", "task_type", "description", "importance",
        "created_at", "expires_at", "last_attempt_at", "attempt_count",
    ],
    "web_chat_logs": [
        "id", "user_id", "bot_id", "session_id", "filename", "content",
        "created_at", "updated_at",
    ],
}

EXPECTED_ENUMS = {"knapp_stage"}

# 遗留列：旧库可能有，不影响运行，不视为差异
LEGACY_EXTRA_COLUMNS: dict[str, set[str]] = {"users": {"mood_state"}}


async def main() -> int:
    url = os.getenv("RENDER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: 请设置 RENDER_DATABASE_URL 或 DATABASE_URL")
        return 1

    source = "RENDER_DATABASE_URL" if os.getenv("RENDER_DATABASE_URL") else "DATABASE_URL"
    print(f"连接: {source}")
    print()

    engine = _create_async_engine_from_database_url(url)
    try:
        async with engine.connect() as conn:
            # 1) 自定义类型（枚举）
            r = await conn.execute(text("""
                SELECT t.typname
                FROM pg_type t
                JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace
                WHERE n.nspname = 'public' AND t.typtype = 'e'
                ORDER BY t.typname
            """))
            db_enums = {row[0] for row in r.fetchall()}
            print("--- 枚举类型 (public) ---")
            for e in sorted(db_enums):
                print(f"  {e}")
            for e in EXPECTED_ENUMS:
                if e not in db_enums:
                    print(f"  缺失: {e}")
            print()

            # 2) 表与列
            r = await conn.execute(text("""
                SELECT table_name, column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_catalog = current_database()
                ORDER BY table_name, ordinal_position
            """))
            rows = r.fetchall()
            by_table: dict[str, list[tuple[str, str, str, str]]] = {}
            for tname, cname, dtype, nullable, default in rows:
                by_table.setdefault(tname, []).append((cname, dtype, nullable, default or ""))

            print("--- 表与列 ---")
            all_ok = True
            for exp_table, exp_cols in sorted(EXPECTED_TABLES.items()):
                if exp_table not in by_table:
                    print(f"表缺失: {exp_table}")
                    all_ok = False
                    continue
                db_cols = [c[0] for c in by_table[exp_table]]
                missing = [c for c in exp_cols if c not in db_cols]
                legacy = LEGACY_EXTRA_COLUMNS.get(exp_table, set())
                extra = [c for c in db_cols if c not in exp_cols and c not in legacy]
                extra_legacy = [c for c in db_cols if c in legacy]
                if missing:
                    print(f"  {exp_table}: 缺少列 {missing}")
                    all_ok = False
                if extra:
                    print(f"  {exp_table}: 多出列 {extra}")
                if extra_legacy:
                    print(f"  {exp_table}: 遗留列（可忽略） {extra_legacy}")
                if not missing and not extra:
                    status = "OK" if not extra_legacy else "OK（含遗留列）"
                    print(f"  {exp_table}: {status} ({len(db_cols)} 列)")

            # 列出库里多出来的表（非预期）
            extra_tables = set(by_table.keys()) - set(EXPECTED_TABLES.keys())
            if extra_tables:
                print(f"  库中多出的表: {sorted(extra_tables)}")
            print()

            # 3) 索引（简要）
            r = await conn.execute(text("""
                SELECT tablename, indexname
                FROM pg_indexes
                WHERE schemaname = 'public'
                ORDER BY tablename, indexname
            """))
            indexes = r.fetchall()
            print("--- 索引 (public) ---")
            for tname, iname in indexes:
                print(f"  {tname}: {iname}")
            print()

            if all_ok and EXPECTED_ENUMS <= db_enums:
                print("结论: Render 数据库结构与 init_schema + 迁移 一致。")
            else:
                print("结论: 存在差异，请对照上方缺失项执行 init_schema.sql 或迁移。")
    finally:
        await engine.dispose()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

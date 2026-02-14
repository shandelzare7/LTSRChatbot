"""
ensure_schema.py

Render/production friendly bootstrap:
- Ensure DB schema exists by executing `init_schema.sql` (best-effort, idempotent-ish).
- Ensure at least one Bot exists so `/api/bots` is not empty on a fresh database.

Usage (from repo root or EmotionalChatBot_V5/):
  python3 EmotionalChatBot_V5/devtools/ensure_schema.py
  # or
  cd EmotionalChatBot_V5 && python3 devtools/ensure_schema.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

from sqlalchemy import select, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env (no override)
try:
    from utils.env_loader import load_project_env

    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from app.core.database import Bot, DBManager  # noqa: E402


def _split_sql_statements(sql: str) -> list[str]:
    parts: list[str] = []
    for chunk in sql.split(";"):
        stmt = chunk.strip()
        if stmt:
            parts.append(stmt)
    return parts


async def _ensure_schema(db: DBManager) -> None:
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
                # best-effort ignore for repeated runs
                if "already exists" in msg or "duplicate" in msg:
                    continue
                # CREATE EXTENSION may require superuser; allow continuing
                if "create extension" in stmt.lower():
                    continue
                raise


async def _ensure_default_bot(db: DBManager) -> None:
    """
    Fresh Render Postgres starts empty; without at least one Bot, the web UI can't start a session.
    """
    async with db.Session() as session:
        async with session.begin():
            existing = (await session.execute(select(Bot).limit(1))).scalars().first()
            if existing:
                return

            # Create a deterministic-ish bot with a fixed UUID so repeated deploys won't fan out too many bots.
            bot_uuid = uuid.UUID("9630ebfa-ada2-4013-8ce5-40c3729322ad")
            # Reuse DBManager helper: treat UUID string as bot_id to trigger profile generation.
            _ = await db._get_or_create_bot(session, str(bot_uuid))


async def main() -> None:
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL 未设置。请在部署环境设置 Render Postgres 的连接串。")

    db = DBManager.from_env()

    # Ensure schema first
    await _ensure_schema(db)

    # Ensure at least one bot exists
    await _ensure_default_bot(db)

    print("[ensure_schema] ok")


if __name__ == "__main__":
    asyncio.run(main())


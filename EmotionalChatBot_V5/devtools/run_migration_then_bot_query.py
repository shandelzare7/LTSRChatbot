"""临时脚本：执行迁移后立刻用 ORM 查 Bot，验证是否同一库。"""
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from sqlalchemy import text
from sqlalchemy import select
from app.core.database import Bot, DBManager


async def main():
    if not os.getenv("DATABASE_URL"):
        print("DATABASE_URL 未设置")
        return
    db = DBManager.from_env()
    migration_path = Path(__file__).resolve().parent / "migrate_add_bot_sidewrite_backlog.sql"
    print("Migration file exists:", migration_path.exists())
    if migration_path.exists():
        sql = migration_path.read_text(encoding="utf-8")
        statements = []
        for s in sql.split(";"):
            stmt = s.strip()
            if not stmt:
                continue
            lines = [line for line in stmt.splitlines() if line.strip() and not line.strip().startswith("--")]
            stmt = " ".join(lines).strip()
            if stmt:
                statements.append(stmt)
        print("Statements to run:", len(statements))
        async with db.engine.connect() as conn:
            ac = await conn.execution_options(isolation_level="AUTOCOMMIT")
            for stmt in statements:
                full = stmt + ";"
                print("Executing:", full[:80], "...")
                await ac.execute(text(full))
        print("Migration executed.")
    async with db.Session() as session:
        r = await session.execute(select(Bot).limit(1))
        bot = r.scalars().first()
        print("Bot query OK. First bot:", bot.name if bot else None)


if __name__ == "__main__":
    asyncio.run(main())

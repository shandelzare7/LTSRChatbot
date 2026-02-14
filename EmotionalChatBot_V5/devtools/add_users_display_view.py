"""
在已有数据库中创建视图 users_with_bot_names，便于查看每个 user 属于哪个 bot（显示 bot 名、user 名而非 uuid）。
执行一次即可，新库由 init_schema.sql 已包含该视图。

  python devtools/add_users_display_view.py
"""

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
from app.core.database import DBManager

VIEW_SQL = """
CREATE OR REPLACE VIEW users_with_bot_names AS
SELECT
  u.id AS user_id,
  u.bot_id,
  b.name AS bot_name,
  u.external_id,
  COALESCE(u.basic_info->>'name', u.basic_info->>'nickname', u.external_id) AS user_name
FROM users u
JOIN bots b ON b.id = u.bot_id;
"""


async def main() -> None:
    if not os.getenv("DATABASE_URL"):
        print("DATABASE_URL 未设置。")
        sys.exit(1)
    db = DBManager.from_env()
    async with db.engine.connect() as conn:
        await conn.execute(text(VIEW_SQL))
        await conn.commit()
    print("已创建视图 users_with_bot_names。在数据库里打开该视图即可看到 bot_name、user_name 与对应 id。")


if __name__ == "__main__":
    asyncio.run(main())

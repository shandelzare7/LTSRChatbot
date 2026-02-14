"""
fix_user_dimensions.py

修复数据库中 users.dimensions 的量纲与缺失字段问题：
- 统一到 0-1 范围（兼容旧 points=0-100 写入）
- 补齐缺失 key（closeness/trust/liking/respect/warmth/power）
- 钳位到 [0,1]

用法：
  python tools/fix_user_dimensions.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env

    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from sqlalchemy import select

from app.core.database import DBManager, User


def _norm01(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return 0.0
    if x > 1.0:
        if x <= 100.0:
            x = x / 100.0
        else:
            x = 1.0
    return float(max(0.0, min(1.0, x)))


async def fix_user_dimensions() -> int:
    if not os.getenv("DATABASE_URL"):
        print("[fix_user_dimensions] ❌ 未设置 DATABASE_URL，跳过。")
        return 0

    db = DBManager.from_env()
    updated = 0
    async with db.Session() as session:
        async with session.begin():
            users = (await session.execute(select(User))).scalars().all()
            for u in users:
                dims_in: Dict[str, Any] = dict(u.dimensions or {})
                dims_out: Dict[str, float] = {}
                for k, default in (
                    ("closeness", 0.3),
                    ("trust", 0.3),
                    ("liking", 0.3),
                    ("respect", 0.3),
                    ("warmth", 0.3),
                    ("power", 0.5),
                ):
                    dims_out[k] = round(_norm01(dims_in.get(k, default)), 4)

                if dims_out != dims_in:
                    u.dimensions = dims_out
                    updated += 1

    print(f"[fix_user_dimensions] ✅ 更新 users.dimensions: {updated} 条")
    return updated


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Detected running event loop; please run in a normal terminal.")


if __name__ == "__main__":
    raise SystemExit(_run_async(fix_user_dimensions()))


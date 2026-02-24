"""
查询 Render（或 DATABASE_URL）中所有 user 的 respect 值列表及平均值。

respect 取自 users.dimensions->>'respect'（六维关系之一，库内多为 0~1 浮点；若为 0~100 会按 0~1 参与平均）。

用法:
  RENDER_DATABASE_URL=postgresql+asyncpg://... python -m devtools.query_all_users_respect
  或: DATABASE_URL=postgresql+asyncpg://... python -m devtools.query_all_users_respect
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from sqlalchemy import select
from app.core.database import (
    DBManager,
    User,
    Bot,
    _create_async_engine_from_database_url,
)


def _normalize_respect(raw: Any) -> float | None:
    """将 dimensions 中的 respect 转为 0~1；若缺失或非法返回 None。"""
    if raw is None:
        return None
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return None
    if x < 0:
        return None
    if x > 1.0 and x <= 100.0:
        x = x / 100.0
    elif x > 100.0:
        x = 1.0
    return round(min(1.0, max(0.0, x)), 4)


async def main() -> None:
    url = os.getenv("RENDER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: 请设置 RENDER_DATABASE_URL 或 DATABASE_URL")
        sys.exit(1)

    source = "RENDER_DATABASE_URL" if os.getenv("RENDER_DATABASE_URL") else "DATABASE_URL"
    print(f"数据来源: {source}\n")

    engine = _create_async_engine_from_database_url(url)
    db = DBManager(engine)

    async with db.Session() as session:
        result = await session.execute(
            select(User, Bot.name.label("bot_name"))
            .join(Bot, Bot.id == User.bot_id)
            .order_by(Bot.name, User.external_id)
        )
        rows = result.all()

    if not rows:
        print("库中没有任何 user。")
        await engine.dispose()
        return

    # 每行: (bot_name, user_display, external_id, respect_0_1)
    out: list[tuple[str, str, str, float]] = []
    for user, bot_name in rows:
        dims = user.dimensions if isinstance(user.dimensions, dict) else {}
        raw = dims.get("respect")
        r = _normalize_respect(raw)
        if r is None:
            r = float("nan")  # 缺失或非法，后面平均时排除
        basic = user.basic_info if isinstance(user.basic_info, dict) else {}
        name_from_basic = (basic.get("name") or "").strip()
        user_display = name_from_basic or (user.external_id or str(user.id))
        bot_str = (bot_name or "").strip() or "?"
        out.append((bot_str, user_display, user.external_id or "", r))

    # 打印列表（按 bot、user 排序已由 SQL 保证）
    print(f"共 {len(out)} 个 user，respect 列表（0~1）：\n")
    print("=" * 80)
    valid: list[float] = []
    for i, (bot_name, user_display, external_id, r) in enumerate(out, 1):
        r_str = f"{r:.4f}" if not (r != r) else "(缺失)"
        if r == r:
            valid.append(r)
        print(f"{i:4}. respect = {r_str:>8}  | bot = {bot_name!r}  user = {user_display!r}  external_id = {external_id}")
    print("=" * 80)

    if valid:
        avg = sum(valid) / len(valid)
        print(f"\n有效 respect 数量: {len(valid)}")
        print(f"respect 平均值（0~1）: {avg:.4f}")
    else:
        print("\n没有有效的 respect 值，无法计算平均值。")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

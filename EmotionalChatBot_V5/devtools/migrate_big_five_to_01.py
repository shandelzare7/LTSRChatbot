"""
数据库迁移脚本：将大五人格从 [-1.0, 1.0] 范围迁移到 [0.0, 1.0] 范围
将负值转换为绝对值（例如 -0.9 -> 0.9）

使用方法：
  RENDER_DATABASE_URL=postgresql+asyncpg://... python -m devtools.migrate_big_five_to_01
  或: DATABASE_URL=postgresql+asyncpg://... python -m devtools.migrate_big_five_to_01
"""
from __future__ import annotations

import asyncio
import json
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

from sqlalchemy import select, update
from app.core.database import (
    DBManager,
    Bot,
    _create_async_engine_from_database_url,
)


def normalize_big_five_value(val: any) -> float:
    """将大五人格值从 [-1.0, 1.0] 转换为 [0.0, 1.0]，负值转为绝对值"""
    try:
        v = float(val)
        # 如果是负数，转换为绝对值
        if v < 0.0:
            v = abs(v)
        # 确保在 [0.0, 1.0] 范围内
        return max(0.0, min(1.0, v))
    except (ValueError, TypeError):
        return 0.5  # 默认值


def migrate_big_five_dict(big_five: dict) -> dict:
    """迁移大五人格字典，将所有负值转换为绝对值"""
    if not isinstance(big_five, dict):
        return {}
    
    keys = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    result = {}
    changed = False
    
    for key in keys:
        if key in big_five:
            old_val = big_five[key]
            new_val = normalize_big_five_value(old_val)
            result[key] = new_val
            if old_val != new_val:
                changed = True
        else:
            result[key] = 0.5  # 默认值
    
    return result if changed else big_five


async def main() -> None:
    url = os.getenv("RENDER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: 请设置 RENDER_DATABASE_URL 或 DATABASE_URL")
        sys.exit(1)

    source = "RENDER_DATABASE_URL" if os.getenv("RENDER_DATABASE_URL") else "DATABASE_URL"
    print(f"数据来源: {source}")
    print("正在迁移大五人格数据：将 [-1.0, 1.0] 范围转换为 [0.0, 1.0]，负值转为绝对值")
    print()

    engine = _create_async_engine_from_database_url(url)
    db = DBManager(engine)

    async with db.Session() as session:
        result = await session.execute(select(Bot))
        bots = list(result.scalars().all())

    if not bots:
        print("未找到任何 bot")
        await engine.dispose()
        return

    print(f"共找到 {len(bots)} 个 bot，开始迁移...")
    print()

    migrated_count = 0
    for bot in bots:
        big_five = bot.big_five if isinstance(bot.big_five, dict) else {}
        new_big_five = migrate_big_five_dict(big_five)
        
        if new_big_five != big_five:
            async with db.Session() as session:
                await session.execute(
                    update(Bot)
                    .where(Bot.id == bot.id)
                    .values(big_five=new_big_five)
                )
                await session.commit()
            
            migrated_count += 1
            print(f"  ✓ {bot.name} (ID: {bot.id})")
            print(f"    旧值: {json.dumps(big_five, ensure_ascii=False)}")
            print(f"    新值: {json.dumps(new_big_five, ensure_ascii=False)}")
        else:
            print(f"  - {bot.name} (ID: {bot.id}) - 无需迁移（已在 [0.0, 1.0] 范围）")

    print()
    print(f"✅ 迁移完成！共迁移 {migrated_count} 个 bot")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

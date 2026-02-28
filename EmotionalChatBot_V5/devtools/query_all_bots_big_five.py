"""
查询 Render（或 DATABASE_URL）上所有 bot 的大五人格 + PADB (mood_state)。

用法:
  RENDER_DATABASE_URL=postgresql+asyncpg://... python -m devtools.query_all_bots_big_five
  或: DATABASE_URL=postgresql+asyncpg://... python -m devtools.query_all_bots_big_five
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

from sqlalchemy import select
from app.core import (
    DBManager,
    Bot,
    _create_async_engine_from_database_url,
)


async def main() -> None:
    url = os.getenv("RENDER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: 请设置 RENDER_DATABASE_URL 或 DATABASE_URL")
        sys.exit(1)

    source = "RENDER_DATABASE_URL" if os.getenv("RENDER_DATABASE_URL") else "DATABASE_URL"
    print(f"数据来源: {source}")
    print()

    engine = _create_async_engine_from_database_url(url)
    db = DBManager(engine)

    async with db.Session() as session:
        result = await session.execute(select(Bot).order_by(Bot.name))
        bots = list(result.scalars().all())

    if not bots:
        print("未找到任何 bot")
        await engine.dispose()
        return

    print(f"共找到 {len(bots)} 个 bot：\n")
    print("=" * 100)
    
    for i, bot in enumerate(bots, 1):
        print(f"\n[{i}] Bot ID: {bot.id}")
        print(f"    名称: {bot.name}")
        
        big_five = bot.big_five if isinstance(bot.big_five, dict) else {}
        
        print(f"    大五人格:")
        print(f"      - Openness (开放性):        {big_five.get('openness', 'N/A'):>6}")
        print(f"      - Conscientiousness (尽责性): {big_five.get('conscientiousness', 'N/A'):>6}")
        print(f"      - Extraversion (外向性):     {big_five.get('extraversion', 'N/A'):>6}")
        print(f"      - Agreeableness (宜人性):    {big_five.get('agreeableness', 'N/A'):>6}")
        print(f"      - Neuroticism (神经质):      {big_five.get('neuroticism', 'N/A'):>6}")
        
        if big_five:
            print(f"    完整 JSON: {json.dumps(big_five, ensure_ascii=False, indent=2)}")
        else:
            print(f"    ⚠️  大五人格数据为空")

        mood = bot.mood_state if isinstance(getattr(bot, "mood_state", None), dict) else {}
        print(f"    PADB (mood_state):")
        print(f"      - pleasure (愉悦):   {mood.get('pleasure', 'N/A'):>6}")
        print(f"      - arousal (唤醒):    {mood.get('arousal', 'N/A'):>6}")
        print(f"      - dominance (支配):  {mood.get('dominance', 'N/A'):>6}")
        print(f"      - busyness (繁忙):   {mood.get('busyness', 'N/A'):>6}")
        if mood:
            print(f"    完整 JSON: {json.dumps(mood, ensure_ascii=False, indent=2)}")
        else:
            print(f"    ⚠️  PADB 数据为空")

        print("-" * 100)

    print(f"\n✅ 查询完成，共 {len(bots)} 个 bot")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

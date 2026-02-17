"""
查询 Render 数据库中 user 的数量
"""
import os
import sys
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass

async def main():
    render_url = os.getenv("RENDER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not render_url:
        print("ERROR: RENDER_DATABASE_URL 或 DATABASE_URL 未设置")
        print("请设置环境变量后再运行")
        sys.exit(1)

    # 显示连接信息（隐藏密码）
    if "@" in render_url:
        parts = render_url.split("@")
        if len(parts) >= 2:
            print(f"连接数据库: ...@{parts[-1]}")
    else:
        print(f"连接数据库: (URL格式已隐藏)")

    from app.core.database import DBManager, _create_async_engine_from_database_url
    from sqlalchemy import select, func
    from app.core.database import User

    engine = _create_async_engine_from_database_url(render_url)
    db = DBManager(engine)

    try:
        async with db.Session() as session:
            # 总 user 数
            result = await session.execute(select(func.count(User.id)))
            total_count = result.scalar() or 0
            print(f"\n总共有 {total_count} 个不同的 user")

            # 按 bot 分组统计
            result2 = await session.execute(
                select(User.bot_id, func.count(User.id).label("user_count")).group_by(User.bot_id)
            )
            rows = result2.all()
            if rows:
                print("\n按 bot 分组:")
                for bot_id, cnt in rows:
                    print(f"  bot_id={bot_id}: {cnt} 个 user")
            else:
                print("\n(无 user 记录)")

            # 显示每个 bot 下的 external_id 示例（最多5个）
            if total_count > 0:
                result3 = await session.execute(
                    select(User.bot_id, User.external_id).limit(20)
                )
                rows3 = result3.all()
                if rows3:
                    print("\n示例 user (前20个):")
                    for bot_id, ext_id in rows3:
                        print(f"  bot_id={bot_id}, external_id={ext_id}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

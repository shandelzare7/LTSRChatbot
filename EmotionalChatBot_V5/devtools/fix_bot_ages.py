"""
修复数据库中 bot 的年龄问题

将年龄不在 18-35 范围内的 bot 修正为合理值（20-25之间随机）
"""
from __future__ import annotations

import asyncio
import os
import sys
import random
from pathlib import Path

# 添加项目根目录到路径
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

try:
    from utils.env_loader import load_project_env
    load_project_env(root)
except Exception:
    pass

from app.core import DBManager
from sqlalchemy import select
from app.core import Bot


async def fix_bot_ages():
    """修复所有 bot 的年龄问题"""
    if not os.getenv("DATABASE_URL"):
        print("❌ 未设置 DATABASE_URL，跳过数据库修复")
        return
    
    db = DBManager.from_env()
    
    async with db.Session() as session:
        async with session.begin():
            # 获取所有 bot
            result = await session.execute(select(Bot))
            bots = result.scalars().all()
            
            fixed_count = 0
            for bot in bots:
                basic_info = bot.basic_info or {}
                age = basic_info.get("age")
                
                # 检查年龄是否有效
                needs_fix = False
                if age is None:
                    needs_fix = True
                    new_age = random.choice([20, 21, 22, 23, 24, 25])
                else:
                    try:
                        age_int = int(age)
                        if age_int < 18 or age_int > 35:
                            needs_fix = True
                            new_age = random.choice([20, 21, 22, 23, 24, 25])
                        else:
                            continue  # 年龄正常，跳过
                    except (ValueError, TypeError):
                        needs_fix = True
                        new_age = random.choice([20, 21, 22, 23, 24, 25])
                
                if needs_fix:
                    old_age = age
                    basic_info["age"] = new_age
                    bot.basic_info = basic_info
                    fixed_count += 1
                    print(f"  ✅ 修复 Bot {bot.name} (ID: {str(bot.id)[:8]}...): 年龄 {old_age} -> {new_age}")
            
            if fixed_count == 0:
                print("✅ 所有 bot 的年龄都在合理范围内（18-35）")
            else:
                print(f"\n✅ 共修复 {fixed_count} 个 bot 的年龄")
    
    print("\n💡 提示：")
    print("   - 年龄范围已修正为 18-35 岁")
    print("   - 新创建的 bot 会自动验证年龄范围")


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
        raise RuntimeError("检测到运行中的事件循环，请在普通终端运行此脚本")
    except RuntimeError:
        return asyncio.run(coro)


if __name__ == "__main__":
    _run_async(fix_bot_ages())

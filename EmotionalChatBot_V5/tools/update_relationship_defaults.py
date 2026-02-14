"""
更新所有现有用户的关系维度默认值为 0.3（power 保持 0.5）

用法：
    python tools/update_relationship_defaults.py
"""
from __future__ import annotations

import asyncio
import os
import sys
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

from app.core.database import DBManager
from sqlalchemy import select, update
from app.core.database import User


async def update_all_relationship_defaults():
    """将所有用户的关系维度更新为默认值 0.3（power 保持 0.5）"""
    if not os.getenv("DATABASE_URL"):
        print("❌ 未设置 DATABASE_URL，跳过数据库更新")
        print("   本地存储的关系值会在下次创建新关系时自动使用新默认值")
        return
    
    db = DBManager.from_env()
    default_dims = {
        "closeness": 0.3,
        "trust": 0.3,
        "liking": 0.3,
        "respect": 0.3,
        "warmth": 0.3,
        "power": 0.5
    }
    
    async with db.Session() as session:
        async with session.begin():
            # 获取所有用户
            result = await session.execute(select(User))
            users = result.scalars().all()
            
            updated_count = 0
            for user in users:
                current_dims = user.dimensions or {}
                
                # 检查是否需要更新（如果任何维度是 0.0，则更新）
                needs_update = False
                for key in ["closeness", "trust", "liking", "respect", "warmth"]:
                    if current_dims.get(key, 0.0) == 0.0:
                        needs_update = True
                        break
                
                if needs_update:
                    # 合并：保留非零值，将 0.0 替换为 0.3
                    new_dims = dict(current_dims)
                    for key in ["closeness", "trust", "liking", "respect", "warmth"]:
                        if new_dims.get(key, 0.0) == 0.0:
                            new_dims[key] = 0.3
                    # 确保 power 存在且为 0.5
                    if "power" not in new_dims or new_dims.get("power", 0.5) != 0.5:
                        new_dims["power"] = 0.5
                    
                    user.dimensions = new_dims
                    updated_count += 1
                    print(f"  ✅ 更新用户 {user.external_id} (bot: {user.bot_name or 'N/A'})")
            
            if updated_count == 0:
                print("✅ 没有需要更新的用户（所有关系值都已非零）")
            else:
                print(f"\n✅ 共更新 {updated_count} 个用户的关系维度默认值")
    
    print("\n💡 提示：")
    print("   - 数据库中的关系值已更新")
    print("   - 本地存储的关系值会在下次创建新关系时自动使用新默认值")
    print("   - 如果使用本地存储，可以手动删除 local_data 目录下的 relationship.json 文件来重置")


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
        raise RuntimeError("检测到运行中的事件循环，请在普通终端运行此脚本")
    except RuntimeError:
        return asyncio.run(coro)


if __name__ == "__main__":
    _run_async(update_all_relationship_defaults())

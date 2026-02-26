"""
性能分析脚本 - 逐个节点计时
使用: LTSR_PROFILE_STEPS=1 python devtools/profile_nodes.py
"""
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import time
import json

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

try:
    from utils.env_loader import load_project_env
    load_project_env(root)
except:
    pass

# 启用性能分析
os.environ["LTSR_PROFILE_STEPS"] = "1"

from langchain_core.messages import HumanMessage
from sqlalchemy import select
from app.core.database import Bot, DBManager, User
from app.graph import build_graph
from main import _make_initial_state


async def profile_single_message():
    """性能分析 - 单条消息"""
    if not os.getenv("DATABASE_URL"):
        print("DATABASE_URL not set")
        return

    db = DBManager.from_env()

    # 获取第一个 Bot
    async with db.Session() as session:
        result = await session.execute(select(Bot).limit(1))
        bot = result.scalar_one()

    bot_id, user_name = str(bot.id), "PerfUser"

    # 获取或创建用户
    import uuid as uuid_lib
    bot_uuid = uuid_lib.UUID(bot_id)

    async with db.Session() as session:
        async with session.begin():
            result = await session.execute(
                select(User).where(User.bot_id == bot_uuid, User.external_id == user_name)
            )
            user = result.scalar_one_or_none()

            if not user:
                from app.core.profile_factory import generate_user_profile
                from app.core.relationship_templates import get_random_relationship_template

                user_basic_info, user_inferred = generate_user_profile(user_name)
                relationship_template = get_random_relationship_template()
                bot_result = await session.execute(select(Bot).where(Bot.id == bot_uuid))
                bot = bot_result.scalar_one()

                user = User(
                    bot_id=bot_uuid,
                    bot_name=bot.name,
                    external_id=user_name,
                    basic_info=user_basic_info,
                    current_stage="initiating",
                    dimensions=relationship_template,
                    inferred_profile=user_inferred,
                    assets={"topic_history": [], "breadth_score": 0, "max_spt_depth": 1},
                    spt_info={},
                    conversation_summary="",
                )
                session.add(user)
                await session.flush()

    user_id = str(user.id)
    await db.clear_messages_for(user_id, bot_id)

    # 构建图
    print("\n构建 graph...")
    t0 = time.time()
    app = build_graph()
    print(f"  Graph 构建: {time.time() - t0:.2f}s\n")

    # 运行推理
    print("=" * 80)
    print("运行推理...")
    print("=" * 80)
    state = _make_initial_state(user_id, bot_id)
    state["messages"] = [HumanMessage(content="你好")]
    state["current_time"] = datetime.now().isoformat()

    t_start = time.time()
    result = await app.ainvoke(state, config={"recursion_limit": 50})
    t_total = time.time() - t_start

    # 提取性能数据
    profile = result.get("_profile", {})
    nodes_data = profile.get("nodes", [])

    if nodes_data:
        print("\n📊 节点执行耗时 (ms):\n")
        print(f"{'节点名':<25} {'耗时 (ms)':<15} {'百分比':<10}")
        print("-" * 50)

        total_ms = sum(n["dt_ms"] for n in nodes_data)

        for node_info in nodes_data:
            name = node_info["name"]
            dt_ms = node_info["dt_ms"]
            pct = (dt_ms / total_ms * 100) if total_ms > 0 else 0
            print(f"{name:<25} {dt_ms:<15.2f} {pct:<10.1f}%")

        print("-" * 50)
        print(f"{'总计':<25} {total_ms:<15.2f} {'100.0%':<10}")
        print(f"\n⏱️  总耗时: {t_total:.2f}s")

        # 显示最慢的 5 个节点
        sorted_nodes = sorted(nodes_data, key=lambda x: x["dt_ms"], reverse=True)
        print(f"\n🔍 最慢的 5 个节点:")
        for i, node_info in enumerate(sorted_nodes[:5], 1):
            print(f"  {i}. {node_info['name']:<25} {node_info['dt_ms']:>8.2f}ms")
    else:
        print(f"⚠️  未捕获到性能数据。请确保设置了 LTSR_PROFILE_STEPS=1")
        print(f"\n总耗时: {t_total:.2f}s")

    print(f"\n最终回复: {result.get('final_response', '（无）')}\n")


if __name__ == "__main__":
    asyncio.run(profile_single_message())


"""
简单生成测试脚本
用于快速测试 inner_monologue、generate、judge 节点的输出
直接使用本地数据库的 Bot 和 User 数据

用法：
  python devtools/test_simple_generation.py [--bot-name "Bot名字"] [--user-name "用户名"] [--messages 消息1,消息2,...]

示例：
  python devtools/test_simple_generation.py --messages "你好,在吗,想聊聊"
"""
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

# 加载 .env
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
try:
    from utils.env_loader import load_project_env
    load_project_env(root)
except Exception:
    pass

from langchain_core.messages import HumanMessage
from sqlalchemy import select

from app.core.database import Bot, DBManager, User
from app.graph import build_graph
from app.state import AgentState
from main import _make_initial_state


async def get_first_bot(db: DBManager) -> Optional[tuple[str, Bot]]:
    """获取数据库中的第一个 Bot"""
    async with db.Session() as session:
        result = await session.execute(select(Bot).limit(1))
        bot = result.scalar_one_or_none()
        if bot:
            return str(bot.id), bot
        return None


async def get_or_create_user(db: DBManager, bot_id: str, user_name: str) -> tuple[str, User]:
    """获取或创建用户"""
    import uuid as uuid_lib
    bot_uuid = uuid_lib.UUID(bot_id)

    async with db.Session() as session:
        async with session.begin():
            result = await session.execute(
                select(User).where(User.bot_id == bot_uuid, User.external_id == user_name)
            )
            user = result.scalar_one_or_none()

            if user:
                return str(user.id), user

            # 创建新用户
            from app.core.profile_factory import generate_user_profile
            from app.core.relationship_templates import get_random_relationship_template

            user_basic_info, user_inferred = generate_user_profile(user_name)
            relationship_template = get_random_relationship_template()

            # 重新获取 bot 信息
            bot_result = await session.execute(select(Bot).where(Bot.id == bot_uuid))
            bot = bot_result.scalar_one()

            new_user = User(
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
            session.add(new_user)
            await session.flush()
            return str(new_user.id), new_user


def format_output(label: str, content: str, width: int = 80):
    """格式化输出"""
    print(f"\n{'=' * width}")
    print(f"  {label}")
    print(f"{'=' * width}")
    if isinstance(content, dict):
        print(json.dumps(content, ensure_ascii=False, indent=2))
    elif isinstance(content, list):
        for i, item in enumerate(content, 1):
            if isinstance(item, dict):
                print(f"\n  [{i}] {json.dumps(item, ensure_ascii=False)}")
            else:
                print(f"  [{i}] {item}")
    else:
        print(content)


async def run_single_message(
    bot_id: str,
    user_id: str,
    user_message: str,
    app,
    db: DBManager,
) -> dict:
    """运行单条消息"""
    print(f"\n{'#' * 80}")
    print(f"# 消息: {user_message}")
    print(f"{'#' * 80}")

    state = _make_initial_state(user_id, bot_id)
    state["messages"] = [HumanMessage(content=user_message)]
    state["current_time"] = datetime.now().isoformat()

    try:
        result = await app.ainvoke(state, config={"recursion_limit": 50})
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {}

    # 提取关键信息
    inner_monologue = result.get("inner_monologue", "")
    monologue_extract = result.get("monologue_extract", {})
    generation_candidates = result.get("generation_candidates", [])
    final_response = result.get("final_response", "")
    judge_result = result.get("judge_result", {})

    # 打印关键节点
    if inner_monologue:
        format_output("内心独白 (Inner Monologue)", inner_monologue)

    if monologue_extract:
        format_output("独白提取结果 (Monologue Extract)", monologue_extract)

    if generation_candidates:
        format_output("生成候选 (Candidates)", generation_candidates)

    if judge_result:
        format_output("评判结果 (Judge Result)", judge_result)

    if final_response:
        format_output("最终回复 (Final Response)", final_response)

    # 显示路由决策
    if result.get("safety_triggered"):
        print(f"\n⚠️  [安全触发] strategy_id = {result.get('safety_strategy_id')}")

    return result


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="简单生成测试")
    parser.add_argument("--bot-name", type=str, default=None, help="指定 Bot 名字")
    parser.add_argument("--user-name", type=str, default="TestUser", help="用户名")
    parser.add_argument(
        "--messages",
        type=str,
        default="你好,在吗,想聊聊",
        help="逗号分隔的消息列表"
    )
    parser.add_argument("--no-clear", action="store_true", help="不清空历史记录")

    args = parser.parse_args()

    # 检查数据库
    if not os.getenv("DATABASE_URL"):
        print("⚠️  DATABASE_URL 未设置")
        sys.exit(1)

    db = DBManager.from_env()

    # 获取 Bot
    bot_result = await get_first_bot(db)
    if not bot_result:
        print("❌ 数据库中找不到任何 Bot")
        sys.exit(1)

    bot_id, bot = bot_result
    print(f"\n✓ Bot: {bot.name} (ID: {bot_id[:8]}...)")

    # 获取或创建用户
    user_id, user = await get_or_create_user(db, bot_id, args.user_name)
    print(f"✓ User: {args.user_name} (ID: {user_id[:8]}...)")

    # 清空历史（可选）
    if not args.no_clear:
        try:
            n = await db.clear_messages_for(user_id, bot_id)
            if n > 0:
                print(f"✓ 已清空 {n} 条历史消息")
        except Exception as e:
            print(f"⚠️  清空历史失败: {e}")

    # 构建图
    print("\n正在构建 graph...")
    app = build_graph()
    print("✓ Graph 构建完成")

    # 解析消息
    messages = [msg.strip() for msg in args.messages.split(",") if msg.strip()]

    # 运行测试
    print(f"\n开始运行 {len(messages)} 条消息...")
    for msg in messages:
        await run_single_message(bot_id, user_id, msg, app, db)

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

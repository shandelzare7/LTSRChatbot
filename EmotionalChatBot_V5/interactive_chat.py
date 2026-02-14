"""
Interactive Chat Entry
交互式对话入口

功能：
1. 启动时列出数据库中的所有 Bot，让用户选择
2. 让用户输入名字（作为 external_id）
3. 如果名字匹配已有用户，使用该用户；否则创建新用户
4. 开始多轮对话
"""
import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# 加载 .env
root = Path(__file__).resolve().parent
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
from main import _make_initial_state, _open_session_log, FileOnlyWriter, TeeWriter


async def list_all_bots(db: DBManager) -> list[Bot]:
    """列出数据库中的所有 Bot"""
    async with db.Session() as session:
        result = await session.execute(select(Bot).order_by(Bot.name))
        bots = result.scalars().all()
        return list(bots)


async def select_bot_interactive(db: DBManager) -> tuple[str, Bot]:
    """交互式选择 Bot"""
    bots = await list_all_bots(db)
    
    if not bots:
        print("⚠️  数据库中没有找到任何 Bot。")
        print("请先创建 Bot 或运行 seed 脚本。")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("请选择一个 Bot 进行对话：")
    print("=" * 60)
    
    for i, bot in enumerate(bots, 1):
        bot_name = bot.name or f"Bot {i}"
        bot_id = str(bot.id)
        # 显示基本信息
        basic_info = bot.basic_info or {}
        age = basic_info.get("age", "未知")
        occupation = basic_info.get("occupation", "未知")
        print(f"  [{i}] {bot_name} (ID: {bot_id[:8]}...)")
        print(f"      年龄: {age}, 职业: {occupation}")
    
    print("=" * 60)
    
    while True:
        try:
            choice = input("\n请输入序号 (1-{}): ".format(len(bots))).strip()
            idx = int(choice) - 1
            if 0 <= idx < len(bots):
                selected_bot = bots[idx]
                bot_id = str(selected_bot.id)
                print(f"\n✓ 已选择: {selected_bot.name} (ID: {bot_id})")
                return bot_id, selected_bot
            else:
                print(f"⚠️  请输入 1-{len(bots)} 之间的数字")
        except ValueError:
            print("⚠️  请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n再见。")
            sys.exit(0)


async def get_or_create_user_by_name(db: DBManager, name: str, bot_id: str) -> tuple[str, User]:
    """根据名字在指定 bot 下获取或创建用户（user 挂在 bot 下）。返回 (external_id, user)。"""
    import uuid as uuid_lib
    external_id = name.strip()
    bot_uuid = uuid_lib.UUID(bot_id)

    async with db.Session() as session:
        async with session.begin():
            result = await session.execute(
                select(User).where(User.bot_id == bot_uuid, User.external_id == external_id)
            )
            existing_user = result.scalar_one_or_none()

            if existing_user:
                print(f"✓ 找到已存在的用户: {name} (ID: {str(existing_user.id)[:8]}...)")
                basic_info = existing_user.basic_info or {}
                if basic_info:
                    print(f"  用户信息: {basic_info}")
                return external_id, existing_user
            else:
                from app.core.profile_factory import generate_user_profile
                from app.core.relationship_templates import get_random_relationship_template
                user_basic_info, user_inferred = generate_user_profile(external_id)
                # 随机选择一个关系维度模板
                relationship_template = get_random_relationship_template()
                new_user = User(
                    bot_id=bot_uuid,
                    bot_name=selected_bot.name,
                    external_id=external_id,
                    basic_info=user_basic_info,
                    current_stage="initiating",
                    dimensions=relationship_template,
                    mood_state={"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0},
                    inferred_profile=user_inferred,
                    assets={"topic_history": [], "breadth_score": 0, "max_spt_depth": 1},
                    spt_info={},
                    conversation_summary="",
                )
                session.add(new_user)
                await session.flush()
                print(f"✓ 创建新用户: {name} (ID: {str(new_user.id)[:8]}...)")
                print(f"  用户信息: {user_basic_info}")
                return external_id, new_user


async def run_interactive_chat():
    """运行交互式对话"""
    # 检查数据库连接
    if not os.getenv("DATABASE_URL"):
        print("⚠️  DATABASE_URL 未设置：请在 .env 里配置本地 PostgreSQL 连接串。")
        sys.exit(1)
    
    db = DBManager.from_env()
    
    # 选择 Bot
    bot_id, selected_bot = await select_bot_interactive(db)
    
    # 输入用户名
    print("\n" + "=" * 60)
    print("请输入您的名字：")
    print("=" * 60)
    while True:
        try:
            user_name = input("\n名字: ").strip()
            if user_name:
                break
            print("⚠️  名字不能为空，请重新输入")
        except KeyboardInterrupt:
            print("\n\n再见。")
            sys.exit(0)
    
    # 获取或创建用户（在该 bot 下）
    user_id, user = await get_or_create_user_by_name(db, user_name, bot_id)
    
    # 开始对话
    print("\n" + "=" * 60)
    print("开始对话")
    print("=" * 60)
    print(f"Bot: {selected_bot.name}")
    print(f"用户: {user_name}")
    print("=" * 60)
    
    # 构建 graph
    app = build_graph()
    
    # 创建日志文件
    log_file, log_path = _open_session_log()
    original_stdout = sys.stdout
    tee = TeeWriter(log_file, original_stdout)
    sys.stdout = tee
    
    def log_line(msg: str):
        """写一行到日志文件并打印到控制台。"""
        print(msg)
    
    # 清空该 user/bot 的对话记录（可选，如果需要新会话）
    try:
        n = await db.clear_messages_for(user_id, bot_id)
        if n > 0:
            log_line(f"[Session] 已清空本会话历史（共 {n} 条），当前为全新对话。")
    except Exception as e:
        log_line(f"[Session] 清空历史失败（继续运行）: {e}")
    
    try:
        log_line("=" * 50)
        log_line("EmotionalChatBot V5.0 交互式对话")
        log_line(f"   Bot: {selected_bot.name} (ID: {bot_id})")
        log_line(f"   用户: {user_name} (ID: {user_id})")
        log_line("   输入一行内容回车发送，空行或 Ctrl+C 退出")
        log_line(f"   日志文件: {log_path}")
        log_line("=" * 50)
        
        loop = asyncio.get_running_loop()
        
        while True:
            try:
                line = await loop.run_in_executor(None, lambda: input("\n你: ").strip())
            except (KeyboardInterrupt, EOFError):
                log_line("\n再见。")
                break
            if not line:
                log_line("（空输入退出）")
                break
            
            now_iso = datetime.now().isoformat()
            log_line("")
            log_line(f"[{now_iso}] === 你: {line}")
            log_line("-" * 40)
            
            state = _make_initial_state(user_id, bot_id)
            state["messages"] = [HumanMessage(content=line, additional_kwargs={"timestamp": now_iso})]
            state["current_time"] = now_iso
            
            try:
                # graph 内部所有 print 只写日志文件，不输出到控制台
                sys.stdout = FileOnlyWriter(log_file)
                try:
                    result = await app.ainvoke(state, config={"recursion_limit": 50})
                finally:
                    sys.stdout = tee
            except Exception as e:
                log_line(f"Bot: [出错] {e}")
                continue
            
            reply = result.get("final_response") or ""
            if not reply and result.get("final_segments"):
                reply = " ".join(result["final_segments"])
            if not reply:
                reply = result.get("draft_response") or "（无回复）"
            
            log_line(f"=== Bot: {reply}")
            log_line("")
            print("Bot:", reply)
    finally:
        sys.stdout = original_stdout
        try:
            log_file.close()
            print(f"\n日志已保存: {log_path}")
        except OSError:
            pass


def main():
    """入口函数"""
    asyncio.run(run_interactive_chat())


if __name__ == "__main__":
    main()

"""
生成所有 Bot 的分享链接
运行此脚本可以快速获取所有 Bot 的分享链接
"""
import os
import sys
import asyncio
from pathlib import Path

# 加载 .env
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
try:
    from utils.env_loader import load_project_env
    load_project_env(root)
except Exception:
    pass

from app.core import DBManager, Bot
from sqlalchemy import select


async def generate_all_share_links():
    """生成所有bot的分享链接"""
    # 获取基础URL（从环境变量或使用默认值）
    base_url = os.getenv("WEB_DOMAIN", "localhost:8000")
    if not base_url.startswith("http"):
        # 判断是否为生产环境
        if os.getenv("ENVIRONMENT") == "production":
            base_url = f"https://{base_url}"
        else:
            base_url = f"http://{base_url}"
    
    try:
        db = DBManager.from_env()
        async with db.Session() as session:
            result = await session.execute(select(Bot).order_by(Bot.name))
            bots = result.scalars().all()
        
        if not bots:
            print("⚠️  数据库中没有找到任何 Bot")
            return
        
        print("=" * 80)
        print("🤖 Chatbot 分享链接")
        print("=" * 80)
        print(f"基础URL: {base_url}")
        print("=" * 80)
        print()
        
        for bot in bots:
            bot_id = str(bot.id)
            bot_name = bot.name or "Unnamed Bot"
            share_link = f"{base_url}/chat/{bot_id}"
            qr_code_url = f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={share_link}"
            
            print(f"📱 {bot_name}")
            print(f"   ID: {bot_id}")
            print(f"   链接: {share_link}")
            print(f"   二维码: {qr_code_url}")
            print()
        
        print("=" * 80)
        print("💡 提示:")
        print("   1. 将链接分享给朋友即可直接开始对话")
        print("   2. 每个链接对应一个特定的 Bot")
        print("   3. 首次访问会自动创建会话")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(generate_all_share_links())

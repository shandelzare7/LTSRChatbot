#!/usr/bin/env python3
"""
端到端测试：询问 Bot 会触发搜索的问题，记录用时、返回内容、加入 chat 后的效果。

用法（需 .env 中 DATABASE_URL、OPENAI/LLM 相关配置，以及至少一个 Bot）：
  cd EmotionalChatBot_V5
  .venv/bin/python devtools/test_search_e2e.py

可选环境变量：
  SEARCH_TEST_QUERY：触发搜索的用户问题，默认「北京今天天气怎么样」
  BOT_ID / USER_ID：不设则从 DB 取第一个 bot 并创建临时 user
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)

# 触发搜索的测试问题（需让 detection 输出 knowledge_gap=True + search_keywords）
DEFAULT_QUERY = "北京今天天气怎么样"
QUERY = os.getenv("SEARCH_TEST_QUERY", "").strip() or DEFAULT_QUERY

# 让 knowledge_fetcher 把搜索耗时写入 state，便于本脚本打印
os.environ["LTSR_SEARCH_TIMING"] = "1"


async def main() -> None:
    if not os.getenv("DATABASE_URL"):
        print("请设置 DATABASE_URL（.env 或环境变量）")
        sys.exit(1)

    from langchain_core.messages import HumanMessage
    from app.graph import build_graph
    from app.core.database import DBManager
    from main import _make_initial_state

    db = DBManager.from_env()
    bot_id = (os.getenv("BOT_ID") or "").strip()
    user_id = (os.getenv("USER_ID") or "").strip()

    if not bot_id or not user_id:
        async with db.Session() as session:
            from sqlalchemy import select
            from app.core.database import Bot
            row = (await session.execute(select(Bot).limit(1))).scalars().first()
            if not row:
                print("数据库中无 Bot，请先创建至少一个 Bot（例如运行 bot_to_bot_chat 会自动创建）")
                sys.exit(1)
            bot_id = str(row.id)
            u = await db._get_or_create_user(session, bot_id, "search_test_user", visit_source="test")
            user_id = str(u.id)
            await session.commit()
        print(f"使用 Bot ID: {bot_id[:8]}..., User ID: {user_id[:8]}...")

    app = build_graph()
    state = _make_initial_state(user_id, bot_id)
    state["user_input"] = QUERY
    state["external_user_text"] = QUERY
    state["messages"] = [HumanMessage(content=QUERY)]
    state["current_time"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    print("=" * 60)
    print("搜索功能端到端测试")
    print("=" * 60)
    print(f"用户问题: {QUERY}")
    print()

    t0 = time.perf_counter()
    try:
        result = await app.ainvoke(state, config={"recursion_limit": 50})
    except Exception as e:
        print(f"invoke 失败: {e}")
        raise
    total_elapsed = time.perf_counter() - t0

    detection = result.get("detection") or {}
    knowledge_gap = detection.get("knowledge_gap", False)
    search_keywords = detection.get("search_keywords") or ""
    retrieved = (result.get("retrieved_external_knowledge") or "").strip()
    search_elapsed = result.get("retrieved_external_knowledge_elapsed_seconds")
    final_response = result.get("final_response") or ""

    print()
    print("【1. Detection 是否触发搜索】")
    print(f"  knowledge_gap: {knowledge_gap}")
    print(f"  search_keywords: {search_keywords!r}")
    print()

    print("【2. 用时】")
    print(f"  全轮总用时: {total_elapsed:.2f}s")
    if search_elapsed is not None:
        print(f"  搜索步骤用时: {search_elapsed:.2f}s")
    print()

    print("【3. 搜索返回内容】")
    if retrieved:
        print(f"  长度: {len(retrieved)} 字符")
        print("  内容:")
        for line in retrieved.split("\n")[:15]:
            print(f"    {line}")
        if retrieved.count("\n") >= 15:
            print("    ...")
    else:
        print("  （无内容，可能未触发搜索或搜索无结果）")
    print()

    print("【4. 加入 Chat 后的效果】")
    print("  注入到独白的区块为: 「## 你刚好知道的背景（来自搜索）」+ 上述内容")
    print("  Bot 最终回复:")
    if final_response:
        for line in (final_response.strip() or "（无）").split("\n"):
            print(f"    {line}")
    else:
        print("    （无）")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

"""
独立测试 memory_manager 话题提取：
验证 prompt + schema 修改后 gpt-4o-mini 能否正确返回 new_topics。
"""
import asyncio
import os
import sys

# 项目根路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.schemas import MemoryManagerOutput


# 复用 memory_manager.py 中的 system prompt（话题检测部分）
SYS_PROMPT = """你是经验丰富的记录总结专家，擅长从对话中提炼关键信息并形成结构化记录。
你将基于【旧摘要】+【本轮对话】输出严格 JSON，用于更新摘要、沉淀稳定记忆与抽取用户基础信息。

通用要求（影响稳定性）：
1) 摘要要"可持续更新"：在旧摘要基础上增量更新，不要推翻重写；保持精炼、客观、可复用。
2) 只写"稳定事实/偏好/正在做什么/已决策/关键约束"，不要猜测、不要心理分析。
3) notes（Derived Notes）要少而精（0~5 条），每条必须是可检验/可复用的信息。
4) entities/topic/short_context 保守：不确定就留空或更泛化；importance 建议 0~1。
5) 任何字段都不允许凭空补全；不确定就输出 null/空。

【话题历史检测（Topic History）】
你需要判断本轮对话是否引入了与现有话题历史不同的新话题。

规则：
- 比较现有 topic_history 和本轮对话内容（user_input + bot_text）
- 话题粒度应为中等层级的主题词，例如："冷笑话"、"诗歌"、"暗恋经历"、"工作压力"、"旅行"、"美食"、"童年回忆"
- 禁止使用过于笼统的类别（如"日常闲聊"、"生活"、"杂谈"），也不要用过于具体的细节（如"海子的面朝大海"）
- 如果对话内容可以归入一个已有话题，则不要添加；但如果涉及了一个明显不同领域的中层主题，应识别为新话题
- 如果本轮没有新话题，返回空数组 []"""


# 测试用例：明显不同话题的对话轮
TEST_CASES = [
    {
        "desc": "第1轮：自我介绍（空topic_history）",
        "topic_history": [],
        "user_input": "你好呀！我叫小明，今天天气真不错，你觉得呢？",
        "bot_text": "你好小明！天气确实不错呢，适合出去走走~",
        "prev_summary": "（空）",
        "expect_topics": True,
    },
    {
        "desc": "第2轮：聊美食（已有'自我介绍'）",
        "topic_history": ["自我介绍"],
        "user_input": "话说你知道哪里有好吃的火锅吗？我最近超想吃辣的！",
        "bot_text": "火锅的话，重庆风味的很不错！你喜欢什么口味？",
        "prev_summary": "用户叫小明，聊了天气。",
        "expect_topics": True,
    },
    {
        "desc": "第3轮：聊旅行（已有'自我介绍','美食'）",
        "topic_history": ["自我介绍", "美食"],
        "user_input": "下个月我打算去云南旅行，你有什么推荐的地方吗？",
        "bot_text": "云南很美！大理和丽江都值得去，你喜欢自然风光还是人文景点？",
        "prev_summary": "用户叫小明，聊了天气和火锅。",
        "expect_topics": True,
    },
    {
        "desc": "第4轮：继续聊美食（不应添加新话题）",
        "topic_history": ["自我介绍", "美食", "旅行"],
        "user_input": "说到吃的，我昨天试了一家新开的日料店，刺身特别新鲜！",
        "bot_text": "听起来不错！日料讲究新鲜，你喜欢哪种鱼的刺身？",
        "prev_summary": "用户叫小明，聊了天气、火锅和云南旅行。",
        "expect_topics": False,
    },
    {
        "desc": "第5轮：聊工作压力（新话题）",
        "topic_history": ["自我介绍", "美食", "旅行"],
        "user_input": "最近加班好多，老板一直催项目进度，感觉压力好大",
        "bot_text": "听起来你挺辛苦的，工作压力大的时候要注意休息哦",
        "prev_summary": "用户叫小明，聊了天气、火锅和云南旅行。",
        "expect_topics": True,
    },
]


async def run_test():
    # 使用与 get_llm(role="fast") 相同的配置
    api_key = os.getenv("OPENAI_API_KEY_OPENAI") or os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=api_key,
        base_url="https://api.openai.com/v1",
    )
    structured = llm.with_structured_output(MemoryManagerOutput)

    print("=" * 60)
    print("Memory Manager 话题提取独立测试")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, tc in enumerate(TEST_CASES):
        topic_history_str = ", ".join(tc["topic_history"]) if tc["topic_history"] else "（空）"
        human_prompt = f"""【旧摘要】
{tc["prev_summary"]}

【本轮对话】
- time: 2026-03-08T14:00:00
- user_input: {tc["user_input"]}
- bot: {tc["bot_text"]}

【现有话题历史】
{topic_history_str}

请判断本轮对话是否引入了新话题。如果有，在 new_topics 中返回新话题列表。new_summary 建议 150~600 字，保留足够细节，但避免逐字复述。
（输出格式由系统约束。）"""

        try:
            obj = await structured.ainvoke(
                [SystemMessage(content=SYS_PROMPT), HumanMessage(content=human_prompt)]
            )
            data = obj.model_dump()
            new_topics = data.get("new_topics", [])
        except Exception as e:
            new_topics = f"ERROR: {e}"

        has_topics = bool(new_topics) and new_topics != []
        status = "PASS" if has_topics == tc["expect_topics"] else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"\n--- Test {i+1}: {tc['desc']} ---")
        print(f"  existing_history: {tc['topic_history']}")
        print(f"  new_topics: {new_topics}")
        print(f"  expected_new: {tc['expect_topics']}, got: {has_topics}")
        print(f"  => {status}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(TEST_CASES)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(run_test())

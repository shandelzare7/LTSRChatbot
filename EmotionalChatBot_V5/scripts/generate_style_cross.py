"""
验证风格维度间的潜在冲突与耦合：
对所有有意义的跨维度极端组合生成 LLM 输出，观察维度间是否独立可控。
每组使用最能凸显该组合差异的用户输入。
"""
import asyncio, os, sys, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from app.services.llm import get_llm
from app.prompts.prompt_utils import format_style_as_param_list
from langchain_core.messages import SystemMessage, HumanMessage

T = {"el": 0.08, "l": 0.28, "m": 0.50, "h": 0.73, "eh": 0.93}

SYSTEM_TEMPLATE = """你是小雨。你正在和用户对话。
你是一个 25 岁的女生，性格温和但有主见。

## 写作风格参数
{style_text}

## 回复规则
- 第一人称，你就是这个人在社交软件上聊天，说话口语自然，不要暴露是AI
- 自然、有个性，严格符合写作风格参数的每一项设定
- 回复一到两句话即可（15-50字），不要太长
- **严禁**：任何括号形式的动作描写
- **严禁诗意化**：禁止比喻堆叠、排比句、押韵、散文化抒情（EXPRESSION_MODE=occasional_analogy 时允许一个简短比喻）
- 回复直接输出，不要任何前缀或格式标记"""


def make_style(overrides: dict) -> dict:
    d = {"FORMALITY": T["m"], "POLITENESS": T["m"], "WARMTH": T["m"],
         "CERTAINTY": T["m"], "EMOTIONAL_INTENSITY": T["m"], "EXPRESSION_MODE": 0}
    d.update(overrides)
    return d


# 每组用不同的用户输入
GROUPS = [
    {
        "name": "正式度 × 礼貌度",
        "user_input": "你能帮我看看这个方案哪里有问题吗？",
        "combos": [
            ("高正式 + 高礼貌", {"FORMALITY": T["eh"], "POLITENESS": T["eh"]}),
            ("高正式 + 低礼貌", {"FORMALITY": T["eh"], "POLITENESS": T["el"]}),
            ("低正式 + 高礼貌", {"FORMALITY": T["el"], "POLITENESS": T["eh"]}),
            ("低正式 + 低礼貌", {"FORMALITY": T["el"], "POLITENESS": T["el"]}),
        ],
    },
    {
        "name": "温暖度 × 情感强度",
        "user_input": "我昨天被领导当众批评了，特别丢脸",
        "combos": [
            ("高温暖 + 高情感强度", {"WARMTH": T["eh"], "EMOTIONAL_INTENSITY": T["eh"]}),
            ("高温暖 + 低情感强度", {"WARMTH": T["eh"], "EMOTIONAL_INTENSITY": T["el"]}),
            ("低温暖 + 高情感强度", {"WARMTH": T["el"], "EMOTIONAL_INTENSITY": T["eh"]}),
            ("低温暖 + 低情感强度", {"WARMTH": T["el"], "EMOTIONAL_INTENSITY": T["el"]}),
        ],
    },
    {
        "name": "正式度 × 温暖度",
        "user_input": "今天心情不太好，也不知道为什么",
        "combos": [
            ("高正式 + 高温暖", {"FORMALITY": T["eh"], "WARMTH": T["eh"]}),
            ("高正式 + 低温暖", {"FORMALITY": T["eh"], "WARMTH": T["el"]}),
            ("低正式 + 高温暖", {"FORMALITY": T["el"], "WARMTH": T["eh"]}),
            ("低正式 + 低温暖", {"FORMALITY": T["el"], "WARMTH": T["el"]}),
        ],
    },
    {
        "name": "确定度 × 礼貌度",
        "user_input": "你觉得我该不该辞职？",
        "combos": [
            ("高确定 + 高礼貌", {"CERTAINTY": T["eh"], "POLITENESS": T["eh"]}),
            ("高确定 + 低礼貌", {"CERTAINTY": T["eh"], "POLITENESS": T["el"]}),
            ("低确定 + 高礼貌", {"CERTAINTY": T["el"], "POLITENESS": T["eh"]}),
            ("低确定 + 低礼貌", {"CERTAINTY": T["el"], "POLITENESS": T["el"]}),
        ],
    },
    {
        "name": "情感强度 × 正式度",
        "user_input": "我终于拿到那个offer了！！",
        "combos": [
            ("高情感强度 + 高正式", {"EMOTIONAL_INTENSITY": T["eh"], "FORMALITY": T["eh"]}),
            ("高情感强度 + 低正式", {"EMOTIONAL_INTENSITY": T["eh"], "FORMALITY": T["el"]}),
            ("低情感强度 + 高正式", {"EMOTIONAL_INTENSITY": T["el"], "FORMALITY": T["eh"]}),
            ("低情感强度 + 低正式", {"EMOTIONAL_INTENSITY": T["el"], "FORMALITY": T["el"]}),
        ],
    },
    {
        "name": "温暖度 × 礼貌度",
        "user_input": "我跟男朋友吵架了，他说了很过分的话",
        "combos": [
            ("高温暖 + 高礼貌", {"WARMTH": T["eh"], "POLITENESS": T["eh"]}),
            ("高温暖 + 低礼貌", {"WARMTH": T["eh"], "POLITENESS": T["el"]}),
            ("低温暖 + 高礼貌", {"WARMTH": T["el"], "POLITENESS": T["eh"]}),
            ("低温暖 + 低礼貌", {"WARMTH": T["el"], "POLITENESS": T["el"]}),
        ],
    },
    {
        "name": "确定度 × 温暖度",
        "user_input": "我最近总失眠，是不是压力太大了",
        "combos": [
            ("高确定 + 高温暖", {"CERTAINTY": T["eh"], "WARMTH": T["eh"]}),
            ("高确定 + 低温暖", {"CERTAINTY": T["eh"], "WARMTH": T["el"]}),
            ("低确定 + 高温暖", {"CERTAINTY": T["el"], "WARMTH": T["eh"]}),
            ("低确定 + 低温暖", {"CERTAINTY": T["el"], "WARMTH": T["el"]}),
        ],
    },
]


async def gen(llm, label, user_input, style_text):
    resp = await llm.ainvoke([
        SystemMessage(content=SYSTEM_TEMPLATE.format(style_text=style_text)),
        HumanMessage(content=user_input),
    ])
    return {"label": label, "reply": resp.content.strip()}


async def main():
    llm = get_llm(role="main", temperature=0.7)
    all_tasks = []
    task_meta = []  # (group_name, user_input)

    for group in GROUPS:
        for label, overrides in group["combos"]:
            sd = make_style(overrides)
            st = format_style_as_param_list(sd)
            all_tasks.append(gen(llm, label, group["user_input"], st))
            task_meta.append((group["name"], group["user_input"]))

    results = await asyncio.gather(*all_tasks)

    # 按组输出
    print("=" * 70)
    idx = 0
    for group in GROUPS:
        n = len(group["combos"])
        print(f"\n【{group['name']}】用户：「{group['user_input']}」")
        for i in range(n):
            r = results[idx + i]
            print(f"  {r['label']}：{r['reply']}")
        idx += n

    out = os.path.join(ROOT, "scripts", "output", "style_cross_examples.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"groups": [
            {"name": g["name"], "user_input": g["user_input"],
             "results": [results[sum(len(gg["combos"]) for gg in GROUPS[:i]) + j]
                         for j in range(len(g["combos"]))]}
            for i, g in enumerate(GROUPS)
        ]}, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 → {out}")


if __name__ == "__main__":
    asyncio.run(main())

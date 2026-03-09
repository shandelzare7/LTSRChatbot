"""
7 档裸标签（very 版）× 5 句不同用户输入 × 3 次重复。
每个维度用最能凸显该维度差异的用户输入。
"""
import asyncio, os, sys, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from app.services.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

CONTINUOUS_DIMS = ["FORMALITY", "POLITENESS", "WARMTH", "CERTAINTY", "EMOTIONAL_INTENSITY"]
DIM_ZH = {"FORMALITY": "正式度", "POLITENESS": "礼貌度", "WARMTH": "温暖度",
           "CERTAINTY": "确定度", "EMOTIONAL_INTENSITY": "情感强度"}

# 每个维度用不同的用户输入，选最能区分该维度的场景
DIM_INPUTS = {
    "FORMALITY": "你能帮我看看这个方案哪里有问题吗？",
    "POLITENESS": "我觉得你说的不太对吧",
    "WARMTH": "我昨天被领导当众批评了，特别丢脸",
    "CERTAINTY": "你觉得我该不该辞职？",
    "EMOTIONAL_INTENSITY": "我终于拿到那个offer了！！",
}

TIERS_7 = ["extremely_low", "very_low", "low", "mid",
           "high", "very_high", "extremely_high"]
TIER_ZH_7 = {"extremely_low": "极低", "very_low": "很低", "low": "低",
             "mid": "中", "high": "高", "very_high": "很高", "extremely_high": "极高"}

SYSTEM_TEMPLATE = """你是小雨。你正在和用户对话。
你是一个 25 岁的女生，性格温和但有主见。

## 写作风格参数
{style_text}

## 回复规则
- 第一人称，你就是这个人在社交软件上聊天，说话口语自然，不要暴露是AI
- 自然、有个性，严格符合写作风格参数的每一项设定
- 回复一句话即可（15-40字），不要太长
- **严禁**：任何括号形式的动作描写
- **严禁诗意化**：禁止比喻堆叠、排比句、押韵、散文化抒情（EXPRESSION_MODE=occasional_analogy 时允许一个简短比喻）
- 回复直接输出，不要任何前缀或格式标记"""

REPEATS = 3


def make_style(target_dim: str, target_tier: str) -> str:
    parts = []
    for dim in CONTINUOUS_DIMS:
        tier = target_tier if dim == target_dim else "mid"
        parts.append(f"{dim}={tier}")
    parts.append("EXPRESSION_MODE=literal")
    parts.append("文学性=zero")
    return "\n".join(parts)


async def gen(llm, dim_label, tier_label, user_input, style_text, run_id):
    resp = await llm.ainvoke([
        SystemMessage(content=SYSTEM_TEMPLATE.format(style_text=style_text)),
        HumanMessage(content=user_input),
    ])
    return {"dim": dim_label, "tier": tier_label, "run": run_id,
            "user_input": user_input, "reply": resp.content.strip()}


async def main():
    llm = get_llm(role="main", temperature=0.7)
    tasks = []

    for dim in CONTINUOUS_DIMS:
        user_input = DIM_INPUTS[dim]
        for tier in TIERS_7:
            st = make_style(dim, tier)
            for r in range(1, REPEATS + 1):
                tasks.append(gen(llm, DIM_ZH[dim], TIER_ZH_7[tier], user_input, st, r))

    results = await asyncio.gather(*tasks)

    print("=" * 70)
    print(f"7 档裸标签测试（very 版 · 多句）× {REPEATS} 次")
    print("=" * 70)

    idx = 0
    for dim in CONTINUOUS_DIMS:
        print(f"\n{'='*50}")
        print(f"【{DIM_ZH[dim]}】用户：「{DIM_INPUTS[dim]}」")
        print(f"{'='*50}")
        for tier in TIERS_7:
            label = TIER_ZH_7[tier]
            print(f"\n  [{label}] ({tier})")
            for r in range(REPEATS):
                res = results[idx]
                print(f"    #{res['run']}：{res['reply']}")
                idx += 1

    out = os.path.join(ROOT, "scripts", "output", "style_7tier_very_multi.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"dim_inputs": DIM_INPUTS, "repeats": REPEATS, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n已保存 → {out}")


if __name__ == "__main__":
    asyncio.run(main())

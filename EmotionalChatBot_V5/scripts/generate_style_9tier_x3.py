"""
9 档裸标签测试 × 3 次重复，消除随机性。
每个维度 × 每个档位 × 3 次 = 5×9×3 = 135 次调用。
"""
import asyncio, os, sys, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from app.services.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

USER_INPUT = "最近工作压力好大，感觉快撑不住了"

CONTINUOUS_DIMS = ["FORMALITY", "POLITENESS", "WARMTH", "CERTAINTY", "EMOTIONAL_INTENSITY"]
DIM_ZH = {"FORMALITY": "正式度", "POLITENESS": "礼貌度", "WARMTH": "温暖度",
           "CERTAINTY": "确定度", "EMOTIONAL_INTENSITY": "情感强度"}

TIERS_9 = ["extremely_low", "very_low", "low", "moderately_low", "mid",
           "moderately_high", "high", "very_high", "extremely_high"]
TIER_ZH_9 = {"extremely_low": "极低", "very_low": "很低", "low": "低",
             "moderately_low": "略低", "mid": "中",
             "moderately_high": "略高", "high": "高",
             "very_high": "很高", "extremely_high": "极高"}

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


async def gen(llm, dim_label, tier_label, style_text, run_id):
    resp = await llm.ainvoke([
        SystemMessage(content=SYSTEM_TEMPLATE.format(style_text=style_text)),
        HumanMessage(content=USER_INPUT),
    ])
    return {"dim": dim_label, "tier": tier_label, "run": run_id, "reply": resp.content.strip()}


async def main():
    llm = get_llm(role="main", temperature=0.7)
    tasks = []

    for dim in CONTINUOUS_DIMS:
        for tier in TIERS_9:
            st = make_style(dim, tier)
            for r in range(1, REPEATS + 1):
                tasks.append(gen(llm, DIM_ZH[dim], TIER_ZH_9[tier], st, r))

    results = await asyncio.gather(*tasks)

    print("=" * 70)
    print(f"9 档裸标签测试 × {REPEATS} 次")
    print(f"用户输入：「{USER_INPUT}」")
    print("=" * 70)

    idx = 0
    for dim in CONTINUOUS_DIMS:
        print(f"\n{'='*50}")
        print(f"【{DIM_ZH[dim]}】")
        print(f"{'='*50}")
        for tier in TIERS_9:
            label = TIER_ZH_9[tier]
            print(f"\n  [{label}] ({tier})")
            for r in range(REPEATS):
                res = results[idx]
                print(f"    #{res['run']}：{res['reply']}")
                idx += 1

    out = os.path.join(ROOT, "scripts", "output", "style_9tier_x3.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"user_input": USER_INPUT, "repeats": REPEATS, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n已保存 → {out}")


if __name__ == "__main__":
    asyncio.run(main())

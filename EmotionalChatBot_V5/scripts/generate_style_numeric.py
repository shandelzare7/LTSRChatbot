"""
对比测试：用 0-1 数值（而非五档标签）注入 style 参数，观察 LLM 输出效果。
与 generate_style_examples.py 的标签版本做对照。
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

# 五档对应数值
TIERS = [0.05, 0.25, 0.50, 0.75, 0.95]
TIER_LABELS = ["0.05", "0.25", "0.50", "0.75", "0.95"]

# 表达模式用标签（离散维度，数值没意义）
EM_MODES = {0: "literal", 1: "implicit_withholding",
            2: "occasional_analogy", 3: "ironic_teasing"}
EM_ZH = {0: "直述", 1: "含蓄", 2: "比喻", 3: "调侃"}

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


def make_numeric_style(target_dim: str, target_val: float) -> str:
    """所有维度用数值，目标维度设为 target_val，其余 0.50。"""
    parts = []
    for dim in CONTINUOUS_DIMS:
        v = target_val if dim == target_dim else 0.50
        parts.append(f"{dim}={v:.2f}")
    parts.append("EXPRESSION_MODE=literal")
    parts.append("文学性=zero")
    return "\n".join(parts)


def make_em_numeric_style(em_mode: int) -> str:
    parts = [f"{dim}=0.50" for dim in CONTINUOUS_DIMS]
    parts.append(f"EXPRESSION_MODE={EM_MODES[em_mode]}")
    parts.append("文学性=zero")
    return "\n".join(parts)


async def gen(llm, dim_label, tier_label, style_text):
    resp = await llm.ainvoke([
        SystemMessage(content=SYSTEM_TEMPLATE.format(style_text=style_text)),
        HumanMessage(content=USER_INPUT),
    ])
    return {"dim": dim_label, "tier": tier_label, "reply": resp.content.strip()}


async def main():
    llm = get_llm(role="main", temperature=0.7)
    tasks = []

    for dim in CONTINUOUS_DIMS:
        for val, label in zip(TIERS, TIER_LABELS):
            st = make_numeric_style(dim, val)
            tasks.append(gen(llm, DIM_ZH[dim], label, st))

    # 表达模式（标签不变，作为对照基线）
    for em in [0, 1, 2, 3]:
        st = make_em_numeric_style(em)
        tasks.append(gen(llm, "表达模式", EM_ZH[em], st))

    results = await asyncio.gather(*tasks)

    print("=" * 70)
    print(f"数值版 style 注入测试")
    print(f"用户输入：「{USER_INPUT}」")
    print("=" * 70)
    cur = None
    for r in results:
        if r["dim"] != cur:
            cur = r["dim"]
            print(f"\n【{cur}】")
        print(f"  {r['tier']:<6s}：{r['reply']}")

    out = os.path.join(ROOT, "scripts", "output", "style_numeric_examples.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"user_input": USER_INPUT, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 → {out}")


if __name__ == "__main__":
    asyncio.run(main())

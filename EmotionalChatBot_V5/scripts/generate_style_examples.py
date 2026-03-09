"""
验证 6D 风格参数各档位对 LLM 输出的实际控制效果。
直接设置 style label（extremely_low/low/mid/high/extremely_high），
使用实际的 get_llm + format_style_as_param_list + 系统 prompt 模板生成回复。
"""
import asyncio, os, sys, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from app.services.llm import get_llm
from app.prompts.prompt_utils import format_style_as_param_list
from langchain_core.messages import SystemMessage, HumanMessage

USER_INPUT = "最近工作压力好大，感觉快撑不住了"

TIERS = ["extremely_low", "low", "mid", "high", "extremely_high"]
TIER_ZH = {"extremely_low": "极低", "low": "低", "mid": "中", "high": "高", "extremely_high": "极高"}

CONTINUOUS_DIMS = ["FORMALITY", "POLITENESS", "WARMTH", "CERTAINTY", "EMOTIONAL_INTENSITY"]
DIM_ZH = {"FORMALITY": "正式度", "POLITENESS": "礼貌度", "WARMTH": "温暖度",
           "CERTAINTY": "确定度", "EMOTIONAL_INTENSITY": "情感强度"}

EM_MODES = {0: "literal", 1: "implicit_withholding",
            2: "occasional_analogy (use ONE brief, everyday analogy at most—never pile up metaphors or write poetically)",
            3: "ironic_teasing"}
EM_ZH = {0: "直述", 1: "含蓄", 2: "比喻", 3: "调侃"}

# 与 prompt_utils 一致的锚点
ANCHORS_EN = {
    "WARMTH": {
        "extremely_low": "cold baseline, little warmth even when positive",
        "low": "restrained, warmth only when clearly triggered",
        "mid": "neutral baseline, affect follows context",
        "high": "warm baseline, fluent positive affect",
        "extremely_high": "intimate baseline, low threshold for affect",
    },
    "EMOTIONAL_INTENSITY": {
        "extremely_low": "completely flat delivery, no intensifiers or exclamations",
        "low": "calm, measured, minimal emotional markers",
        "mid": "moderate intensity, occasional emphasis",
        "high": "noticeably animated, frequent intensifiers and exclamations",
        "extremely_high": "highly activated, heavy use of intensifiers/repetition/exclamations",
    },
}


def make_style_dict(target_dim: str, target_tier: str) -> dict:
    """构造 style dict：目标维度设为 target_tier，其余连续维度 mid，EM=0。"""
    # format_style_as_param_list 需要数值，用各档中点值
    tier_to_val = {"extremely_low": 0.08, "low": 0.28, "mid": 0.50, "high": 0.73, "extremely_high": 0.93}
    d = {}
    for dim in CONTINUOUS_DIMS:
        d[dim] = tier_to_val[target_tier] if dim == target_dim else tier_to_val["mid"]
    d["EXPRESSION_MODE"] = 0
    return d


def make_em_style_dict(em_mode: int) -> dict:
    """构造 style dict：所有连续维度 mid，EM 设为指定模式。"""
    tier_to_val = {"mid": 0.50}
    d = {dim: 0.50 for dim in CONTINUOUS_DIMS}
    d["EXPRESSION_MODE"] = em_mode
    return d


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


async def gen(llm, dim_label, tier_label, style_text):
    resp = await llm.ainvoke([
        SystemMessage(content=SYSTEM_TEMPLATE.format(style_text=style_text)),
        HumanMessage(content=USER_INPUT),
    ])
    return {"dim": dim_label, "tier": tier_label, "reply": resp.content.strip(), "style_text": style_text}


async def main():
    llm = get_llm(role="main", temperature=0.7)
    tasks = []

    # 连续维度 × 5 档
    for dim in CONTINUOUS_DIMS:
        for tier in TIERS:
            sd = make_style_dict(dim, tier)
            st = format_style_as_param_list(sd)
            tasks.append(gen(llm, DIM_ZH[dim], TIER_ZH[tier], st))

    # 表达模式 × 4 档
    for em in [0, 1, 2, 3]:
        sd = make_em_style_dict(em)
        st = format_style_as_param_list(sd)
        tasks.append(gen(llm, "表达模式", EM_ZH[em], st))

    results = await asyncio.gather(*tasks)

    print("=" * 70)
    print(f"用户输入：「{USER_INPUT}」")
    print("=" * 70)
    cur = None
    for r in results:
        if r["dim"] != cur:
            cur = r["dim"]
            print(f"\n【{cur}】")
        print(f"  {r['tier']:<6s}：{r['reply']}")

    out = os.path.join(ROOT, "scripts", "output", "style_examples.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"user_input": USER_INPUT, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 → {out}")


if __name__ == "__main__":
    asyncio.run(main())

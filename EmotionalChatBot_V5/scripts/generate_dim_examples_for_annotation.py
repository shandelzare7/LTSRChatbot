"""
为标注页五维（FORMALITY / POLITENESS / FRIENDLINESS / CERTAINTY / EMOTIONAL_TONE）
各生成一组易区分的示例句。

- 与 generate 一致：不传任何描述/回答目标，只传 style block（维度名=档位标签）+ 用户消息。
- 每个维度一个用户问题作为 HumanMessage，仅此而已。
- 输出可直接用于 static/annotation/index.html 的 DIM_EXAMPLES

用法（在 EmotionalChatBot_V5 目录或项目根执行）:
  python -m scripts.generate_dim_examples_for_annotation
  输出: scripts/output/dim_examples_for_annotation.json
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from langchain_core.messages import HumanMessage, SystemMessage

from app.prompts.prompt_utils import format_style_as_param_list
from app.services.llm import get_llm

# 显式指定 Qwen 生成模型（与其它 generate_style_* 脚本一致）
QWEN_MODEL = "qwen3-next-80b-a3b-instruct"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 与 annotation 页一致的五维 + 五档
CONTINUOUS_DIMS = ["FORMALITY", "POLITENESS", "FRIENDLINESS", "CERTAINTY", "EMOTIONAL_TONE"]
TIERS = ["extremely_low", "low", "mid", "high", "extremely_high"]
TIER_TO_VAL = {"extremely_low": 0.08, "low": 0.28, "mid": 0.50, "high": 0.73, "extremely_high": 0.93}

# 每个维度：仅用户问题（user_input）。传入 LLM 的只有「维度名=档位标签」，无描述、无 anchor
DIM_PROMPTS = {
    "FORMALITY": {
        "desc": "说话的书面感 / 口语感程度",
        "user_input": "你的论文写好了吗？什么时候能交终稿？",
    },
    "POLITENESS": {
        "desc": "措辞是否体现礼节和客气",
        "user_input": "能帮我看看这个报告哪里要改吗？",
    },
    "FRIENDLINESS": {
        "desc": "语气是亲近温暖还是疏远冷淡",
        "user_input": "没事我就随便说说，你忙吧。",
    },
    "CERTAINTY": {
        "desc": "说话者对自己所说内容的笃定程度",
        "user_input": "你还记得我们上次去游乐园的事吗？",
    },
    "EMOTIONAL_TONE": {
        "desc": "情绪表达的激烈程度，不区分正负",
        "user_input": "你今天过得怎么样？",
    },
}


def make_style_dict(target_dim: str, target_tier: str) -> dict:
    """目标维度=target_tier，其余维度=mid，EXPRESSION_MODE=0。仅用标签，无 anchor。"""
    d = {}
    for dim in CONTINUOUS_DIMS:
        d[dim] = TIER_TO_VAL[target_tier] if dim == target_dim else TIER_TO_VAL["mid"]
    d["EXPRESSION_MODE"] = 0
    return d


# 与 generate 一致：不传描述/回答目标，只传 style block（维度名=档位标签）
SYSTEM_TEMPLATE = """你是小雨。你正在和用户对话。

## 写作风格参数
{style_text}

## 回复规则
- 第一人称，口语自然，不要暴露是AI
- 回复一句话即可（15–40 字），不要太长
- 严禁括号动作描写、比喻堆叠、排比、押韵、散文化
- 回复直接输出，不要任何前缀或格式标记"""


async def gen_one(llm, dim: str, tier: str) -> dict:
    cfg = DIM_PROMPTS[dim]
    style_dict = make_style_dict(dim, tier)
    style_text = format_style_as_param_list(style_dict)
    system = SYSTEM_TEMPLATE.format(style_text=style_text)
    resp = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=cfg["user_input"]),
    ])
    return {"dim": dim, "tier": tier, "reply": (resp.content or "").strip()}


async def main() -> None:
    llm = get_llm(
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL,
        api_key=os.getenv("QWEN_API_KEY") or os.getenv("LTSR_GEN_API_KEY"),
        temperature=0.2,
    )
    results: list[dict] = []

    for dim in CONTINUOUS_DIMS:
        print(f"\n【{dim}】")
        for tier in TIERS:
            r = await gen_one(llm, dim, tier)
            results.append(r)
            print(f"  {tier:14s}: {r['reply']}")

    # 组装为 DIM_EXAMPLES 结构：{ DIM: { desc, EL, L, M, H, EH } }
    tier_key = {"extremely_low": "EL", "low": "L", "mid": "M", "high": "H", "extremely_high": "EH"}
    dim_examples = {}
    for dim in CONTINUOUS_DIMS:
        dim_examples[dim] = {"desc": DIM_PROMPTS[dim]["desc"]}
        for tier in TIERS:
            key = tier_key[tier]
            row = next((x for x in results if x["dim"] == dim and x["tier"] == tier), None)
            dim_examples[dim][key] = (row["reply"] if row else "")

    out_dir = ROOT / "scripts" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dim_examples_for_annotation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dim_examples": dim_examples,
                "raw_results": results,
                "dim_prompts": {k: {"user_input": v["user_input"]} for k, v in DIM_PROMPTS.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n已保存 → {out_path}")
    print("\n可把 dim_examples 复制到 static/annotation/index.html 的 DIM_EXAMPLES 中。")


if __name__ == "__main__":
    asyncio.run(main())

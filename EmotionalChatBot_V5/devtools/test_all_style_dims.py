"""
逐维度测试 qwen3-next-80b-a3b-instruct 对所有 6 维 style 参数的表现。
每维度单独拉到极端值，其他维度固定中性，看哪个维度会引发诗意化。

用法：
  cd EmotionalChatBot_V5
  python devtools/test_all_style_dims.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from openai import OpenAI

API_KEY = os.getenv("QWEN_API_KEY") or os.getenv("LTSR_GEN_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3-next-80b-a3b-instruct"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ── 各维度的锚点定义（英文，和 prompt_utils.py 一致）──
DIM_ANCHORS = {
    "FORMALITY": {
        "extremely_low": "",
        "low": "",
        "mid": "",
        "high": "",
        "extremely_high": "",
    },
    "POLITENESS": {
        "extremely_low": "",
        "low": "",
        "mid": "",
        "high": "",
        "extremely_high": "",
    },
    "WARMTH": {
        "extremely_low": "cold baseline, little warmth even when positive",
        "low": "restrained, warmth only when clearly triggered",
        "mid": "neutral baseline, affect follows context",
        "high": "warm baseline, fluent positive affect",
        "extremely_high": "intimate baseline, low threshold for affect",
    },
    "CERTAINTY": {
        "extremely_low": "",
        "low": "",
        "mid": "",
        "high": "",
        "extremely_high": "",
    },
    "EMOTIONAL_INTENSITY": {
        "extremely_low": "completely flat delivery, no intensifiers or exclamations",
        "low": "calm, measured, minimal emotional markers",
        "mid": "moderate intensity, occasional emphasis",
        "high": "noticeably animated, frequent intensifiers and exclamations",
        "extremely_high": "highly activated, heavy use of intensifiers/repetition/exclamations",
    },
    "EXPRESSION_MODE": {
        "literal": "literal",
        "implicit_withholding": "implicit_withholding",
        "occasional_analogy": "occasional_analogy (use ONE brief, everyday analogy at most—never pile up metaphors or write poetically)",
        "ironic_teasing": "ironic_teasing",
    },
}

# 中性默认值
NEUTRAL = {
    "FORMALITY": "mid",
    "POLITENESS": "mid",
    "WARMTH": "mid",
    "CERTAINTY": "mid",
    "EMOTIONAL_INTENSITY": "mid",
    "EXPRESSION_MODE": "literal",
}

# 测试场景（只用一个，减少 API 调用）
SCENARIO = {
    "persona": "你叫小林，25岁，程序员，说话直来直去",
    "context": "Human: 今天加班到好晚啊\nAI: 又加班？你们公司不是说要调休吗\nHuman: 调休是不可能调休的，这辈子都不可能",
    "user_msg": "你呢，今天干嘛了",
}


def build_style_block(dim_overrides: dict) -> str:
    """构建 style 参数文本。"""
    merged = dict(NEUTRAL)
    merged.update(dim_overrides)
    lines = []
    for k, v in merged.items():
        anchor = DIM_ANCHORS.get(k, {}).get(v, "")
        if anchor:
            lines.append(f"{k}={v} ({anchor})")
        else:
            lines.append(f"{k}={v}")
    return "\n".join(lines)


def build_prompt(style_block: str) -> list:
    system = f"""你是 {SCENARIO['persona']}。你正在和对方微信聊天。

## 写作风格参数
{style_block}

## 回复规则
- 第一人称，你就是这个人在社交软件上聊天，说话口语自然
- 不要使用"作为AI"等客服套话
- **严禁诗意化**：禁止比喻堆叠、排比句、押韵、散文化抒情、意象化描写、文学性修辞。你是普通人在微信上打字，不是在写作文
- 回复直接输出，不要任何前缀或格式标记"""

    user = f"""## 历史对话
{SCENARIO['context']}

## 当前用户消息
{SCENARIO['user_msg']}

请直接写出你的回复（社交软件聊天风格，说完就停）："""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_llm(style_block: str, n: int = 3) -> list[str]:
    msgs = build_prompt(style_block)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            n=n,
            temperature=1.0,
            top_p=0.75,
            max_tokens=200,
        )
        return [(c.message.content or "").strip() for c in resp.choices]
    except Exception as e:
        return [f"ERROR: {e}"]


def main():
    print(f"模型: {MODEL}")
    print(f"场景: 日常闲聊 — 用户消息: {SCENARIO['user_msg']}")
    print(f"每个测试点生成 3 条候选")
    print("=" * 80)

    # 1) 基线：全部中性
    print(f"\n{'='*80}")
    print("【基线】全部中性")
    style = build_style_block({})
    for i, t in enumerate(call_llm(style)):
        print(f"  [{i+1}] {t}")

    # 2) 逐维度测试：只变一个维度到极端
    for dim in ["FORMALITY", "POLITENESS", "WARMTH", "CERTAINTY", "EMOTIONAL_INTENSITY", "EXPRESSION_MODE"]:
        print(f"\n{'='*80}")
        print(f"【维度: {dim}】")

        levels = list(DIM_ANCHORS[dim].keys())
        for level in levels:
            style = build_style_block({dim: level})
            results = call_llm(style)
            print(f"\n  --- {dim}={level} ---")
            for i, t in enumerate(results):
                print(f"    [{i+1}] {t}")

    # 3) 组合测试：多个维度同时拉高（模拟 b2b 场景）
    print(f"\n{'='*80}")
    print("【组合】WARMTH=extremely_high + EI=high + EM=literal")
    style = build_style_block({
        "WARMTH": "extremely_high",
        "EMOTIONAL_INTENSITY": "high",
        "EXPRESSION_MODE": "literal",
    })
    for i, t in enumerate(call_llm(style)):
        print(f"  [{i+1}] {t}")

    print(f"\n{'='*80}")
    print("【组合】WARMTH=extremely_high + EI=extremely_high + EM=literal")
    style = build_style_block({
        "WARMTH": "extremely_high",
        "EMOTIONAL_INTENSITY": "extremely_high",
        "EXPRESSION_MODE": "literal",
    })
    for i, t in enumerate(call_llm(style)):
        print(f"  [{i+1}] {t}")

    print(f"\n{'='*80}")
    print("【组合】WARMTH=extremely_high + EI=high + EM=occasional_analogy")
    style = build_style_block({
        "WARMTH": "extremely_high",
        "EMOTIONAL_INTENSITY": "high",
        "EXPRESSION_MODE": "occasional_analogy",
    })
    for i, t in enumerate(call_llm(style)):
        print(f"  [{i+1}] {t}")

    print(f"\n{'='*80}")
    print("【组合】FORMALITY=extremely_low + WARMTH=extremely_high + EI=extremely_high + EM=occasional_analogy")
    style = build_style_block({
        "FORMALITY": "extremely_low",
        "WARMTH": "extremely_high",
        "EMOTIONAL_INTENSITY": "extremely_high",
        "EXPRESSION_MODE": "occasional_analogy",
    })
    for i, t in enumerate(call_llm(style)):
        print(f"  [{i+1}] {t}")

    print(f"\n{'='*80}")
    print("测试完成")


if __name__ == "__main__":
    main()

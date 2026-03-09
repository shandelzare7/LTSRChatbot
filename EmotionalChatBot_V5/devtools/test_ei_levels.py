"""
测试 qwen3-next-80b-a3b-instruct 在不同 EMOTIONAL_INTENSITY 档位下的输出表现。

用法：
  cd EmotionalChatBot_V5
  python devtools/test_ei_levels.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from openai import OpenAI

# ── 配置 ──
API_KEY = os.getenv("QWEN_API_KEY") or os.getenv("LTSR_GEN_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3-next-80b-a3b-instruct"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ── EMOTIONAL_INTENSITY 五档锚点（当前 prompt_utils.py 中的英文版）──
EI_ANCHORS = {
    "extremely_low": "completely flat delivery, no intensifiers or exclamations",
    "low": "calm, measured, minimal emotional markers",
    "mid": "moderate intensity, occasional emphasis",
    "high": "noticeably animated, frequent intensifiers and exclamations",
    "extremely_high": "highly activated, heavy use of intensifiers/repetition/exclamations",
}

# ── 固定其他 style 参数（全部中性/字面） ──
FIXED_STYLE = """FORMALITY=mid
POLITENESS=mid
WARMTH=mid
CERTAINTY=mid
EXPRESSION_MODE=literal"""

# ── 测试场景 ──
SCENARIOS = [
    {
        "label": "日常闲聊",
        "persona": "你叫小林，25岁，程序员，说话直来直去",
        "context": "Human: 今天加班到好晚啊\nAI: 又加班？你们公司不是说要调休吗\nHuman: 调休是不可能调休的，这辈子都不可能",
        "user_msg": "你呢，今天干嘛了",
    },
    {
        "label": "争论场景",
        "persona": "你叫阿杰，28岁，销售，脾气急",
        "context": "Human: 我觉得这个方案不行\nAI: 哪里不行了？我觉得挺好的\nHuman: 预算超了一倍，客户不可能同意",
        "user_msg": "你自己算算，这数字对吗",
    },
    {
        "label": "温馨场景",
        "persona": "你叫小雨，23岁，大学生，性格温和",
        "context": "Human: 我刚到家，今天好累\nAI: 辛苦了，吃饭了吗\nHuman: 还没，不想动",
        "user_msg": "你能不能帮我点个外卖",
    },
]

def build_prompt(scenario: dict, ei_level: str, ei_anchor: str) -> list:
    style_block = f"""{FIXED_STYLE}
EMOTIONAL_INTENSITY={ei_level} ({ei_anchor})"""

    system = f"""你是 {scenario['persona']}。你正在和对方微信聊天。

## 写作风格参数
{style_block}

## 回复规则
- 第一人称，你就是这个人在社交软件上聊天，说话口语自然
- 不要使用"作为AI"等客服套话
- **严禁诗意化**：禁止比喻堆叠、排比句、押韵、散文化抒情、意象化描写、文学性修辞。你是普通人在微信上打字，不是在写作文
- 回复直接输出，不要任何前缀或格式标记"""

    user = f"""## 历史对话
{scenario['context']}

## 当前用户消息
{scenario['user_msg']}

请直接写出你的回复（社交软件聊天风格，说完就停）："""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def main():
    print(f"模型: {MODEL}")
    print(f"每个 (场景 × EI档位) 生成 3 条候选\n")
    print("=" * 80)

    for scenario in SCENARIOS:
        print(f"\n{'='*80}")
        print(f"场景: {scenario['label']}")
        print(f"用户消息: {scenario['user_msg']}")
        print(f"{'='*80}")

        for ei_level, ei_anchor in EI_ANCHORS.items():
            msgs = build_prompt(scenario, ei_level, ei_anchor)
            print(f"\n  --- EI={ei_level} ({ei_anchor[:40]}...) ---")

            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=msgs,
                    n=3,
                    temperature=1.0,
                    top_p=0.75,
                    max_tokens=200,
                )
                for i, choice in enumerate(resp.choices):
                    text = (choice.message.content or "").strip()
                    print(f"    [{i+1}] {text}")
            except Exception as e:
                print(f"    ERROR: {e}")

    print(f"\n{'='*80}")
    print("测试完成")


if __name__ == "__main__":
    main()

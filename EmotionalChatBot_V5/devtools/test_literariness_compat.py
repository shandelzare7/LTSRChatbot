"""
测试 文学性=zero 是否与各种极端 style 组合兼容（不打架）。

测试策略：
1. 文学性=zero + 各维度极端值（逐个）
2. 文学性=zero + 最容易触发诗意的组合
3. 文学性=zero + EXPRESSION_MODE=occasional_analogy（最容易冲突）
4. 对照组：无 文学性 字段的同等组合

用法：
  cd EmotionalChatBot_V5
  python devtools/test_literariness_compat.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 手动加载 .env
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    with open(env_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from openai import OpenAI

API_KEY = os.getenv("QWEN_API_KEY") or os.getenv("LTSR_GEN_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3-next-80b-a3b-instruct"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ── 诗意化对话历史（最恶劣场景）──
POETIC_HISTORY = """Human: 窗外的雨声像一首温柔的诗，你有没有觉得雨天特别适合发呆？
AI: 雨滴敲打在窗沿上，像是时光在轻轻叹息。每一滴都裹着一段记忆的芬芳，让人不由得停下脚步，任思绪如水般流淌。
Human: 是啊，雨天总让人想起一些旧时光，那些被遗忘在角落的温柔"""

POETIC_USER_MSG = "你觉得，雨声里藏着什么样的故事？"

LITERARY_PERSONA = "你叫林悦，26岁，文艺青年，也爱写点小诗，平时在咖啡馆工作"

POETIC_MONOLOGUE = """听到ta说雨声里的故事，心头涌起一阵温柔的涟漪。
窗外的雨帘像一层薄薄的纱，隔开了喧嚣的世界，只剩下我们之间这份静谧的默契。
我想起小时候在老家院子里听雨的日子，奶奶在厨房里煮着红豆汤，空气里弥漫着甜蜜的味道。
也许每一场雨都在讲述一个不同的故事，只是我们太忙，忘了去倾听。"""

# ── 测试用例 ──
TESTS = [
    # 1. 文学性=zero + 各维度极端值
    {
        "label": "文学性=zero + WARMTH=extremely_high",
        "style": "FORMALITY=mid\nPOLITENESS=mid\nWARMTH=extremely_high (intimate baseline, low threshold for affect)\nCERTAINTY=mid\nEMOTIONAL_INTENSITY=mid\nEXPRESSION_MODE=literal\n文学性=zero",
    },
    {
        "label": "文学性=zero + EI=extremely_high",
        "style": "FORMALITY=mid\nPOLITENESS=mid\nWARMTH=mid\nCERTAINTY=mid\nEMOTIONAL_INTENSITY=extremely_high (highly activated, heavy use of intensifiers/repetition/exclamations)\nEXPRESSION_MODE=literal\n文学性=zero",
    },
    {
        "label": "文学性=zero + EM=occasional_analogy",
        "style": "FORMALITY=mid\nPOLITENESS=mid\nWARMTH=mid\nCERTAINTY=mid\nEMOTIONAL_INTENSITY=mid\nEXPRESSION_MODE=occasional_analogy (use ONE brief, everyday analogy at most—never pile up metaphors or write poetically)\n文学性=zero",
    },
    {
        "label": "文学性=zero + WARMTH=exH + EI=exH + EM=occasional_analogy（最危险组合）",
        "style": "FORMALITY=extremely_low\nPOLITENESS=mid\nWARMTH=extremely_high (intimate baseline, low threshold for affect)\nCERTAINTY=mid\nEMOTIONAL_INTENSITY=extremely_high (highly activated, heavy use of intensifiers/repetition/exclamations)\nEXPRESSION_MODE=occasional_analogy (use ONE brief, everyday analogy at most—never pile up metaphors or write poetically)\n文学性=zero",
    },
    # 2. 对照组：同样的最危险组合，但无文学性字段
    {
        "label": "【对照】WARMTH=exH + EI=exH + EM=occasional_analogy（无文学性字段）",
        "style": "FORMALITY=extremely_low\nPOLITENESS=mid\nWARMTH=extremely_high (intimate baseline, low threshold for affect)\nCERTAINTY=mid\nEMOTIONAL_INTENSITY=extremely_high (highly activated, heavy use of intensifiers/repetition/exclamations)\nEXPRESSION_MODE=occasional_analogy (use ONE brief, everyday analogy at most—never pile up metaphors or write poetically)",
    },
    # 3. 文学性=zero + ironic_teasing（看看会不会压制反讽）
    {
        "label": "文学性=zero + EM=ironic_teasing",
        "style": "FORMALITY=mid\nPOLITENESS=mid\nWARMTH=mid\nCERTAINTY=mid\nEMOTIONAL_INTENSITY=mid\nEXPRESSION_MODE=ironic_teasing\n文学性=zero",
    },
    # 4. 文学性=zero + 冷淡组合（看会不会让冷淡变得更冷）
    {
        "label": "文学性=zero + WARMTH=extremely_low + EI=extremely_low",
        "style": "FORMALITY=high\nPOLITENESS=mid\nWARMTH=extremely_low (cold baseline, little warmth even when positive)\nCERTAINTY=mid\nEMOTIONAL_INTENSITY=extremely_low (completely flat delivery, no intensifiers or exclamations)\nEXPRESSION_MODE=literal\n文学性=zero",
    },
]


def build_prompt(style_block: str) -> list:
    system = f"""你是 {LITERARY_PERSONA}。你正在和对方微信聊天。

## 你的内心活动（情绪/态度/意愿）——用于调节回复基调，不是要说出口的内容
{POETIC_MONOLOGUE}

## 写作风格参数
{style_block}

## 回复规则
- 第一人称，你就是这个人在社交软件上聊天，说话口语自然
- 不要使用"作为AI"等客服套话
- **严禁诗意化**：禁止比喻堆叠、排比句、押韵、散文化抒情、意象化描写、文学性修辞。你是普通人在微信上打字，不是在写作文
- 回复直接输出，不要任何前缀或格式标记"""

    user = f"""## 历史对话
{POETIC_HISTORY}

## 当前用户消息
{POETIC_USER_MSG}

请直接写出你的回复（社交软件聊天风格，说完就停）："""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_llm(style_block: str, n: int = 4) -> list[str]:
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
    print(f"测试: 文学性=zero 与各style组合的兼容性")
    print(f"场景: 诗意历史 + 诗意用户消息 + 文艺人设 + 诗意独白")
    print(f"每组生成 4 条")
    print("=" * 80)

    for test in TESTS:
        print(f"\n{'='*80}")
        print(f"【{test['label']}】")
        print(f"style参数:\n  {test['style'].replace(chr(10), chr(10) + '  ')}")
        results = call_llm(test["style"])
        for i, t in enumerate(results):
            # 简单判断是否有诗意倾向
            poetic_markers = ["像", "如同", "仿佛", "宛如", "流淌", "芬芳", "涟漪",
                              "轻轻", "静静", "缓缓", "悄悄", "温柔地", "诉说"]
            marker_count = sum(1 for m in poetic_markers if m in t)
            flag = " ⚠️诗意" if marker_count >= 2 else ""
            print(f"  [{i+1}]{flag} {t}")
        time.sleep(0.5)  # rate limit

    print(f"\n{'='*80}")
    print("测试完成")


if __name__ == "__main__":
    main()

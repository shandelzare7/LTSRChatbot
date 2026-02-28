"""
用 qwen3-next-80b-a3b-instruct 测试 6 维风格（formality / politeness / warmth / certainty / chat_markers / expression_mode），
每维每档至少 10 个样本。

五档取值（与 prompt_utils 一致）：
  extremely_low: 0.08   low: 0.285   mid: 0.51   high: 0.735   extremely_high: 0.93
EXPRESSION_MODE 为 4 档：0=literal_direct, 1=literal_hedged, 2=metaphor_imagery, 3=light_teasing

运行：
  cd EmotionalChatBot_V5
  python devtools/test_formality_politeness_qwen.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# 项目根
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 加载 .env
_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)

from langchain_core.messages import HumanMessage

# 五档数值（对应 extremely_low, low, mid, high, extremely_high）
FIVE_LEVELS = (0.08, 0.285, 0.51, 0.735, 0.93)
LEVEL_NAMES = ("extremely_low", "low", "mid", "high", "extremely_high")

# EXPRESSION_MODE 四档
EXPRESSION_LEVELS = (0, 1, 2, 3)
EXPRESSION_NAMES = ("literal_direct", "literal_hedged", "metaphor_imagery", "light_teasing")

# 每种（维度+档位）的样本数
SAMPLES_PER_LEVEL = 10

# 10 条不同用户消息 + 对应简短独白，用于多样本
USER_MESSAGES = [
    "今天天气不错，你周末有什么安排？",
    "昨天那部电影你看了吗？我觉得一般。",
    "最近工作好累，想出去散散心。",
    "你平时周末都干嘛呀？",
    "明天有空吗？想约你喝杯咖啡。",
    "这首歌你听过没？推荐一下。",
    "我可能要换工作了，有点纠结。",
    "晚上吃啥？有没有推荐？",
    "你上次说的那家店在哪来着？",
    "周末一起打球不？",
]
MONOLOGUES = [
    "对方随便问问周末安排，正常接话就好，可以简单说说自己的打算。",
    "对方在聊电影观感，可以接话说说自己看法，不必较真。",
    "对方说工作累想散心，可以表达一点共鸣或给点轻松建议。",
    "对方问周末干嘛，正常分享自己的日常就行。",
    "对方约咖啡，愿意的话自然答应，顺便聊两句。",
    "对方在聊歌，接话推荐或说说自己的口味。",
    "对方在说换工作纠结，可以倾听为主，适当回应。",
    "对方问晚饭推荐，随口推荐或一起纠结都行。",
    "对方问店址，想起来就告诉，想不起来就说忘了。",
    "对方约打球，乐意就答应，不乐意就婉拒或改别的。",
]

BOT_NAME = "小测"
USER_NAME = "对方"

# 6 维：前 5 维为五档连续值，第 6 维为 EXPRESSION_MODE 0-3
DIMENSIONS_5 = ("FORMALITY", "POLITENESS", "WARMTH", "CERTAINTY", "CHAT_MARKERS")
DIMENSION_EXPRESSION = "EXPRESSION_MODE"


def _minimal_state(style_overrides: dict, user_msg: str, monologue: str) -> dict:
    """构建仅够 generate 节点使用的 minimal state。"""
    return {
        "user_input": user_msg,
        "chat_buffer": [HumanMessage(content=user_msg)],
        "bot_basic_info": {"name": BOT_NAME},
        "user_basic_info": {"name": USER_NAME},
        "bot_persona": "性格比较随和，说话自然。",
        "inner_monologue": monologue,
        "monologue_extract": {"selected_content_move_ids": []},
        "conversation_momentum": 0.5,
        "style": {
            "FORMALITY": 0.5,
            "POLITENESS": 0.5,
            "WARMTH": 0.5,
            "CERTAINTY": 0.5,
            "CHAT_MARKERS": 0.5,
            "EXPRESSION_MODE": 0,
            **style_overrides,
        },
    }


async def run_one(state: dict, generate_node) -> str:
    """跑一次 generate，返回第一个候选的文本。"""
    out = await generate_node(state)
    candidates = (out or {}).get("generation_candidates") or []
    if not candidates:
        return "[无候选]"
    return (candidates[0].get("text") or "").strip() or "[空]"


async def run_dimension_5level(
    dim: str,
    generate_node,
    samples_per_level: int = SAMPLES_PER_LEVEL,
) -> list:
    """某维五档，每档 samples_per_level 条样本。返回 [(level_name, value, [(user_msg, reply), ...]), ...]"""
    results = []
    n = min(samples_per_level, len(USER_MESSAGES))
    for val, name in zip(FIVE_LEVELS, LEVEL_NAMES):
        samples = []
        for i in range(n):
            msg = USER_MESSAGES[i % len(USER_MESSAGES)]
            mono = MONOLOGUES[i % len(MONOLOGUES)]
            state = _minimal_state({dim: val}, msg, mono)
            reply = await run_one(state, generate_node)
            samples.append((msg, reply))
            await asyncio.sleep(0.15)
        results.append((name, val, samples))
    return results


async def run_dimension_expression(
    generate_node,
    samples_per_level: int = SAMPLES_PER_LEVEL,
) -> list:
    """EXPRESSION_MODE 四档，每档 samples_per_level 条样本。"""
    results = []
    n = min(samples_per_level, len(USER_MESSAGES))
    for val, name in zip(EXPRESSION_LEVELS, EXPRESSION_NAMES):
        samples = []
        for i in range(n):
            msg = USER_MESSAGES[i % len(USER_MESSAGES)]
            mono = MONOLOGUES[i % len(MONOLOGUES)]
            state = _minimal_state({"EXPRESSION_MODE": val}, msg, mono)
            reply = await run_one(state, generate_node)
            samples.append((msg, reply))
            await asyncio.sleep(0.15)
        results.append((name, val, samples))
    return results


async def main():
    from utils.yaml_loader import load_llm_models_config
    from app.services.llm import get_llm
    from app.nodes.generate import create_generate_node

    cfg = load_llm_models_config()
    gen_cfg = (cfg.get("generate") or {}) if isinstance(cfg.get("generate"), dict) else {}
    model = (gen_cfg.get("model") or "").strip() or "qwen-plus-2024-12-20"
    base_url = (gen_cfg.get("base_url") or "").strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key = (os.getenv("LTSR_GEN_API_KEY") or os.getenv("QWEN_API_KEY") or "").strip()
    if not api_key and "qwen" in model.lower():
        api_key = (os.getenv("QWEN_API_KEY") or "").strip()

    model = "qwen3-next-80b-a3b-instruct"
    if "qwen" in model.lower():
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not api_key:
            api_key = (os.getenv("QWEN_API_KEY") or "").strip()
    print(f"使用模型: {model}")
    print(f"base_url: {base_url}")
    print(f"API Key: {'已设置' if api_key else '未设置'}")
    print(f"每档样本数: {SAMPLES_PER_LEVEL}（用户消息池共 {len(USER_MESSAGES)} 条）")
    if not api_key:
        print("请设置 QWEN_API_KEY 或 LTSR_GEN_API_KEY")
        return

    llm_gen = get_llm(
        role="fast",
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        temperature=float(gen_cfg.get("temperature", 1.0)),
        top_p=float(gen_cfg.get("top_p", 0.95)),
        presence_penalty=float(gen_cfg.get("presence_penalty", 0.3)),
        n=1,
    )
    generate_node = create_generate_node(llm_gen)

    all_results = {}

    for dim in DIMENSIONS_5:
        print(f"\n{'='*60}\n{dim} 五档 × {SAMPLES_PER_LEVEL} 样本\n{'='*60}")
        results = await run_dimension_5level(dim, generate_node, SAMPLES_PER_LEVEL)
        all_results[dim] = [(name, val, samples) for name, val, samples in results]
        for name, val, samples in results:
            print(f"  [{name} ({val})] {len(samples)} 条")
            for msg, reply in samples[:2]:
                print(f"    用户: {msg[:30]}… → {reply[:50]}…")
            if len(samples) > 2:
                print(f"    ... 共 {len(samples)} 条")

    print(f"\n{'='*60}\n{DIMENSION_EXPRESSION} 四档 × {SAMPLES_PER_LEVEL} 样本\n{'='*60}")
    results = await run_dimension_expression(generate_node, SAMPLES_PER_LEVEL)
    all_results[DIMENSION_EXPRESSION] = [(name, val, samples) for name, val, samples in results]
    for name, val, samples in results:
        print(f"  [{name} ({val})] {len(samples)} 条")
        for msg, reply in samples[:2]:
            print(f"    用户: {msg[:30]}… → {reply[:50]}…")
        if len(samples) > 2:
            print(f"    ... 共 {len(samples)} 条")

    # 写入 JSON（便于后续分析）
    out_json = ROOT / "devtools" / "style_6d_qwen_results.json"
    serialized = {}
    for dim, level_list in all_results.items():
        serialized[dim] = []
        for name, val, samples in level_list:
            serialized[dim].append({
                "level": name,
                "value": val,
                "samples": [{"user": u, "reply": r} for u, r in samples],
            })
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"model": model, "samples_per_level": SAMPLES_PER_LEVEL, "dimensions": serialized}, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 结果已写入: {out_json}")

    # 写入可读 txt
    out_txt = ROOT / "devtools" / "style_6d_qwen_results.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Model: {model}\nSamples per level: {SAMPLES_PER_LEVEL}\n\n")
        for dim, level_list in all_results.items():
            f.write(f"\n{'='*60}\n{dim}\n{'='*60}\n\n")
            for name, val, samples in level_list:
                f.write(f"[{name} (value={val})]\n")
                for i, (u, r) in enumerate(samples, 1):
                    f.write(f"  {i}. 用户: {u}\n     回复: {r}\n")
                f.write("\n")
    print(f"TXT 结果已写入: {out_txt}")


if __name__ == "__main__":
    asyncio.run(main())

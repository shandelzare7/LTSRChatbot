"""
为标注 HTML 的 DIM_EXAMPLES 生成五档梯度示例。
使用 Qwen（dashscope）模型，每维度每档生成 3 个候选，人工（或脚本）选最优。

运行方式：
    python scripts/generate_dim_examples_for_html.py

输出：
    scripts/output/dim_examples_for_html.json   （候选集）
    标准输出打印可直接粘贴到 index.html 的 DIM_EXAMPLES 块
"""

import asyncio, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ── Qwen 配置 ──────────────────────────────────────────────────────────────────
QWEN_API_KEY  = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL    = "qwen-plus"          # qwen-plus 更稳定，可改为 qwen-turbo

# 用户说的那句话（上下文锚点，全程不变）
USER_INPUT = "最近工作压力好大，感觉快撑不住了"

# ── 维度定义 ────────────────────────────────────────────────────────────────────
TIERS_EN = {
    "EL": "extremely_low",
    "L":  "low",
    "M":  "mid",
    "H":  "high",
    "EH": "extremely_high",
}

DIMS = {
    "FORMALITY":      {"zh": "正式",    "desc": "说话的书面感 / 口语感程度",
                       "input": "最近工作压力好大，感觉快撑不住了"},
    "POLITENESS":     {"zh": "礼貌",    "desc": "措辞是否体现礼节和客气",
                       "input": "最近工作压力好大，感觉快撑不住了"},
    "FRIENDLINESS":   {"zh": "友善",    "desc": "语气是亲近温暖还是疏远冷淡",
                       "input": "最近工作压力好大，感觉快撑不住了"},
    # 确定：用"要我意见"的场景，更容易体现笃定/犹豫梯度
    "CERTAINTY":      {"zh": "确定",    "desc": "说话者对自己所说内容的笃定程度",
                       "input": "我在想要不要辞职，你觉得我该怎么办？"},
    # 情感浓度：用平静事件，避免原情境本身带情绪"污染"梯度
    "EMOTIONAL_TONE": {"zh": "情感浓度", "desc": "情绪表达的激烈程度，不区分正负",
                       "input": "我今天去公园散步了"},
}

TIER_ZH = {"EL": "极低", "L": "低", "M": "中", "H": "高", "EH": "极高"}

N_CANDIDATES = 3   # 每档生成几个候选

SYSTEM_PROMPT = """You are having a conversation on a social messaging app. The user just said:
「{user_input}」

Reply to this message. Your reply must strictly follow the style parameter below:
{dim_en}={tier_en}

Output the reply directly in Chinese. No prefix, no explanation."""

USER_PROMPT = """Generate {n} different replies, one per line, no numbering:"""


def get_llm():
    if not QWEN_API_KEY:
        raise RuntimeError("QWEN_API_KEY 未设置，请检查 .env 文件")
    return ChatOpenAI(
        model=QWEN_MODEL,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        temperature=0.85,
        max_retries=2,
        timeout=60,
    )


async def gen_candidates(llm, dim_key: str, tier_key: str) -> list[str]:
    dim = DIMS[dim_key]
    sys_msg = SYSTEM_PROMPT.format(
        user_input=dim.get("input", USER_INPUT),
        dim_en=dim_key,
        tier_en=TIERS_EN[tier_key],
    )
    usr_msg = USER_PROMPT.format(n=N_CANDIDATES)
    resp = await llm.ainvoke([
        SystemMessage(content=sys_msg),
        HumanMessage(content=usr_msg),
    ])
    lines = [l.strip() for l in resp.content.strip().splitlines() if l.strip()]
    # 去掉可能出现的序号前缀 "1. " 或 "1、"
    cleaned = []
    for l in lines:
        if len(l) > 2 and l[0].isdigit() and l[1] in '.、）)':
            l = l[2:].strip()
        cleaned.append(l)
    return cleaned[:N_CANDIDATES]


async def main():
    llm = get_llm()
    TIER_ORDER = ["EL", "L", "M", "H", "EH"]

    tasks = []
    keys = []
    for dim_key in DIMS:
        for tier_key in TIER_ORDER:
            tasks.append(gen_candidates(llm, dim_key, tier_key))
            keys.append((dim_key, tier_key))

    print(f"生成 {len(tasks)} 组样本（{N_CANDIDATES} 候选/组），请稍候…\n")
    results_raw = await asyncio.gather(*tasks, return_exceptions=True)

    # 整理结果
    output: dict[str, dict] = {}
    for (dim_key, tier_key), res in zip(keys, results_raw):
        if dim_key not in output:
            output[dim_key] = {}
        if isinstance(res, Exception):
            print(f"  ⚠ {dim_key} {tier_key} 生成失败：{res}")
            output[dim_key][tier_key] = ["（生成失败）"]
        else:
            output[dim_key][tier_key] = res

    # ── 打印候选集 ──────────────────────────────────────────────────────────────
    print("=" * 72)
    for dim_key, dim_cfg in DIMS.items():
        print(f"\n【{dim_cfg['zh']} — {dim_cfg['desc']}】（输入：「{dim_cfg.get('input', USER_INPUT)}」）")
        for tier_key in TIER_ORDER:
            candidates = output[dim_key].get(tier_key, [])
            print(f"  {TIER_ZH[tier_key]:<3s}  候选：")
            for i, c in enumerate(candidates, 1):
                print(f"    {i}. {c}")

    # ── 打印可直接粘贴的 JS 代码块（取每档第 1 个候选）──────────────────────────
    print("\n\n" + "=" * 72)
    print("// ── 可直接粘贴到 index.html 的 DIM_EXAMPLES（取每档第1候选）─────────")
    print("const DIM_EXAMPLES = {")
    for dim_key, dim_cfg in DIMS.items():
        print(f"  {dim_key}: {{")
        print(f"    desc: '{dim_cfg['desc']}',")
        for tier_key in TIER_ORDER:
            candidates = output[dim_key].get(tier_key, [""])
            best = candidates[0] if candidates else ""
            # 转义单引号
            best = best.replace("'", "\\'")
            print(f"    {tier_key}:  '{best}',")
        print("  },")
    print("};")

    # ── 保存候选集 JSON ─────────────────────────────────────────────────────────
    out_path = os.path.join(ROOT, "scripts", "output", "dim_examples_for_html.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_obj = {
        "user_input": USER_INPUT,
        "model": QWEN_MODEL,
        "dims": {
            dim_key: {
                "desc": dim_cfg["desc"],
                "tiers": output[dim_key],
            }
            for dim_key, dim_cfg in DIMS.items()
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=2)
    print(f"\n候选集已保存 → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())

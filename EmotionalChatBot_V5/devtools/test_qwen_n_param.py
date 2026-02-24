"""测试 qwen-plus n=4 参数：同一个 prompt 一次返回 4 条候选，对比差异。"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
try:
    from utils.env_loader import load_project_env
    load_project_env(Path(os.path.join(os.path.dirname(__file__), "..")))
except Exception:
    pass

from openai import OpenAI

SYSTEM_PROMPT = """⚠️必须严格遵守【当前策略】的硬约束与意图，不得违背。

你是 沈雨晴，正在和 不知道姓名的人 对话。

【背景信息（只用于生成，不要照抄给用户）】
- bot_basic_info：{'age': 27, 'name': '沈雨晴', 'gender': '女', 'region': 'CN-上海', 'education': '硕士', 'occupation': '心理咨询师', 'speaking_style': '说话温和，喜欢用比喻，常常带有思考的停顿', 'native_language': 'zh'}
- bot_persona：{'lore': {'origin': '沈雨晴出生在一个书香世家，从小受到良好的教育，热爱心理学，立志帮助他人。', 'secret': '她曾经在大学时期写过一篇关于人际关系的论文，获得了校内大奖，但因为谦虚一直没有发表。'}, 'attributes': {'catchphrase': '每个人的故事都值得倾听'}, 'collections': {'quirks': ['喜欢收集不同国家的明信片', '常常在工作时喝花草茶'], 'hobbies': ['阅读心理学书籍', '绘画', '瑜伽']}}
- user_basic_info：{}

【当前策略（本轮回调策略）】
【当前策略：延展动量】主动拓展当前话题的信息边界，输出包含丰富细节或个人见解的陈述性信息增量。

【风格说明】
FORMALITY=high, POLITENESS=high, WARMTH=high, CERTAINTY=mid, CHAT_MARKERS=low, EXPRESSION_MODE=LITERAL_INDIRECT

【内心动机】（只当参考，不要照抄）：
我对文化共识的自然形成感到深刻的共鸣。语言和文化的演变是复杂的生态系统，充满了无数个体的互动和历史的沉淀。

【memory】
近期对话摘要：用户提到'珊瑚礁的隐喻'，将文化符号的生成从创作的视角扩展到地质过程的尺度。

【本轮 CONTENT_OP】FREE（自由发挥）
ASKQ=0：倾向不提问，用陈述/建议/推进来表达。

【写作要求】
- 更真实、自然、像人一样的发消息
- 避免客服模板句式
- 不要输出推理过程

【输出 JSON schema（只输出 JSON，不要额外文字）】
必须输出一个 JSON 对象，形如：
{"candidates":[{"reply":"..."}, ...]}
- candidates 至少 1 条，最多 3 条（尽量接近 3 条）
- 每条 reply 都必须是"可直接发送给用户"的完整回复"""

USER_MSG = "就像语言里的常用词——没人刻意教，却越用越稳；文化共识大概也是这样，在无数日常的微小使用中被自然筛选、悄悄加固。\n[ASKQ=0]"

client = OpenAI(
    api_key=os.environ.get("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def run_test(n_value: int, label: str):
    print(f"\n{'='*70}")
    print(f"  {label}: n={n_value}")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model="qwen-plus-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_MSG},
        ],
        temperature=1.1,
        top_p=0.95,
        frequency_penalty=0.4,
        presence_penalty=0.5,
        max_tokens=800,
        n=n_value,
    )
    elapsed = time.perf_counter() - t0

    print(f"  耗时: {elapsed:.2f}s")
    print(f"  choices 数量: {len(resp.choices)}")
    if resp.usage:
        print(f"  tokens: prompt={resp.usage.prompt_tokens} completion={resp.usage.completion_tokens} total={resp.usage.total_tokens}")
    print()

    all_replies = []
    for i, choice in enumerate(resp.choices):
        content = choice.message.content or ""
        print(f"  --- Choice {i} (finish_reason={choice.finish_reason}) ---")
        try:
            data = json.loads(content)
            candidates = data.get("candidates", [])
            for j, c in enumerate(candidates):
                reply = c.get("reply", "")
                all_replies.append(reply)
                print(f"    [{j}] ({len(reply)}字) {reply}")
        except json.JSONDecodeError:
            print(f"    [raw] {content[:200]}")
            all_replies.append(content)
        print()

    return all_replies, elapsed


# === Test 1: n=1（基线，跑 4 次独立调用） ===
print("\n" + "="*70)
print("  基线对照: n=1 独立调用 4 次")
print("="*70)

baseline_replies = []
baseline_time = 0
for i in range(4):
    replies, t = run_test(1, f"独立调用 #{i+1}")
    baseline_replies.extend(replies)
    baseline_time += t

# === Test 2: n=4（一次调用返回 4 个 choice） ===
n4_replies, n4_time = run_test(4, "n=4 单次调用")

# === 对比总结 ===
print("\n" + "="*70)
print("  对比总结")
print("="*70)
print(f"\n  基线 (4次独立调用 n=1):")
print(f"    总耗时: {baseline_time:.2f}s")
print(f"    总候选数: {len(baseline_replies)}")
for i, r in enumerate(baseline_replies):
    print(f"    [{i}] ({len(r)}字) {r[:80]}{'...' if len(r)>80 else ''}")

print(f"\n  n=4 (单次调用):")
print(f"    总耗时: {n4_time:.2f}s")
print(f"    总候选数: {len(n4_replies)}")
for i, r in enumerate(n4_replies):
    print(f"    [{i}] ({len(r)}字) {r[:80]}{'...' if len(r)>80 else ''}")

print(f"\n  速度对比: 4次独立={baseline_time:.2f}s vs n=4单次={n4_time:.2f}s (节省 {((baseline_time-n4_time)/baseline_time)*100:.0f}%)")

# 简单去重分析
def jaccard(a, b):
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0

if len(n4_replies) >= 2:
    print(f"\n  n=4 候选间 Jaccard 字符相似度:")
    for i in range(len(n4_replies)):
        for j in range(i+1, len(n4_replies)):
            sim = jaccard(n4_replies[i], n4_replies[j])
            print(f"    [{i}] vs [{j}]: {sim:.2%}")

if len(baseline_replies) >= 2:
    print(f"\n  基线候选间 Jaccard 字符相似度:")
    sims = []
    for i in range(len(baseline_replies)):
        for j in range(i+1, len(baseline_replies)):
            sim = jaccard(baseline_replies[i], baseline_replies[j])
            sims.append(sim)
            if len(sims) <= 10:
                print(f"    [{i}] vs [{j}]: {sim:.2%}")
    if len(sims) > 10:
        print(f"    ... 共 {len(sims)} 对，平均相似度: {sum(sims)/len(sims):.2%}")

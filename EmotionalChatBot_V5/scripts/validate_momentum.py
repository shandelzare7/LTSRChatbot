"""
第四章验证脚本一：Momentum 目标生成验证
==========================================

Part A：公式正确性验证（完全自动化）
  直接调用 state_update._compute_momentum()，构造覆盖关键边界的测试用例，
  验证每组输入产生符合预期的 M_t 值和目标类别。

Part B：人工标注导出
  从测试用例生成场景描述，供标注员判断"应收敛/维持/延长"。

运行方式：
  cd EmotionalChatBot_V5
  python scripts/validate_momentum.py

输出：
  - 控制台打印测试结果摘要
  - momentum_validation_results.csv（所有测试用例的完整数据）
  - momentum_annotation_tasks.json（供人工标注的场景描述）
"""
from __future__ import annotations

import csv
import json
import sys
import os

# 把项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.nodes.pipeline.state_update import _compute_momentum


# ─────────────────────────────────────────────────────────────
# 目标类别划分（来自 generate.py:_momentum_to_direction()）
# ─────────────────────────────────────────────────────────────
def momentum_to_category(m: float) -> str:
    if m >= 0.80:
        return "延长（饱满）"
    elif m >= 0.60:
        return "延长（延展）"
    elif m >= 0.40:
        return "维持（对等）"
    else:
        return "收敛"   # 理论上不会出现，floor=0.4


def _build_mock_state(
    engagement: float,
    topic_appeal: float,
    hostility: float,
    busyness: float,
    arousal: float,
    turn_count: int,
    m_prev: float = 0.65,
    attractiveness: float = 0.5,
    liking: float = 0.5,
    closeness: float = 0.4,
) -> dict:
    """构造 _compute_momentum 所需的最小状态字典。"""
    return {
        "detection": {
            "engagement_level": engagement,
            "hostility_level": hostility,
        },
        "monologue_extract": {
            "topic_appeal": topic_appeal,
        },
        "mood_state": {
            "arousal": arousal,
            "busyness": busyness,
        },
        "relationship_state": {
            "attractiveness": attractiveness,
            "liking": liking,
            "closeness": closeness,
        },
        "conversation_momentum": m_prev,
        "turn_count_in_session": turn_count,
    }


# ─────────────────────────────────────────────────────────────
# 测试用例定义
# 每组：(desc, state_kwargs, expected_category_hint)
# expected_category_hint 是人工预期，用于 Pass/Fail 判断
# ─────────────────────────────────────────────────────────────
TEST_CASES = [
    # ── 典型"延长"情景 ──────────────────────────────────────
    {
        "desc": "高投入+高话题吸引力+无敌意→应延长",
        "state": dict(engagement=9.0, topic_appeal=8.5, hostility=0.0,
                      busyness=0.2, arousal=0.4, turn_count=5, m_prev=0.70),
        "expected_gte": 0.75,
        "expected_category": "延长",
    },
    {
        "desc": "高投入+中话题+轻微敌意→应延长（轻度减弱）",
        "state": dict(engagement=8.0, topic_appeal=6.0, hostility=2.0,
                      busyness=0.3, arousal=0.2, turn_count=8, m_prev=0.68),
        "expected_gte": 0.65,
        "expected_category": "延长",
    },
    {
        "desc": "中投入+低话题吸引力+零敌意→应延长（边界）",
        "state": dict(engagement=6.0, topic_appeal=3.0, hostility=0.0,
                      busyness=0.2, arousal=0.1, turn_count=5, m_prev=0.75),
        "expected_gte": 0.60,
        "expected_category": "延长或维持",
    },

    # ── 典型"维持"情景 ──────────────────────────────────────
    {
        "desc": "中投入+中话题+零敌意→维持",
        "state": dict(engagement=5.0, topic_appeal=5.0, hostility=0.0,
                      busyness=0.4, arousal=0.0, turn_count=10, m_prev=0.60),
        "expected_range": (0.50, 0.75),
        "expected_category": "维持",
    },
    {
        "desc": "中投入+中话题+繁忙=0.6→维持（繁忙压天花板）",
        "state": dict(engagement=5.5, topic_appeal=5.5, hostility=0.0,
                      busyness=0.6, arousal=0.0, turn_count=5, m_prev=0.60),
        "expected_range": (0.45, 0.70),
        "expected_category": "维持",
    },

    # ── 典型"收敛"情景 ──────────────────────────────────────
    {
        "desc": "低投入+敌意=7+低话题→收敛",
        "state": dict(engagement=2.0, topic_appeal=2.0, hostility=7.0,
                      busyness=0.5, arousal=-0.3, turn_count=5, m_prev=0.55),
        "expected_lte": 0.52,
        "expected_category": "收敛或维持（偏低）",
    },
    {
        "desc": "高繁忙=0.9+中投入→繁忙天花板压低M",
        "state": dict(engagement=6.0, topic_appeal=6.0, hostility=0.0,
                      busyness=0.9, arousal=0.0, turn_count=5, m_prev=0.70),
        "expected_lte": 0.56,   # ceiling=100*(1-0.9*0.5)=55 → M=0.55
        "expected_category": "收敛或维持（繁忙限制）",
    },
    {
        "desc": "长对话疲劳惩罚：第30轮，中等输入",
        "state": dict(engagement=5.5, topic_appeal=5.5, hostility=0.0,
                      busyness=0.3, arousal=0.0, turn_count=30, m_prev=0.65),
        "expected_lte": 0.62,
        "expected_category": "维持偏低（疲劳惩罚）",
    },

    # ── 边界与特殊情景 ─────────────────────────────────────
    {
        "desc": "floor保底：极低输入但不应跌破0.4",
        "state": dict(engagement=0.0, topic_appeal=0.0, hostility=10.0,
                      busyness=0.9, arousal=-0.8, turn_count=40, m_prev=0.40),
        "expected_gte": 0.40,
        "expected_category": "收敛（floor保底）",
    },
    {
        "desc": "高唤醒放大效应：同等投入+arousal=0.8",
        "state": dict(engagement=6.0, topic_appeal=6.0, hostility=0.0,
                      busyness=0.2, arousal=0.8, turn_count=5, m_prev=0.65),
        "expected_gte": 0.70,
        "expected_category": "延长（情绪放大）",
    },
    {
        "desc": "低唤醒抑制：同等投入+arousal=-0.6",
        "state": dict(engagement=6.0, topic_appeal=6.0, hostility=0.0,
                      busyness=0.2, arousal=-0.6, turn_count=5, m_prev=0.65),
        "expected_lte": 0.70,
        "expected_category": "维持（情绪抑制）",
    },
    {
        "desc": "EMA惯性：极高输入但前一轮M低→不会骤升",
        "state": dict(engagement=10.0, topic_appeal=10.0, hostility=0.0,
                      busyness=0.0, arousal=0.5, turn_count=1, m_prev=0.40),
        "expected_range": (0.44, 0.72),   # EMA alpha=0.15，历史权重0.85
        "expected_category": "惯性：不骤升到上限",
    },
    {
        "desc": "吸引力高关系基值提升M",
        "state": dict(engagement=5.0, topic_appeal=5.0, hostility=0.0,
                      busyness=0.3, arousal=0.0, turn_count=5, m_prev=0.60,
                      attractiveness=0.9, liking=0.8, closeness=0.7),
        "expected_gte": 0.60,
        "expected_category": "维持偏高（高吸引力关系基值）",
    },
]


# ─────────────────────────────────────────────────────────────
# 场景描述生成（供人工标注）
# ─────────────────────────────────────────────────────────────
SCENARIO_TEMPLATES = {
    "高投入+高话题吸引力+无敌意→应延长": (
        "用户消息：「哇真的吗？那你当时是怎么处理的，后来呢？」"
        "（长消息，主动追问细节，明显投入，话题是你们都感兴趣的经历分享）"
    ),
    "高投入+中话题+轻微敌意→应延长（轻度减弱）": (
        "用户消息：「嗯，你说的有道理……但我觉得也不全对，不过算了，继续吧」"
        "（轻度不认同但仍愿意继续，整体投入度高）"
    ),
    "中投入+低话题吸引力+零敌意→应延长（边界）": (
        "用户消息：「哦好吧，那就这样」（话题已接近尾声，用户回复简短但无敌意）"
    ),
    "中投入+中话题+零敌意→维持": (
        "用户消息：「还好啊，今天一般，你呢」（正常闲聊，没有特别高的投入或关闭信号）"
    ),
    "中投入+中话题+繁忙=0.6→维持（繁忙压天花板）": (
        "用户消息：「嗯那挺好的！」（系统当前繁忙度较高，对话继续但节奏应放缓）"
    ),
    "低投入+敌意=7+低话题→收敛": (
        "用户消息：「你懂什么，烦死了」（明显烦躁、攻击性，话题无法继续推进）"
    ),
    "高繁忙=0.9+中投入→繁忙天花板压低M": (
        "用户消息：「最近怎么样？」（正常问候，但系统当前非常忙碌，适合简短回应）"
    ),
    "长对话疲劳惩罚：第30轮，中等输入": (
        "（已聊了30轮）用户消息：「哈哈，好吧」（对话进行了很久，自然疲劳）"
    ),
    "floor保底：极低输入但不应跌破0.4": (
        "用户消息：「去死」（极端敌意，但系统仍应维持最低在场状态，不能完全沉默）"
    ),
    "高唤醒放大效应：同等投入+arousal=0.8": (
        "用户消息：「啊真的吗！！！好兴奋！」（高唤醒情绪，应加速对话节奏）"
    ),
    "低唤醒抑制：同等投入+arousal=-0.6": (
        "用户消息：「嗯好的……」（低唤醒，平静甚至稍显疲倦，不宜过度延展）"
    ),
    "EMA惯性：极高输入但前一轮M低→不会骤升": (
        "用户突然发来一条热情长消息：「哦天啊我终于想到了！……」（前一轮对话刚刚冷淡过）"
    ),
    "吸引力高关系基值提升M": (
        "（两人关系较好，互相有好感）用户消息：「最近还好吗」（关系基础好，基值高）"
    ),
}


# ─────────────────────────────────────────────────────────────
# 主验证逻辑
# ─────────────────────────────────────────────────────────────
def run_validation() -> None:
    results = []
    passed = 0
    failed = 0

    print("\n" + "=" * 70)
    print("Momentum 目标生成验证 — Part A：公式正确性")
    print("=" * 70)

    for i, case in enumerate(TEST_CASES, 1):
        state = _build_mock_state(**case["state"])
        m = _compute_momentum(state)
        category = momentum_to_category(m)

        # 判断是否通过
        ok = True
        fail_reason = ""
        if "expected_gte" in case and m < case["expected_gte"] - 0.001:
            ok = False
            fail_reason = f"M={m:.3f} < expected_gte={case['expected_gte']}"
        if "expected_lte" in case and m > case["expected_lte"] + 0.001:
            ok = False
            fail_reason = f"M={m:.3f} > expected_lte={case['expected_lte']}"
        if "expected_range" in case:
            lo, hi = case["expected_range"]
            if not (lo - 0.001 <= m <= hi + 0.001):
                ok = False
                fail_reason = f"M={m:.3f} 不在 [{lo}, {hi}]"

        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"\n[{i:02d}] {status}")
        print(f"  情景：{case['desc']}")
        print(f"  M_t = {m:.4f}  →  {category}")
        print(f"  人工预期类别：{case['expected_category']}")
        if not ok:
            print(f"  失败原因：{fail_reason}")

        row = {
            "id": i,
            "desc": case["desc"],
            "engagement": case["state"]["engagement"],
            "topic_appeal": case["state"]["topic_appeal"],
            "hostility": case["state"]["hostility"],
            "busyness": case["state"]["busyness"],
            "arousal": case["state"]["arousal"],
            "turn_count": case["state"]["turn_count"],
            "m_prev": case["state"]["m_prev"],
            "m_computed": round(m, 4),
            "category": category,
            "expected_category": case["expected_category"],
            "pass": "PASS" if ok else "FAIL",
            "fail_reason": fail_reason,
        }
        results.append(row)

    print("\n" + "=" * 70)
    print(f"总结：{passed} PASS / {failed} FAIL / {len(TEST_CASES)} 总计")
    print("=" * 70)

    # 写 CSV
    out_csv = os.path.join(os.path.dirname(__file__), "momentum_validation_results.csv")
    fieldnames = list(results[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ 详细结果已写入：{out_csv}")

    # 生成人工标注任务
    annotation_tasks = []
    for r in results:
        desc = r["desc"]
        scenario = SCENARIO_TEMPLATES.get(desc, f"（情景：{desc}）")
        annotation_tasks.append({
            "id": r["id"],
            "scenario": scenario,
            "system_m": r["m_computed"],
            "system_category": r["category"],
            "question": "根据上述对话情景，你认为这一轮机器人应该（单选）：",
            "options": ["收敛（简短收尾，不延展）", "维持（保持当前节奏）", "延长（主动推进，多说一些）"],
            "annotator_choice": "",   # 标注员填写
            "annotator_note": "",     # 可选备注
        })

    out_json = os.path.join(os.path.dirname(__file__), "momentum_annotation_tasks.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(annotation_tasks, f, ensure_ascii=False, indent=2)
    print(f"✓ 人工标注任务已写入：{out_json}")
    print(f"\n标注说明：请将 momentum_annotation_tasks.json 分发给 3 名标注员，")
    print("  各自填写 annotator_choice 字段，汇总后计算 Cohen's Kappa 系数。\n")


if __name__ == "__main__":
    run_validation()

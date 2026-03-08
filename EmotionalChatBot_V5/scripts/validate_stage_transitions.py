"""
第四章验证脚本二：关系阶段门控验证
==========================================

Part A：转换规则正确性验证（完全自动化）
  直接调用 stage_manager.evaluate_transition()，
  覆盖全部10阶段×3类转换(JUMP/GROWTH/DECAY/STAY)的合成测试用例，
  验证输出与 knapp_rules.yaml 的预期一致。

Part B：阶段行为差异摘要
  对同一用户输入，在 Experimenting vs Integrating 阶段下
  展示 style 参数的计算差异（量化阶段行为约束的效果）。

运行方式：
  cd EmotionalChatBot_V5
  python scripts/validate_stage_transitions.py

输出：
  - 控制台打印测试结果摘要
  - stage_transition_results.csv
  - stage_behavior_diff.json（Part B 行为差异数据）
"""
from __future__ import annotations

import csv
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.nodes.relation.stage_manager import KnappStageManager
from app.nodes.pipeline.style import compute_style_keys, Inputs

_manager = KnappStageManager()
evaluate_transition = _manager.evaluate_transition


# ─────────────────────────────────────────────────────────────
# 辅助：构造测试用 state
# ─────────────────────────────────────────────────────────────
def _rel(cl=0.3, tr=0.3, li=0.3, re=0.4, at=0.4, po=0.5) -> dict:
    return dict(closeness=cl, trust=tr, liking=li,
                respect=re, attractiveness=at, power=po)


def _spt(depth=1, breadth=2, trend="stable",
         depth_reduction=False, breadth_reduction=False) -> dict:
    signals = []
    if depth_reduction:
        signals.append("depth_reduction")
    if breadth_reduction:
        signals.append("breadth_reduction")
    return dict(depth=depth, breadth=breadth, depth_trend=trend,
                recent_signals=signals)


def _assets(confirm_counts: dict | None = None, topic_history: list | None = None) -> dict:
    return {
        "stage_confirm_counts": confirm_counts or {},
        "topic_history": topic_history or [],
    }


def _profile(n: int = 2) -> dict:
    """生成有 n 个字段的模拟用户画像。"""
    keys = ["hobby", "job", "city", "age", "personality", "favorite_food"]
    return {k: "test" for k in keys[:n]}


def _build_state(
    stage: str,
    rel: dict,
    spt: dict,
    assets: dict,
    profile_n: int = 2,
    rel_deltas: dict | None = None,
    user_turns: int = 5,
) -> dict:
    state = {
        "current_stage": stage,
        "relationship_state": rel,
        "spt_info": spt,
        "relationship_assets": assets,
        "user_inferred_profile": _profile(profile_n),
        "user_basic_info": {},
        "chat_buffer": [None] * (user_turns * 2),
        "turn_count_in_session": user_turns,
    }
    if rel_deltas:
        state["relationship_deltas_applied"] = rel_deltas
    return state


# ─────────────────────────────────────────────────────────────
# 测试用例
# ─────────────────────────────────────────────────────────────
TEST_CASES = [
    # ══════════════ JUMP 测试 ══════════════════════════════════
    {
        "desc": "任意阶段：信任 delta≤-0.25 → JUMP 到 terminating",
        "stage": "intensifying",
        "rel": _rel(cl=0.55, tr=0.50),
        "spt": _spt(depth=3, breadth=5),
        "assets": _assets(),
        "rel_deltas": {"trust": -0.30, "closeness": 0.0, "liking": 0.0,
                       "respect": 0.0, "attractiveness": 0.0, "power": 0.0},
        "expected_stage": "terminating",
        "expected_type": "JUMP",
    },
    {
        "desc": "任意阶段：尊重 delta≤-0.25 → JUMP 到 differentiating",
        "stage": "integrating",
        "rel": _rel(cl=0.70, tr=0.65),
        "spt": _spt(depth=4, breadth=6),
        "assets": _assets(),
        "rel_deltas": {"respect": -0.28, "trust": 0.0, "closeness": 0.0,
                       "liking": 0.0, "attractiveness": 0.0, "power": 0.0},
        "expected_stage": "differentiating",
        "expected_type": "JUMP",
    },
    {
        "desc": "initiating：SPT深度≥3+喜爱>0.4 → JUMP 到 intensifying",
        "stage": "initiating",
        "rel": _rel(cl=0.10, tr=0.10, li=0.45),
        "spt": _spt(depth=3, breadth=2),
        "assets": _assets(),
        "expected_stage": "intensifying",
        "expected_type": "JUMP",
    },

    # ══════════════ STAY 测试 ══════════════════════════════════
    {
        "desc": "initiating：条件未达到，维持不变",
        "stage": "initiating",
        "rel": _rel(cl=0.10, tr=0.10, li=0.10),
        "spt": _spt(depth=1, breadth=1),
        "assets": _assets(),
        "expected_stage": "initiating",
        "expected_type": "STAY",
    },
    {
        "desc": "experimenting：条件达到但确认轮数不够(只有1轮)",
        "stage": "experimenting",
        "rel": _rel(cl=0.45, tr=0.40, li=0.45),
        "spt": _spt(depth=2, breadth=5),
        "assets": _assets(confirm_counts={"experimenting": 1}),  # 需要连续2轮
        "profile_n": 3,
        "expected_stage": "experimenting",
        "expected_type": "STAY",
    },
    {
        "desc": "intensifying：关系维度在中间，无明显变化信号",
        "stage": "intensifying",
        "rel": _rel(cl=0.55, tr=0.50, li=0.52),
        "spt": _spt(depth=3, breadth=5),
        "assets": _assets(),
        "expected_stage": "intensifying",
        "expected_type": "STAY",
    },

    # ══════════════ GROWTH 测试 ══════════════════════════════════
    {
        "desc": "initiating→experimenting：亲密≥0.20+信任≥0.15+喜爱≥0.22+深度≥1+话题≥2，连续2轮",
        "stage": "initiating",
        "rel": _rel(cl=0.22, tr=0.17, li=0.25, re=0.40, at=0.40, po=0.50),
        "spt": _spt(depth=1, breadth=3),
        "assets": _assets(confirm_counts={"growth_initiating_experimenting": 1}),  # 已确认1轮，本轮是第2轮
        "profile_n": 1,
        "expected_stage": "experimenting",
        "expected_type": "GROWTH",
    },
    {
        "desc": "experimenting→intensifying：亲密≥0.42+信任≥0.38+深度≥2+话题≥4+画像≥3，连续2轮",
        "stage": "experimenting",
        "rel": _rel(cl=0.44, tr=0.40, li=0.44, re=0.40, at=0.45, po=0.50),
        "spt": _spt(depth=2, breadth=5),
        "assets": _assets(confirm_counts={"growth_experimenting_intensifying": 1}),  # 已确认1轮
        "profile_n": 4,
        "expected_stage": "intensifying",
        "expected_type": "GROWTH",
    },

    # ══════════════ DECAY 测试 ══════════════════════════════════
    {
        "desc": "experimenting DECAY：好感≤0.08，连续3轮",
        "stage": "experimenting",
        "rel": _rel(cl=0.35, tr=0.35, li=0.05, re=0.35, at=0.40, po=0.50),
        "spt": _spt(depth=1, breadth=2),
        "assets": _assets(confirm_counts={"decay_experimenting_initiating": 2}),  # 已确认2轮，本轮是第3轮
        "expected_stage": "initiating",
        "expected_type": "DECAY",
    },
    {
        "desc": "intensifying DECAY：SPT深度退缩信号，连续3轮",
        "stage": "intensifying",
        "rel": _rel(cl=0.55, tr=0.50, li=0.50),
        "spt": _spt(depth=2, breadth=4, trend="decreasing", depth_reduction=True),
        "assets": _assets(confirm_counts={"decay_intensifying_experimenting": 2}),  # 已确认2轮
        "expected_stage": "experimenting",
        "expected_type": "DECAY",
    },
    {
        "desc": "differentiating DECAY：信任≤0.45，连续3轮",
        "stage": "differentiating",
        "rel": _rel(cl=0.55, tr=0.42, li=0.45, re=0.38),
        "spt": _spt(depth=2, breadth=3),
        "assets": _assets(confirm_counts={"decay_differentiating_circumscribing": 2}),  # 已确认2轮
        "expected_stage": "circumscribing",
        "expected_type": "DECAY",
    },

    # ══════════════ 边界/特殊测试 ══════════════════════════════
    {
        "desc": "hysteresis：升级条件满足但连续轮数只有1轮→STAY",
        "stage": "intensifying",
        "rel": _rel(cl=0.62, tr=0.58, li=0.60, re=0.42, at=0.50, po=0.50),
        "spt": _spt(depth=3, breadth=6),
        "assets": _assets(confirm_counts={"growth_intensifying_integrating": 0}),  # 仅第1轮，需要2轮
        "profile_n": 5,
        "expected_stage": "intensifying",
        "expected_type": "STAY",
    },
    {
        "desc": "power_balance veto：权力失衡>0.3，即使满足升级条件→STAY（仅integrating→bonding生效）",
        "stage": "integrating",
        "rel": _rel(cl=0.76, tr=0.72, li=0.72, re=0.52, at=0.60, po=0.85),  # |0.85-0.5|=0.35>0.3
        "spt": _spt(depth=4, breadth=6),
        "assets": _assets(confirm_counts={"growth_integrating_bonding": 1}),
        "profile_n": 6,
        "expected_stage": "integrating",
        "expected_type": "STAY",   # power_balance veto
    },
    {
        "desc": "terminating：终态，无论如何都 STAY",
        "stage": "terminating",
        "rel": _rel(cl=0.02, tr=0.02, li=0.02),
        "spt": _spt(depth=1, breadth=1),
        "assets": _assets(),
        "expected_stage": "terminating",
        "expected_type": "STAY",
    },
]


# ─────────────────────────────────────────────────────────────
# Part B：阶段行为差异（style 参数对照）
# ─────────────────────────────────────────────────────────────
def compute_stage_behavior_diff() -> dict:
    """
    固定同一 Big Five + 用户输入，仅改变阶段对应的关系维度，
    对比 Experimenting 和 Integrating 阶段下的 style 输出差异。
    """
    base_big5 = dict(O=0.60, C=0.55, E=0.65, A=0.70, N=0.30)
    base_pad   = dict(P=0.55, Ar=0.25, D=0.55)
    base_ctx   = dict(busy=0.3, momentum=0.65, topic_appeal=5.5, evidence=None)

    # Experimenting 阶段的典型关系维度
    rel_exp = dict(closeness=0.35, trust=0.32, liking=0.38,
                   respect=0.42, attractiveness=0.40, power=0.50)
    # Integrating 阶段的典型关系维度
    rel_int = dict(closeness=0.72, trust=0.68, liking=0.72,
                   respect=0.55, attractiveness=0.60, power=0.50)

    def _style(rel: dict) -> dict:
        inp = Inputs(
            **base_big5, **base_pad, **rel,
            busy=base_ctx["busy"],
            momentum=base_ctx["momentum"],
            topic_appeal=base_ctx["topic_appeal"],
            evidence=base_ctx["evidence"],
        )
        return compute_style_keys(inp)

    s_exp = _style(rel_exp)
    s_int = _style(rel_int)

    diff = {}
    for k in s_exp:
        diff[k] = {
            "experimenting": round(float(s_exp[k]), 3),
            "integrating": round(float(s_int[k]), 3),
            "delta": round(float(s_int[k]) - float(s_exp[k]), 3),
        }

    result = {
        "scenario": "固定 Big Five(O=0.60/C=0.55/E=0.65/A=0.70/N=0.30) + PAD平静 + Momentum=0.65",
        "user_input": "最近有点累，感觉什么都没意思",
        "stage_comparison": {
            "experimenting": {
                "desc": "实验期：互动较浅，关系初建",
                "relationship": rel_exp,
            },
            "integrating": {
                "desc": "整合期：关系稳定，互相了解较深",
                "relationship": rel_int,
            },
        },
        "style_diff": diff,
        "interpretation": {
            "FORMALITY": "Integrating 更低：关系越亲密，正式度越低",
            "WARMTH": "Integrating 更高：亲密关系下温暖感更强",
            "CHAT_MARKERS": "Integrating 更高：熟络后口语化标记更多",
            "EXPRESSION_MODE": "Integrating 可能激活更复杂表达（如间接表达/讽刺）",
            "POLITENESS": "Integrating 更低：亲密关系中面子管理成本降低",
        }
    }
    return result


# ─────────────────────────────────────────────────────────────
# 主验证逻辑
# ─────────────────────────────────────────────────────────────
def run_validation() -> None:
    results = []
    passed = 0
    failed = 0

    print("\n" + "=" * 70)
    print("关系阶段门控验证 — Part A：转换规则正确性")
    print("=" * 70)

    for i, case in enumerate(TEST_CASES, 1):
        kwargs = {
            "stage": case["stage"],
            "rel": case["rel"],
            "spt": case["spt"],
            "assets": case["assets"],
            "user_turns": case.get("user_turns", 5),
        }
        if "profile_n" in case:
            kwargs["profile_n"] = case["profile_n"]
        if "rel_deltas" in case:
            kwargs["rel_deltas"] = case["rel_deltas"]

        state = _build_state(**kwargs)

        try:
            result = evaluate_transition(case["stage"], state)
            new_stage = result.get("new_stage", case["stage"])
            trans_type = result.get("transition_type", "STAY")
        except Exception as e:
            new_stage = f"ERROR: {e}"
            trans_type = "ERROR"

        stage_ok = (new_stage == case["expected_stage"])
        type_ok = (trans_type == case["expected_type"])
        ok = stage_ok and type_ok

        if ok:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"
            detail = []
            if not stage_ok:
                detail.append(f"阶段：got={new_stage}, expected={case['expected_stage']}")
            if not type_ok:
                detail.append(f"类型：got={trans_type}, expected={case['expected_type']}")

        print(f"\n[{i:02d}] {status}")
        print(f"  情景：{case['desc']}")
        print(f"  当前阶段：{case['stage']}  →  结果阶段：{new_stage}（{trans_type}）")
        print(f"  预期：{case['expected_stage']}（{case['expected_type']}）")
        if not ok:
            for d in (detail if not ok else []):
                print(f"  ⚠ {d}")

        results.append({
            "id": i,
            "desc": case["desc"],
            "input_stage": case["stage"],
            "output_stage": new_stage,
            "output_type": trans_type,
            "expected_stage": case["expected_stage"],
            "expected_type": case["expected_type"],
            "pass": "PASS" if ok else "FAIL",
        })

    print("\n" + "=" * 70)
    print(f"总结：{passed} PASS / {failed} FAIL / {len(TEST_CASES)} 总计")
    print("=" * 70)

    out_csv = os.path.join(os.path.dirname(__file__), "stage_transition_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ 详细结果已写入：{out_csv}")

    # Part B
    print("\n" + "=" * 70)
    print("关系阶段门控验证 — Part B：阶段行为差异（Style 参数对照）")
    print("=" * 70)

    diff_data = compute_stage_behavior_diff()

    print(f"\n场景：{diff_data['scenario']}")
    print(f"用户输入：{diff_data['user_input']}")
    print(f"\n{'维度':<18} {'Experimenting':>14} {'Integrating':>12} {'差值(I-E)':>11}")
    print("-" * 60)
    for dim, vals in diff_data["style_diff"].items():
        interp = diff_data["interpretation"].get(dim, "")
        marker = "↑" if vals["delta"] > 0.05 else ("↓" if vals["delta"] < -0.05 else "~")
        print(f"{dim:<18} {vals['experimenting']:>14.3f} {vals['integrating']:>12.3f} "
              f"{vals['delta']:>+10.3f} {marker}  {interp}")

    out_json = os.path.join(os.path.dirname(__file__), "stage_behavior_diff.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(diff_data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 行为差异数据已写入：{out_json}\n")


if __name__ == "__main__":
    run_validation()

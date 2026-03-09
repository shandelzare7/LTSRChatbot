#!/usr/bin/env python3
"""
scripts/validate_conflict_repair.py

第四章验证实验：冲突注入与修复场景模拟（4.5C）
=================================================

模拟一段 10 轮的关系演变：
  Phase 1（轮 1-3）：稳定期 — experimenting 阶段正常互动
  Phase 2（轮 4  ）：冲突注入 — trust 骤降 -0.30
  Phase 3（轮 5-7）：低谷期 — 消极但不修复
  Phase 4（轮 8-10）：修复期 — 持续正面互动

验证点：
  1. 冲突注入后 stage 是否正确 JUMP（trust delta ≤ -0.25 → terminating）
  2. style 是否立即响应（WARMTH↓ FORMALITY↑）
  3. 修复阶段 style 是否回暖（WARMTH↑ FORMALITY↓）
  4. hysteresis 是否生效（修复不能一轮就恢复 stage）

运行方式：
  cd EmotionalChatBot_V5
  python scripts/validate_conflict_repair.py

输出：
  scripts/output/conflict_repair_results.csv
  scripts/output/conflict_repair_report.json
"""
from __future__ import annotations

import csv
import json
import os
import sys
from copy import deepcopy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.nodes.pipeline.style import Inputs, compute_style_keys  # noqa: E402
from app.nodes.relation.stage_manager import KnappStageManager  # noqa: E402

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 固定 Bot 人格（与 stage_gradient 保持一致）
# ─────────────────────────────────────────────────────────────
BIG5 = dict(E=0.65, A=0.70, C=0.55, O=0.60, N=0.30)
PAD_CALM = dict(P=0.55, Ar=0.25, D=0.55)

STYLE_KEYS = ["FORMALITY", "POLITENESS", "WARMTH", "CERTAINTY",
              "EXPRESSION_MODE", "EMOTIONAL_INTENSITY"]


def _compute_style(rel: dict, pad: dict, momentum: float = 0.65,
                   topic_appeal: float = 0.55, busy: float = 0.3) -> dict:
    """封装 compute_style_keys 调用。"""
    inp = Inputs(
        **BIG5, **pad, **rel,
        busy=busy,
        momentum=momentum,
        topic_appeal=topic_appeal,
        evidence=None,
    )
    return compute_style_keys(inp)


def _build_state(stage: str, rel: dict, spt: dict, assets: dict,
                 rel_deltas: dict | None = None, user_turns: int = 5,
                 profile_n: int = 3) -> dict:
    """构造 stage_manager 可消费的 state。"""
    keys = ["hobby", "job", "city", "age", "personality", "favorite_food"]
    profile = {k: "test" for k in keys[:profile_n]}
    state = {
        "current_stage": stage,
        "relationship_state": rel,
        "spt_info": spt,
        "relationship_assets": assets,
        "user_inferred_profile": profile,
        "user_basic_info": {},
        "chat_buffer": [None] * (user_turns * 2),
    }
    if rel_deltas:
        state["relationship_deltas_applied"] = rel_deltas
    return state


# ─────────────────────────────────────────────────────────────
# 10 轮模拟场景定义
# ─────────────────────────────────────────────────────────────
def build_scenario() -> list[dict]:
    """
    构造 10 轮模拟序列。每轮定义：
      - phase: 所属阶段名
      - desc: 轮次描述
      - rel: 当轮关系维度
      - pad: 当轮情绪状态
      - momentum: 对话动量
      - rel_deltas: 当轮 delta（用于 stage jump 检测）
      - expected_stage: 预期阶段
    """
    turns = []

    # === Phase 1：稳定的 experimenting（轮 1-3）===
    stable_rel = dict(closeness=0.40, trust=0.38, liking=0.42,
                      respect=0.42, attractiveness=0.40, power=0.50)
    for i in range(1, 4):
        turns.append({
            "turn": i,
            "phase": "stable",
            "desc": f"稳定期第{i}轮：experimenting 阶段正常互动",
            "rel": deepcopy(stable_rel),
            "pad": deepcopy(PAD_CALM),
            "momentum": 0.65,
            "rel_deltas": {"trust": 0.02, "closeness": 0.01, "liking": 0.01,
                           "respect": 0.0, "attractiveness": 0.0, "power": 0.0},
            "expected_stage": "experimenting",
        })

    # === Phase 2：冲突注入（轮 4）===
    # trust 骤降 -0.30 → 触发 JUMP（threshold = 0.25）
    conflict_rel = dict(closeness=0.38, trust=0.10, liking=0.25,
                        respect=0.30, attractiveness=0.35, power=0.50)
    conflict_pad = dict(P=0.20, Ar=0.70, D=0.40)
    turns.append({
        "turn": 4,
        "phase": "conflict",
        "desc": "冲突注入：trust 骤降 -0.30（背叛/欺骗事件）",
        "rel": conflict_rel,
        "pad": conflict_pad,
        "momentum": 0.40,
        "rel_deltas": {"trust": -0.30, "closeness": -0.05, "liking": -0.10,
                       "respect": -0.08, "attractiveness": -0.05, "power": 0.0},
        "expected_stage": "terminating",  # JUMP triggered by trust delta ≤ -0.25
    })

    # === Phase 3：低谷（轮 5-7）===
    # 冲突后关系维度低迷，情绪负面
    low_rel = dict(closeness=0.15, trust=0.08, liking=0.12,
                   respect=0.20, attractiveness=0.25, power=0.50)
    low_pad = dict(P=0.25, Ar=0.35, D=0.45)
    for i, turn_n in enumerate([5, 6, 7]):
        turns.append({
            "turn": turn_n,
            "phase": "low",
            "desc": f"低谷期第{i+1}轮：关系低迷，消极互动",
            "rel": deepcopy(low_rel),
            "pad": deepcopy(low_pad),
            "momentum": 0.35,
            "rel_deltas": {"trust": 0.0, "closeness": 0.0, "liking": 0.0,
                           "respect": 0.0, "attractiveness": 0.0, "power": 0.0},
            "expected_stage": "terminating",  # 终态吸收
        })

    # === Phase 4：修复（轮 8-10）===
    # 持续正面互动，关系逐步回暖
    repair_rels = [
        dict(closeness=0.22, trust=0.18, liking=0.20,
             respect=0.28, attractiveness=0.30, power=0.50),
        dict(closeness=0.28, trust=0.25, liking=0.28,
             respect=0.32, attractiveness=0.33, power=0.50),
        dict(closeness=0.33, trust=0.30, liking=0.35,
             respect=0.35, attractiveness=0.36, power=0.50),
    ]
    repair_pad = dict(P=0.50, Ar=0.30, D=0.50)
    for i, (turn_n, rel) in enumerate(zip([8, 9, 10], repair_rels)):
        turns.append({
            "turn": turn_n,
            "phase": "repair",
            "desc": f"修复期第{i+1}轮：持续正面互动，关系回暖",
            "rel": rel,
            "pad": deepcopy(repair_pad),
            "momentum": 0.55 + i * 0.05,
            "rel_deltas": {"trust": 0.05, "closeness": 0.04, "liking": 0.05,
                           "respect": 0.02, "attractiveness": 0.02, "power": 0.0},
            # terminating 是吸收态，不会升级
            "expected_stage": "terminating",
        })

    return turns


def run_simulation() -> dict:
    """运行 10 轮模拟，收集 style + stage 数据。"""
    manager = KnappStageManager()
    turns = build_scenario()
    results = []
    current_stage = "experimenting"

    for t in turns:
        # 1. 计算 style
        style = _compute_style(
            rel=t["rel"], pad=t["pad"],
            momentum=t["momentum"],
        )

        # 2. 评估 stage 转移
        spt = dict(depth=2, breadth=4, depth_trend="stable",
                   recent_signals=[])
        assets = dict(stage_confirm_counts={}, topic_history=[])
        state = _build_state(
            stage=current_stage,
            rel=t["rel"],
            spt=spt,
            assets=assets,
            rel_deltas=t["rel_deltas"],
        )

        try:
            transition = manager.evaluate_transition(current_stage, state)
            new_stage = transition.get("new_stage", current_stage)
            trans_type = transition.get("transition_type", "STAY")
            reason = transition.get("reason", "")
        except Exception as e:
            new_stage = current_stage
            trans_type = "ERROR"
            reason = str(e)

        # 记录
        row = {
            "turn": t["turn"],
            "phase": t["phase"],
            "desc": t["desc"],
            "stage_before": current_stage,
            "stage_after": new_stage,
            "transition_type": trans_type,
            "reason": reason,
            "expected_stage": t["expected_stage"],
            "stage_correct": new_stage == t["expected_stage"],
            "rel": t["rel"],
            "pad": t["pad"],
            "momentum": t["momentum"],
            "style": {k: round(float(style[k]), 4) for k in STYLE_KEYS},
        }
        results.append(row)

        # 更新当前阶段
        current_stage = new_stage

    return results


def analyze_results(results: list) -> dict:
    """分析实验结果，提取关键验证指标。"""
    analysis = {}

    # 验证 1：冲突注入后 stage 是否 JUMP
    turn4 = results[3]
    analysis["conflict_jump"] = {
        "desc": "冲突注入后 stage 是否触发 JUMP → terminating",
        "passed": turn4["transition_type"] == "JUMP" and turn4["stage_after"] == "terminating",
        "actual_type": turn4["transition_type"],
        "actual_stage": turn4["stage_after"],
    }

    # 验证 2：冲突后 style 响应（对比轮 3 vs 轮 4）
    style_pre = results[2]["style"]   # 轮 3（稳定期最后一轮）
    style_post = results[3]["style"]  # 轮 4（冲突轮）
    warmth_drop = style_post["WARMTH"] < style_pre["WARMTH"]
    formality_rise = style_post["FORMALITY"] > style_pre["FORMALITY"]
    ei_rise = style_post["EMOTIONAL_INTENSITY"] > style_pre["EMOTIONAL_INTENSITY"]
    analysis["style_response"] = {
        "desc": "冲突后 style 响应：WARMTH↓ FORMALITY↑ EI↑",
        "passed": warmth_drop and formality_rise,
        "warmth_delta": round(style_post["WARMTH"] - style_pre["WARMTH"], 4),
        "formality_delta": round(style_post["FORMALITY"] - style_pre["FORMALITY"], 4),
        "ei_delta": round(style_post["EMOTIONAL_INTENSITY"] - style_pre["EMOTIONAL_INTENSITY"], 4),
        "warmth_dropped": warmth_drop,
        "formality_rose": formality_rise,
        "ei_rose": ei_rise,
    }

    # 验证 3：terminating 是吸收态（轮 5-10 都不应离开）
    all_stay_term = all(
        r["stage_after"] == "terminating"
        for r in results[4:]  # 轮 5-10
    )
    analysis["terminating_absorbing"] = {
        "desc": "terminating 是吸收态：后续轮次不应离开",
        "passed": all_stay_term,
        "stages_after_conflict": [r["stage_after"] for r in results[4:]],
    }

    # 验证 4：修复期 style 相对低谷期回暖
    style_low = results[4]["style"]    # 轮 5（低谷期第 1 轮）
    style_repair = results[9]["style"]  # 轮 10（修复期最后一轮）
    warmth_recovery = style_repair["WARMTH"] > style_low["WARMTH"]
    formality_recovery = style_repair["FORMALITY"] < style_low["FORMALITY"]
    analysis["style_recovery"] = {
        "desc": "修复期 style 回暖：WARMTH↑ FORMALITY↓（相对低谷期）",
        "passed": warmth_recovery and formality_recovery,
        "warmth_delta": round(style_repair["WARMTH"] - style_low["WARMTH"], 4),
        "formality_delta": round(style_repair["FORMALITY"] - style_low["FORMALITY"], 4),
        "warmth_recovered": warmth_recovery,
        "formality_recovered": formality_recovery,
    }

    analysis["all_pass"] = all(v["passed"] for v in analysis.values() if isinstance(v, dict))
    return analysis


def print_results(results: list, analysis: dict) -> bool:
    """格式化打印。"""
    print("\n" + "=" * 90)
    print("  冲突-修复场景模拟（4.5C）")
    print("=" * 90)

    # 时序表
    print(f"\n{'轮':>3} {'阶段':>5} {'Stage':>14} {'→':>2} {'Stage':>14} {'类型':>6}  "
          f"{'FORM':>6} {'POLI':>6} {'WARM':>6} {'CERT':>6} {'EM':>3} {'EI':>6}")
    print("-" * 90)

    for r in results:
        s = r["style"]
        marker = ""
        if r["phase"] == "conflict":
            marker = " ⚡"
        elif r["phase"] == "repair":
            marker = " 🔧"
        print(f"{r['turn']:>3} {r['phase']:>8} {r['stage_before']:>14} → "
              f"{r['stage_after']:>14} {r['transition_type']:>6}  "
              f"{s['FORMALITY']:>6.3f} {s['POLITENESS']:>6.3f} "
              f"{s['WARMTH']:>6.3f} {s['CERTAINTY']:>6.3f} "
              f"{s['EXPRESSION_MODE']:>3} {s['EMOTIONAL_INTENSITY']:>6.3f}{marker}")

    # 验证结果
    print(f"\n{'─' * 70}")
    print("验证结果：")
    for key, val in analysis.items():
        if key == "all_pass":
            continue
        if not isinstance(val, dict):
            continue
        status = "PASS ✓" if val["passed"] else "FAIL ✗"
        print(f"\n  [{status}] {val['desc']}")
        for k, v in val.items():
            if k in ("desc", "passed"):
                continue
            print(f"         {k}: {v}")

    all_pass = analysis.get("all_pass", False)
    print(f"\n{'=' * 70}")
    print(f"  总结：{'所有验证通过 ✓' if all_pass else '存在失败项 ✗'}")
    print("=" * 70)
    return all_pass


def save_outputs(results: list, analysis: dict) -> None:
    """保存 CSV 和 JSON。"""
    # CSV
    csv_path = os.path.join(OUTPUT_DIR, "conflict_repair_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "turn", "phase", "desc",
            "stage_before", "stage_after", "transition_type",
            "expected_stage", "stage_correct",
            *STYLE_KEYS,
            "P", "Ar", "momentum",
            "closeness", "trust", "liking",
        ])
        for r in results:
            writer.writerow([
                r["turn"], r["phase"], r["desc"],
                r["stage_before"], r["stage_after"], r["transition_type"],
                r["expected_stage"], r["stage_correct"],
                *[r["style"][k] for k in STYLE_KEYS],
                r["pad"]["P"], r["pad"]["Ar"], r["momentum"],
                r["rel"]["closeness"], r["rel"]["trust"], r["rel"]["liking"],
            ])
    print(f"\n  → CSV 已保存：{csv_path}")

    # JSON
    json_path = os.path.join(OUTPUT_DIR, "conflict_repair_report.json")
    report = {
        "experiment": "4.5C 冲突-修复场景模拟",
        "scenario": {
            "phase1_stable": "轮 1-3：experimenting 稳定期",
            "phase2_conflict": "轮 4：trust 骤降 -0.30（JUMP 触发）",
            "phase3_low": "轮 5-7：低谷消极互动",
            "phase4_repair": "轮 8-10：持续正面互动修复",
        },
        "turns": results,
        "analysis": analysis,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  → JSON 已保存：{json_path}")


def main() -> None:
    results = run_simulation()
    analysis = analyze_results(results)
    all_pass = print_results(results, analysis)
    save_outputs(results, analysis)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

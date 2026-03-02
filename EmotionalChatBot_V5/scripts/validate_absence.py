#!/usr/bin/env python3
"""
scripts/validate_absence.py

时间化缺席机制自动化规则验证脚本

本脚本验证 HumanizationProcessor.calculate_absence() 的触发逻辑。
共 4 类触发场景 + 1 类在线场景，每类 3-4 组测试用例。

触发优先级：sleep → ghosting → busy → cooling → online

验证策略：
  - sleep / online：触发条件完全确定性，运行 1 次即可验证（100%）
  - ghosting / busy / cooling：含随机采样，每组运行 N_TRIALS=60 次，
    使用固定随机种子保证跨运行可重现，要求触发率达到门控阈值

输出：absence_validation_results.csv + absence_validation_report.json
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple

# ── 路径初始化 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.nodes.pipeline.processor import HumanizationProcessor  # noqa: E402

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 全局配置 ────────────────────────────────────────────────────────────────
N_TRIALS = 60            # 随机场景每组运行次数
MASTER_SEED_BASE = 2024  # 固定种子基准（保证跨运行可重现）

# 各类别的通过率门控阈值（根据理论概率设定）
PASS_THRESHOLDS: Dict[str, float] = {
    "sleep": 1.00,      # 完全确定性
    "online": 1.00,     # 完全确定性
    "ghosting": 0.80,   # 理论概率 ≥ 0.92 → 期望 55/60
    "busy": 0.60,       # 理论概率 ≈ 0.70 → 期望 42/60
    "cooling": 0.15,    # 理论概率 0.22-0.47（短消息场景偏低）→ 以最低情形为门控下限
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  State 构造工具                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_state(
    *,
    current_time: str,
    stage: str = "experimenting",
    big5: Dict[str, float] | None = None,
    mood: Dict[str, Any] | None = None,
    relationship: Dict[str, float] | None = None,
    user_input: str = "在吗",
    momentum: float = 0.5,
) -> Dict[str, Any]:
    """构造 HumanizationProcessor 所需的最小 state 字典。"""
    _big5 = dict(
        extraversion=0.5, agreeableness=0.5, conscientiousness=0.5,
        openness=0.5, neuroticism=0.3,
    )
    if big5:
        _big5.update(big5)

    _mood = dict(pleasure=0.0, arousal=0.0, dominance=0.0, busyness=0.3, pad_scale="m1_1")
    if mood:
        _mood.update(mood)

    _rel = dict(closeness=0.5, trust=0.5, liking=0.5, respect=0.5, attractiveness=0.5, power=0.5)
    if relationship:
        _rel.update(relationship)

    return {
        "current_time": current_time,
        "current_stage": stage,
        "bot_big_five": _big5,
        "mood_state": _mood,
        "relationship_state": _rel,
        "user_input": user_input,
        "conversation_momentum": momentum,
    }


def run_absence(state: Dict[str, Any]) -> Tuple[float, str, str]:
    """调用 calculate_absence 并返回 (seconds, reason, sub_reason)。"""
    proc = HumanizationProcessor(state)
    dyn = proc.calculate_dynamics_modifiers()
    return proc.calculate_absence(dyn)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  测试用例定义                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

TEST_CASES: List[Dict[str, Any]] = [

    # ────────────────────────────────────────────────────────────────────────
    #  SLEEP 场景（完全确定性：当前小时 < wakeup，不依赖随机分支）
    # ────────────────────────────────────────────────────────────────────────

    {
        "id": "S01",
        "category": "sleep",
        "desc": "凌晨3点，中性人格 → sleep（跨午夜判断）",
        "state_kwargs": {
            "current_time": "2024-03-15T03:00:00",
            "stage": "experimenting",
            "big5": {"extraversion": 0.5, "conscientiousness": 0.5, "neuroticism": 0.3},
            "mood": {"pleasure": 0.0, "arousal": 0.0, "busyness": 0.2, "pad_scale": "m1_1"},
        },
        "expected_reason": "sleep",
        "deterministic": True,
        "logic_note": "03:00 < wakeup≈6.75 → is_sleeping=True（跨午夜区间 bedtime>wakeup）",
    },
    {
        "id": "S02",
        "category": "sleep",
        "desc": "凌晨2点，低外向早睡型人格 → sleep",
        "state_kwargs": {
            "current_time": "2024-03-15T02:00:00",
            "stage": "experimenting",
            "big5": {"extraversion": 0.2, "conscientiousness": 0.7, "neuroticism": 0.2},
            "mood": {"pleasure": 0.0, "arousal": 0.0, "busyness": 0.1, "pad_scale": "m1_1"},
        },
        "expected_reason": "sleep",
        "deterministic": True,
        "logic_note": "E=0.2, C=0.7 → bedtime≈22.3, wakeup≈6.4 → 02:00 in [22.3,∞)∪[0,6.4) → 睡眠中",
    },
    {
        "id": "S03",
        "category": "sleep",
        "desc": "凌晨1点，高神经质高外向（晚睡型，1点仍在睡眠窗口） → sleep",
        "state_kwargs": {
            "current_time": "2024-03-15T01:00:00",
            "stage": "experimenting",
            "big5": {"extraversion": 0.9, "conscientiousness": 0.2, "neuroticism": 0.8},
            "mood": {"pleasure": 0.0, "arousal": 0.0, "busyness": 0.0, "pad_scale": "m1_1"},
        },
        "expected_reason": "sleep",
        "deterministic": True,
        "logic_note": "E=0.9 → bedtime≈23.6+noise；但1:00仍在睡眠区间（bedtime≈23.6→1点已过bedtime，wakeup≈7.4→1<7.4）",
    },
    {
        "id": "S04",
        "category": "sleep",
        "desc": "凌晨5点，中性人格 → sleep（起床前）",
        "state_kwargs": {
            "current_time": "2024-03-15T05:00:00",
            "stage": "experimenting",
            "big5": {"extraversion": 0.5, "conscientiousness": 0.5, "neuroticism": 0.3},
            "mood": {"pleasure": 0.0, "arousal": 0.0, "busyness": 0.2, "pad_scale": "m1_1"},
        },
        "expected_reason": "sleep",
        "deterministic": True,
        "logic_note": "wakeup≈6.75 → 05:00 < 6.75 → is_sleeping=True",
    },

    # ────────────────────────────────────────────────────────────────────────
    #  ONLINE 场景（完全确定性：所有随机分支概率=0）
    # ────────────────────────────────────────────────────────────────────────

    {
        "id": "O01",
        "category": "online",
        "desc": "下午2点，低忙碌，关系正常，非敏感阶段 → online",
        "state_kwargs": {
            "current_time": "2024-03-15T14:00:00",
            "stage": "experimenting",
            "big5": {"extraversion": 0.5, "conscientiousness": 0.6, "neuroticism": 0.3},
            "mood": {"pleasure": 0.0, "arousal": 0.0, "busyness": 0.2, "pad_scale": "m1_1"},
        },
        "expected_reason": "online",
        "deterministic": True,
        "logic_note": (
            "14:00非睡眠；stage=experimenting→base_ghost=0，pleasure中性→ghost=0；"
            "busyness=0.2<0.5→busy=0；N=0.3<0.55→cooling条件不满足→online"
        ),
    },
    {
        "id": "O02",
        "category": "online",
        "desc": "上午10点，关系紧密，bonding阶段 → online",
        "state_kwargs": {
            "current_time": "2024-03-15T10:00:00",
            "stage": "bonding",
            "big5": {"extraversion": 0.6, "conscientiousness": 0.6, "neuroticism": 0.4},
            "mood": {"pleasure": 0.3, "arousal": 0.1, "busyness": 0.3, "pad_scale": "m1_1"},
            "relationship": {"closeness": 0.8, "liking": 0.8, "attractiveness": 0.8},
        },
        "expected_reason": "online",
        "deterministic": True,
        "logic_note": (
            "stage=bonding→base_ghost=0，高liking&attractiveness使ghost更低；"
            "busyness=0.3<0.5→busy=0；stage=bonding使cooling cond_stage=False→online"
        ),
    },
    {
        "id": "O03",
        "category": "online",
        "desc": "傍晚6点，integrating阶段，中性忙碌 → online",
        "state_kwargs": {
            "current_time": "2024-03-15T18:00:00",
            "stage": "integrating",
            "big5": {"extraversion": 0.5, "conscientiousness": 0.5, "neuroticism": 0.4},
            "mood": {"pleasure": 0.2, "arousal": 0.0, "busyness": 0.4, "pad_scale": "m1_1"},
        },
        "expected_reason": "online",
        "deterministic": True,
        "logic_note": (
            "18:00非睡眠；stage=integrating→base_ghost=0，pleasure正常；"
            "busyness=0.4<0.5→busy=0；stage=integrating使cooling cond_stage=False→online"
        ),
    },

    # ────────────────────────────────────────────────────────────────────────
    #  GHOSTING 场景（高概率随机，ghost_prob ≥ 0.90，N_TRIALS 次验证）
    # ────────────────────────────────────────────────────────────────────────

    {
        "id": "G01",
        "category": "ghosting",
        "desc": "stage=avoiding，低愉悦(-0.7)，高神经质(N=0.8) → ghosting（概率≈0.92）",
        "state_kwargs": {
            "current_time": "2024-03-15T14:00:00",
            "stage": "avoiding",
            "big5": {"extraversion": 0.5, "conscientiousness": 0.5, "neuroticism": 0.8},
            "mood": {"pleasure": -0.7, "arousal": 0.0, "busyness": 0.1, "pad_scale": "m1_1"},
        },
        "expected_reason": "ghosting",
        "deterministic": False,
        "logic_note": (
            "base_ghost=0.65(avoiding)+0.30(P<0.35)+0.15(N>0.65)=1.10→clamp→0.92"
        ),
    },
    {
        "id": "G02",
        "category": "ghosting",
        "desc": "stage=stagnating，低愉悦(-0.8)，高神经质(N=0.75) → ghosting（概率≈0.92）",
        "state_kwargs": {
            "current_time": "2024-03-15T15:00:00",
            "stage": "stagnating",
            "big5": {"extraversion": 0.5, "conscientiousness": 0.5, "neuroticism": 0.75},
            "mood": {"pleasure": -0.8, "arousal": 0.0, "busyness": 0.1, "pad_scale": "m1_1"},
        },
        "expected_reason": "ghosting",
        "deterministic": False,
        "logic_note": (
            "base_ghost=0.50(stagnating)+0.30(P<0.35)+0.15(N>0.65)=0.95→clamp→0.92"
        ),
    },
    {
        "id": "G03",
        "category": "ghosting",
        "desc": "stage=terminating，极低愉悦(-0.9)，N=0.7 → ghosting（概率≈0.92）",
        "state_kwargs": {
            "current_time": "2024-03-15T16:00:00",
            "stage": "terminating",
            "big5": {"extraversion": 0.5, "conscientiousness": 0.5, "neuroticism": 0.7},
            "mood": {"pleasure": -0.9, "arousal": 0.0, "busyness": 0.1, "pad_scale": "m1_1"},
        },
        "expected_reason": "ghosting",
        "deterministic": False,
        "logic_note": (
            "base_ghost=0.65(terminating)+0.30(P<0.35)+0.15(N>0.65)=1.10→clamp→0.92"
        ),
    },

    # ────────────────────────────────────────────────────────────────────────
    #  BUSY 场景（高忙碌度，ghost概率为0，N_TRIALS 次验证）
    # ────────────────────────────────────────────────────────────────────────

    {
        "id": "B01",
        "category": "busy",
        "desc": "busyness=0.95，非敏感阶段，愉悦正常 → busy（概率≈0.80）",
        "state_kwargs": {
            "current_time": "2024-03-15T11:00:00",
            "stage": "experimenting",
            "big5": {"extraversion": 0.5, "conscientiousness": 0.5, "neuroticism": 0.3},
            "mood": {"pleasure": 0.0, "arousal": 0.0, "busyness": 0.95, "pad_scale": "m1_1"},
        },
        "expected_reason": "busy",
        "deterministic": False,
        "logic_note": (
            "ghost_prob=0（stage=experimenting，pleasure中性，N<0.65）；"
            "busy_base=0.50+(0.95-0.85)/0.15*0.30≈0.70→clamp→0.70"
        ),
    },
    {
        "id": "B02",
        "category": "busy",
        "desc": "busyness=0.88，上班时间，closeness低 → busy（概率≈0.57）",
        "state_kwargs": {
            "current_time": "2024-03-15T15:00:00",
            "stage": "intensifying",
            "big5": {"extraversion": 0.5, "conscientiousness": 0.5, "neuroticism": 0.3},
            "mood": {"pleasure": 0.0, "arousal": 0.0, "busyness": 0.88, "pad_scale": "m1_1"},
            "relationship": {"closeness": 0.4, "liking": 0.4},
        },
        "expected_reason": "busy",
        "deterministic": False,
        "logic_note": (
            "ghost_prob=0；busy_base=0.50+(0.88-0.85)/0.15*0.30≈0.56，"
            "closeness=0.4<0.65/liking<0.65 → 无减免 → busy_prob≈0.56"
        ),
    },
    {
        "id": "B03",
        "category": "busy",
        "desc": "busyness=0.95，关系一般，无紧急消息 → busy（概率≈0.80）",
        "state_kwargs": {
            "current_time": "2024-03-15T09:30:00",
            "stage": "circumscribing",
            "big5": {"extraversion": 0.5, "conscientiousness": 0.6, "neuroticism": 0.4},
            "mood": {"pleasure": 0.0, "arousal": 0.0, "busyness": 0.95, "pad_scale": "m1_1"},
            "relationship": {"closeness": 0.5},
        },
        "user_input": "嗯",
        "expected_reason": "busy",
        "deterministic": False,
        "logic_note": (
            "ghost_prob=0（stage=circumscribing，pleasure中性，N<0.65）；"
            "busy_base≈0.70，关系中等无减免 → busy_prob≈0.70"
        ),
    },

    # ────────────────────────────────────────────────────────────────────────
    #  COOLING 场景（高神经质 + 低愉悦 + 高唤起，ghost/busy 概率为0）
    # ────────────────────────────────────────────────────────────────────────

    {
        "id": "C01",
        "category": "cooling",
        "desc": "N=0.9，低愉悦(-0.8)，高唤起(Ar=0.6)，高liking阻断ghost → cooling",
        "state_kwargs": {
            "current_time": "2024-03-15T20:00:00",
            "stage": "circumscribing",
            "big5": {"extraversion": 0.5, "conscientiousness": 0.5, "neuroticism": 0.9},
            "mood": {"pleasure": -0.8, "arousal": 0.6, "busyness": 0.1, "pad_scale": "m1_1"},
            "relationship": {"closeness": 0.5, "liking": 0.9, "attractiveness": 0.9, "power": 0.9},
        },
        # 长消息提升 emotional_weight
        "user_input": "我今天真的好难受，什么事情都不顺，感觉整个人都崩了，" * 4,
        "expected_reason": "cooling",
        "deterministic": False,
        "logic_note": (
            "ghost_prob=0（liking/attractiveness/power高，抵消pleasure/N带来的基础值）；"
            "busy_prob=0（busyness=0.1）；"
            "N=0.9>0.55, arousal01=0.8>0.78→cond_emotional=True；"
            "cool_base=0.9*0.4*0.7=0.252+0.20(emotional_weight)=0.452→prob≈0.45"
        ),
    },
    {
        "id": "C02",
        "category": "cooling",
        "desc": "N=0.85，极低愉悦(-0.9)，Ar=0.7，避免ghost/busy → cooling",
        "state_kwargs": {
            "current_time": "2024-03-15T21:00:00",
            "stage": "differentiating",
            "big5": {"extraversion": 0.5, "conscientiousness": 0.5, "neuroticism": 0.85},
            "mood": {"pleasure": -0.9, "arousal": 0.7, "busyness": 0.1, "pad_scale": "m1_1"},
            "relationship": {"closeness": 0.5, "liking": 0.9, "attractiveness": 0.9, "power": 0.85},
        },
        # msg_len ≥ 95 使 emotional_weight = min(1, len/200) * 0.85 > 0.4
        "user_input": "你能不能帮我，今天发生了好多事，压力超级大，好难受" * 5,
        "expected_reason": "cooling",
        "deterministic": False,
        "logic_note": (
            "ghost_prob≈0（高liking/attr/power抵消低pleasure/高N）；"
            "N=0.85>0.55, pleasure01=0.05<0.35→cond_emotional=True；"
            "cool_base=0.85*0.45*0.7=0.267；msg_len=125→ew=0.625*0.85=0.531>0.4→+0.20→prob≈0.467"
        ),
    },
    {
        "id": "C03",
        "category": "cooling",
        "desc": "N=0.9，低愉悦，短消息（低情绪权重） → cooling（边界低概率场景≈0.22）",
        "state_kwargs": {
            "current_time": "2024-03-15T19:00:00",
            "stage": "circumscribing",
            "big5": {"extraversion": 0.5, "conscientiousness": 0.5, "neuroticism": 0.9},
            "mood": {"pleasure": -0.7, "arousal": 0.3, "busyness": 0.1, "pad_scale": "m1_1"},
            "relationship": {"closeness": 0.5, "liking": 0.9, "attractiveness": 0.9, "power": 0.9},
        },
        "user_input": "嗯",
        "expected_reason": "cooling",
        "deterministic": False,
        "logic_note": (
            "cool_base=0.9*(0.5-0.15)*0.7=0.220，短消息无emotional_weight加成→prob≈0.22；"
            "此为冷却机制低端边界验证，阈值调低至0.15"
        ),
    },
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  运行逻辑                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_test_case(case: Dict[str, Any], case_idx: int) -> Dict[str, Any]:
    """
    运行单个测试用例。
    确定性用例：运行 1 次，100% 准确率要求。
    随机用例：运行 N_TRIALS 次，统计触发率。
    """
    state_kwargs = dict(case["state_kwargs"])
    if "user_input" in case:
        state_kwargs["user_input"] = case["user_input"]

    expected = case["expected_reason"]
    is_det = case.get("deterministic", True)
    category = case["category"]
    threshold = PASS_THRESHOLDS[category]

    if is_det:
        # ── 确定性测试 ────────────────────────────────────────────────────
        random.seed(MASTER_SEED_BASE + case_idx * 1000)
        state = build_state(**state_kwargs)
        _, reason, sub_reason = run_absence(state)
        passed = (reason == expected)
        return {
            "id": case["id"],
            "category": category,
            "desc": case["desc"],
            "expected": expected,
            "trials": 1,
            "trigger_count": 1 if passed else 0,
            "trigger_rate": 1.0 if passed else 0.0,
            "threshold": threshold,
            "sample_reason": reason,
            "sample_sub_reason": sub_reason,
            "passed": passed,
            "logic_note": case.get("logic_note", ""),
        }
    else:
        # ── 随机采样测试 ──────────────────────────────────────────────────
        random.seed(MASTER_SEED_BASE + case_idx * 1000)
        counts: Dict[str, int] = {}
        last_sub = ""
        for _ in range(N_TRIALS):
            state = build_state(**state_kwargs)
            _, reason, sub_reason = run_absence(state)
            counts[reason] = counts.get(reason, 0) + 1
            if reason == expected:
                last_sub = sub_reason

        trigger_count = counts.get(expected, 0)
        trigger_rate = trigger_count / N_TRIALS
        passed = trigger_rate >= threshold

        dominant_reason = max(counts, key=lambda k: counts[k])
        return {
            "id": case["id"],
            "category": category,
            "desc": case["desc"],
            "expected": expected,
            "trials": N_TRIALS,
            "trigger_count": trigger_count,
            "trigger_rate": round(trigger_rate, 3),
            "threshold": threshold,
            "sample_reason": dominant_reason,
            "sample_sub_reason": last_sub,
            "reason_distribution": counts,
            "passed": passed,
            "logic_note": case.get("logic_note", ""),
        }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  输出                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def save_csv(results: List[Dict]) -> None:
    csv_path = os.path.join(OUTPUT_DIR, "absence_validation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "test_id", "category", "desc",
            "expected_reason", "trials", "trigger_count",
            "trigger_rate", "threshold", "dominant_reason",
            "result",
        ])
        for r in results:
            writer.writerow([
                r["id"], r["category"], r["desc"],
                r["expected"], r["trials"], r["trigger_count"],
                f"{r['trigger_rate']:.1%}",
                f"{r['threshold']:.0%}",
                r["sample_reason"],
                "PASS ✓" if r["passed"] else "FAIL ✗",
            ])
    print(f"\n  → CSV 已保存：{csv_path}")


def save_json(results: List[Dict]) -> None:
    json_path = os.path.join(OUTPUT_DIR, "absence_validation_report.json")
    report = {
        "config": {
            "N_TRIALS": N_TRIALS,
            "MASTER_SEED_BASE": MASTER_SEED_BASE,
            "PASS_THRESHOLDS": PASS_THRESHOLDS,
        },
        "results": results,
        "summary": {
            cat: {
                "total": sum(1 for r in results if r["category"] == cat),
                "passed": sum(1 for r in results if r["category"] == cat and r["passed"]),
            }
            for cat in ["sleep", "online", "ghosting", "busy", "cooling"]
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  → JSON 已保存：{json_path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  入口                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    print("=" * 70)
    print("  时间化缺席机制规则验证脚本")
    print(f"  随机场景：每组 {N_TRIALS} 次试验 | 随机种子基准：{MASTER_SEED_BASE}")
    print("=" * 70)

    results: List[Dict] = []

    categories = ["sleep", "online", "ghosting", "busy", "cooling"]
    category_labels = {
        "sleep": "SLEEP   —— 完全确定性（时间窗口判断）",
        "online": "ONLINE  —— 完全确定性（所有分支概率=0）",
        "ghosting": f"GHOSTING —— 随机采样（阈值≥{PASS_THRESHOLDS['ghosting']:.0%}）",
        "busy": f"BUSY    —— 随机采样（阈值≥{PASS_THRESHOLDS['busy']:.0%}）",
        "cooling": f"COOLING —— 随机采样（阈值≥{PASS_THRESHOLDS['cooling']:.0%}）",
    }

    for cat in categories:
        print(f"\n[{category_labels[cat]}]")
        print("-" * 50)
        cat_cases = [c for c in TEST_CASES if c["category"] == cat]
        for idx, case in enumerate(cat_cases):
            global_idx = TEST_CASES.index(case)
            result = run_test_case(case, global_idx)
            results.append(result)

            if result["trials"] == 1:
                status = "PASS ✓" if result["passed"] else "FAIL ✗"
                print(f"  [{status}] {result['id']} {result['desc']}")
                print(f"         reason={result['sample_reason']} sub={result['sample_sub_reason']}")
            else:
                status = "PASS ✓" if result["passed"] else "FAIL ✗"
                rate_str = f"{result['trigger_rate']:.1%}"
                print(f"  [{status}] {result['id']} {result['desc']}")
                print(
                    f"         trigger_rate={rate_str} ({result['trigger_count']}/{result['trials']}) "
                    f"threshold={result['threshold']:.0%} "
                    f"dist={result.get('reason_distribution', {})}"
                )

    # ── 汇总 ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  验证汇总")
    print("=" * 70)
    total = len(results)
    passed_total = sum(r["passed"] for r in results)
    for cat in categories:
        cat_res = [r for r in results if r["category"] == cat]
        n = len(cat_res)
        p = sum(r["passed"] for r in cat_res)
        print(f"  {cat.upper():<10}: {p}/{n} PASS")
    print(f"\n  总计: {passed_total}/{total} PASS  {'✓ 全部通过' if passed_total == total else '✗ 存在失败项'}")

    print("\n保存输出文件……")
    save_csv(results)
    save_json(results)

    sys.exit(0 if passed_total == total else 1)


if __name__ == "__main__":
    main()

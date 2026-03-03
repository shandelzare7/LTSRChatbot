#!/usr/bin/env python3
"""
scripts/validate_style.py

Style 6D 风格参数公式验证脚本

Part A: 单变量扫描（单调性验证）
  - 扫描 closeness 0→1：WARMTH↑, FORMALITY↓
  - 扫描 momentum 0→1：EMOTIONAL_INTENSITY↑, FORMALITY↓
  - 扫描 arousal（低 P 制造高张力）：WARMTH↓, EMOTIONAL_INTENSITY↑

Part B: 边界/守卫条件验证
  - respect=0.85 → EXPRESSION_MODE ≤ 1（讽刺被禁用）
  - trust=0.30  → EXPRESSION_MODE ≤ 1（比喻/讽刺被禁用）
  - busy=0.85   → EMOTIONAL_INTENSITY 偏低（busy -0.20 权重）
  - 四重联合条件全满足 → EXPRESSION_MODE = 3（讽刺激活）

输出：style_validation_results.csv + style_validation_report.json
"""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

# ── 路径初始化 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.nodes.pipeline.style import Inputs, compute_style_keys  # noqa: E402

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 工具函数 ────────────────────────────────────────────────────────────────

def neutral_inputs(**overrides) -> Inputs:
    """构造全中性基准（所有维度 0.5），再按 overrides 覆盖。"""
    defaults = dict(
        E=0.5, A=0.5, C=0.5, O=0.5, N=0.5,
        P=0.5, Ar=0.5, D=0.5,
        busy=0.5,
        momentum=0.5, topic_appeal=0.5,
        closeness=0.5, trust=0.5, liking=0.5,
        respect=0.5, attractiveness=0.5, power=0.5,
        evidence=None,
    )
    defaults.update(overrides)
    return Inputs(**defaults)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Part A: 单变量扫描（单调性验证）                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

SWEEP_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

SWEEP_SPECS: List[Dict[str, Any]] = [
    {
        "id": "A01",
        "name": "closeness_sweep",
        "label": "closeness 0→1 (momentum=topic_appeal=0.5，其余中性)",
        "var": "closeness",
        "extra_fixed": {},
        "expected_increase": ["WARMTH"],
        "expected_decrease": ["FORMALITY"],
        "hypothesis": (
            "closeness 提升 → familiarity 上升 → "
            "WARMTH↑ (familiarity +0.25), FORMALITY↓ (familiarity -0.40)"
        ),
    },
    {
        "id": "A02",
        "name": "momentum_sweep",
        "label": "momentum 0→1 (closeness=0.5，其余中性)",
        "var": "momentum",
        "extra_fixed": {},
        "expected_increase": ["EMOTIONAL_INTENSITY"],
        "expected_decrease": ["FORMALITY"],
        "hypothesis": (
            "momentum 提升 → EMOTIONAL_INTENSITY↑ (momentum +0.20), "
            "FORMALITY↓ (momentum -0.10)"
        ),
    },
    {
        "id": "A03",
        "name": "tension_via_arousal_sweep",
        "label": "Ar 0→1 (P=0.2 低愉悦制造高张力，其余中性)",
        "var": "Ar",
        "extra_fixed": {"P": 0.2},
        "expected_increase": ["EMOTIONAL_INTENSITY"],
        "expected_decrease": ["WARMTH"],
        "hypothesis": (
            "Ar↑ + 低P → tension↑ → WARMTH↓ (tension -0.30); "
            "Ar↑ → EMOTIONAL_INTENSITY↑ (Ar +0.50)"
        ),
    },
]


def _check_global_trend(
    sweep_data: List[Dict[str, Any]],
    key: str,
    direction: str,
) -> Tuple[bool, float, float]:
    """
    检验全局趋势（首尾比较）。
    direction: "increase" | "decrease"
    返回 (passed, first_val, last_val)
    """
    vals = [d[key] for d in sweep_data]
    first, last = vals[0], vals[-1]
    if direction == "increase":
        passed = last > first
    else:
        passed = last < first
    return passed, first, last


def run_part_a() -> Tuple[List[Dict], List[Dict]]:
    """
    运行单变量扫描测试。
    返回 (sweep_rows_for_csv, summary_rows)
    """
    sweep_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for spec in SWEEP_SPECS:
        var = spec["var"]
        extra = spec.get("extra_fixed", {})

        # ── 构造扫描数据 ──────────────────────────────────────────────────
        scan_data: List[Dict] = []
        for val in SWEEP_VALUES:
            kwargs = {var: val, **extra}
            inp = neutral_inputs(**kwargs)
            style = compute_style_keys(inp)
            row = {
                "part": "A",
                "test_id": spec["id"],
                "sweep_var": var,
                "sweep_value": val,
                **style,
            }
            scan_data.append(row)
            sweep_rows.append(row)

        # ── 检验单调趋势 ──────────────────────────────────────────────────
        checks: Dict[str, Any] = {}
        all_pass = True

        for key in spec.get("expected_increase", []):
            passed, fv, lv = _check_global_trend(scan_data, key, "increase")
            checks[f"{key}_increase"] = {
                "passed": passed,
                "first": round(fv, 4),
                "last": round(lv, 4),
                "delta": round(lv - fv, 4),
            }
            if not passed:
                all_pass = False

        for key in spec.get("expected_decrease", []):
            passed, fv, lv = _check_global_trend(scan_data, key, "decrease")
            checks[f"{key}_decrease"] = {
                "passed": passed,
                "first": round(fv, 4),
                "last": round(lv, 4),
                "delta": round(lv - fv, 4),
            }
            if not passed:
                all_pass = False

        summary_rows.append({
            "test_id": spec["id"],
            "name": spec["name"],
            "label": spec["label"],
            "hypothesis": spec["hypothesis"],
            "all_checks_pass": all_pass,
            "checks": checks,
        })
        status = "PASS ✓" if all_pass else "FAIL ✗"
        print(f"  [{status}] {spec['id']} {spec['name']}")
        for ck, cv in checks.items():
            arrow = "↑" if cv["delta"] > 0 else "↓"
            print(
                f"         {ck}: first={cv['first']:.3f} → last={cv['last']:.3f} "
                f"({arrow}{abs(cv['delta']):.3f})  {'OK' if cv['passed'] else 'FAILED'}"
            )

    return sweep_rows, summary_rows


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Part B: 边界/守卫条件验证                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# 高亲密度基准（本应触发讽刺的场景）
_IRONY_BASE = dict(
    E=0.8, A=0.7, C=0.3, O=1.0, N=0.3,
    P=0.9, Ar=0.5, D=0.5,
    busy=0.1,
    momentum=0.9, topic_appeal=0.9,
    closeness=0.9, trust=0.9, liking=0.9,
    attractiveness=0.9, power=0.5,
    evidence=None,
)

GUARDRAIL_SPECS: List[Dict[str, Any]] = [
    # ── B01: 四重联合条件全满足 → EXPRESSION_MODE = 3 ────────────────────
    {
        "id": "B01",
        "name": "irony_activation",
        "label": "四重联合条件全满足 → EXPRESSION_MODE = 3",
        "hypothesis": (
            "irony_propensity≥0.78, closeness≥0.68, trust≥0.62, "
            "P≥0.58, tension≤0.62, figurative_bias≥0.70 → EXPRESSION_MODE=3"
        ),
        "inputs": {**_IRONY_BASE, "respect": 0.20},
        "checks": [
            ("EXPRESSION_MODE", "==", 3),
        ],
    },
    # ── B02: respect=0.85 → 讽刺被 guardrail 截断为 EXPRESSION_MODE ≤ 1 ─
    {
        "id": "B02",
        "name": "respect_guardrail",
        "label": "respect=0.85 → EXPRESSION_MODE ≤ 1 (讽刺模式被截断)",
        "hypothesis": (
            "guardrail: if respect≥0.80 and EXPRESSION_MODE==3 → EXPRESSION_MODE=1"
        ),
        "inputs": {**_IRONY_BASE, "respect": 0.85},
        "checks": [
            ("EXPRESSION_MODE", "<=", 1),
        ],
    },
    # ── B03: trust=0.30 → 比喻/讽刺被 guardrail 截断 ────────────────────
    {
        "id": "B03",
        "name": "trust_guardrail",
        "label": "trust=0.30 → EXPRESSION_MODE ≤ 1 (低信任禁用比喻/讽刺)",
        "hypothesis": (
            "guardrail: if trust≤0.35 → if EXPRESSION_MODE in (2,3) → EXPRESSION_MODE=1"
        ),
        "inputs": {**_IRONY_BASE, "respect": 0.20, "trust": 0.30},
        "checks": [
            ("EXPRESSION_MODE", "<=", 1),
        ],
    },
    # ── B04: closeness=0.20 → 低亲密度禁用比喻/讽刺 ─────────────────────
    {
        "id": "B04",
        "name": "closeness_guardrail",
        "label": "closeness=0.20 → EXPRESSION_MODE ≤ 1 (低亲密度禁用比喻/讽刺)",
        "hypothesis": (
            "guardrail: if closeness≤0.35 → if EXPRESSION_MODE in (2,3) → EXPRESSION_MODE=1"
        ),
        "inputs": {**_IRONY_BASE, "respect": 0.20, "closeness": 0.20},
        "checks": [
            ("EXPRESSION_MODE", "<=", 1),
        ],
    },
    # ── B05: busy=0.85 → EMOTIONAL_INTENSITY 偏低 ────────────────────────
    {
        "id": "B05",
        "name": "busy_emotional_intensity_low",
        "label": "busy=0.85 → EMOTIONAL_INTENSITY ≤ 0.55",
        "hypothesis": (
            "busy 高 → EMOTIONAL_INTENSITY↓ (busy -0.20 权重)"
        ),
        "inputs": {"busy": 0.85},
        "checks": [
            ("EMOTIONAL_INTENSITY", "<=", 0.55),
        ],
    },
    # ── B06: busy=0.85 → EXPRESSION_MODE = 0 or 1 ───────────────────────
    {
        "id": "B06",
        "name": "busy_expression_mode_guardrail",
        "label": "busy=0.85 → EXPRESSION_MODE ≤ 1 (忙碌禁用比喻/讽刺)",
        "hypothesis": (
            "guardrail: if busy≥0.80 → EXPRESSION_MODE = 0 (if CERTAINTY≥0.55) or 1"
        ),
        "inputs": {"busy": 0.85},
        "checks": [
            ("EXPRESSION_MODE", "<=", 1),
        ],
    },
    # ── B07: 低亲密度场景 FORMALITY 显著高于高亲密度 ─────────────────────
    {
        "id": "B07",
        "name": "closeness_formality_contrast",
        "label": "closeness=0.1 vs 0.9：FORMALITY 应显著更高",
        "hypothesis": (
            "低亲密度(closeness=0.1)的 FORMALITY 应显著高于高亲密度(closeness=0.9)"
        ),
        "inputs_a": {"closeness": 0.1},
        "inputs_b": {"closeness": 0.9},
        "contrast_key": "FORMALITY",
        "contrast_direction": "a_greater",
        "contrast_threshold": 0.10,  # 要求差值 ≥ 0.10
        "checks": [],  # 特殊处理
    },
    # ── B08: 高 momentum 场景 EMOTIONAL_INTENSITY 显著高于低 momentum ──────
    {
        "id": "B08",
        "name": "momentum_emotional_intensity_contrast",
        "label": "momentum=0.9 vs 0.1：EMOTIONAL_INTENSITY 应显著更高",
        "hypothesis": (
            "高动量(momentum=0.9)的 EMOTIONAL_INTENSITY 应显著高于低动量(momentum=0.1)"
        ),
        "inputs_a": {"momentum": 0.9},
        "inputs_b": {"momentum": 0.1},
        "contrast_key": "EMOTIONAL_INTENSITY",
        "contrast_direction": "a_greater",
        "contrast_threshold": 0.05,
        "checks": [],
    },
]


def _eval_check(val: Any, op: str, threshold: Any) -> bool:
    if op == "==":
        return val == threshold
    elif op == "<=":
        return val <= threshold
    elif op == ">=":
        return val >= threshold
    elif op == "<":
        return val < threshold
    elif op == ">":
        return val > threshold
    return False


def run_part_b() -> List[Dict]:
    """运行边界/守卫条件测试，返回结果列表。"""
    results: List[Dict] = []

    for spec in GUARDRAIL_SPECS:
        test_id = spec["id"]

        # ── 对照差异测试（B07/B08）────────────────────────────────────────
        if "inputs_a" in spec:
            inp_a = neutral_inputs(**spec["inputs_a"])
            inp_b = neutral_inputs(**spec["inputs_b"])
            style_a = compute_style_keys(inp_a)
            style_b = compute_style_keys(inp_b)
            key = spec["contrast_key"]
            va, vb = style_a[key], style_b[key]
            diff = va - vb
            if spec["contrast_direction"] == "a_greater":
                passed = diff >= spec["contrast_threshold"]
            else:
                passed = diff <= -spec["contrast_threshold"]

            result = {
                "test_id": test_id,
                "name": spec["name"],
                "label": spec["label"],
                "hypothesis": spec["hypothesis"],
                "type": "contrast",
                "contrast_key": key,
                "value_a": round(va, 4),
                "value_b": round(vb, 4),
                "diff": round(diff, 4),
                "threshold": spec["contrast_threshold"],
                "passed": passed,
                "style_a": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in style_a.items()},
                "style_b": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in style_b.items()},
            }
            results.append(result)
            status = "PASS ✓" if passed else "FAIL ✗"
            print(f"  [{status}] {test_id} {spec['name']}")
            print(f"         {key}: A={va:.3f} vs B={vb:.3f}  diff={diff:+.3f}  (需≥{spec['contrast_threshold']})")
            continue

        # ── 普通单场景守卫测试 ────────────────────────────────────────────
        kwargs = spec.get("inputs", {})
        inp = neutral_inputs(**kwargs)
        style = compute_style_keys(inp)

        check_results: List[Dict] = []
        all_pass = True
        for key, op, threshold in spec.get("checks", []):
            val = style[key]
            passed_chk = _eval_check(val, op, threshold)
            check_results.append({
                "key": key,
                "op": op,
                "threshold": threshold,
                "actual": round(val, 4) if isinstance(val, float) else val,
                "passed": passed_chk,
            })
            if not passed_chk:
                all_pass = False

        result = {
            "test_id": test_id,
            "name": spec["name"],
            "label": spec["label"],
            "hypothesis": spec["hypothesis"],
            "type": "single",
            "style_output": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in style.items()},
            "checks": check_results,
            "passed": all_pass,
        }
        results.append(result)

        status = "PASS ✓" if all_pass else "FAIL ✗"
        print(f"  [{status}] {test_id} {spec['name']}")
        for ck in check_results:
            print(
                f"         {ck['key']} {ck['op']} {ck['threshold']}: "
                f"actual={ck['actual']}  {'OK' if ck['passed'] else 'FAILED'}"
            )

    return results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  输出                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def save_csv(sweep_rows: List[Dict], part_b_results: List[Dict]) -> None:
    """保存 CSV：Part A 扫描数据 + Part B 摘要。"""
    csv_path = os.path.join(OUTPUT_DIR, "style_validation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Part A
        writer.writerow(["=== Part A: 单变量扫描数据 ==="])
        writer.writerow([
            "test_id", "sweep_var", "sweep_value",
            "FORMALITY", "POLITENESS", "WARMTH",
            "CERTAINTY", "EXPRESSION_MODE", "EMOTIONAL_INTENSITY",
        ])
        for row in sweep_rows:
            writer.writerow([
                row["test_id"],
                row["sweep_var"],
                row["sweep_value"],
                round(row["FORMALITY"], 4),
                round(row["POLITENESS"], 4),
                round(row["WARMTH"], 4),
                round(row["CERTAINTY"], 4),
                row["EXPRESSION_MODE"],
                round(row["EMOTIONAL_INTENSITY"], 4),
            ])

        writer.writerow([])
        writer.writerow(["=== Part B: 边界/守卫条件验证摘要 ==="])
        writer.writerow([
            "test_id", "name", "label", "result",
            "check_key", "op", "threshold", "actual",
        ])
        for r in part_b_results:
            if r.get("type") == "contrast":
                writer.writerow([
                    r["test_id"], r["name"], r["label"],
                    "PASS ✓" if r["passed"] else "FAIL ✗",
                    f"{r['contrast_key']} diff",
                    ">=",
                    r["threshold"],
                    f"{r['diff']:+.4f}",
                ])
            else:
                for ck in r.get("checks", []):
                    writer.writerow([
                        r["test_id"], r["name"], r["label"],
                        "PASS ✓" if r["passed"] else "FAIL ✗",
                        ck["key"], ck["op"], ck["threshold"], ck["actual"],
                    ])

    print(f"\n  → CSV 已保存：{csv_path}")


def save_json(sweep_summary: List[Dict], part_b_results: List[Dict]) -> None:
    """保存完整 JSON 报告。"""
    json_path = os.path.join(OUTPUT_DIR, "style_validation_report.json")
    report = {
        "part_a_summary": sweep_summary,
        "part_b_results": part_b_results,
        "part_a_pass": all(r["all_checks_pass"] for r in sweep_summary),
        "part_b_pass": all(r["passed"] for r in part_b_results),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  → JSON 已保存：{json_path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  入口                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    print("=" * 70)
    print("  Style 6D 公式验证脚本")
    print("=" * 70)

    print("\n[Part A] 单变量扫描 —— 单调性验证")
    print("-" * 50)
    sweep_rows, sweep_summary = run_part_a()

    a_pass = all(r["all_checks_pass"] for r in sweep_summary)
    a_str = f"PASS ({sum(r['all_checks_pass'] for r in sweep_summary)}/{len(sweep_summary)})"
    print(f"\nPart A 总结：{a_str}")

    print("\n[Part B] 边界/守卫条件验证")
    print("-" * 50)
    part_b_results = run_part_b()

    b_total = len(part_b_results)
    b_pass_count = sum(r["passed"] for r in part_b_results)
    b_str = f"PASS ({b_pass_count}/{b_total})"
    print(f"\nPart B 总结：{b_str}")

    print("\n保存输出文件……")
    save_csv(sweep_rows, part_b_results)
    save_json(sweep_summary, part_b_results)

    all_ok = a_pass and (b_pass_count == b_total)
    print("\n" + "=" * 70)
    print(f"  整体结果：{'所有测试通过 ✓' if all_ok else '存在失败测试项 ✗'}")
    print("=" * 70)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/validate_stage_gradient.py

第四章验证实验：跨 5 阶段的风格梯度对比（4.5A）
=================================================

固定 Bot Big Five + PAD + momentum，仅改变关系维度参数
（对应 Knapp 5 个成长阶段的典型值），调用 compute_style_keys
输出 6D 风格参数，验证阶段递进时风格的系统性变化。

5 个阶段点：
  initiating / experimenting / intensifying / integrating / bonding

预期趋势（随亲密度递增）：
  FORMALITY ↓  POLITENESS ↓  WARMTH ↑
  CERTAINTY / EMOTIONAL_INTENSITY 不变（不含关系维度驱动因子）

运行方式：
  cd EmotionalChatBot_V5
  python scripts/validate_stage_gradient.py

输出：
  scripts/output/stage_gradient_results.csv
  scripts/output/stage_gradient_report.json
"""
from __future__ import annotations

import csv
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.nodes.pipeline.style import Inputs, compute_style_keys  # noqa: E402

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 固定参数：中等外向、较高宜人性的 Bot 人格
# PAD 处于轻松偏正面状态（P=0.55, Ar=0.25）
# ─────────────────────────────────────────────────────────────
FIXED_BIG5 = dict(E=0.65, A=0.70, C=0.55, O=0.60, N=0.30)
FIXED_PAD = dict(P=0.55, Ar=0.25, D=0.55)
FIXED_CTX = dict(busy=0.3, momentum=0.65, topic_appeal=0.55, evidence=None)

# ─────────────────────────────────────────────────────────────
# 5 个阶段的典型关系维度（参考 knapp_rules.yaml 的 up_entry 阈值）
#
# 设计原则：
#   - initiating：接近初始默认值（0.30）
#   - experimenting：略超过 initiating→experimenting 升级阈值
#   - intensifying：略超过 experimenting→intensifying 升级阈值
#   - integrating：略超过 intensifying→integrating 升级阈值
#   - bonding：略超过 integrating→bonding 升级阈值
# ─────────────────────────────────────────────────────────────
STAGE_PROFILES = [
    {
        "stage": "initiating",
        "desc": "初识期：默认初始值，关系尚未建立",
        "rel": dict(closeness=0.30, trust=0.30, liking=0.30,
                    respect=0.30, attractiveness=0.30, power=0.50),
    },
    {
        "stage": "experimenting",
        "desc": "实验期：试探性互动，关系初建（超过 init→exp 阈值）",
        "rel": dict(closeness=0.35, trust=0.32, liking=0.38,
                    respect=0.42, attractiveness=0.40, power=0.50),
    },
    {
        "stage": "intensifying",
        "desc": "强化期：关系加深，自我暴露增多（超过 exp→int 阈值）",
        "rel": dict(closeness=0.50, trust=0.45, liking=0.50,
                    respect=0.45, attractiveness=0.48, power=0.50),
    },
    {
        "stage": "integrating",
        "desc": "整合期：关系稳定，深度互相了解（超过 int→integ 阈值）",
        "rel": dict(closeness=0.68, trust=0.62, liking=0.65,
                    respect=0.52, attractiveness=0.55, power=0.50),
    },
    {
        "stage": "bonding",
        "desc": "结合期：高度亲密，彼此信赖（超过 integ→bond 阈值）",
        "rel": dict(closeness=0.80, trust=0.75, liking=0.78,
                    respect=0.55, attractiveness=0.60, power=0.50),
    },
]

# 预期的全局趋势（initiating → bonding）
# 注：EMOTIONAL_INTENSITY 由 Ar/tension/momentum 驱动，不直接依赖关系维度，
#     在 PAD 和 momentum 固定时应保持不变。
# 同理 CERTAINTY 由 D/N/C/power/busy/momentum 驱动，不含关系维度，
#     在固定参数下也应保持不变。两者的不变性支持维度独立性。
EXPECTED_TRENDS = {
    "FORMALITY":  "decrease",
    "POLITENESS":  "decrease",
    "WARMTH":      "increase",
    # CERTAINTY 不受关系维度影响 → 不做趋势检验（独立性证据）
    # EMOTIONAL_INTENSITY 不受关系维度影响 → 不做趋势检验（独立性证据）
    # EXPRESSION_MODE 是离散值，不做连续趋势检验
}

STYLE_KEYS = ["FORMALITY", "POLITENESS", "WARMTH", "CERTAINTY",
              "EXPRESSION_MODE", "EMOTIONAL_INTENSITY"]


def run_gradient() -> dict:
    """对 5 个阶段计算 style 参数，返回完整结果数据。"""
    rows = []
    for sp in STAGE_PROFILES:
        inp = Inputs(
            **FIXED_BIG5, **FIXED_PAD,
            **sp["rel"],
            busy=FIXED_CTX["busy"],
            momentum=FIXED_CTX["momentum"],
            topic_appeal=FIXED_CTX["topic_appeal"],
            evidence=FIXED_CTX["evidence"],
        )
        style = compute_style_keys(inp)
        rows.append({
            "stage": sp["stage"],
            "desc": sp["desc"],
            "relationship": sp["rel"],
            "style": {k: round(float(style[k]), 4) for k in STYLE_KEYS},
        })
    return rows


def check_trends(rows: list) -> list:
    """检验全局趋势（首→尾比较），返回检查结果列表。"""
    checks = []
    first = rows[0]["style"]
    last = rows[-1]["style"]
    for dim, direction in EXPECTED_TRENDS.items():
        fv = first[dim]
        lv = last[dim]
        if direction == "increase":
            passed = lv > fv
        else:
            passed = lv < fv
        checks.append({
            "dimension": dim,
            "expected": direction,
            "first_stage": rows[0]["stage"],
            "last_stage": rows[-1]["stage"],
            "first_value": fv,
            "last_value": lv,
            "delta": round(lv - fv, 4),
            "passed": passed,
        })
    return checks


def print_results(rows: list, checks: list) -> None:
    """格式化打印结果。"""
    print("\n" + "=" * 80)
    print("  跨阶段风格梯度对比（4.5A）")
    print("=" * 80)

    print(f"\n固定参数：")
    print(f"  Big Five: E={FIXED_BIG5['E']} A={FIXED_BIG5['A']} C={FIXED_BIG5['C']} "
          f"O={FIXED_BIG5['O']} N={FIXED_BIG5['N']}")
    print(f"  PAD: P={FIXED_PAD['P']} Ar={FIXED_PAD['Ar']} D={FIXED_PAD['D']}")
    print(f"  Context: busy={FIXED_CTX['busy']} momentum={FIXED_CTX['momentum']} "
          f"topic_appeal={FIXED_CTX['topic_appeal']}")

    # 表头
    print(f"\n{'阶段':<16}", end="")
    for k in STYLE_KEYS:
        print(f"{k:>18}", end="")
    print()
    print("-" * (16 + 18 * len(STYLE_KEYS)))

    # 数据行
    for row in rows:
        print(f"{row['stage']:<16}", end="")
        for k in STYLE_KEYS:
            v = row["style"][k]
            print(f"{v:>18.4f}" if isinstance(v, float) else f"{v:>18}", end="")
        print()

    # 趋势变化（箭头标注）
    print(f"\n{'变化趋势':<16}", end="")
    first = rows[0]["style"]
    last = rows[-1]["style"]
    for k in STYLE_KEYS:
        delta = float(last[k]) - float(first[k])
        if abs(delta) < 0.01:
            marker = "  ~"
        elif delta > 0:
            marker = f"  ↑{delta:+.3f}"
        else:
            marker = f"  ↓{delta:+.3f}"
        print(f"{marker:>18}", end="")
    print()

    # 检验结果
    print(f"\n{'─' * 60}")
    print("趋势检验结果：")
    all_pass = True
    for ck in checks:
        status = "PASS ✓" if ck["passed"] else "FAIL ✗"
        arrow = "↑" if ck["expected"] == "increase" else "↓"
        print(f"  [{status}] {ck['dimension']}: 预期{arrow}  "
              f"{ck['first_stage']}={ck['first_value']:.4f} → "
              f"{ck['last_stage']}={ck['last_value']:.4f}  "
              f"(delta={ck['delta']:+.4f})")
        if not ck["passed"]:
            all_pass = False

    print(f"\n总结：{'所有趋势检验通过 ✓' if all_pass else '存在失败项 ✗'}")
    return all_pass


def save_outputs(rows: list, checks: list) -> None:
    """保存 CSV 和 JSON。"""
    # CSV
    csv_path = os.path.join(OUTPUT_DIR, "stage_gradient_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "desc"] + STYLE_KEYS +
                        ["closeness", "trust", "liking", "respect",
                         "attractiveness", "power"])
        for row in rows:
            writer.writerow([
                row["stage"], row["desc"],
                *[row["style"][k] for k in STYLE_KEYS],
                *[row["relationship"][k] for k in
                  ["closeness", "trust", "liking", "respect",
                   "attractiveness", "power"]],
            ])
    print(f"\n  → CSV 已保存：{csv_path}")

    # JSON
    json_path = os.path.join(OUTPUT_DIR, "stage_gradient_report.json")
    report = {
        "experiment": "4.5A 跨阶段风格梯度对比",
        "fixed_params": {
            "big_five": FIXED_BIG5,
            "pad": FIXED_PAD,
            "context": FIXED_CTX,
        },
        "stages": rows,
        "trend_checks": checks,
        "all_pass": all(ck["passed"] for ck in checks),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  → JSON 已保存：{json_path}")


def main() -> None:
    rows = run_gradient()
    checks = check_trends(rows)
    all_pass = print_results(rows, checks)
    save_outputs(rows, checks)
    print("=" * 80)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

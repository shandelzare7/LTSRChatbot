"""
生成第四章所需的可视化图表。

输出目录：scripts/output/figures/
  fig4_1_sweep_trends.png   — 梯度扫描折线图（实验 A）
  fig4_2_correlation_heatmap.png — 正交性热力图（N=1000）
  fig4_3_stage_gradient.png — 跨阶段风格梯度折线图
  fig4_4_conflict_repair.png — 冲突-修复场景时序图

运行：
  cd EmotionalChatBot_V5
  python scripts/generate_chapter4_figures.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ─── 中文字体 ───
for font in ["PingFang SC", "Heiti SC", "SimHei", "Microsoft YaHei"]:
    try:
        rcParams["font.sans-serif"] = [font] + rcParams["font.sans-serif"]
        break
    except Exception:
        pass
rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "output", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================================
# 图 4-1：梯度扫描折线图
# =====================================================================
def fig4_1():
    csv_path = os.path.join(os.path.dirname(__file__), "output", "style_validation_results.csv")
    with open(csv_path, encoding="utf-8") as f:
        lines = f.readlines()

    # 解析 Part A 数据
    header = None
    data = {}  # test_id -> list of rows
    for line in lines:
        line = line.strip()
        if line.startswith("===") or not line:
            continue
        if line.startswith("test_id,"):
            header = line.split(",")
            continue
        if header and line[0] == "A":
            parts = line.split(",")
            row = dict(zip(header, parts))
            tid = row["test_id"]
            data.setdefault(tid, []).append(row)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    configs = [
        ("A01", "closeness", ["WARMTH", "FORMALITY"], "扫描 A01: closeness 0.1→0.9"),
        ("A02", "momentum", ["EMOTIONAL_INTENSITY", "FORMALITY"], "扫描 A02: momentum 0.1→0.9"),
        ("A03", "Ar", ["EMOTIONAL_INTENSITY", "WARMTH"], "扫描 A03: Arousal 0.1→0.9 (P=0.2)"),
    ]

    colors = {"WARMTH": "#e74c3c", "FORMALITY": "#3498db",
              "EMOTIONAL_INTENSITY": "#e67e22", "POLITENESS": "#2ecc71",
              "CERTAINTY": "#9b59b6"}

    for ax, (tid, xvar, dims, title) in zip(axes, configs):
        rows = data[tid]
        xs = [float(r["sweep_value"]) for r in rows]
        for dim in dims:
            ys = [float(r[dim]) for r in rows]
            ax.plot(xs, ys, "o-", label=dim, color=colors.get(dim, "#333"),
                    markersize=5, linewidth=2)
        ax.set_xlabel(xvar, fontsize=11)
        ax.set_ylabel("输出值", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(0.05, 0.95)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_1_sweep_trends.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"✓ {path}")


# =====================================================================
# 图 4-2：正交性热力图（N=1000 随机采样）
# =====================================================================
def fig4_2():
    from app.nodes.pipeline.style import compute_style_keys, Inputs

    rng = np.random.default_rng(42)
    N = 1000
    dims_out = ["FORMALITY", "POLITENESS", "WARMTH", "CERTAINTY",
                "EXPRESSION_MODE", "EMOTIONAL_INTENSITY"]
    labels_short = ["FORM", "POLIT", "WARM", "CERT", "EM", "EI"]
    results = {d: [] for d in dims_out}

    for _ in range(N):
        vals = rng.uniform(0, 1, 17)
        inp = Inputs(
            E=vals[0], A=vals[1], C=vals[2], O=vals[3], N=vals[4],
            P=vals[5], Ar=vals[6], D=vals[7],
            closeness=vals[8], trust=vals[9], liking=vals[10],
            respect=vals[11], attractiveness=vals[12], power=vals[13],
            busy=vals[14], momentum=vals[15], topic_appeal=vals[16],
            evidence=None,
        )
        style = compute_style_keys(inp)
        for d in dims_out:
            results[d].append(float(style[d]))

    mat = np.zeros((6, 6))
    for i, di in enumerate(dims_out):
        for j, dj in enumerate(dims_out):
            if i == j:
                mat[i][j] = 1.0
            else:
                mat[i][j] = abs(np.corrcoef(results[di], results[dj])[0, 1])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(labels_short, fontsize=10)
    ax.set_yticklabels(labels_short, fontsize=10)

    for i in range(6):
        for j in range(6):
            val = mat[i][j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_title("6D 风格维度 Pearson |r| 相关矩阵 (N=1000)", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="|r|")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_2_correlation_heatmap.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"✓ {path}")


# =====================================================================
# 图 4-3：跨阶段风格梯度折线图
# =====================================================================
def fig4_3():
    json_path = os.path.join(os.path.dirname(__file__), "output", "stage_gradient_report.json")
    with open(json_path, encoding="utf-8") as f:
        report = json.load(f)

    stages_en = [s["stage"] for s in report["stages"]]
    stages_cn = ["初识", "探索", "强化", "整合", "结合"]
    labels = [f"{cn}\n{en}" for cn, en in zip(stages_cn, stages_en)]

    dims = ["FORMALITY", "POLITENESS", "WARMTH", "CERTAINTY", "EMOTIONAL_INTENSITY"]
    colors = {"WARMTH": "#e74c3c", "FORMALITY": "#3498db",
              "POLITENESS": "#2ecc71", "CERTAINTY": "#9b59b6",
              "EMOTIONAL_INTENSITY": "#e67e22"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(5)

    for dim in dims:
        ys = [s["style"][dim] for s in report["stages"]]
        ax.plot(x, ys, "o-", label=dim, color=colors[dim], markersize=7, linewidth=2.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("风格参数值", fontsize=12)
    ax.set_title("Knapp 5 阶段风格梯度变化", fontsize=13)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.15, 1.0)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_3_stage_gradient.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"✓ {path}")


# =====================================================================
# 图 4-4：冲突-修复场景时序图
# =====================================================================
def fig4_4():
    json_path = os.path.join(os.path.dirname(__file__), "output", "conflict_repair_report.json")
    with open(json_path, encoding="utf-8") as f:
        report = json.load(f)

    turns = report["turns"]
    x = [t["turn"] for t in turns]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # 上图：style 参数
    for dim, color, label in [
        ("WARMTH", "#e74c3c", "WARMTH"),
        ("FORMALITY", "#3498db", "FORMALITY"),
        ("EMOTIONAL_INTENSITY", "#e67e22", "EI"),
    ]:
        ys = [t["style"][dim] for t in turns]
        ax1.plot(x, ys, "o-", color=color, label=label, markersize=6, linewidth=2)

    ax1.axvline(x=4, color="red", linestyle="--", alpha=0.5, label="冲突注入")
    ax1.axvspan(5, 7, alpha=0.08, color="gray")
    ax1.axvspan(8, 10, alpha=0.08, color="green")
    ax1.set_ylabel("风格参数值", fontsize=11)
    ax1.set_title("冲突-修复场景：风格参数与关系维度变化", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 下图：关系维度
    for dim, color, label in [
        ("closeness", "#2c3e50", "closeness"),
        ("trust", "#8e44ad", "trust"),
        ("liking", "#16a085", "liking"),
    ]:
        ys = [t["rel"][dim] for t in turns]
        ax2.plot(x, ys, "s-", color=color, label=label, markersize=6, linewidth=2)

    ax2.axvline(x=4, color="red", linestyle="--", alpha=0.5)
    ax2.axvspan(5, 7, alpha=0.08, color="gray")
    ax2.axvspan(8, 10, alpha=0.08, color="green")
    ax2.set_xlabel("轮次", fontsize=11)
    ax2.set_ylabel("关系维度值", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 阶段标注
    for t in turns:
        if t["turn"] == 3:
            ax2.annotate("稳定期", xy=(2, 0.44), fontsize=9, color="gray", ha="center")
        elif t["turn"] == 6:
            ax2.annotate("低谷期", xy=(6, 0.03), fontsize=9, color="gray", ha="center")
        elif t["turn"] == 9:
            ax2.annotate("修复期", xy=(9, 0.33), fontsize=9, color="green", ha="center")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_4_conflict_repair.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"✓ {path}")


if __name__ == "__main__":
    fig4_1()
    fig4_2()
    fig4_3()
    fig4_4()
    print(f"\n全部图表已生成到 {OUT_DIR}/")

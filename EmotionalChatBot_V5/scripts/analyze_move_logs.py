#!/usr/bin/env python3
"""
scripts/analyze_move_logs.py

从 bot-to-bot 日志中提取 Content Move 选择数据，进行多维度分析：

1. 总体 Move 分布（跨所有日志）
2. 按阶段（stage）分组的 Move 分布
3. Move 共现矩阵（哪些 Move 经常一起被选）
4. 按轮次位置的 Move 分布（前 1/3, 中 1/3, 后 1/3）

输出：scripts/output/move_analysis_report.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 与 config/content_moves.yaml 中 moves 保持一致
MOVE_NAMES = {
    1: "追问细节",
    2: "复述确认",
    3: "反问回去",
    4: "给个判断",
    5: "说句体感",
    6: "接一句自己的事",
    7: "拉个对比",
    8: "带一嘴当下",
    9: "说个原因",
    10: "换个如果",
    11: "往后推一步",
    12: "出个主意",
    13: "唱个反调",
}


def _find_log_files() -> List[str]:
    """找到所有包含 selected_content_move_ids 的日志文件。"""
    files = []
    for fn in sorted(os.listdir(LOGS_DIR)):
        if not fn.endswith(".log"):
            continue
        path = os.path.join(LOGS_DIR, fn)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                head = f.read(50000)
            if "selected_content_move_ids" in head:
                files.append(path)
        except Exception:
            pass
    return files


def _extract_rounds(log_path: str) -> List[Dict[str, Any]]:
    """
    从单个日志中提取每轮的 (stage, move_ids, bot_stance, round_idx) 数据。

    解析策略：
    - stage: 从 "阶段ID": xxx 行提取
    - move_ids: 从 【解析结果 (Parsed)】 后的 JSON 中提取 selected_content_move_ids
    - 排除 prompt 模板中的 move_ids（只取 Parsed 结果块）
    """
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    rounds = []
    current_stage = None

    # 找所有 stage 声明
    stage_pattern = re.compile(r"\*\*阶段ID\*\*:\s*(\w+)")
    # 找所有 Parsed 结果块
    parsed_pattern = re.compile(
        r"【解析结果 \(Parsed\)】\s*={10,}\s*\n(.*?)={10,}",
        re.DOTALL,
    )

    # 先收集所有 stage 出现的位置
    stage_positions = [(m.start(), m.group(1)) for m in stage_pattern.finditer(content)]

    # 收集所有 parsed block
    parsed_blocks = list(parsed_pattern.finditer(content))

    # 过滤：只要 Extract 节点的 parsed block（包含 selected_content_move_ids）
    round_idx = 0
    for match in parsed_blocks:
        block_text = match.group(1)
        if "selected_content_move_ids" not in block_text:
            continue

        # 找这个 block 之前最近的 stage
        block_pos = match.start()
        stage = None
        for spos, sname in reversed(stage_positions):
            if spos < block_pos:
                stage = sname
                break

        # 解析 JSON
        try:
            # 清理可能的注释
            clean = re.sub(r"//.*?$", "", block_text, flags=re.MULTILINE)
            # 尝试提取 JSON 对象
            json_match = re.search(r"\{.*\}", clean, re.DOTALL)
            if not json_match:
                continue
            data = json.loads(json_match.group())
        except (json.JSONDecodeError, Exception):
            # 尝试手动提取 move_ids
            ids_match = re.search(
                r'"selected_content_move_ids"\s*:\s*\[([\d\s,]+)\]', block_text
            )
            if not ids_match:
                continue
            data = {
                "selected_content_move_ids": [
                    int(x.strip()) for x in ids_match.group(1).split(",") if x.strip()
                ]
            }

        move_ids = data.get("selected_content_move_ids", [])
        if not move_ids or not isinstance(move_ids, list):
            continue

        # 过滤非法 id
        move_ids = [mid for mid in move_ids if isinstance(mid, int) and 1 <= mid <= 8]
        if not move_ids:
            continue

        bot_stance = data.get("bot_stance", "")

        rounds.append({
            "stage": stage or "unknown",
            "move_ids": move_ids,
            "bot_stance": bot_stance,
            "round_idx": round_idx,
        })
        round_idx += 1

    return rounds


def analyze(all_rounds: List[Dict]) -> Dict[str, Any]:
    """汇总分析。"""
    total_rounds = len(all_rounds)
    total_slots = sum(len(r["move_ids"]) for r in all_rounds)

    # 1. 总体 Move 分布
    move_counter = Counter()
    for r in all_rounds:
        for mid in r["move_ids"]:
            move_counter[mid] += 1

    overall_dist = []
    for mid in sorted(move_counter.keys()):
        count = move_counter[mid]
        overall_dist.append({
            "move_id": mid,
            "name": MOVE_NAMES.get(mid, f"move_{mid}"),
            "count": count,
            "pct": round(count / total_slots * 100, 1) if total_slots else 0,
            "rounds_with": sum(1 for r in all_rounds if mid in r["move_ids"]),
            "rounds_pct": round(
                sum(1 for r in all_rounds if mid in r["move_ids"]) / total_rounds * 100, 1
            ) if total_rounds else 0,
        })

    # 2. 按阶段分组
    stage_groups = defaultdict(list)
    for r in all_rounds:
        stage_groups[r["stage"]].append(r)

    stage_dist = {}
    for stage, rounds in sorted(stage_groups.items()):
        n = len(rounds)
        slots = sum(len(r["move_ids"]) for r in rounds)
        counter = Counter()
        for r in rounds:
            for mid in r["move_ids"]:
                counter[mid] += 1
        stage_dist[stage] = {
            "n_rounds": n,
            "n_slots": slots,
            "distribution": {
                mid: {
                    "count": counter.get(mid, 0),
                    "pct": round(counter.get(mid, 0) / slots * 100, 1) if slots else 0,
                }
                for mid in range(1, 9)
            },
        }

    # 3. 共现矩阵（哪些 move 经常一起出现）
    cooccurrence = [[0] * 8 for _ in range(8)]
    for r in all_rounds:
        ids = set(r["move_ids"])
        for i in ids:
            for j in ids:
                if 1 <= i <= 8 and 1 <= j <= 8:
                    cooccurrence[i - 1][j - 1] += 1

    # 4. 按轮次位置分组（早期/中期/晚期）
    # 对每个日志会话，按 round_idx 分三段
    position_dist = {"early": Counter(), "mid": Counter(), "late": Counter()}
    # 按日志文件分组（用 round_idx 重置判断）
    # 简化：直接用 all_rounds 中的 round_idx，按文件分组
    # 实际上 all_rounds 已经是跨文件的，需要按文件追踪
    # 这里用另一种方式：对每个文件单独处理
    # ... 在主函数中处理

    # 5. bot_stance 分布
    stance_counter = Counter(r["bot_stance"] for r in all_rounds if r.get("bot_stance"))

    # 6. 每轮选择的 move 数量分布
    move_count_dist = Counter(len(r["move_ids"]) for r in all_rounds)

    return {
        "total_logs": None,  # 在外部填充
        "total_rounds": total_rounds,
        "total_slots": total_slots,
        "avg_moves_per_round": round(total_slots / total_rounds, 2) if total_rounds else 0,
        "move_count_distribution": dict(sorted(move_count_dist.items())),
        "overall_distribution": overall_dist,
        "stage_distribution": stage_dist,
        "cooccurrence_matrix": {
            "labels": [f"move_{i}" for i in range(1, 9)],
            "matrix": cooccurrence,
        },
        "bot_stance_distribution": dict(stance_counter.most_common()),
    }


def analyze_position(file_rounds_map: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """按轮次位置（早/中/晚）分析 Move 分布。"""
    position_dist = {"early": Counter(), "mid": Counter(), "late": Counter()}
    position_counts = {"early": 0, "mid": 0, "late": 0}

    for filepath, rounds in file_rounds_map.items():
        n = len(rounds)
        if n < 3:
            continue
        third = n / 3
        for i, r in enumerate(rounds):
            if i < third:
                pos = "early"
            elif i < 2 * third:
                pos = "mid"
            else:
                pos = "late"
            for mid in r["move_ids"]:
                position_dist[pos][mid] += 1
            position_counts[pos] += 1

    result = {}
    for pos in ["early", "mid", "late"]:
        total = sum(position_dist[pos].values())
        result[pos] = {
            "n_rounds": position_counts[pos],
            "distribution": {
                mid: {
                    "count": position_dist[pos].get(mid, 0),
                    "pct": round(position_dist[pos].get(mid, 0) / total * 100, 1) if total else 0,
                }
                for mid in range(1, 9)
            },
        }
    return result


def print_report(report: Dict) -> None:
    """格式化打印报告。"""
    print("=" * 70)
    print("  Content Move 日志分析报告")
    print("=" * 70)
    print(f"\n  日志文件数: {report['total_logs']}")
    print(f"  总轮次数: {report['total_rounds']}")
    print(f"  总 Move 槽位: {report['total_slots']}")
    print(f"  平均每轮选择 Move 数: {report['avg_moves_per_round']}")

    print(f"\n  每轮 Move 数量分布:")
    for k, v in sorted(report["move_count_distribution"].items()):
        print(f"    {k} 个: {v} 轮 ({v/report['total_rounds']*100:.1f}%)")

    print(f"\n{'─' * 70}")
    print("  总体 Move 分布")
    print(f"{'─' * 70}")
    print(f"  {'Move':<20} {'次数':>6} {'占比':>8} {'出现轮次':>10} {'轮次占比':>10}")
    for d in report["overall_distribution"]:
        print(
            f"  move_{d['move_id']}({d['name']:<6})"
            f" {d['count']:>6}"
            f" {d['pct']:>7.1f}%"
            f" {d['rounds_with']:>10}"
            f" {d['rounds_pct']:>9.1f}%"
        )

    print(f"\n{'─' * 70}")
    print("  按阶段分组的 Move 分布")
    print(f"{'─' * 70}")
    for stage, data in report["stage_distribution"].items():
        print(f"\n  [{stage}] ({data['n_rounds']} 轮, {data['n_slots']} 槽位)")
        for mid in range(1, 9):
            d = data["distribution"][mid]
            if d["count"] > 0:
                bar = "#" * int(d["pct"] / 2)
                print(f"    move_{mid}: {d['count']:>4} ({d['pct']:>5.1f}%) {bar}")

    if "position_distribution" in report:
        print(f"\n{'─' * 70}")
        print("  按轮次位置（早/中/晚期）的 Move 分布")
        print(f"{'─' * 70}")
        for pos in ["early", "mid", "late"]:
            data = report["position_distribution"][pos]
            label = {"early": "前1/3", "mid": "中1/3", "late": "后1/3"}[pos]
            print(f"\n  [{label}] ({data['n_rounds']} 轮)")
            for mid in range(1, 9):
                d = data["distribution"][mid]
                if d["count"] > 0:
                    print(f"    move_{mid}({MOVE_NAMES.get(mid, '?'):<6}): {d['count']:>4} ({d['pct']:>5.1f}%)")

    print(f"\n{'─' * 70}")
    print("  bot_stance 分布")
    print(f"{'─' * 70}")
    for stance, count in report.get("bot_stance_distribution", {}).items():
        print(f"  {stance:<20} {count:>6} ({count/report['total_rounds']*100:.1f}%)")

    print("\n" + "=" * 70)


def main() -> None:
    log_files = _find_log_files()
    print(f"找到 {len(log_files)} 个含 Move 数据的日志文件")

    all_rounds = []
    file_rounds_map = {}
    for lf in log_files:
        rounds = _extract_rounds(lf)
        if rounds:
            all_rounds.extend(rounds)
            file_rounds_map[lf] = rounds

    print(f"共提取 {len(all_rounds)} 轮有效 Move 数据")

    report = analyze(all_rounds)
    report["total_logs"] = len(log_files)
    report["position_distribution"] = analyze_position(file_rounds_map)

    print_report(report)

    json_path = os.path.join(OUTPUT_DIR, "move_analysis_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  -> JSON 已保存: {json_path}")


if __name__ == "__main__":
    main()

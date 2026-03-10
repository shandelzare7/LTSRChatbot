#!/usr/bin/env python3
"""
Pre-allocate annotation tasks for 10 annotators.
Loads annotation JSONL → Jaccard dedup → stratified sampling → per-annotator JSON.

Usage:
    python allocate_tasks.py [--threshold 0.2] [--seed 42] [--no-cache]
"""

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import jieba

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
OUTPUT_DIR = ROOT / "scripts" / "output"

# ─── 配置 ───
N_ANNOTATORS = 10
TASK_A_COUNT = 300        # Move 识别
TASK_B1_COUNT = 400       # Style A/B 对比（对数）
TASK_B2_COUNT = 200       # Style 直接标签
TASK_C_COUNT = 100        # ExprMode
REPEAT_RATIO = 0.05       # 5% 隐藏重复

STYLE_DIMS = ["FORMALITY", "POLITENESS", "FRIENDLINESS", "CERTAINTY", "EMOTIONAL_TONE"]
TIERS = ["EL", "L", "M", "H", "EH"]
EM_VALUES = [0, 1, 2, 3]
ROUTES = ["move_1", "move_2", "move_3", "move_4", "move_5",
          "move_6", "move_7", "move_8", "free"]

# B1：每维度 80 对，按档距分配
#   档距1 (4组合×6对=24)  档距2 (3×8=24)  档距3 (2×8=16)  档距4 (1×16=16)  = 80
B1_GAP_SPEC = [
    (1, [("EL", "L"), ("L", "M"), ("M", "H"), ("H", "EH")], 6),
    (2, [("EL", "M"), ("L", "H"), ("M", "EH")], 8),
    (3, [("EL", "H"), ("L", "EH")], 8),
    (4, [("EL", "EH")], 16),
]


def tier_of(v: float) -> str:
    if v < 0.2:
        return "EL"
    if v < 0.4:
        return "L"
    if v < 0.6:
        return "M"
    if v < 0.8:
        return "H"
    return "EH"


# ─── 分词 & 去重 ───
_PUNC = re.compile(
    r"[，。！？、；：\u201c\u201d\u2018\u2019（）【】《》…—"
    r"\s\u3000,.!?;:\"'()\[\]{}<>\-_/\\@#$%^&*+=|~`]+"
)


def _tokenize(text: str) -> set[str]:
    return {
        _PUNC.sub("", t).strip().lower()
        for t in jieba.lcut(text)
        if _PUNC.sub("", t).strip()
    }


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ─── 加载 ───
def load_candidates(log_dir: Path, min_len: int = 5) -> list[dict]:
    records = []
    files = sorted(log_dir.glob("*_annotation.jsonl"))
    print(f"Loading from {len(files)} annotation files...")
    for f in files:
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            style = row.get("style", {})
            em = int(style.get("EXPRESSION_MODE", 0))
            context = row.get("user_message", row.get("context", ""))
            session = row.get("session_label", "")
            for cand in row.get("candidates", []):
                text = cand.get("text", "")
                if len(text) < min_len:
                    continue
                style_vals = {d: style.get(d, 0.5) for d in STYLE_DIMS}
                records.append({
                    "text": text,
                    "context": context,
                    "route": cand.get("route", "unknown"),
                    "session": session,
                    "round": row.get("round", 0),
                    "style": style_vals,
                    "em": em,
                    "tiers": {d: tier_of(v) for d, v in style_vals.items()},
                })
    print(f"Loaded {len(records)} candidates")
    return records


def dedup(records: list[dict], threshold: float) -> list[dict]:
    print(f"Deduplicating (Jaccard >= {threshold})...")
    t0 = time.time()
    token_sets = [_tokenize(r["text"]) for r in records]
    print(f"  Tokenized in {time.time() - t0:.1f}s")

    kept_idx: list[int] = []
    kept_ts: list[set] = []
    t1 = time.time()
    for i, ts in enumerate(token_sets):
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(records)}, kept {len(kept_idx)}, {time.time() - t1:.0f}s")
        dup = False
        for kt in kept_ts:
            if _jaccard(ts, kt) >= threshold:
                dup = True
                break
        if not dup:
            kept_idx.append(i)
            kept_ts.append(ts)

    kept = [records[i] for i in kept_idx]
    removed = len(records) - len(kept)
    print(f"  Result: {len(records)} → {len(kept)} (removed {removed}, {removed / len(records) * 100:.1f}%)")
    print(f"  Time: {time.time() - t1:.1f}s")
    return kept


# ─── 建索引 ───
def build_indices(pool: list[dict]):
    by_route: dict[str, list[int]] = defaultdict(list)
    by_dim_tier: dict[str, dict[str, list[int]]] = {d: defaultdict(list) for d in STYLE_DIMS}
    by_em: dict[int, list[int]] = defaultdict(list)

    for i, r in enumerate(pool):
        by_route[r["route"]].append(i)
        for d in STYLE_DIMS:
            by_dim_tier[d][r["tiers"][d]].append(i)
        by_em[r["em"]].append(i)

    return by_route, by_dim_tier, by_em


# ─── 抽样辅助 ───
def _sample(rng: random.Random, indices: list[int], n: int,
            exclude: set[int]) -> list[int]:
    avail = [i for i in indices if i not in exclude]
    rng.shuffle(avail)
    return avail[:n]


# ─── 单个标注员分配 ───
def allocate(pool, by_route, by_dim_tier, by_em, annotator_id, base_seed):
    rng = random.Random(base_seed + annotator_id * 42)
    used: set[int] = set()
    tasks: list[dict] = []
    warnings: list[str] = []

    def _rec(idx, task_type, **extra):
        r = pool[idx]
        t = {"task_type": task_type, **extra, "is_repeat": False}
        if task_type == "move":
            t["context_user_text"] = r["context"]
            t["bot_text"] = r["text"]
            t["ground_truth"] = {"route": r["route"]}
        elif task_type == "style_label":
            t["context_user_text"] = r["context"]
            t["bot_text"] = r["text"]
        elif task_type == "expr_mode":
            t["context_user_text"] = r["context"]
            t["bot_text"] = r["text"]
            t["ground_truth"] = {"em": r["em"]}
        return t

    # ── Task C: ExprMode (100) ──
    per_em = TASK_C_COUNT // len(EM_VALUES)  # 25
    for em in EM_VALUES:
        sampled = _sample(rng, by_em[em], per_em, used)
        if len(sampled) < per_em:
            warnings.append(f"EM={em}: wanted {per_em}, got {len(sampled)}")
        for idx in sampled:
            used.add(idx)
            tasks.append(_rec(idx, "expr_mode"))

    # ── Task A: Move (300) ──
    per_route = TASK_A_COUNT // len(ROUTES)  # 33
    remainder = TASK_A_COUNT - per_route * len(ROUTES)  # 3
    for ri, route in enumerate(ROUTES):
        n = per_route + (1 if ri < remainder else 0)
        sampled = _sample(rng, by_route[route], n, used)
        if len(sampled) < n:
            warnings.append(f"Move {route}: wanted {n}, got {len(sampled)}")
        for idx in sampled:
            used.add(idx)
            tasks.append(_rec(idx, "move"))

    # ── Task B2: Style label (200) ──
    per_dim = TASK_B2_COUNT // len(STYLE_DIMS)  # 40
    per_tier = per_dim // len(TIERS)  # 8
    for dim in STYLE_DIMS:
        for tier in TIERS:
            sampled = _sample(rng, by_dim_tier[dim][tier], per_tier, used)
            if len(sampled) < per_tier:
                warnings.append(f"B2 {dim}/{tier}: wanted {per_tier}, got {len(sampled)}")
            for idx in sampled:
                used.add(idx)
                tasks.append(_rec(idx, "style_label",
                                  dimension=dim,
                                  ground_truth={"tier": tier,
                                                "value": round(pool[idx]["style"][dim], 4)}))

    # ── Task B1: Style A/B comparison (400 pairs) ──
    for dim in STYLE_DIMS:
        for gap, combos, per_combo in B1_GAP_SPEC:
            for tier_lo, tier_hi in combos:
                avail_lo = [i for i in by_dim_tier[dim][tier_lo] if i not in used]
                avail_hi = [i for i in by_dim_tier[dim][tier_hi] if i not in used]
                rng.shuffle(avail_lo)
                rng.shuffle(avail_hi)

                # 尽量不同 session 配对
                n = min(per_combo, len(avail_lo), len(avail_hi))
                if n < per_combo:
                    warnings.append(
                        f"B1 {dim} {tier_lo}↔{tier_hi} (gap={gap}): "
                        f"wanted {per_combo}, got {n}"
                    )

                for p in range(n):
                    idx_lo = avail_lo[p]
                    idx_hi = avail_hi[p]
                    used.add(idx_lo)
                    used.add(idx_hi)

                    r_lo = pool[idx_lo]
                    r_hi = pool[idx_hi]

                    # 随机交换展示顺序
                    swap = rng.random() < 0.5
                    if swap:
                        da, db = r_hi, r_lo
                        higher = "a"
                    else:
                        da, db = r_lo, r_hi
                        higher = "b"

                    tasks.append({
                        "task_type": "style_compare",
                        "dimension": dim,
                        "text_a_context": da["context"],
                        "text_a_bot": da["text"],
                        "text_b_context": db["context"],
                        "text_b_bot": db["text"],
                        "ground_truth": {
                            "tier_lo": tier_lo,
                            "tier_hi": tier_hi,
                            "gap": gap,
                            "higher": higher,
                        },
                        "is_qc_anchor": gap >= 3,
                        "is_repeat": False,
                    })

    # ── 5% 隐藏重复 ──
    by_type: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        by_type[t["task_type"]].append(t)

    repeats = []
    for ttype, ttasks in by_type.items():
        n_rep = max(1, int(len(ttasks) * REPEAT_RATIO))
        sources = rng.sample(ttasks, min(n_rep, len(ttasks)))
        for src in sources:
            rep = dict(src)
            rep["is_repeat"] = True
            repeats.append(rep)
    tasks.extend(repeats)

    # ── 编号 & 打乱 ──
    rng.shuffle(tasks)
    for i, t in enumerate(tasks):
        t["task_id"] = f"ann{annotator_id}_{i + 1:04d}"

    return tasks, warnings


# ─── main ───
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pool_file = OUTPUT_DIR / "deduped_pool.json"

    if pool_file.exists() and not args.no_cache:
        print(f"Loading cached pool from {pool_file}")
        pool = json.loads(pool_file.read_text())
    else:
        records = load_candidates(LOG_DIR)
        pool = dedup(records, args.threshold)
        pool_file.write_text(json.dumps(pool, ensure_ascii=False))
        print(f"Saved pool ({len(pool)} records) to {pool_file}")

    print(f"\nPool: {len(pool)} candidates")

    # 打印池子分布
    rc = Counter(r["route"] for r in pool)
    print("  Routes:", {k: v for k, v in rc.most_common()})
    ec = Counter(r["em"] for r in pool)
    print("  EM:", dict(sorted(ec.items())))

    by_route, by_dim_tier, by_em = build_indices(pool)

    # 打印每维度各档位数量
    print("  Style tiers:")
    for dim in STYLE_DIMS:
        parts = " ".join(f"{t}={len(by_dim_tier[dim][t])}" for t in TIERS)
        print(f"    {dim}: {parts}")

    print()

    # 为每个标注员分配
    for ann_id in range(1, N_ANNOTATORS + 1):
        tasks, warnings = allocate(pool, by_route, by_dim_tier, by_em, ann_id, args.seed)

        out = OUTPUT_DIR / f"annotator_{ann_id}_tasks.json"
        out.write_text(json.dumps(tasks, ensure_ascii=False, indent=1))

        tc = Counter(t["task_type"] for t in tasks)
        n_rep = sum(1 for t in tasks if t["is_repeat"])
        n_qc = sum(1 for t in tasks if t.get("is_qc_anchor"))
        print(
            f"Annotator {ann_id:2d}: {len(tasks):4d} tasks | "
            f"move={tc['move']:3d}  compare={tc['style_compare']:3d}  "
            f"label={tc['style_label']:3d}  expr={tc['expr_mode']:3d} | "
            f"repeats={n_rep:2d}  qc_anchors={n_qc:3d}"
        )
        for w in warnings:
            print(f"  ⚠ {w}")

    print(f"\nDone! Files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

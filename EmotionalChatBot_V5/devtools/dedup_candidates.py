#!/usr/bin/env python3
"""
Jaccard token-level dedup across all annotation JSONL files.
Usage:
    python dedup_candidates.py [--threshold 0.2]
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import jieba

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"

# ---------- tokenisation ----------
STOPWORDS: set[str] = set()
_STOP_FILE = ROOT / "devtools" / "stopwords.txt"
if _STOP_FILE.exists():
    STOPWORDS = set(_STOP_FILE.read_text().splitlines())

_PUNC = re.compile(r"[，。！？、；：""''（）【】《》…—\s\u3000,.!?;:\"'()\[\]{}<>\-_/\\@#$%^&*+=|~`]+")


def tokenize(text: str) -> list[str]:
    """jieba cut → remove stopwords & punctuation → lowercase."""
    tokens = jieba.lcut(text)
    out = []
    for t in tokens:
        t = _PUNC.sub("", t).strip()
        if t and t not in STOPWORDS and len(t) >= 1:
            out.append(t.lower())
    return out


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------- load ----------
def load_candidates(log_dir: Path, min_len: int = 5) -> list[dict]:
    """Load all candidate texts from annotation JSONL files.
    Returns list of {text, route, session, round, style, em, ...}."""
    records = []
    files = sorted(log_dir.glob("*_annotation.jsonl"))
    for f in files:
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            style = row.get("style", {})
            em = style.get("EXPRESSION_MODE", 0)
            for cand in row.get("candidates", []):
                text = cand.get("text", "")
                if len(text) < min_len:
                    continue
                records.append({
                    "text": text,
                    "context": row.get("user_message") or row.get("context") or "",
                    "route": cand.get("route", "unknown"),
                    "session": row.get("session_label", ""),
                    "round": row.get("round", 0),
                    "style": {k: v for k, v in style.items() if k != "EXPRESSION_MODE"},
                    "em": em,
                })
    return records


# ---------- dedup ----------
def dedup(records: list[dict], threshold: float) -> list[dict]:
    """Remove records whose Jaccard similarity with any kept record >= threshold."""
    print(f"Total candidates (>=5 chars): {len(records)}")

    t0 = time.time()
    # tokenize all
    token_sets = []
    for r in records:
        token_sets.append(set(tokenize(r["text"])))
    print(f"Tokenized in {time.time()-t0:.1f}s")

    kept_indices: list[int] = []
    kept_tokens: list[set] = []
    t1 = time.time()

    for i, ts in enumerate(token_sets):
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(records)}, kept {len(kept_indices)}, {time.time()-t1:.0f}s")
        is_dup = False
        for kt in kept_tokens:
            if jaccard(ts, kt) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept_indices.append(i)
            kept_tokens.append(ts)

    kept = [records[i] for i in kept_indices]
    elapsed = time.time() - t1
    removed = len(records) - len(kept)
    print(f"\n=== Dedup Result (Jaccard >= {threshold} = dup) ===")
    print(f"Before: {len(records)}")
    print(f"After:  {len(kept)}")
    print(f"Removed: {removed} ({removed/len(records)*100:.1f}%)")
    print(f"Keep rate: {len(kept)/len(records)*100:.1f}%")
    print(f"Time: {elapsed:.1f}s")
    return kept


# ---------- stats ----------
def print_stats(records: list[dict], label: str = ""):
    if label:
        print(f"\n=== {label} ===")

    # route
    route_counts = Counter(r["route"] for r in records)
    print(f"\n=== Route {label} ===")
    for route, cnt in route_counts.most_common():
        print(f"  {route}: {cnt}")

    # EM
    em_counts = Counter(r["em"] for r in records)
    print(f"\n=== EM {label} ===")
    for em in sorted(em_counts):
        print(f"  EM={em}: {em_counts[em]}")

    # Style tiers
    def tier(v: float) -> str:
        if v < 0.2: return "EL"
        if v < 0.4: return "L"
        if v < 0.6: return "M"
        if v < 0.8: return "H"
        return "EH"

    dims = ["FORMALITY", "POLITENESS", "FRIENDLINESS", "CERTAINTY", "EMOTIONAL_TONE"]
    print(f"\n=== Style Tier {label} ===")
    for dim in dims:
        tier_counts = Counter()
        for r in records:
            v = r["style"].get(dim)
            if v is not None:
                tier_counts[tier(v)] += 1
        parts = "  ".join(f"{t}={tier_counts.get(t,0)}" for t in ["EL","L","M","H","EH"])
        print(f"  {dim}: {parts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--min-len", type=int, default=5)
    args = parser.parse_args()

    records = load_candidates(LOG_DIR, min_len=args.min_len)
    print_stats(records, "before dedup")

    kept = dedup(records, threshold=args.threshold)
    print_stats(kept, "after dedup")

    # 添加 tiers 字段
    def tier(v: float) -> str:
        if v < 0.2: return "EL"
        if v < 0.4: return "L"
        if v < 0.6: return "M"
        if v < 0.8: return "H"
        return "EH"

    dims = ["FORMALITY", "POLITENESS", "FRIENDLINESS", "CERTAINTY", "EMOTIONAL_TONE"]
    for r in kept:
        r["tiers"] = {d: tier(r["style"].get(d, 0.5)) for d in dims}

    # 保存去重后的数据池
    out_dir = ROOT / "scripts" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "deduped_pool.json"
    pool_path.write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✓ 已保存 {len(kept)} 条到 {pool_path}")


if __name__ == "__main__":
    main()

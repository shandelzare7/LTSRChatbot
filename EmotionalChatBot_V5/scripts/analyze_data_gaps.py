#!/usr/bin/env python3
"""
Phase 9 Data Distribution Gap Analysis
=======================================
Loads all *_annotation.jsonl files, extracts route (content move),
expression_mode (EM), and style dimensions, then reports counts and
identifies combinations with < 100 samples.

Usage:
    python scripts/analyze_data_gaps.py
"""

import json
import glob
import os
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
STYLE_DIMS = ["FORMALITY", "WARMTH", "CERTAINTY", "POLITENESS", "EMOTIONAL_INTENSITY"]
TIER_LABELS = ["EL", "L", "M", "H", "EH"]
GAP_THRESHOLD = 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def value_to_tier(v: float) -> str:
    """Bucket a [0,1] value into 5 tiers."""
    if v < 0.2:
        return "EL"
    elif v < 0.4:
        return "L"
    elif v < 0.6:
        return "M"
    elif v < 0.8:
        return "H"
    else:
        return "EH"


def load_all_records(logs_dir: str):
    """Load all annotation JSONL files, yielding parsed dicts."""
    pattern = os.path.join(logs_dir, "*_annotation.jsonl")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} annotation files in {os.path.abspath(logs_dir)}")

    total_lines = 0
    parsed = 0
    errors = 0

    for fpath in files:
        with open(fpath, "rb") as f:
            content = f.read().decode("utf-8", errors="replace")
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                rec = json.loads(line)
                parsed += 1
                yield rec
            except json.JSONDecodeError:
                errors += 1

    print(f"Total lines: {total_lines}, parsed OK: {parsed}, errors: {errors}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Counters
    route_counter = Counter()          # route (move type) -> count
    em_counter = Counter()             # EM value -> count
    dim_tier_counter = defaultdict(Counter)  # dim -> {tier: count}
    route_em_counter = Counter()       # (route, em) -> count
    route_dim_tier = Counter()         # (route, dim, tier) -> count
    em_dim_tier = Counter()            # (em, dim, tier) -> count

    total_records = 0

    for rec in load_all_records(LOGS_DIR):
        style = rec.get("style", {})
        if not style:
            continue

        em = style.get("EXPRESSION_MODE")
        if em is None:
            continue

        em = int(em)

        # Determine the route of the SELECTED / final response
        # selected_move_ids tells which moves were used; candidates have route
        # We attribute the record to all selected routes
        selected_ids = rec.get("selected_move_ids", [])
        candidates = rec.get("candidates", [])

        # Collect unique routes for this record's selected moves
        routes_in_record = set()
        for cand in candidates:
            mid = cand.get("move_id")
            route = cand.get("route", "")
            if mid in selected_ids:
                routes_in_record.add(route)
        # If nothing matched, use "unknown"
        if not routes_in_record:
            routes_in_record.add("unknown")

        total_records += 1

        # EM count (once per record)
        em_counter[em] += 1

        # Style dimension tiers
        tiers = {}
        for dim in STYLE_DIMS:
            v = style.get(dim)
            if v is not None:
                t = value_to_tier(float(v))
                tiers[dim] = t
                dim_tier_counter[dim][t] += 1

        # Route counts (once per unique route in this record)
        for route in routes_in_record:
            route_counter[route] += 1
            route_em_counter[(route, em)] += 1
            for dim in STYLE_DIMS:
                if dim in tiers:
                    route_dim_tier[(route, dim, tiers[dim])] += 1
                    em_dim_tier[(em, dim, tiers[dim])] += 1

    print(f"\nTotal usable records: {total_records}")
    print("=" * 70)

    # ----- 1. Route / Move type counts -----
    print("\n[1] ROUTE (Content Move) Counts")
    print("-" * 40)
    for route, cnt in sorted(route_counter.items(), key=lambda x: -x[1]):
        print(f"  {route:12s}  {cnt:>6d}")

    # ----- 2. EM value counts -----
    print("\n[2] EXPRESSION_MODE (EM) Counts")
    print("-" * 40)
    for em in sorted(em_counter.keys()):
        print(f"  EM={em}  {em_counter[em]:>6d}")

    # ----- 3. Style dimension × tier -----
    print("\n[3] Style Dimension × Tier Counts")
    print("-" * 60)
    header = f"  {'Dimension':25s}" + "".join(f"{t:>8s}" for t in TIER_LABELS)
    print(header)
    print("  " + "-" * (25 + 8 * len(TIER_LABELS)))
    for dim in STYLE_DIMS:
        row = f"  {dim:25s}"
        for t in TIER_LABELS:
            cnt = dim_tier_counter[dim].get(t, 0)
            row += f"{cnt:>8d}"
        print(row)

    # ----- 4. Gaps: combinations with < GAP_THRESHOLD -----
    print(f"\n[4] GAPS: Combinations with < {GAP_THRESHOLD} samples")
    print("=" * 70)

    # 4a. Route gaps
    print(f"\n  4a. Routes with < {GAP_THRESHOLD} samples:")
    route_gaps = [(r, c) for r, c in route_counter.items() if c < GAP_THRESHOLD]
    if route_gaps:
        for r, c in sorted(route_gaps, key=lambda x: x[1]):
            print(f"      {r:12s}  {c:>6d}")
    else:
        print("      (none)")

    # 4b. EM gaps
    print(f"\n  4b. EM values with < {GAP_THRESHOLD} samples:")
    em_gaps = [(em, c) for em, c in em_counter.items() if c < GAP_THRESHOLD]
    if em_gaps:
        for em, c in sorted(em_gaps, key=lambda x: x[1]):
            print(f"      EM={em}  {c:>6d}")
    else:
        print("      (none)")

    # 4c. Dimension × tier gaps
    print(f"\n  4c. Dimension × Tier with < {GAP_THRESHOLD} samples:")
    dim_tier_gaps = []
    for dim in STYLE_DIMS:
        for t in TIER_LABELS:
            cnt = dim_tier_counter[dim].get(t, 0)
            if cnt < GAP_THRESHOLD:
                dim_tier_gaps.append((dim, t, cnt))
    if dim_tier_gaps:
        for dim, t, cnt in sorted(dim_tier_gaps, key=lambda x: x[2]):
            print(f"      {dim:25s} × {t:3s}  {cnt:>6d}")
    else:
        print("      (none)")

    # 4d. Route × EM gaps
    print(f"\n  4d. Route × EM with < {GAP_THRESHOLD} samples:")
    re_gaps = [(r, em, c) for (r, em), c in route_em_counter.items() if c < GAP_THRESHOLD]
    if re_gaps:
        for r, em, c in sorted(re_gaps, key=lambda x: x[2]):
            print(f"      {r:12s} × EM={em}  {c:>6d}")
    else:
        print("      (none)")

    # 4e. EM × Dimension × Tier gaps
    print(f"\n  4e. EM × Dimension × Tier with < {GAP_THRESHOLD} samples (showing first 40):")
    edt_gaps = []
    for (em, dim, t), c in em_dim_tier.items():
        if c < GAP_THRESHOLD:
            edt_gaps.append((em, dim, t, c))
    # Also add zero-count combos
    for em in range(4):
        for dim in STYLE_DIMS:
            for t in TIER_LABELS:
                key = (em, dim, t)
                if key not in em_dim_tier:
                    edt_gaps.append((em, dim, t, 0))

    edt_gaps.sort(key=lambda x: x[3])
    for em, dim, t, cnt in edt_gaps[:40]:
        print(f"      EM={em} × {dim:25s} × {t:3s}  {cnt:>6d}")
    if len(edt_gaps) > 40:
        print(f"      ... and {len(edt_gaps) - 40} more gaps")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Total records:             {total_records}")
    print(f"  Unique routes:             {len(route_counter)}")
    print(f"  Route gaps (<{GAP_THRESHOLD}):         {len(route_gaps)}")
    print(f"  EM gaps (<{GAP_THRESHOLD}):            {len(em_gaps)}")
    print(f"  Dim×Tier gaps (<{GAP_THRESHOLD}):      {len(dim_tier_gaps)}")
    print(f"  Route×EM gaps (<{GAP_THRESHOLD}):      {len(re_gaps)}")
    print(f"  EM×Dim×Tier gaps (<{GAP_THRESHOLD}):   {len(edt_gaps)}")


if __name__ == "__main__":
    main()

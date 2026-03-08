"""
run_b2b_batch.py

批量运行 bot_to_bot_chat.py 的 30 组会话配置，确保：
1. Big Five 多样性（随机 / 极低 / 极高 / 混合）
2. Knapp 阶段多样性（通过不同初始关系分数实现）
3. 基本信息多样性（每次创建新 Bot，LLM 生成不同人设）

使用:
  cd EmotionalChatBot_V5
  python devtools/run_b2b_batch.py                    # 运行全部 30 组
  python devtools/run_b2b_batch.py --start 16 --end 20  # 只运行 16-20 组
  python devtools/run_b2b_batch.py --ids 1,5,26        # 只运行指定 id
  python devtools/run_b2b_batch.py --group extreme_low  # 只运行某组
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "b2b_batch_configs.json"


def load_configs(
    config_path: Path | None = None,
    start: int | None = None,
    end: int | None = None,
    ids: list[int] | None = None,
    group: str | None = None,
) -> list[dict]:
    with open(config_path or DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    sessions = [s for s in data["sessions"] if "id" in s]
    if ids:
        sessions = [s for s in sessions if s["id"] in ids]
    elif group:
        sessions = [s for s in sessions if s.get("group") == group]
    elif start is not None or end is not None:
        s = start or 1
        e = end or 999
        sessions = [sess for sess in sessions if s <= sess["id"] <= e]
    return sessions


def build_env(session: dict) -> dict:
    """Build environment variables for a single session."""
    env = os.environ.copy()
    env["BOT2BOT_NUM_RUNS"] = "1"
    # 轮数：使用预分配的 _rounds（main 里已确定）
    env["BOT2BOT_ROUNDS_PER_RUN"] = str(session.get("_rounds") or session.get("rounds") or random.randint(10, 30))
    env["BOT2BOT_SESSION_LABEL"] = session.get("label", "")

    # Big Five
    b5a = session.get("big_five_a", "random")
    b5b = session.get("big_five_b", "random")
    if isinstance(b5a, dict):
        env["BOT2BOT_BIG_FIVE_A"] = json.dumps(b5a)
    else:
        env.pop("BOT2BOT_BIG_FIVE_A", None)
    if isinstance(b5b, dict):
        env["BOT2BOT_BIG_FIVE_B"] = json.dumps(b5b)
    else:
        env.pop("BOT2BOT_BIG_FIVE_B", None)

    # Relationship template / custom dims
    custom_dims = session.get("custom_dims")
    if custom_dims:
        env["BOT2BOT_CUSTOM_DIMS"] = json.dumps(custom_dims)
        env.pop("BOT2BOT_USER_DIMENSIONS_TEMPLATE", None)
    else:
        env.pop("BOT2BOT_CUSTOM_DIMS", None)
        template = session.get("template", "friendly_icebreaker")
        env["BOT2BOT_USER_DIMENSIONS_TEMPLATE"] = template

    # Phase 2: 起始阶段、行为资产、PAD 预设
    initial_stage = session.get("initial_stage")
    if initial_stage:
        env["BOT2BOT_INITIAL_STAGE"] = str(initial_stage)
    else:
        env.pop("BOT2BOT_INITIAL_STAGE", None)

    initial_assets = session.get("initial_assets")
    if initial_assets and isinstance(initial_assets, dict):
        env["BOT2BOT_INITIAL_ASSETS"] = json.dumps(initial_assets)
    else:
        env.pop("BOT2BOT_INITIAL_ASSETS", None)

    initial_pad = session.get("initial_pad")
    if initial_pad and isinstance(initial_pad, dict):
        env["BOT2BOT_INITIAL_PAD"] = json.dumps(initial_pad)
    else:
        env.pop("BOT2BOT_INITIAL_PAD", None)

    # Each session uses a different random seed for bot creation diversity
    env.pop("BOT2BOT_SEED", None)

    # Ensure no stale bot IDs (force new bot creation each session)
    env.pop("BOT2BOT_BOT_A_ID", None)
    env.pop("BOT2BOT_BOT_B_ID", None)

    return env


def main():
    parser = argparse.ArgumentParser(description="Batch runner for b2b annotation sessions")
    parser.add_argument("--start", type=int, help="Start session id (inclusive)")
    parser.add_argument("--end", type=int, help="End session id (inclusive)")
    parser.add_argument("--ids", type=str, help="Comma-separated session ids")
    parser.add_argument("--group", type=str, help="Run only sessions in this group")
    parser.add_argument("--config", type=str, help="Path to config JSON file (default: b2b_batch_configs.json)")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    ids_list = [int(x) for x in args.ids.split(",")] if args.ids else None
    sessions = load_configs(config_path, args.start, args.end, ids_list, args.group)

    if not sessions:
        print("No sessions matched the filter criteria.")
        sys.exit(1)

    # 预分配轮数（提前确定，便于预览总轮数）
    for s in sessions:
        if not s.get("_rounds"):
            s["_rounds"] = s.get("rounds") or random.randint(10, 30)

    total_rounds = sum(s["_rounds"] for s in sessions)
    print(f"Will run {len(sessions)} session(s), total ~{total_rounds} rounds:")
    for s in sessions:
        dims = s.get("custom_dims") or s.get("template", "friendly_icebreaker")
        stage = s.get("initial_stage", "initiating")
        print(f"  [{s['id']:2d}] {s['label']:<30s}  group={s.get('group',''):<12s}  rounds={s['_rounds']:<3d}  stage={stage:<16s}  dims={dims if isinstance(dims, str) else 'custom'}")
    print()

    if args.dry_run:
        print("Dry run mode, not executing.")
        for s in sessions:
            env = build_env(s)
            print(f"--- Session {s['id']} ({s['label']}) ---")
            for k in sorted(env):
                if k.startswith("BOT2BOT_"):
                    print(f"  {k}={env[k]}")
            print()
        return

    script = str(SCRIPT_DIR / "bot_to_bot_chat.py")
    total = len(sessions)
    results = []

    for i, session in enumerate(sessions, 1):
        sid = session["id"]
        label = session.get("label", f"session_{sid}")
        print("=" * 70)
        print(f"[{i}/{total}] Session {sid}: {label} (group={session.get('group', '')})")
        print("=" * 70)

        env = build_env(session)
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, script],
                cwd=str(PROJECT_ROOT),
                env=env,
                timeout=3600,  # 1 hour max per session
            )
            elapsed = time.time() - t0
            status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            status = "TIMEOUT"
        except Exception as e:
            elapsed = time.time() - t0
            status = f"ERROR({e})"

        results.append({"id": sid, "label": label, "status": status, "elapsed_s": round(elapsed, 1)})
        print(f"\n  -> {status} ({elapsed:.1f}s)\n")

    # Summary
    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    ok = sum(1 for r in results if r["status"] == "OK")
    for r in results:
        print(f"  [{r['id']:2d}] {r['label']:<30s}  {r['status']:<20s}  {r['elapsed_s']}s")
    print(f"\n  Total: {ok}/{total} succeeded, {total - ok} failed")
    total_time = sum(r["elapsed_s"] for r in results)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""实时监控 bot_to_bot_chat 运行状态：轮次、最新回复、耗时等。用法: python devtools/monitor_bot2bot.py [日志路径]"""
import os
import sys
import time
import re

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEFAULT_PATTERN = "bot_to_bot_chat_*.log"


def find_latest_log(path=None):
    if path and os.path.isfile(path):
        return path
    import glob
    files = sorted(glob.glob(os.path.join(LOG_DIR, "bot_to_bot_chat_*.log")), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else None
    log_path = find_latest_log(log_path)
    if not log_path:
        print("未找到 bot_to_bot_chat_*.log")
        return
    print(f"监控: {log_path}")
    print("按 Ctrl+C 停止\n")

    last_size = 0
    round_re = re.compile(r"第 (\d+) 次会话 / 第 (\d+) 轮")
    reply_re = re.compile(r"\[ROUND (\d+) REPLY\]")
    momentum_re = re.compile(r"3\. Momentum: ([\d.]+)")
    turn_re = re.compile(r"2\. Turn count: (\d+)")

    try:
        while True:
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(max(0, last_size))
                    new = f.read()
                    last_size = f.tell()
            except (IOError, OSError) as e:
                print(f"读日志失败: {e}")
                time.sleep(2)
                continue

            if new:
                for line in new.splitlines():
                    m = round_re.search(line)
                    if m:
                        print(f"\r[进度] 会话 {m.group(1)} / 第 {m.group(2)} 轮", end="", flush=True)
                    m = reply_re.search(line)
                    if m:
                        print(f"\n  → 第 {m.group(1)} 轮已产出回复")
                    m = momentum_re.search(line)
                    if m:
                        print(f"  → Momentum: {m.group(1)}", end="", flush=True)
                    m = turn_re.search(line)
                    if m:
                        print(f"  Turn count: {m.group(1)}")

            time.sleep(3)
    except KeyboardInterrupt:
        print("\n监控已停止")


if __name__ == "__main__":
    main()

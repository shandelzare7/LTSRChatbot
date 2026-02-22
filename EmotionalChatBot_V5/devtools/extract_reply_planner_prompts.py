"""
从 bot_to_bot_chat 的 .log 中提取每次 ReplyPlanner 调用的完整提示词列表。

用法（在 EmotionalChatBot_V5 目录下）：
  python -m devtools.extract_reply_planner_prompts [日志路径]
  不传参数时使用 logs/ 下最新的 bot_to_bot_chat_*.log。

输出：在日志同目录下生成 reply_planner_prompts_<原日志名>.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def find_blocks(content: str) -> list[tuple[str, str]]:
    """
    找到所有 [ReplyPlanGen] 或 [ReplyPlanGen (Candidates)] 的提示词块。
    返回 [(block_type, block_text), ...]
    """
    # 开始: "[ReplyPlanGen] ========== 提示词与参数 ==========" 或 "[ReplyPlanGen (Candidates)] ..."
    start_pattern = re.compile(
        r"^\[(ReplyPlanGen(?: \(Candidates\))?)\] ={10,}\s*提示词与参数\s*={10,}\s*$",
        re.MULTILINE,
    )
    # 结束: "[ReplyPlanGen] =========================================="
    end_pattern = re.compile(
        r"^\[(ReplyPlanGen(?: \(Candidates\))?)\] ={30,}\s*$",
        re.MULTILINE,
    )

    blocks: list[tuple[str, str]] = []
    for start_m in start_pattern.finditer(content):
        label = start_m.group(1)
        block_start = start_m.end()
        # 找下一个同 label 的结束行
        end_m = end_pattern.search(content, block_start)
        if not end_m:
            continue
        if end_m.group(1) != label:
            continue
        block_text = content[block_start : end_m.start()].strip()
        blocks.append((label, block_text))
    return blocks


def extract_section(block: str, header: str) -> str:
    """从 block 中提取 【header】 与下一个 【 或字符串结尾之间的内容（去掉前导 ===== 和缩进）。"""
    pattern = re.compile(
        re.escape(header) + r"\s*\n\s*={10,}\s*\n(.*?)(?=\n\s*【|\Z)",
        re.DOTALL,
    )
    m = pattern.search(block)
    if not m:
        return ""
    text = m.group(1).strip()
    # 去掉每行前导两个空格（indent）
    lines = [line[2:] if line.startswith("  ") else line for line in text.split("\n")]
    return "\n".join(lines).strip()


def extract_sections(block_text: str) -> dict[str, str]:
    out = {}
    for key, header in [
        ("system_prompt", "【System Prompt】"),
        ("user_prompt", "【User Prompt / Task】"),
        ("messages_body", "【Messages (Body)】"),
        ("params", "【输入参数】"),
    ]:
        out[key] = extract_section(block_text, header)
    return out


def main() -> None:
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_dir = PROJECT_ROOT / "logs"
        if not log_dir.exists():
            print(f"日志目录不存在: {log_dir}")
            sys.exit(1)
        log_files = sorted(
            log_dir.glob("bot_to_bot_chat_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not log_files:
            print(f"未找到 bot_to_bot_chat_*.log（目录: {log_dir}）")
            sys.exit(1)
        log_path = log_files[0]

    if not log_path.exists():
        print(f"文件不存在: {log_path}")
        sys.exit(1)

    content = log_path.read_text(encoding="utf-8", errors="replace")
    blocks = find_blocks(content)
    print(f"共找到 {len(blocks)} 次 ReplyPlanner 调用（ReplyPlanGen / ReplyPlanGen (Candidates)）")

    out_path = log_path.parent / f"reply_planner_prompts_{log_path.name}"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# 来源: {log_path.name}\n")
        f.write(f"# 共 {len(blocks)} 次调用\n\n")

        for i, (label, block_text) in enumerate(blocks, 1):
            sec = extract_sections(block_text)
            f.write("=" * 80 + "\n")
            f.write(f"第 {i} 次调用 [{label}]\n")
            f.write("=" * 80 + "\n\n")

            f.write("## System Prompt\n")
            f.write("-" * 60 + "\n")
            f.write(sec["system_prompt"] or "(无)\n")
            f.write("\n")

            f.write("## User Prompt / Task\n")
            f.write("-" * 60 + "\n")
            f.write(sec["user_prompt"] or "(无)\n")
            f.write("\n")

            f.write("## Messages (Body)\n")
            f.write("-" * 60 + "\n")
            f.write(sec["messages_body"] or "(无)\n")
            f.write("\n")

            f.write("## 输入参数\n")
            f.write("-" * 60 + "\n")
            f.write(sec["params"] or "(无)\n")
            f.write("\n\n")

    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()

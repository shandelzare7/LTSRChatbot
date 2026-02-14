"""
跑一遍 normal 分支，记录每个 node 的输入 state 与输出 update，写入报告文件。

chat_buffer：最近 N 条对话（来自 DB），供 detection/reasoner/generator 做上下文。
每次跑完图 memory_writer 会把本轮 user+ai 写入 DB，所以不清理的话会越积越多、出现重复。
本脚本在跑图前会先清空该 bot+user 的 messages，使本次测试的 chat_buffer 只含本轮一条用户消息。

用法：
  cd EmotionalChatBot_V5
  .venv/bin/python devtools/log_normal_branch_io.py

输出：devtools/normal_branch_io_report.md（以及 normal_branch_io.json）
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass

DEFAULT_BOT_ID = "4d803b5a-cb30-4d14-89eb-88d259564610"
DEFAULT_USER_ID = "local_user_5128d1c1"
MAX_STR = 500  # 单字段字符串截断长度
MAX_LIST = 20  # 列表最多展示项数


def _serialize_for_log(obj, depth=0):
    """把 state/update 转成可 JSON 序列化且可读的结构（截断过长内容）。"""
    if depth > 8:
        return "<<max_depth>>"
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _serialize_for_log(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        out = []
        for i, v in enumerate(obj):
            if i >= MAX_LIST:
                out.append(f"... (+{len(obj) - MAX_LIST} more)")
                break
            out.append(_serialize_for_log(v, depth + 1))
        return out
    if isinstance(obj, str):
        return obj if len(obj) <= MAX_STR else obj[:MAX_STR] + f"... ({len(obj)} chars)"
    if isinstance(obj, (int, float, bool)):
        return obj
    # BaseMessage 等（带时间戳）
    if hasattr(obj, "content") and hasattr(obj, "type"):
        d = {"type": getattr(obj, "type", "message"), "content": _serialize_for_log(getattr(obj, "content", ""), depth + 1)}
        if hasattr(obj, "additional_kwargs") and isinstance(getattr(obj, "additional_kwargs"), dict):
            ts = (obj.additional_kwargs or {}).get("timestamp")
            if ts:
                d["timestamp"] = ts
        return d
    if hasattr(obj, "__dict__"):
        return _serialize_for_log(getattr(obj, "__dict__", str(obj)), depth + 1)
    return str(obj)[:MAX_STR]


def _merge_state(state: dict, update: dict) -> dict:
    """简单合并：新 key 覆盖；不处理 messages 的 add_messages，仅用于记录顺序。"""
    out = dict(state)
    for k, v in update.items():
        out[k] = v
    return out


async def _clear_chat_buffer(bot_id: str, user_id: str) -> int:
    """清空该 bot+user 在 DB 中的 messages，避免 chat_buffer 重复堆积。返回删除条数。"""
    from app.core.database import DBManager
    db = DBManager.from_env()
    return await db.clear_messages_for(user_id, bot_id)


def main():
    bot_id = os.getenv("TEST_BOT_ID") or DEFAULT_BOT_ID
    user_id = os.getenv("TEST_USER_ID") or DEFAULT_USER_ID

    if not os.getenv("DATABASE_URL"):
        print("DATABASE_URL 未设置，请配置 .env 后重试。")
        sys.exit(1)

    # 测试前清空该会话的 messages，使 chat_buffer 只含本轮一条用户消息
    n = asyncio.run(_clear_chat_buffer(bot_id, user_id))
    print(f"已清空该会话历史消息: {n} 条（chat_buffer 将只含本轮输入）")

    from langchain_core.messages import HumanMessage
    from app.graph import build_graph
    from app.state import AgentState

    app = build_graph()
    now_iso = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0).isoformat()
    initial: AgentState = {
        "messages": [HumanMessage(content="今天天气不错，你平时周末会做什么？", additional_kwargs={"timestamp": now_iso})],
        "current_time": now_iso,
        "user_id": user_id,
        "bot_id": bot_id,
        "user_profile": {},
        "memories": "",
        "deep_reasoning_trace": {},
        "style_analysis": "",
        "draft_response": "",
        "critique_feedback": "",
        "retry_count": 0,
        "final_segments": [],
        "final_delay": 0.0,
    }

    # 用 stream_mode="updates" 拿到每个节点返回的 update
    # 同时维护当前 state（作为下一节点的 input）
    current_state = deepcopy(initial)
    records = []  # list of { "node": str, "input": dict, "output": dict }

    try:
        for chunk in app.stream(initial, config={"recursion_limit": 50}, stream_mode="updates"):
            if not isinstance(chunk, dict):
                continue
            for node_name, update in chunk.items():
                if not isinstance(update, dict):
                    update = {}
                input_snapshot = deepcopy(current_state)
                records.append({
                    "node": node_name,
                    "input": input_snapshot,
                    "output": update,
                })
                current_state = _merge_state(current_state, update)
    except Exception as e:
        records.append({"node": "__error__", "input": {}, "output": {"error": str(e)}})

    # 写出 Markdown 报告
    report_path = PROJECT_ROOT / "devtools" / "normal_branch_io_report.md"
    lines = [
        "# Normal 分支各 Node 输入/输出记录",
        "",
        f"运行时间: {datetime.now().isoformat()}",
        f"user_id: {user_id}, bot_id: {bot_id}",
        "",
        "---",
        "",
    ]

    for i, rec in enumerate(records):
        node = rec["node"]
        inp = _serialize_for_log(rec["input"])
        out = _serialize_for_log(rec["output"])
        lines.append(f"## {i + 1}. {node}")
        lines.append("")
        lines.append("### Input (进入该节点时的 state 快照)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(inp, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("### Output (该节点返回的 state 更新)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(out, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("已写入:", report_path)

    # 可选：完整 JSON（便于程序化分析）
    json_path = PROJECT_ROOT / "devtools" / "normal_branch_io.json"
    json_records = [
        {"node": r["node"], "input": _serialize_for_log(r["input"]), "output": _serialize_for_log(r["output"])}
        for r in records
    ]
    json_path.write_text(json.dumps(json_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写入:", json_path)
    print("节点顺序:", [r["node"] for r in records])


if __name__ == "__main__":
    main()

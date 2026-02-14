"""
按顺序跑一遍 graph，验证 state 是否从 DB 正确加载，并记录各环节问题。

用法：
  cd EmotionalChatBot_V5
  .venv/bin/python devtools/graph_run_report.py

要求：.env 中 DATABASE_URL 已配置，且已执行过 reset_and_seed 或 seed_local_postgres（存在 bot/user）。
"""

from __future__ import annotations

import asyncio
import os
import sys
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

# 使用上次 reset_and_seed 的 bot_id / user_id（若库中只有一条，可改从 DB 查）
DEFAULT_BOT_ID = "4d803b5a-cb30-4d14-89eb-88d259564610"
DEFAULT_USER_ID = "local_user_5128d1c1"


def _section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _ok(msg: str) -> None:
    print("[OK]", msg)


def _fail(msg: str) -> None:
    print("[FAIL]", msg)


def _warn(msg: str) -> None:
    print("[WARN]", msg)


async def _check_db_load(bot_id: str, user_id: str) -> dict:
    """1) 直接从 DB load_state，验证返回内容是否完整、正确。"""
    report = {"ok": True, "issues": [], "state_keys": [], "samples": {}}
    if not os.getenv("DATABASE_URL"):
        report["ok"] = False
        report["issues"].append("DATABASE_URL 未设置，无法从数据库加载")
        return report

    try:
        from app.core.database import DBManager
        db = DBManager.from_env()
        data = await db.load_state(user_id, bot_id)
    except Exception as e:
        report["ok"] = False
        report["issues"].append(f"DBManager.load_state 异常: {e}")
        return report

    report["state_keys"] = list(data.keys())
    required = [
        "bot_basic_info", "bot_big_five", "bot_persona",
        "user_basic_info", "user_inferred_profile",
        "relationship_state", "mood_state", "current_stage",
        "chat_buffer", "relationship_assets", "spt_info", "conversation_summary",
    ]
    for k in required:
        if k not in data:
            report["issues"].append(f"load_state 缺少字段: {k}")
            report["ok"] = False
        else:
            report["samples"][k] = data[k] if k != "chat_buffer" else f"list(len={len(data.get('chat_buffer') or [])})"

    # 抽样检查 bot/user 是否有内容
    bi = data.get("bot_basic_info") or {}
    if not bi.get("name"):
        report["issues"].append("bot_basic_info 缺少 name")
        report["ok"] = False
    else:
        report["samples"]["bot_name"] = bi.get("name")

    persona = data.get("bot_persona") or {}
    if not persona:
        report["warn"] = "bot_persona 为空"
    else:
        report["samples"]["persona_lore"] = persona.get("lore")
        report["samples"]["persona_collections"] = persona.get("collections")

    uf = data.get("user_inferred_profile") or {}
    report["samples"]["user_inferred_profile_keys"] = list(uf.keys())

    rel = data.get("relationship_state") or {}
    if not rel or set(rel.keys()) != {"closeness", "trust", "liking", "respect", "warmth", "power"}:
        report["issues"].append("relationship_state 维度不完整或键名不符")
        report["ok"] = False

    return report


def _run_graph_and_collect_issues(bot_id: str, user_id: str) -> dict:
    """2) 构建图并 invoke 一轮，从最终 state 反推各环节是否正常，并记录问题。"""
    report = {"ok": True, "issues": [], "final_keys": [], "error": None}
    if not os.getenv("DATABASE_URL"):
        report["ok"] = False
        report["issues"].append("DATABASE_URL 未设置")
        return report

    try:
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

        result = app.invoke(initial, config={"recursion_limit": 50})
        report["final_keys"] = list(result.keys()) if isinstance(result, dict) else []
        issues = []

        # 按环节检查：loader -> detection -> reasoner -> style -> generator -> critic -> processor -> evolver -> stage_manager -> memory_writer
        if not (result.get("bot_basic_info") and (result.get("bot_basic_info") or {}).get("name")):
            issues.append("loader: 最终 state 缺少 bot_basic_info 或 bot name（state 未从 DB 正确获得 bot 信息）")
        if not result.get("bot_persona"):
            issues.append("loader: 最终 state 缺少 bot_persona")
        if result.get("user_inferred_profile") is None and "user_inferred_profile" not in result:
            issues.append("loader: 最终 state 缺少 user_inferred_profile")
        if not result.get("relationship_state") or set((result.get("relationship_state") or {}).keys()) != {"closeness", "trust", "liking", "respect", "warmth", "power"}:
            issues.append("loader: relationship_state 缺失或 6 维不完整")
        if result.get("current_stage") is None:
            issues.append("loader: current_stage 缺失")

        cat = result.get("detection_category") or result.get("detection_result")
        if cat is None:
            issues.append("detection: 未写入 detection_category/detection_result")
        else:
            report["detection_category"] = cat

        if not result.get("inner_monologue") and not result.get("response_strategy"):
            issues.append("reasoner: 未写入 inner_monologue 或 response_strategy")
        if not result.get("style_analysis"):
            issues.append("style: 未写入 style_analysis")
        if not result.get("draft_response") and not result.get("final_response"):
            issues.append("generator: 未写入 draft_response/final_response")
        if not result.get("humanized_output") and not result.get("final_segments"):
            issues.append("processor: 未写入 humanized_output 或 final_segments")
        if not result.get("final_response"):
            issues.append("最终未得到 final_response")

        report["has_final_response"] = bool(result.get("final_response"))
        report["has_humanized_output"] = bool(result.get("humanized_output"))
        report["issues"] = issues
        if issues:
            report["ok"] = False

    except Exception as e:
        report["ok"] = False
        report["error"] = str(e)
        import traceback
        report["traceback"] = traceback.format_exc()

    return report


def main():
    bot_id = os.getenv("TEST_BOT_ID") or DEFAULT_BOT_ID
    user_id = os.getenv("TEST_USER_ID") or DEFAULT_USER_ID

    print("Graph 各环节验证报告")
    print("bot_id:", bot_id)
    print("user_id:", user_id)

    # 1) State 是否能从数据库正确获得 bot/user 等信息？
    _section("1) State 从数据库加载 (load_state)")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    db_report = loop.run_until_complete(_check_db_load(bot_id, user_id))
    loop.close()

    if db_report["ok"]:
        _ok("load_state 返回完整，所需字段均存在")
        print("  返回键:", db_report["state_keys"])
        print("  抽样: bot_name =", db_report["samples"].get("bot_name"))
        print("  persona.lore =", db_report["samples"].get("persona_lore"))
        print("  persona.collections =", db_report["samples"].get("persona_collections"))
    else:
        _fail("load_state 存在问题")
        for i in db_report["issues"]:
            print("  -", i)
    if db_report.get("issues"):
        for i in db_report["issues"]:
            print("  -", i)

    # 2) 按顺序跑一遍图，记录问题
    _section("2) 按顺序跑 Graph (loader -> detection -> ... -> memory_writer)")
    graph_report = _run_graph_and_collect_issues(bot_id, user_id)

    if graph_report.get("error"):
        _fail("Graph 运行异常: " + graph_report["error"])
        print(graph_report.get("traceback", ""))
    else:
        _ok("Graph 跑通")
        if graph_report.get("issues"):
            for i in graph_report["issues"]:
                _warn(i)
        print("  最终 state 键:", graph_report.get("final_keys", [])[:25])
        print("  final_response 存在:", graph_report.get("has_final_response"))
        print("  humanized_output 存在:", graph_report.get("has_humanized_output"))
        print("  detection_category:", graph_report.get("detection_category"))

    _section("总结")
    if db_report["ok"] and graph_report["ok"] and not graph_report.get("issues"):
        print("State 能从数据库正确获得 bot/user 等信息；Graph 各环节按顺序跑通，未记录到问题。")
    else:
        print("存在问题汇总:")
        if not db_report["ok"]:
            for i in db_report["issues"]:
                print("  [DB]", i)
        if graph_report.get("error"):
            print("  [Graph]", graph_report["error"])
        for i in graph_report.get("issues", []):
            print("  [Graph]", i)
    print()


if __name__ == "__main__":
    main()

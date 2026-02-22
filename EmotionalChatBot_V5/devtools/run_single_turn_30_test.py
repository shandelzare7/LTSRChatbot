"""
run_single_turn_30_test.py

用途：新 Bot 模拟 User，单轮对话 × 30 次；关闭「常规走 fast」开关，全量打出 log，用于评估：
  1. 5 路并行信号检测（detection / inner_monologue / 3 个 router）是否正常、返回值与冲量是否合理
  2. 不知道姓名时紧急任务「姓名」是否正常触发并进入 reply_planner
  3. 新 reply_planner 提示词输出是否合理
  4. 5 并行 judge（rel/stage/mood/task/strategy）是否合理工作
  5. 评估系统（LATS 加权与选回复）是否合理

前置：DATABASE_URL、.env；可选 BOT_ID 使用已有 Bot，否则创建新 Bot。
运行：
  cd EmotionalChatBot_V5
  FAST_ROUTE_WHEN_QUICK_REPLY_ENABLED=0 python3 devtools/run_single_turn_30_test.py
（脚本内也会强制设为 0）
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# 强制关闭「前两条回复总用时走 fast」的常规 fast 开关
os.environ["FAST_ROUTE_WHEN_QUICK_REPLY_ENABLED"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from langchain_core.messages import HumanMessage
from sqlalchemy import select

from app.core.database import Bot, DBManager, User
from app.core.relationship_templates import get_relationship_template_by_name
from app.graph import build_graph
from main import _make_initial_state
from utils.external_text import sanitize_external_text

# 单轮测试首句池（可与 bot_to_bot 一致或简化）
FIRST_MESSAGE_POOL = [
    "你好",
    "今天天气不错。",
    "在吗？",
    "想随便聊几句。",
    "你叫什么名字？",
]

class _TeeStderr:
    """将 stderr 同时写入 log 文件与原始 stderr，便于 [LLM_ELAPSED] 既实时显示又进入日志供报告解析。"""

    def __init__(self, log_file, original_stderr):
        self._log = log_file
        self._err = original_stderr

    def write(self, s: str) -> None:
        self._log.write(s)
        self._log.flush()
        self._err.write(s)
        self._err.flush()

    def flush(self) -> None:
        self._log.flush()
        self._err.flush()


# 从 bot_to_bot_chat 复用 run_one_turn（需传入 log_file 以捕获 graph 内所有 print）
async def _run_one_turn(app, user_id: str, bot_id: str, message: str, log_file, original_stdout, original_stderr) -> tuple[str, dict, float]:
    from main import FileOnlyWriter

    state = _make_initial_state(user_id, bot_id)
    state["lats_rollouts"] = int(os.getenv("BOT2BOT_LATS_ROLLOUTS", "4"))
    state["lats_expand_k"] = int(os.getenv("BOT2BOT_LATS_EXPAND_K", "2"))
    state["lats_early_exit_root_score"] = float(os.getenv("BOT2BOT_EARLY_EXIT_SCORE", "0.82"))
    state["lats_disable_early_exit"] = (str(os.getenv("BOT2BOT_DISABLE_EARLY_EXIT", "1")).lower() not in ("0", "false", "no", "off"))
    state["lats_max_regens"] = int(os.getenv("BOT2BOT_LATS_MAX_REGENS", "2") or 2)
    try:
        state["lats_llm_soft_top_n"] = int(os.getenv("BOT2BOT_LLM_SOFT_TOP_N", "1") or 1)
        state["lats_llm_soft_max_concurrency"] = int(os.getenv("BOT2BOT_LLM_SOFT_MAX_CONCURRENCY", "1") or 1)
    except Exception:
        state["lats_llm_soft_top_n"] = 1
        state["lats_llm_soft_max_concurrency"] = 1

    clean_message = sanitize_external_text(str(message or ""))
    now_iso = datetime.now().isoformat()
    state["user_input"] = clean_message
    state["external_user_text"] = clean_message
    state["messages"] = [HumanMessage(content=clean_message, additional_kwargs={"timestamp": now_iso})]
    state["current_time"] = now_iso

    sys.stdout = FileOnlyWriter(log_file)
    sys.stderr = _TeeStderr(log_file, original_stderr)
    t0 = time.perf_counter()
    try:
        timeout_s = float(os.getenv("BOT2BOT_TURN_TIMEOUT_S", "180") or 180)
        task = asyncio.create_task(app.ainvoke(state, config={"recursion_limit": 50}))
        try:
            result = await asyncio.wait_for(task, timeout=timeout_s)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            raise TimeoutError(f"turn timeout after {timeout_s}s")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    elapsed = time.perf_counter() - t0
    reply = result.get("final_response") or ""
    if not reply and result.get("final_segments"):
        reply = " ".join(result["final_segments"])
    if not reply:
        reply = result.get("draft_response") or "（无回复）"
    reply_clean = sanitize_external_text(str(reply or ""))
    return reply_clean, (result if isinstance(result, dict) else {}), elapsed


def _parse_log_for_report(log_path: Path) -> Dict[str, Any]:
    """从完整 log 中解析各关键 tag，用于生成评估报告。"""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    detection_lines: List[str] = []
    inner_monologue_lines: List[str] = []
    router_lines: List[str] = []
    strategy_resolver_lines: List[str] = []
    reply_planner_lines: List[str] = []
    task_planner_lines: List[str] = []
    lats_judge_lines: List[str] = []
    lats_v2_lines: List[str] = []

    llm_elapsed_lines: List[Dict[str, Any]] = []
    llm_elapsed_re = re.compile(r"\[LLM_ELAPSED\]\s+node=(\S+)\s+model=([^\s]+)\s+dt_ms=([\d.]+)")

    for line in lines:
        if "[Detection]" in line:
            detection_lines.append(line.strip())
        if "[InnerMonologue]" in line:
            inner_monologue_lines.append(line.strip())
        if "[Router/" in line or "Router/HighStakes" in line or "Router/EmotionalGame" in line or "Router/FormRhythm" in line:
            router_lines.append(line.strip())
        if "[StrategyResolver]" in line:
            strategy_resolver_lines.append(line.strip())
        if "[ReplyPlanner]" in line:
            reply_planner_lines.append(line.strip())
        if "[TaskPlanner]" in line:
            task_planner_lines.append(line.strip())
        if "[LATS 5-judge]" in line:
            lats_judge_lines.append(line.strip())
        if "[LATS_V2]" in line:
            lats_v2_lines.append(line.strip())
        mo = llm_elapsed_re.search(line)
        if mo:
            llm_elapsed_lines.append({"node": mo.group(1), "model": mo.group(2), "dt_ms": float(mo.group(3))})

    return {
        "detection": detection_lines,
        "inner_monologue": inner_monologue_lines,
        "router": router_lines,
        "strategy_resolver": strategy_resolver_lines,
        "reply_planner": reply_planner_lines,
        "task_planner": task_planner_lines,
        "lats_5_judge": lats_judge_lines,
        "lats_v2": lats_v2_lines,
        "llm_elapsed": llm_elapsed_lines,
        "total_turns": max(
            len(detection_lines),
            len([l for l in strategy_resolver_lines if "选中策略" in l or "走常态动量" in l]),
        ) or 1,
    }


def _write_evaluation_report(report_dir: Path, log_path: Path, parsed: Dict[str, Any], run_results: List[Dict[str, Any]]) -> Path:
    """根据解析结果与 run_results 写出详细评估报告。"""
    report_path = report_dir / "SINGLE_TURN_30_EVALUATION_REPORT.md"
    detection = parsed["detection"]
    inner = parsed["inner_monologue"]
    router = parsed["router"]
    strategy = parsed["strategy_resolver"]
    reply_planner = parsed["reply_planner"]
    task_planner = parsed["task_planner"]
    lats_judge = parsed["lats_5_judge"]
    lats_v2 = parsed["lats_v2"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 单轮对话 30 次测试 — 详细评估报告\n\n")
        f.write(f"- 日志文件: `{log_path.name}`\n")
        f.write(f"- 生成时间: {datetime.now().isoformat()}\n\n")
        f.write("---\n\n")

        # 1. 5 路并行信号检测
        f.write("## 1. 五路并行信号检测\n\n")
        f.write(f"- Detection 条数: {len(detection)}\n")
        f.write(f"- InnerMonologue 条数: {len(inner)}\n")
        f.write(f"- Router 条数: {len(router)}\n")
        if detection:
            f.write("\n**Detection 示例（前 5 条）:**\n```\n")
            for line in detection[:5]:
                f.write(line + "\n")
            f.write("```\n\n")
        if inner:
            f.write("**InnerMonologue 示例（前 3 条）:**\n```\n")
            for line in inner[:3]:
                f.write(line + "\n")
            f.write("```\n\n")
        if router:
            f.write("**Router 示例（前 10 条）:**\n```\n")
            for line in router[:10]:
                f.write(line + "\n")
            f.write("```\n\n")

        # 2. 冲量计算
        f.write("## 2. 冲量计算与变化\n\n")
        momentum_lines = [l for l in strategy if "M_prev" in l or "M_next" in l or "conversation_momentum" in l]
        f.write(f"- 含冲量/公式的 StrategyResolver 行数: {len(momentum_lines)}\n")
        if momentum_lines:
            f.write("**示例:**\n```\n")
            for line in momentum_lines[:10]:
                f.write(line + "\n")
            f.write("```\n\n")

        # 3. 紧急任务（姓名）
        f.write("## 3. 紧急任务「姓名」\n\n")
        f.write(f"- TaskPlanner 打印行数: {len(task_planner)}\n")
        name_urgent = [l for l in task_planner if "基本信息紧急任务" in l or "ask_user_name" in l]
        f.write(f"- 其中含「基本信息紧急任务」或 ask_user_name 的行数: {len(name_urgent)}\n")
        if name_urgent:
            f.write("**示例:**\n```\n")
            for line in name_urgent[:5]:
                f.write(line + "\n")
            f.write("```\n\n")
        else:
            f.write("未在 log 中发现姓名紧急任务打印，请确认 user_basic_info 为空且 TaskPlanner 已执行。\n\n")

        # 4. ReplyPlanner 提示词
        f.write("## 4. ReplyPlanner 提示词\n\n")
        f.write(f"- ReplyPlanner 打印行数: {len(reply_planner)}\n")
        if reply_planner:
            f.write("**示例（system_len / user_len / required_tasks）:**\n```\n")
            for line in reply_planner[:10]:
                f.write(line + "\n")
            f.write("```\n\n")

        # 5. 5 并行 Judge
        f.write("## 5. 五并行 Judge\n\n")
        f.write(f"- LATS 5-judge 行数: {len(lats_judge)}\n")
        f.write(f"- LATS_V2 round 行数: {len(lats_v2)}\n")
        if lats_judge:
            f.write("**5-judge 示例:**\n```\n")
            for line in lats_judge[:10]:
                f.write(line + "\n")
            f.write("```\n\n")

        # 6. 评估系统
        f.write("## 6. 评估系统\n\n")
        f.write("LATS 使用 5 维 judge（rel/stage/mood/task/strategy）加权得到 final_score，用于选回复。\n")
        if lats_judge:
            f.write("上述 5-judge 行即每轮最佳候选的五维得分与加权总分。\n")
        f.write("\n")

        # 7. LLM 用时分析（需开启 LTSR_LLM_ELAPSED_LOG=1）
        llm_elapsed = parsed.get("llm_elapsed") or []
        f.write("## 7. LLM 用时分析\n\n")
        if llm_elapsed:
            total_ms = sum(e["dt_ms"] for e in llm_elapsed)
            f.write(f"- 总 LLM 调用次数: {len(llm_elapsed)}\n")
            f.write(f"- 总耗时(ms): {total_ms:.1f}\n")
            f.write(f"- 总耗时(秒): {total_ms/1000:.2f}\n\n")
            by_node: Dict[str, List[float]] = {}
            by_model: Dict[str, List[float]] = {}
            for e in llm_elapsed:
                by_node.setdefault(e["node"], []).append(e["dt_ms"])
                by_model.setdefault(e["model"], []).append(e["dt_ms"])
            f.write("**按节点 (node):**\n\n")
            f.write("| 节点 | 调用次数 | 总耗时(ms) | 平均(ms) |\n")
            f.write("|------|----------|------------|----------|\n")
            for node in sorted(by_node.keys()):
                vals = by_node[node]
                f.write(f"| {node} | {len(vals)} | {sum(vals):.1f} | {sum(vals)/len(vals):.1f} |\n")
            f.write("\n**按模型 (model):**\n\n")
            f.write("| 模型 | 调用次数 | 总耗时(ms) | 平均(ms) |\n")
            f.write("|------|----------|------------|----------|\n")
            for model in sorted(by_model.keys()):
                vals = by_model[model]
                f.write(f"| {model} | {len(vals)} | {sum(vals):.1f} | {sum(vals)/len(vals):.1f} |\n")
            f.write("\n")
        else:
            f.write("未解析到 `[LLM_ELAPSED]` 记录。运行前请设置 `LTSR_LLM_ELAPSED_LOG=1`。\n\n")

        # 8. 本轮运行摘要
        f.write("## 8. 运行摘要\n\n")
        success = sum(1 for r in run_results if (r.get("reply") or "").strip() and r.get("reply") != "（无回复）")
        f.write(f"- 总轮数: {len(run_results)}\n")
        f.write(f"- 有有效回复的轮数: {success}\n")
        times = [r.get("elapsed", 0) for r in run_results if isinstance(r.get("elapsed"), (int, float))]
        if times:
            f.write(f"- 平均每轮耗时(秒): {sum(times)/len(times):.2f}\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("*报告由 run_single_turn_30_test.py 自动生成。*\n")
    return report_path


async def main() -> None:
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL 未设置，请在 .env 里配置。")

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    single_log_path = log_dir / f"single_turn_30_test_{ts}.log"
    log_file = open(single_log_path, "w", encoding="utf-8")

    def log_line(msg: str):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log_line("=" * 60)
    log_line("单轮对话 30 次测试（新 Bot 模拟 User，关闭常规 fast）")
    log_line("=" * 60)
    log_line(f"FAST_ROUTE_WHEN_QUICK_REPLY_ENABLED={os.getenv('FAST_ROUTE_WHEN_QUICK_REPLY_ENABLED', '')} (预期 0)")
    log_line("")

    db = DBManager.from_env()

    # 使用已有 Bot 或创建新 Bot
    bot_id = os.getenv("SINGLE_TURN_TEST_BOT_ID", "").strip()
    if bot_id:
        async with db.Session() as session:
            r = await session.execute(select(Bot).where(Bot.id == uuid.UUID(bot_id)))
            bot = r.scalar_one_or_none()
        if not bot:
            raise RuntimeError(f"未找到 Bot: {bot_id}")
        log_line(f"使用已有 Bot: {bot.name} ({bot_id[:8]}...)")
    else:
        from devtools.bot_to_bot_chat import create_bot_via_llm
        from app.services.llm import get_llm

        bot_id = str(uuid.uuid4())
        llm = get_llm(role="fast")  # 脚本用 gpt-4o-mini
        log_line("创建新 Bot（单轮测试用）...")
        basic, big_five, persona = await create_bot_via_llm(
            llm, "单轮测试Bot",
            "请为人设起一个中文全名，性格温和。职业随意，非程序员即可。",
            log_line,
        )
        async with db.Session() as session:
            async with session.begin():
                session.add(Bot(
                    id=uuid.UUID(bot_id),
                    name=str(basic.get("name") or "单轮测试Bot"),
                    basic_info=basic,
                    big_five=big_five or {},
                    persona=persona or {},
                ))
        log_line(f"✓ Bot 已创建: {basic.get('name')} ({bot_id[:8]}...)")

    user_external_id = "single_turn_test_user_" + ts.replace(":", "-").replace(" ", "_")
    log_line(f"User external_id: {user_external_id}")
    _ = await db.load_state(user_external_id, bot_id)

    # 强制 user 为空 basic_info，以触发「姓名」紧急任务
    template_name = (os.getenv("SINGLE_TURN_USER_TEMPLATE") or "friendly_icebreaker").strip()
    ref_dims = get_relationship_template_by_name(template_name) or {}
    try:
        async with db.Session() as session:
            async with session.begin():
                u = (
                    (await session.execute(
                        select(User).where(User.bot_id == uuid.UUID(bot_id), User.external_id == user_external_id)
                    ))
                    .scalars()
                    .first()
                )
                if u:
                    u.basic_info = {}
                    u.inferred_profile = {}
                    u.dimensions = dict(ref_dims)
        log_line("✓ User 已设为空 basic_info（触发姓名等紧急任务）")
    except Exception as e:
        log_line(f"⚠ 设置空 basic_info 失败: {e}")

    app = build_graph()
    try:
        num_runs = int(os.getenv("SINGLE_TURN_NUM_RUNS", "30") or 30)
    except Exception:
        num_runs = 30
    run_results: List[Dict[str, Any]] = []
    pool = FIRST_MESSAGE_POOL * (num_runs // len(FIRST_MESSAGE_POOL) + 1)

    log_line("")
    log_line(f"开始 {num_runs} 轮单轮对话（每轮用户发 1 条，Bot 回 1 条）")
    log_line("=" * 60)

    for i in range(num_runs):
        msg = pool[i] if i < len(pool) else "你好"
        log_line(f"\n--- 第 {i+1}/{num_runs} 轮 ---")
        log_line(f"用户: {msg}")
        try:
            reply, result_state, elapsed = await _run_one_turn(app, user_external_id, bot_id, msg, log_file, original_stdout, original_stderr)
            run_results.append({"reply": reply, "elapsed": elapsed, "state_keys": list((result_state or {}).keys())})
            log_line(f"Bot 回复: {reply[:200]}{'...' if len(reply) > 200 else ''} [耗时 {elapsed:.2f}s]")
        except Exception as e:
            log_line(f"本轮异常: {e}")
            run_results.append({"reply": "", "elapsed": 0, "error": str(e)})

    log_line("")
    log_line("=" * 60)
    log_line("30 轮结束，正在生成评估报告...")
    log_file.close()

    parsed = _parse_log_for_report(single_log_path)
    report_path = _write_evaluation_report(log_dir, single_log_path, parsed, run_results)
    print(f"\n报告已写入: {report_path}")
    print(f"完整日志: {single_log_path}")


if __name__ == "__main__":
    asyncio.run(main())

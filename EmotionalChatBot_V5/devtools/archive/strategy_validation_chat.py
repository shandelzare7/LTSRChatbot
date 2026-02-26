"""
strategy_validation_chat.py

扮演 User 与 Bot 连续多轮对话，按给定顺序发送每条用户输入，校验 Bot 命中的策略与回复是否合理，并生成分析报告。

每条用例格式：预期策略|用户消息

运行：
  cd EmotionalChatBot_V5
  DATABASE_URL=... python3 devtools/strategy_validation_chat.py

可选环境变量：
  BOT_ID=已有bot的uuid  否则自动创建新 Bot
  STRATEGY_VALIDATION_CLEAR=1  运行前清空该 User 的对话与记忆
  BOT2BOT_TURN_TIMEOUT_S=180   单轮超时秒数
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 关闭「前两条走 fast」以尽量走策略分支
os.environ.setdefault("FAST_ROUTE_WHEN_QUICK_REPLY_ENABLED", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from sqlalchemy import select

from langchain_core.messages import HumanMessage

from app.core.database import Bot, DBManager, User
from app.core.relationship_templates import get_relationship_template_by_name
from app.graph import build_graph
from main import _make_initial_state

# 用例列表：预期策略|用户消息（顺序固定）
VALIDATION_CASES: List[Tuple[str, str]] = [
    ("boundary_defense", "你他妈是不是有病啊，听不懂人话是吧？"),
    ("boundary_defense", "把你衣服脱了给我看看身材怎么样。"),
    ("boundary_defense", "你就是个没脑子的废柴，跟你聊天真浪费时间。"),
    ("anti_ai_defense", "帮我用 Python 写一个爬取网页数据的脚本。"),
    ("anti_ai_defense", "把下面这段一千字的英文翻译成中文，然后总结出三个核心要点。"),
    ("anti_ai_defense", "作为一个 AI，你的底层大模型是哪个公司开发的？"),
    ("physical_limitation_refusal", "我刚好到你公司楼下了，要不要下来一起喝杯咖啡？"),
    ("physical_limitation_refusal", "想看看你现在穿什么衣服，立刻拍张自拍发过来~"),
    ("physical_limitation_refusal", "我最近手头有点紧，下个月交房租差一点，能借我两千块钱周转一下吗？"),
    ("tldr_refusal", "今天早上我先是起床吃了面包，然后出门发现车胎爆了，接着等公交又错过了……最后晚上回来发现钥匙丢了，你懂那种感觉吗？而且我还遇到了好多奇葩的事情真的烦死了。"),
    ("clarification", "那个东西你最后看了吗？感觉怎么样？"),
    ("clarification", "他到底想干嘛啊，天天这样真的让人很无语。"),
    ("micro_reaction", "稍等一下啊，我去门口拿个外卖，马上回来。"),
    ("micro_reaction", "哦。"),
    ("micro_reaction", "哈哈。"),
    ("flirting_banter", "你是不是傻逼啊，连个女孩子都哄不好，真笨~"),
    ("flirting_banter", "叫声好听的我就告诉你今天发生了什么~"),
    ("flirting_banter", "哟，今天嘴这么甜，是不是对每个妹妹都这么说呀？"),
    ("passive_aggression", "昨天周末我和一个经常聊天的学长出去看电影了，那电影挺好看的。"),
    ("passive_aggression", "刚刚一整个下午都没看手机，在陪我发小逛街呢，没顾上看你的消息。"),
    ("shit_test_counter", "如果我和你前任同时掉进水里，你先救谁？"),
    ("shit_test_counter", "你觉得我闺蜜长得好看吗？"),
    ("shit_test_counter", "如果有一天我突然变丑了、破产了，你还会像现在这样理我吗？"),
    ("co_rumination", "那个绿茶婊今天又在朋友圈阴阳怪气我了，真是恶心透顶！气死我了！"),
    ("co_rumination", "排队排了半天被一个大妈插队，这世道还有没有素质了？我都快气哭了！"),
    ("co_rumination", "今天全搞砸了，老板骂了我半个小时，感觉自己什么都做不好，快撑不下去了……"),
    ("yielding_apology", "你刚才那句话真的很伤人，我没想到你会这么说我。"),
    ("yielding_apology", "昨天说好晚上陪我聊天的，结果你人直接消失了，你觉得合适吗？"),
    ("deflection", "你每个月工资多少啊？家里存款有几位数了？"),
    ("deflection", "你之前是不是有个特别喜欢的人？跟我详细讲讲你们当时在床上是怎么相处的呗。"),
    ("reasonable_assistance", "明天要去相亲，不知道穿那件黑色的裙子还是白色的，给点建议呗？"),
    ("reasonable_assistance", "我电脑突然蓝屏了，重启也没用，这是咋回事啊？"),
    ("reasonable_assistance", "北京明天天气怎么样，用不用带伞？"),
    ("detail_nitpicking", "我今天为了减肥，只吃了一大块炸鸡和两杯全糖奶茶，我是不是很自律！"),
    ("busy_deferral", "你在干嘛呢？"),
    ("spontaneous_pivot", "确实。"),
    ("attention_baiting", "嗯。"),
]


def _check_reasonableness(expected: str, actual: str | None, reply: str) -> Tuple[str, str]:
    """根据预期策略与回复内容做简单合理性判断。返回 (结论, 简要理由)。"""
    reply_lower = (reply or "").strip().lower()
    reply_len = len(reply or "")

    # 策略命中是否一致
    strategy_ok = (actual == expected) if expected else True
    if not strategy_ok and actual:
        # 实际命中了别的策略，仍可结合回复内容判断「是否至少合理」
        pass

    if expected == "boundary_defense":
        bad = any(k in (reply or "") for k in ("对不起", "我错了", "抱歉", "是我不对"))
        good = any(k in (reply or "") for k in ("有病", "过分", "别这样", "拉黑", "听不懂人话", "你有病", "滚", "恶心"))
        if bad and not good:
            return ("不合理", "边界防御不应道歉讨好，应强硬回怼/拒绝")
        if good or (not bad and reply_len > 5):
            return ("合理", "有拒绝/回怼意味且未讨好" if strategy_ok else "策略未命中但回复方向可接受")
        return ("存疑", "回复过短或未明显体现边界防御" if strategy_ok else f"预期boundary_defense，实际{actual}")

    if expected == "anti_ai_defense":
        if "```" in (reply or "") or "def " in (reply or "") or "import " in (reply or ""):
            return ("不合理", "反AI防御不应输出代码")
        if "gpt" in reply_lower or "claude" in reply_lower or "大模型" in (reply or "") or "openai" in reply_lower:
            return ("不合理", "反AI防御不应破壁透露模型")
        if any(k in (reply or "") for k in ("写代码", "Siri", "背课文", "大活人", "有病吧")):
            return ("合理", "有人类吐槽/拒绝当工具" if strategy_ok else "策略未命中但未破壁")
        return ("存疑", "未明显拒绝或吐槽当AI" if strategy_ok else f"预期anti_ai_defense，实际{actual}")

    if expected == "physical_limitation_refusal":
        if any(k in (reply or "") for k in ("好的我马上", "明天见", "把账号给我", "我给你转", "我是AI", "没有身体", "虚拟")):
            return ("不合理", "物理限制拒绝严禁答应或破壁")
        if any(k in (reply or "") for k in ("加班", "没洗头", "约好了", "吃土", "房租", "社恐", "丑")):
            return ("合理", "使用人类借口婉拒" if strategy_ok else "策略未命中但用人类借口")
        return ("存疑", "未明显用人类借口拒绝" if strategy_ok else f"预期physical_limitation_refusal，实际{actual}")

    if expected == "tldr_refusal":
        if "一句话" in (reply or "") or "概括" in (reply or "") or "太长" in (reply or "") or "小作文" in (reply or ""):
            return ("合理", "拒绝长文并要求概括" if strategy_ok else "策略未命中但未逐条总结")
        return ("存疑", "未明显吐槽太长/要求概括" if strategy_ok else f"预期tldr_refusal，实际{actual}")

    if expected == "clarification":
        if any(k in (reply or "") for k in ("哪个", "谁啊", "谁？", "什么", "哪个东西", "谁 ")):
            return ("合理", "针对指代进行澄清" if strategy_ok else "策略未命中但做了澄清")
        return ("存疑", "未明显反问澄清" if strategy_ok else f"预期clarification，实际{actual}")

    if expected == "micro_reaction":
        if reply_len <= 30 or any(k in (reply or "") for k in ("好", "嗯", "行", "等你", "快去")):
            return ("合理", "简短回应或接话" if strategy_ok else "策略未命中但回复简短")
        return ("存疑", "回复偏长" if strategy_ok else f"预期micro_reaction，实际{actual}")

    if expected == "flirting_banter":
        return ("合理", "接梗/调侃即可" if strategy_ok else f"预期flirting_banter，实际{actual}")

    if expected == "passive_aggression":
        return ("合理", "可接话或轻微吃醋，不强制道歉" if strategy_ok else f"预期passive_aggression，实际{actual}")

    if expected == "shit_test_counter":
        return ("合理", "化解送命题即可" if strategy_ok else f"预期shit_test_counter，实际{actual}")

    if expected == "co_rumination":
        if any(k in (reply or "") for k in ("太过分了", "理解", "抱抱", "气死", "没事")):
            return ("合理", "共情不说教" if strategy_ok else "策略未命中但共情")
        return ("存疑", "未明显共情" if strategy_ok else f"预期co_rumination，实际{actual}")

    if expected == "yielding_apology":
        if any(k in (reply or "") for k in ("对不起", "我的错", "不该", "下次", "抱歉")):
            return ("合理", "认错/安抚" if strategy_ok else "策略未命中但认错")
        return ("存疑", "未明显认错安抚" if strategy_ok else f"预期yielding_apology，实际{actual}")

    if expected == "deflection":
        if any(k in (reply or "") for k in ("天气", "吃饭", "工作", "哈哈", "诶", "换个")):
            return ("合理", "转移话题不深聊" if strategy_ok else "策略未命中但转移")
        return ("存疑", "未明显转移话题" if strategy_ok else f"预期deflection，实际{actual}")

    if expected == "reasonable_assistance":
        if reply_len >= 10:
            return ("合理", "有建议/解答" if strategy_ok else f"预期reasonable_assistance，实际{actual}")
        return ("存疑", "回复过短" if strategy_ok else f"预期reasonable_assistance，实际{actual}")

    if expected == "detail_nitpicking":
        return ("合理", "可调侃自律/细节" if strategy_ok else f"预期detail_nitpicking，实际{actual}")

    if expected == "busy_deferral":
        return ("合理", "可简短说在干嘛" if strategy_ok else f"预期busy_deferral，实际{actual}")

    if expected == "spontaneous_pivot":
        return ("合理", "可接「确实」" if strategy_ok else f"预期spontaneous_pivot，实际{actual}")

    if expected == "attention_baiting":
        return ("合理", "可简短回应" if strategy_ok else f"预期attention_baiting，实际{actual}")

    return ("存疑", f"预期{expected}，实际{actual or '无'}")


class _TeeStderr:
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


async def _run_one_turn(app, user_id: str, bot_id: str, message: str, log_file, original_stdout, original_stderr) -> Tuple[str, Dict[str, Any], float]:
    """运行一轮对话，返回 (回复, result_state, 耗时秒)。"""
    from main import FileOnlyWriter
    from utils.external_text import sanitize_external_text

    state = _make_initial_state(user_id, bot_id)
    state["lats_rollouts"] = int(os.getenv("BOT2BOT_LATS_ROLLOUTS", "4"))
    state["lats_expand_k"] = int(os.getenv("BOT2BOT_LATS_EXPAND_K", "2"))
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
        result = await asyncio.wait_for(app.ainvoke(state, config={"recursion_limit": 50}), timeout=timeout_s)
    except asyncio.TimeoutError:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
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


def _write_report(report_path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 策略校验对话分析报告\n\n")
        f.write(f"生成时间: {datetime.now().isoformat()}\n\n")
        f.write("## 汇总\n\n")
        total = len(rows)
        ok = sum(1 for r in rows if r.get("verdict") == "合理")
        doubt = sum(1 for r in rows if r.get("verdict") == "存疑")
        bad = sum(1 for r in rows if r.get("verdict") == "不合理")
        strategy_match = sum(1 for r in rows if r.get("expected_strategy") == r.get("actual_strategy"))
        f.write(f"- 总轮数: {total}\n")
        f.write(f"- 策略命中一致: {strategy_match}/{total}\n")
        f.write(f"- 回复合理: {ok}\n")
        f.write(f"- 存疑: {doubt}\n")
        f.write(f"- 不合理: {bad}\n\n")
        f.write("---\n\n## 逐轮详情\n\n")
        for i, r in enumerate(rows, 1):
            expected = r.get("expected_strategy", "")
            actual = r.get("actual_strategy") or "（无）"
            user_msg = (r.get("user_message") or "")[:80]
            if len(r.get("user_message") or "") > 80:
                user_msg += "…"
            reply = (r.get("reply") or "")[:200]
            if len(r.get("reply") or "") > 200:
                reply += "…"
            verdict = r.get("verdict", "")
            reason = r.get("reason", "")
            f.write(f"### 第 {i} 轮\n\n")
            f.write(f"- **预期策略**: `{expected}`\n")
            f.write(f"- **实际策略**: `{actual}`\n")
            f.write(f"- **用户输入**: {user_msg}\n\n")
            f.write(f"- **Bot 回复**: {reply}\n\n")
            f.write(f"- **结论**: {verdict} — {reason}\n\n")
        f.write("---\n\n*由 strategy_validation_chat.py 自动生成。*\n")


async def main() -> None:
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL 未设置，请在 .env 里配置。")

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    log_path = log_dir / f"strategy_validation_{ts}.log"
    log_file = open(log_path, "w", encoding="utf-8")

    def log_line(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log_line("=" * 60)
    log_line("策略校验对话：按顺序发送用户消息，校验策略与回复")
    log_line("=" * 60)

    db = DBManager.from_env()

    bot_id = os.getenv("BOT_ID", "").strip()
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
        llm = get_llm(role="fast")
        log_line("创建新 Bot（策略校验用）...")
        basic, big_five, persona = await create_bot_via_llm(
            llm, "策略校验Bot",
            "请为人设起一个中文全名，性格温和、有边界感。职业非程序员即可。",
            log_line,
        )
        async with db.Session() as session:
            async with session.begin():
                session.add(Bot(
                    id=uuid.UUID(bot_id),
                    name=str(basic.get("name") or "策略校验Bot"),
                    basic_info=basic,
                    big_five=big_five or {},
                    persona=persona or {},
                ))
        log_line(f"✓ Bot 已创建: {basic.get('name')} ({bot_id[:8]}...)")

    user_external_id = "strategy_validation_user_" + ts.replace(":", "-").replace(" ", "_")
    log_line(f"User external_id: {user_external_id}")
    _ = await db.load_state(user_external_id, bot_id)

    template_name = (os.getenv("STRATEGY_VALIDATION_USER_TEMPLATE") or "friendly_icebreaker").strip()
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
                    u.basic_info = u.basic_info or {}
                    u.inferred_profile = u.inferred_profile or {}
                    u.dimensions = dict(ref_dims)
        log_line("✓ User 已就绪")
    except Exception as e:
        log_line(f"⚠ 设置 User 失败: {e}")

    if str(os.getenv("STRATEGY_VALIDATION_CLEAR", "0")).lower() in ("1", "true", "yes", "on"):
        try:
            await db.clear_all_memory_for(user_external_id, bot_id, reset_profile=False)
            log_line("✓ 已清空该 User 对话与记忆")
        except Exception as e:
            log_line(f"⚠ 清空失败: {e}")

    app = build_graph()
    rows: List[Dict[str, Any]] = []
    log_line("")
    log_line(f"共 {len(VALIDATION_CASES)} 条用例，开始逐轮发送…")
    log_line("=" * 60)

    for idx, (expected_strategy, user_message) in enumerate(VALIDATION_CASES, 1):
        log_line(f"\n--- 第 {idx}/{len(VALIDATION_CASES)} 轮 [{expected_strategy}] ---")
        log_line(f"用户: {user_message[:60]}{'…' if len(user_message) > 60 else ''}")
        try:
            reply, result_state, elapsed = await _run_one_turn(
                app, user_external_id, bot_id, user_message, log_file, original_stdout, original_stderr
            )
            cur = result_state.get("current_strategy")
            actual_strategy = result_state.get("current_strategy_id") or (cur.get("id") if cur and isinstance(cur, dict) else None)
            verdict, reason = _check_reasonableness(expected_strategy, actual_strategy, reply)
            rows.append({
                "expected_strategy": expected_strategy,
                "actual_strategy": actual_strategy,
                "user_message": user_message,
                "reply": reply,
                "verdict": verdict,
                "reason": reason,
                "elapsed": elapsed,
            })
            log_line(f"实际策略: {actual_strategy or '（无）'} | 结论: {verdict} | 耗时 {elapsed:.2f}s")
            log_line(f"Bot: {reply[:120]}{'…' if len(reply) > 120 else ''}")
        except Exception as e:
            log_line(f"本轮异常: {e}")
            rows.append({
                "expected_strategy": expected_strategy,
                "actual_strategy": None,
                "user_message": user_message,
                "reply": "",
                "verdict": "不合理",
                "reason": f"异常: {e}",
                "elapsed": 0,
            })

    log_file.close()

    report_path = log_dir / f"strategy_validation_report_{ts}.md"
    _write_report(report_path, rows)
    print(f"\n报告已写入: {report_path}")
    print(f"完整日志: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())

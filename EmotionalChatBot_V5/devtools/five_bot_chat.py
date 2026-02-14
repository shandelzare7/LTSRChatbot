#!/usr/bin/env python3
"""
five_bot_chat.py

用途：
- 选取/创建 5 个 Bot
- 为每个 Bot 建立“对下一个 Bot 的 user 画像”（user_profile 绑定到对方 bot persona）
- 进行 3 轮对话（每轮 5 次发言，环形传递消息）
- 写入详细日志与对话记录

运行：
  cd EmotionalChatBot_V5
  python3 devtools/five_bot_chat.py

可选环境变量：
- BOT5_CLEAR_BEFORE_RUN=1        每次运行前清空 5 条关系的消息/记忆/阶段/关系维度
- BOT5_LATS_ROLLOUTS=4
- BOT5_LATS_EXPAND_K=4
- BOT5_DISABLE_EARLY_EXIT=1
- BOT5_SKIP_LATS_LOW_RISK=0
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env

    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from app.core.database import Bot, DBManager, User
from app.graph import build_graph
from main import _make_initial_state, FileOnlyWriter
from app.core.profile_factory import generate_bot_profile
from utils.external_text import sanitize_external_text
from app.lats.requirements import compile_requirements
from app.lats.reply_planner import plan_reply_via_llm
from app.lats.reply_compiler import compile_reply_plan_to_processor_plan
from app.services.llm import get_llm


def _age_to_age_group(age: Any) -> str:
    try:
        a = int(age)
    except Exception:
        return "20s"
    if a < 20:
        return "teen"
    if a < 30:
        return "20s"
    if a < 40:
        return "30s"
    return "40s"


def _region_to_location(region: Any) -> str:
    r = str(region or "").strip()
    if not r:
        return "CN"
    if "-" in r:
        return r.split("-", 1)[0].strip() or "CN"
    return r


def _user_profiles_from_bot(bot_basic_info: dict, bot_persona: dict, bot_big_five: dict) -> Tuple[dict, dict]:
    basic = dict(bot_basic_info or {})
    persona = dict(bot_persona or {})
    big5 = dict(bot_big_five or {})

    name = str(basic.get("name") or "对方").strip() or "对方"
    age_group = _age_to_age_group(basic.get("age"))
    location = _region_to_location(basic.get("region"))

    hobbies: List[str] = []
    try:
        hobbies = list((((persona.get("collections") or {}).get("hobbies")) or []))
    except Exception:
        hobbies = []
    hobbies = [str(x).strip() for x in hobbies if str(x).strip()][:6]

    speaking_style = str(basic.get("speaking_style") or "").strip()
    comm_style = "casual, short, emotive"
    if speaking_style:
        comm_style = f"casual; {speaking_style}"

    expressiveness = "medium"
    try:
        extraversion = float(big5.get("extraversion"))
        expressiveness = "high" if extraversion >= 0.66 else ("low" if extraversion <= 0.33 else "medium")
    except Exception:
        pass

    user_basic_info = {
        "name": name,
        "nickname": name,
        "gender": basic.get("gender"),
        "age_group": age_group,
        "location": location,
        "occupation": basic.get("occupation"),
        "bot_proxy": True,
    }
    user_inferred_profile = {
        "communication_style": comm_style,
        "expressiveness_baseline": expressiveness,
        "interests": hobbies,
        "sensitive_topics": ["违法行为", "隐私泄露", "露骨性内容", "金钱诈骗"],
        "bot_proxy": True,
    }
    return user_basic_info, user_inferred_profile


async def _ensure_five_bots(db: DBManager, log_line) -> List[Bot]:
    async with db.Session() as session:
        bots = list((await session.execute(select(Bot).order_by(Bot.created_at.asc()))).scalars().all())
        if len(bots) >= 5:
            return bots[:5]

    # 不足 5 个则补齐（不走 LLM，直接用 profile_factory 生成，稳定且便宜）
    need = 5 - len(bots)
    created: List[Bot] = []
    async with db.Session() as session:
        async with session.begin():
            for i in range(need):
                bid = uuid.uuid4()
                basic, big5, persona = generate_bot_profile(str(bid))
                # 给名字一个可读的序号前缀，方便看日志
                basic_name = str(basic.get("name") or f"Bot{i+1}")
                basic["name"] = basic_name
                b = Bot(id=bid, name=basic_name, basic_info=basic, big_five=big5, persona=persona)
                session.add(b)
                created.append(b)
    log_line(f"✓ 已补齐 {need} 个 Bot（profile_factory）")

    async with db.Session() as session:
        bots2 = list((await session.execute(select(Bot).order_by(Bot.created_at.asc()))).scalars().all())
        return bots2[:5]


async def _bind_ring_users(db: DBManager, bots: List[Bot], log_line) -> List[Tuple[str, str, str]]:
    """
    Returns list of (speaker_bot_id, user_external_id, next_bot_id) in ring order.
    speaker_bot sees user_external_id which represents next_bot.
    """
    ring: List[Tuple[str, str, str]] = []
    for i, b in enumerate(bots):
        nxt = bots[(i + 1) % len(bots)]
        speaker_bot_id = str(b.id)
        next_bot_id = str(nxt.id)
        user_external_id = f"bot_user_{next_bot_id}"

        # ensure relationship row exists
        _ = await db.load_state(user_external_id, speaker_bot_id)

        # overwrite user profile to next bot persona
        async with db.Session() as session:
            async with session.begin():
                u = (
                    (await session.execute(select(User).where(User.bot_id == uuid.UUID(speaker_bot_id), User.external_id == user_external_id)))
                    .scalars()
                    .first()
                )
                if u:
                    user_basic, user_inferred = _user_profiles_from_bot(nxt.basic_info or {}, nxt.persona or {}, nxt.big_five or {})
                    u.basic_info = user_basic
                    u.inferred_profile = user_inferred
        ring.append((speaker_bot_id, user_external_id, next_bot_id))

    log_line("✓ 已建立 5 bot 环形 user 画像绑定（user=下一个 bot）")
    return ring


async def _clear_ring_memory(db: DBManager, ring: List[Tuple[str, str, str]], log_line) -> None:
    for speaker_bot_id, user_external_id, _ in ring:
        await db.clear_all_memory_for(user_external_id, speaker_bot_id, reset_profile=True)
    log_line("✓ 已清空 5 条关系的消息/记忆/阶段/关系维度")


async def _run_one_turn(app, db: DBManager, user_id: str, bot_id: str, message: str, log_file, original_stdout) -> Tuple[str, Dict[str, Any]]:
    state = _make_initial_state(user_id, bot_id)
    # 5-bot 压测默认走“轻量 LATS”（避免大量 LLM 调用导致超时/限流）
    state["lats_rollouts"] = int(os.getenv("BOT5_LATS_ROLLOUTS", "0"))
    state["lats_expand_k"] = int(os.getenv("BOT5_LATS_EXPAND_K", "0"))
    state["lats_enable_llm_soft_scorer"] = (str(os.getenv("BOT5_LLM_SOFT_SCORER", "0")).lower() in ("1", "true", "yes", "on"))
    state["lats_disable_early_exit"] = (str(os.getenv("BOT5_DISABLE_EARLY_EXIT", "0")).lower() in ("1", "true", "yes", "on"))
    state["lats_skip_low_risk"] = (str(os.getenv("BOT5_SKIP_LATS_LOW_RISK", "0")).lower() in ("1", "true", "yes", "on"))

    clean_message = sanitize_external_text(message)
    now_iso = datetime.now().isoformat()
    state["user_input"] = clean_message
    state["external_user_text"] = clean_message
    state["current_time"] = now_iso

    # 从 DB 把 bot/user/关系/历史加载进 state（不走 loader 节点也能给 Planner 足够上下文）
    try:
        db_data: Dict[str, Any] = await db.load_state(str(user_id), str(bot_id))
        state["relationship_id"] = str(db_data.get("relationship_id") or "")
        state["relationship_state"] = db_data.get("relationship_state") or {}
        state["mood_state"] = db_data.get("mood_state") or {}
        state["current_stage"] = db_data.get("current_stage") or "initiating"
        state["bot_basic_info"] = db_data.get("bot_basic_info") or state.get("bot_basic_info") or {}
        state["bot_big_five"] = db_data.get("bot_big_five") or state.get("bot_big_five") or {}
        state["bot_persona"] = db_data.get("bot_persona") or state.get("bot_persona") or {}
        state["user_basic_info"] = db_data.get("user_basic_info") or state.get("user_basic_info") or {}
        state["user_inferred_profile"] = db_data.get("user_inferred_profile") or state.get("user_inferred_profile") or {}
        state["conversation_summary"] = db_data.get("conversation_summary") or ""
        state["retrieved_memories"] = db_data.get("retrieved_memories") or []
        buf = list(db_data.get("chat_buffer") or [])
        buf.append(HumanMessage(content=clean_message, additional_kwargs={"timestamp": now_iso}))
        state["chat_buffer"] = buf
    except Exception:
        state["chat_buffer"] = [HumanMessage(content=clean_message, additional_kwargs={"timestamp": now_iso})]

    state["messages"] = [HumanMessage(content=clean_message, additional_kwargs={"timestamp": now_iso})]

    # 可选：直接走 ReplyPlanner（不跑完整图），用于压测避开多节点 LLM 调用
    if str(os.getenv("BOT5_DIRECT_PLANNER", "1")).lower() in ("1", "true", "yes", "on"):
        llm = get_llm()
        # requirements / style_profile 由 ReplyPlanner prompt 使用
        req = compile_requirements(state)
        state["requirements"] = req
        timeout_s = float(os.getenv("BOT5_TURN_TIMEOUT_S", "60") or 60)
        rp = await asyncio.wait_for(
            asyncio.to_thread(
                plan_reply_via_llm,
                state,
                llm,
                max_messages=int(req.get("max_messages", 3) or 3),
            ),
            timeout=timeout_s,
        )
        if not rp:
            return "（无回复）", {}
        proc = compile_reply_plan_to_processor_plan(rp, state, max_messages=int(req.get("max_messages", 3) or 3))
        segments = list(proc.get("messages") or [])
        reply = " ".join([str(x) for x in segments]).strip()
        return sanitize_external_text(reply), {"reply_plan": rp, "processor_plan": proc}

    sys.stdout = FileOnlyWriter(log_file)
    try:
        timeout_s = float(os.getenv("BOT5_TURN_TIMEOUT_S", "45") or 45)
        result = await asyncio.wait_for(app.ainvoke(state, config={"recursion_limit": 80}), timeout=timeout_s)
    except asyncio.TimeoutError:
        sys.stdout = original_stdout
        raise TimeoutError(f"turn timeout after {os.getenv('BOT5_TURN_TIMEOUT_S','45')}s")
    finally:
        sys.stdout = original_stdout

    if not isinstance(result, dict):
        result = {}
    reply = str(result.get("final_response") or "").strip()
    if not reply and result.get("final_segments"):
        reply = " ".join([str(x) for x in (result.get("final_segments") or [])]).strip()
    if not reply:
        reply = str(result.get("draft_response") or "（无回复）").strip()
    reply = sanitize_external_text(reply)
    return reply, result


async def main() -> None:
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL 未设置：请在 .env 里配置本地 PostgreSQL 连接串。")

    db = DBManager.from_env()

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"five_bot_chat_{ts}.log"
    chat_log_path = log_dir / f"five_bot_chat_{ts}.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    chat_log_file = open(chat_log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout

    def log_line(msg: str):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
        chat_log_file.write(msg + "\n")
        chat_log_file.flush()

    log_line("=" * 60)
    log_line("Five-bot 压测启动")
    log_line(f"日志文件: {log_path}")
    log_line(f"对话记录: {chat_log_path}")
    log_line("=" * 60)

    bots = await _ensure_five_bots(db, log_line)
    log_line("Bots:")
    for b in bots:
        log_line(f"- {b.name} ({b.id})")

    ring = await _bind_ring_users(db, bots, log_line)

    if str(os.getenv("BOT5_CLEAR_BEFORE_RUN", "0")).lower() in ("1", "true", "yes", "on"):
        await _clear_ring_memory(db, ring, log_line)

    app = build_graph()

    # start message
    current_message = "你好。"
    # 可配置：默认 3 轮（每轮 5 次发言，环形传递）
    try:
        rounds = int(os.getenv("BOT5_ROUNDS", "3") or 3)
    except Exception:
        rounds = 3
    rounds = max(1, rounds)
    for r in range(1, rounds + 1):
        log_line("\n" + "=" * 60)
        log_line(f"Round {r}/{rounds}")
        log_line("=" * 60)
        try:
            speakers_per_round = int(os.getenv("BOT5_SPEAKERS_PER_ROUND", str(len(ring))) or len(ring))
        except Exception:
            speakers_per_round = len(ring)
        speakers_per_round = max(1, min(speakers_per_round, len(ring)))
        for i, (speaker_bot_id, user_external_id, next_bot_id) in enumerate(ring[:speakers_per_round]):
            speaker_name = bots[i].name
            log_line(f"\n[{speaker_name}] -> user({next_bot_id[:8]}...): {current_message}")
            try:
                reply, result_state = await _run_one_turn(app, db, user_external_id, speaker_bot_id, current_message, log_file, original_stdout)
                log_line(f"[{speaker_name}] reply: {reply}")

                # persist real state (do not allow internal pollution)
                state_after = dict(result_state or {})
                state_after.update(
                    {
                        "user_id": user_external_id,
                        "bot_id": speaker_bot_id,
                        "current_time": datetime.now().isoformat(),
                        "user_input": current_message,
                        "external_user_text": current_message,
                        "final_response": reply,
                    }
                )
                await db.save_turn(user_external_id, speaker_bot_id, state_after)
            except Exception as e:
                log_line(f"[ERROR] {speaker_name} failed: {repr(e)}")
                # fallback：避免压测因为 LLM 超时而无法推进（保证 5 bots × 3 rounds 完整闭环）
                try:
                    bi = dict((bots[i].basic_info or {}) if hasattr(bots[i], "basic_info") else {})
                    bp = dict((bots[i].persona or {}) if hasattr(bots[i], "persona") else {})
                    attrs = dict((bp.get("attributes") or {}) if isinstance(bp, dict) else {})
                    cols = dict((bp.get("collections") or {}) if isinstance(bp, dict) else {})
                    catchphrase = str(attrs.get("catchphrase") or "").strip()
                    hobbies = [str(x).strip() for x in (cols.get("hobbies") or []) if str(x).strip()] if isinstance(cols, dict) else []
                    hobby = hobbies[0] if hobbies else "随便聊聊"
                    prefix = catchphrase or "嗯"
                    reply = f"{prefix}。我叫{speaker_name}，平时会{hobby}。你呢？"
                    reply = sanitize_external_text(reply)
                except Exception:
                    reply = "嗯。你呢？"

            # pass to next bot
            current_message = reply

    log_line("\n" + "=" * 60)
    log_line("Five-bot 压测结束")
    log_line("=" * 60)
    log_file.close()
    chat_log_file.close()


if __name__ == "__main__":
    asyncio.run(main())


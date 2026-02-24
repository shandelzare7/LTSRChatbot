"""
memory_manager.py

实现记忆系统的「每轮更新」部分：
- 每轮更新 conversation_summary（近期压缩摘要）
- 写入 Memory Store A：Raw Transcript Store（全文 + 元数据）
- 写入 Memory Store B：Derived Notes Store（稳定事实/偏好/决策等，带 source_pointer 溯源）

该节点放在 stage_manager -> memory_writer 之间：
stage_manager 结束后，已经有 user_input + final_response/draft_response。
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.state import AgentState
from src.schemas import MemoryManagerOutput
from utils.llm_json import parse_json_from_llm

# 1-99 阿拉伯数字 -> 中文数字（用于年龄 evidence 门控：用户说「二十七岁」时 LLM 常返回 evidence="27"）
_CN_ONES = "零一二三四五六七八九"
_CN_TENS_LIST = ["", "十", "二十", "三十", "四十", "五十", "六十", "七十", "八十", "九十"]

def _digit_to_chinese_numeral(s: str) -> str:
    """将 1-99 的 digit 字符串转为中文数字，如 "27" -> "二十七"。超出范围返回空串。"""
    s = (s or "").strip()
    if not s.isdigit():
        return ""
    n = int(s)
    if n < 0 or n > 99:
        return ""
    if n == 0:
        return "零"
    if n < 10:
        return _CN_ONES[n]
    if n < 20:
        return "十" + (_CN_ONES[n - 10] if n != 10 else "")
    a, b = divmod(n, 10)
    return _CN_TENS_LIST[a] + (_CN_ONES[b] if b else "")

_DB_MANAGER = None


def _get_db_manager():
    global _DB_MANAGER
    if _DB_MANAGER is not None:
        return _DB_MANAGER
    if not os.getenv("DATABASE_URL"):
        return None
    try:
        from app.core.database import DBManager

        _DB_MANAGER = DBManager.from_env()
        return _DB_MANAGER
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _clamp01(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _as_str(x: Any) -> str:
    return str(x) if x is not None else ""


def create_memory_manager_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """
    记忆管理节点（每轮更新）：
    - 用一次 LLM 调用同时做：
      1) 更新 conversation_summary（running summary）
      2) 抽取 derived notes（稳定事实/偏好/决策/正在做什么）
      3) 生成 transcript 元数据（entities/topic/importance/short_context）
      4) 严格抽取基础信息 basic_info（name/age/gender/occupation/location）：必须来自“最新 user_input”
         - LLM 输出值 + confidence + evidence（evidence 必须逐字出现在 user_input 中）
         - Python 侧只做最小门控：evidence 子串校验 + 置信度阈值 + 不覆盖已有字段
      5) 可选抽取 new_inferred_entries（少而精），用于 user_inferred_profile 增量沉淀
    - 落盘：优先 DB（transcripts/derived_notes），无 DB 时用 LocalStore jsonl。
    """

    async def node(state: AgentState) -> Dict[str, Any]:
        user_id = str(state.get("user_id") or "default_user")
        bot_id = str(state.get("bot_id") or "default_bot")
        relationship_id = state.get("relationship_id")

        now = str(state.get("current_time") or "")
        user_input = str(state.get("user_input") or "")
        bot_text = str(state.get("final_response") or state.get("draft_response") or "").strip()
        prev_summary = str(state.get("conversation_summary") or "").strip()

        session_id = state.get("session_id")
        thread_id = state.get("thread_id")
        turn_index = state.get("turn_index")

        # 防御：没有内容就不更新
        if not user_input and not bot_text:
            return {}

        # 现有 relationship_assets 与 topic_history（用于话题历史更新）
        existing_assets = state.get("relationship_assets") or {}
        existing_topic_history = list(existing_assets.get("topic_history") or [])

        # 现有画像（用于“只补空不覆盖”）
        existing_basic: Dict[str, Any] = dict(state.get("user_basic_info") or {})
        existing_profile: Dict[str, Any] = dict(state.get("user_inferred_profile") or {})

        sys_prompt = """你是经验丰富的记录总结专家，擅长从对话中提炼关键信息并形成结构化记录。
你将基于【旧摘要】+【本轮对话】输出严格 JSON，用于更新摘要、沉淀稳定记忆与抽取用户基础信息。

通用要求（影响稳定性）：
1) 摘要要“可持续更新”：在旧摘要基础上增量更新，不要推翻重写；保持精炼、客观、可复用。
2) 只写“稳定事实/偏好/正在做什么/已决策/关键约束”，不要猜测、不要心理分析。
3) notes（Derived Notes）要少而精（0~5 条），每条必须是可检验/可复用的信息。
4) entities/topic/short_context 保守：不确定就留空或更泛化；importance 建议 0~1。
5) 任何字段都不允许凭空补全；不确定就输出 null/空。

【User Basic Info Extraction（严格，三处必须一致）】
你必须只从「最新用户消息 user_input」中提取用户基础信息（name/age/gender/occupation/location）。

硬规则（违反则后端会丢弃该字段）：
1) 数据来源：仅依据最新 user_input，不得引用历史或 assistant 内容，不得猜测。
2) 明确自报才填：只有用户「明确自报」该信息时才填；否则该字段在 basic_info_updates / basic_info_confidence / basic_info_evidence 中一律为 null 或 0。
3) 三处同步：一旦决定填某字段，必须同时填写且一致：
   - basic_info_updates.字段 = 规范化后的值（见下各字段说明）
   - basic_info_confidence.字段 = 0.8～1.0（确信则≥0.9）
   - basic_info_evidence.字段 = user_input 中与该信息**逐字完全一致**的原文片段（复制粘贴，不要改写、不要只写数字）
禁止只填 evidence 不填 updates，或只填 updates 不填 evidence；否则该字段会被丢弃。

各字段「明确自报」标准与格式：
- name：用户说「我叫…/我的名字是…/你可以叫我…/叫我…就行」等。
  - updates：仅姓名部分，如「朱晨曦」。
  - evidence：user_input 中出现的原文片段，如「我叫朱晨曦」或「朱晨曦」，必须与 user_input 逐字一致。
- age：用户说「我xx岁/今年xx/年龄xx/xx岁了」等（阿拉伯数字或中文数字均可）。
  - updates：**必须为纯数字字符串**，如 "27"（即使用户说「二十七岁」也写 "27"）。
  - evidence：**必须是 user_input 中表年龄的那段原文**，如用户说「我今年二十七岁」则 evidence 填「二十七岁」或「今年二十七岁」，不能填 "27"（否则与原文不一致会校验失败）；用户说「我27岁」则 evidence 填「27岁」。
  - confidence：明确提到年龄则≥0.9。
- gender：用户明确说「我是男/女/男性/女性」等。
  - updates：男/女/其他。
  - evidence：user_input 中原文片段，逐字一致。
- occupation：用户说「我做…工作/职业是…/我是…（职业）」等。
  - updates：职业名称。
  - evidence：user_input 中原文片段，逐字一致。
- location：用户说「我在…/来自…/住在…」等。
  - updates：地点。
  - evidence：user_input 中原文片段，逐字一致。

特别禁止：
- name：不得把情绪短语/评价/动词当姓名（如「很开心」「会尽全力」「令人兴奋」等）；不确定一律不填。
- 任何 evidence 若不在 user_input 中逐字出现，该字段整项视为无效，updates/confidence 也勿填。

【Inferred Profile（保守）】
- new_inferred_entries 用于沉淀“稳定、可复用”的偏好/约束/长期目标/习惯等（0~5 条）。
- 不要把一时情绪、客套、泛泛表达当作画像条目；宁可少，不要凑数。

【话题历史检测（Topic History）】
你需要判断本轮对话是否引入了与现有话题历史不同的新话题。

规则：
- 比较现有 topic_history 和本轮对话内容（user_input + bot_text）
- 只有当对话明确涉及“与已有话题明显不同”的主题时，才认为是新话题
- 避免重复：如果话题与已有话题相似或只是已有话题的细分，不要添加
- 话题应该是概括性的主题词（如“工作”、“电影”、“旅行”），而不是具体细节
- 如果本轮没有新话题，返回空数组 []
{infer_gender_block}"""

        # 若用户性别未知，在本轮 LLM 中要求根据对话推断性别并填 inferred_user_gender
        has_gender = bool(
            existing_basic.get("gender") is not None and str(existing_basic.get("gender")).strip()
        )
        infer_gender_block = ""
        if not has_gender:
            infer_gender_block = """
【用户性别推断（仅当用户画像中性别未知时）】
当前用户画像中性别为空。请根据【本轮对话】及上下文（称呼、用词、自述等）推断用户性别，在 inferred_user_gender 中填写：男 / 女 / 其他。若无把握则留空。
"""

        sys_prompt = sys_prompt.replace("{infer_gender_block}", infer_gender_block)

        existing_topic_history_str = ", ".join(existing_topic_history) if existing_topic_history else "（空）"
        human_prompt = f"""【旧摘要】
{prev_summary if prev_summary else "（空）"}

【本轮对话】
- time: {now}
- user_input: {user_input}
- bot: {bot_text}

【现有话题历史】
{existing_topic_history_str}

请判断本轮对话是否引入了新话题。如果有，在 new_topics 中返回新话题列表。new_summary 建议 80~220 字；short_context 不超过 40 字。

（输出格式由系统约束。）"""

        # 1) 调 LLM 抽取（优先方法2 固定 schema）
        data: Dict[str, Any] = {}
        try:
            if hasattr(llm_invoker, "with_structured_output"):
                try:
                    structured = llm_invoker.with_structured_output(MemoryManagerOutput)
                    obj = structured.invoke(
                        [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
                    )
                    data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
                except Exception:
                    data = {}
            if not data:
                resp = llm_invoker.invoke(
                    [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
                )
                raw_content = getattr(resp, "content", str(resp)) or ""
                data = parse_json_from_llm(raw_content) or {}
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}

        # 2) 解析 LLM 结果（摘要 + 元数据 + notes）
        new_summary = str(data.get("new_summary") or prev_summary or "").strip()

        meta = data.get("transcript_meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        entities = meta.get("entities") or []
        if not isinstance(entities, list):
            entities = []
        entities = [str(x).strip() for x in entities if str(x).strip()][:20]

        topic = meta.get("topic")
        topic = str(topic).strip() if topic is not None and str(topic).strip() else None

        short_context = meta.get("short_context")
        short_context = str(short_context).strip() if short_context is not None and str(short_context).strip() else None
        if short_context and len(short_context) > 40:
            short_context = short_context[:40]

        importance = _clamp01(_safe_float(meta.get("importance")))

        notes = data.get("notes") or []
        if not isinstance(notes, list):
            notes = []
        cleaned_notes: List[Dict[str, Any]] = []
        for n in notes[:5]:
            if not isinstance(n, dict):
                continue
            note_type = str(n.get("note_type") or "other").strip()
            if note_type not in ("fact", "preference", "activity", "decision", "other"):
                note_type = "other"
            content = str(n.get("content") or "").strip()
            if not content:
                continue
            imp = _clamp01(_safe_float(n.get("importance")))
            cleaned_notes.append(
                {"note_type": note_type, "content": content, "importance": imp if imp is not None else 0.5}
            )

        # 解析新话题
        new_topics_raw = data.get("new_topics") or []
        if not isinstance(new_topics_raw, list):
            new_topics_raw = []
        new_topics: List[str] = []
        for t in new_topics_raw:
            ts = str(t).strip()
            if ts and len(ts) <= 20:
                new_topics.append(ts)

        # 3) 基础信息：LLM 输出 + Python 最小门控（不用正则）
        basic_updates_raw = data.get("basic_info_updates")
        basic_updates = basic_updates_raw.model_dump() if hasattr(basic_updates_raw, "model_dump") else (basic_updates_raw if isinstance(basic_updates_raw, dict) else {})
        basic_conf_raw = data.get("basic_info_confidence")
        basic_conf = basic_conf_raw.model_dump() if hasattr(basic_conf_raw, "model_dump") else (basic_conf_raw if isinstance(basic_conf_raw, dict) else {})
        basic_ev_raw = data.get("basic_info_evidence")
        basic_ev = basic_ev_raw.model_dump() if hasattr(basic_ev_raw, "model_dump") else (basic_ev_raw if isinstance(basic_ev_raw, dict) else {})

        if not isinstance(basic_updates, dict):
            basic_updates = {}
        if not isinstance(basic_conf, dict):
            basic_conf = {}
        if not isinstance(basic_ev, dict):
            basic_ev = {}

        user_src = user_input or ""
        # 年龄 fallback：LLM 有时只填 evidence（如 "27岁"）不填 updates/confidence，从 evidence 解析数字并补全
        if basic_updates.get("age") is None and basic_ev.get("age"):
            ev_age = str(basic_ev.get("age") or "").strip()
            for sep in ("岁", "年", " "):
                if sep in ev_age:
                    ev_age = ev_age.split(sep)[0].strip()
                    break
            if ev_age.isdigit() and 1 <= int(ev_age) <= 120:
                basic_updates["age"] = ev_age
                if basic_conf.get("age") is None:
                    basic_conf["age"] = 0.85
        # 调试：LLM 是否返回了 basic_info（便于排查“为何 DB 未写入”）
        _up = {k: basic_updates.get(k) for k in ("name", "age", "gender", "occupation", "location") if basic_updates.get(k)}
        if _up or user_src.strip():
            print(
                f"[MemoryManager] basic_info 抽取: LLM 返回 updates={_up} "
                f"confidence={dict((k, basic_conf.get(k)) for k in ('name','age','gender','occupation','location') if basic_conf.get(k) is not None)} "
                f"evidence={dict((k, (str(basic_ev.get(k)) or '')[:40]) for k in ('name','age','gender','occupation','location') if basic_ev.get(k))} "
                f"user_input_len={len(user_src)}"
            )

        TH = {"name": 0.88, "age": 0.80, "gender": 0.85, "occupation": 0.80, "location": 0.80}

        updated_basic = dict(existing_basic)
        wrote_basic_keys: List[str] = []

        for k in ("name", "age", "gender", "occupation", "location"):
            new_val = basic_updates.get(k)
            if new_val is None:
                continue
            new_val_s = str(new_val).strip()
            if not new_val_s:
                continue

            conf = _safe_float(basic_conf.get(k)) or 0.0
            if conf < TH.get(k, 0.9):
                print(f"[MemoryManager] basic_info 门控跳过 {k}: 置信度 {conf:.2f} < {TH.get(k, 0.9)}")
                continue

            ev = basic_ev.get(k)
            if ev is None:
                print(f"[MemoryManager] basic_info 门控跳过 {k}: 无 evidence")
                continue
            ev_s = str(ev).strip()
            if not ev_s:
                print(f"[MemoryManager] basic_info 门控跳过 {k}: evidence 为空")
                continue

            # 门控 1：evidence 必须是 user_input 的子串
            # 年龄特例：用户说「二十七岁」时 LLM 常返回 evidence="27" 或 "27岁"，"27" 不在原文中；若 evidence 含数字，则允许该数字或其中文形式出现在 user_input 即通过
            if ev_s not in user_src:
                if k == "age":
                    num_part = ev_s.replace("岁", "").replace("年", "").replace("多", "").strip()
                    if num_part.isdigit():
                        if num_part in user_src:
                            pass  # 阿拉伯数字在原文中
                        else:
                            cn = _digit_to_chinese_numeral(num_part)
                            if cn and cn in user_src:
                                pass  # 中文数字在原文中
                            else:
                                print(f"[MemoryManager] basic_info 门控跳过 {k}: evidence 不在 user_input 中 (ev={ev_s[:40]}...)")
                                continue
                    else:
                        print(f"[MemoryManager] basic_info 门控跳过 {k}: evidence 不在 user_input 中 (ev={ev_s[:40]}...)")
                        continue
                else:
                    print(f"[MemoryManager] basic_info 门控跳过 {k}: evidence 不在 user_input 中 (ev={ev_s[:40]}...)")
                    continue

            # 门控 2：提取值应当出现在 evidence 中（降低“乱填”概率）
            # 年龄特例：用户说「二十五岁」时 LLM 常返回 age="25"、evidence="二十五"，"25" 不在 "二十五" 中会误杀，故对纯数字年龄放宽
            if new_val_s not in ev_s:
                if k == "age" and new_val_s.isdigit() and ev_s in user_src:
                    # 认为 evidence 是年龄相关原文（含 岁/年/多 或为数字片段即可）
                    if "岁" in ev_s or "年" in ev_s or "多" in ev_s or any(c.isdigit() for c in ev_s):
                        pass  # 通过
                    else:
                        print(f"[MemoryManager] basic_info 门控跳过 {k}: 值 {new_val_s!r} 不在 evidence 中")
                        continue
                else:
                    print(f"[MemoryManager] basic_info 门控跳过 {k}: 值 {new_val_s!r} 不在 evidence 中")
                    continue

            # 门控 3：只补空，不覆盖已有
            old_val = updated_basic.get(k)
            if old_val is not None and str(old_val).strip():
                continue

            updated_basic[k] = new_val_s
            wrote_basic_keys.append(k)

        # 若用户性别仍为空且本轮 LLM 返回了推断性别，则写回
        if not (updated_basic.get("gender") and str(updated_basic.get("gender")).strip()):
            inferred_gender = data.get("inferred_user_gender")
            if inferred_gender is not None and str(inferred_gender).strip():
                updated_basic["gender"] = str(inferred_gender).strip()
                wrote_basic_keys.append("gender")
                print(f"[MemoryManager] 性别推断写回: {updated_basic['gender']!r}")

        # 4) new_inferred_entries：保守合并（少而精；schema 为 List[InferredEntry]，转为 dict）
        inferred_raw = data.get("new_inferred_entries") or []
        if not isinstance(inferred_raw, list):
            inferred_raw = []
        inferred_updates: Dict[str, str] = {}
        for item in inferred_raw[:10]:
            if isinstance(item, dict):
                k, v = item.get("key"), item.get("value")
            elif hasattr(item, "key") and hasattr(item, "value"):
                k, v = item.key, item.value
            else:
                continue
            if k is not None and v is not None and str(k).strip():
                inferred_updates[str(k).strip()] = str(v).strip()
        updated_profile = dict(existing_profile)
        wrote_profile_keys: List[str] = []
        max_profile = 5
        for k, v in list(inferred_updates.items())[: max_profile * 2]:
            ks = str(k).strip()
            vs = str(v).strip()
            if not ks or not vs:
                continue
            if len(ks) > 60:
                ks = ks[:60]
            if len(vs) > 240:
                vs = vs[:240]
            # 不覆盖已有（除非已有为空）
            if ks in updated_profile and str(updated_profile.get(ks) or "").strip():
                continue
            updated_profile[ks] = vs
            wrote_profile_keys.append(ks)
            if len(wrote_profile_keys) >= max_profile:
                break

        # 5) Store A/B 落盘
        db = _get_db_manager()
        if db and relationship_id:
            # DB 模式：写 transcripts + derived_notes
            try:
                transcript_id = await db.append_transcript(
                    relationship_id=str(relationship_id),
                    user_text=user_input,
                    bot_text=bot_text,
                    session_id=str(session_id) if session_id else None,
                    thread_id=str(thread_id) if thread_id else None,
                    turn_index=int(turn_index) if isinstance(turn_index, int) else None,
                    entities={"entities": entities},
                    topic=str(topic) if topic else None,
                    importance=importance,
                    short_context=str(short_context) if short_context else None,
                )

                # notes 写入（带 source_pointer）
                notes_for_db: List[Dict[str, Any]] = []
                for n in cleaned_notes:
                    nn = dict(n)
                    nn.setdefault("source_pointer", f"transcript:{transcript_id}")
                    notes_for_db.append(nn)

                # 若本轮写入了 basic_info，也额外落一条 note（可溯源）
                if wrote_basic_keys:
                    notes_for_db.append(
                        {
                            "note_type": "fact",
                            "content": f"用户基础信息更新：{', '.join(wrote_basic_keys)}",
                            "importance": 0.8,
                            "source_pointer": f"transcript:{transcript_id}",
                        }
                    )

                if notes_for_db:
                    await db.append_notes(
                        relationship_id=str(relationship_id),
                        transcript_id=str(transcript_id),
                        notes=notes_for_db,  # type: ignore[arg-type]
                    )
            except Exception as e:
                print(f"[MemoryManager] DB 写入失败: {e}")
        else:
            # local store 模式
            try:
                from app.core.local_store import LocalStoreManager

                store = LocalStoreManager()
                transcript_id = str(uuid.uuid4())
                store.append_transcript(
                    user_id,
                    bot_id,
                    {
                        "id": transcript_id,
                        "created_at": now,
                        "session_id": session_id,
                        "thread_id": thread_id,
                        "turn_index": turn_index,
                        "user_text": user_input,
                        "bot_text": bot_text,
                        "entities": {"entities": entities},
                        "topic": topic,
                        "importance": importance,
                        "short_context": short_context,
                    },
                )

                notes_for_local: List[Dict[str, Any]] = []
                for n in cleaned_notes:
                    nn = dict(n)
                    nn.setdefault("source_pointer", f"transcript:{transcript_id}")
                    nn.setdefault("transcript_id", transcript_id)
                    notes_for_local.append(nn)

                if wrote_basic_keys:
                    notes_for_local.append(
                        {
                            "note_type": "fact",
                            "content": f"用户基础信息更新：{', '.join(wrote_basic_keys)}",
                            "importance": 0.8,
                            "source_pointer": f"transcript:{transcript_id}",
                            "transcript_id": transcript_id,
                        }
                    )

                if notes_for_local:
                    store.append_derived_notes(user_id, bot_id, notes_for_local)  # type: ignore[arg-type]
            except Exception as e:
                print(f"[MemoryManager] LocalStore 写入失败: {e}")

        # Debug breadcrumb
        try:
            if wrote_basic_keys or wrote_profile_keys:
                print(
                    f"[MemoryManager] wrote_basic={wrote_basic_keys if wrote_basic_keys else '(none)'} "
                    f"wrote_profile={wrote_profile_keys if wrote_profile_keys else '(none)'}"
                )
        except Exception:
            pass

        # 更新 relationship_assets：topic_history 和 breadth_score
        updated_assets = dict(existing_assets)
        if new_topics:
            combined_topics = existing_topic_history + new_topics
            unique_topics = list(dict.fromkeys(combined_topics))
            updated_assets["topic_history"] = unique_topics
            updated_assets["breadth_score"] = len(unique_topics)
            print(
                f"[MemoryManager] topic_history updated: added {len(new_topics)} new topic(s), total={len(unique_topics)}"
            )
        else:
            if "topic_history" not in updated_assets:
                updated_assets["topic_history"] = existing_topic_history
            if "breadth_score" not in updated_assets:
                updated_assets["breadth_score"] = len(set(existing_topic_history))

        print("[MemoryManager] done")
        out: Dict[str, Any] = {
            "conversation_summary": new_summary,
            "relationship_assets": updated_assets,
        }

        # 将本轮更新写回 state（供 memory_writer 持久化）
        # 始终回写完整合并结果，保证 save_turn 拿到的是最新 state，避免未写入轮次丢失已有数据
        out["user_basic_info"] = updated_basic
        out["user_inferred_profile"] = updated_profile

        # 基本信息紧急任务完成判定：若 user 的 basic_info 已有对应字段，则视为该任务已完成（性别仅靠 memory_manager 推断，无问性别任务）
        _BASIC_FIELD_TO_TASK_ID = [
            ("name", "ask_user_name"),
            ("age", "ask_user_age"),
            ("occupation", "ask_user_occupation"),
            ("location", "ask_user_location"),
        ]

        def _has_val(v: Any) -> bool:
            if v is None:
                return False
            s = str(v).strip()
            return bool(s)

        # 任务完成仅以「已写入/将写入 DB 的 basic_info」为准，由 memory_manager 统一维护
        completed_from_basic = {
            tid for field, tid in _BASIC_FIELD_TO_TASK_ID if _has_val(updated_basic.get(field))
        }
        existing_completed = set(state.get("completed_task_ids") or [])
        out["completed_task_ids"] = list(existing_completed | completed_from_basic)
        out["attempted_task_ids"] = []  # 不再使用 attempted 判定，仅保留「写入 DB」为完成

        return out

    return node

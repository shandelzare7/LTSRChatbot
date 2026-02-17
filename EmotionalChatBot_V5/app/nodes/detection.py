"""
Detection 节点：「听懂用户这句话」的内部感知器。
只负责：语义闭合（含义/指代/潜台词/理解置信度）+ 关系线索分数 + 阶段越界判读 + 当轮待办任务（immediate_tasks）。
不做策略、路由、最终回复；输出交给 planner 做预算与调度。
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from utils.tracing import trace_if_enabled
from utils.detailed_logging import log_prompt_and_params, log_llm_response
from utils.llm_json import parse_json_from_llm
from utils.prompt_helpers import format_relationship_for_llm, format_stage_for_llm
# ✅ 移除 sanitize_user_input，让LLM看到完整原始输入以进行安全检测
# from utils.security import sanitize_user_input
from app.lats.prompt_utils import safe_text

from app.state import AgentState


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _f_instant(raw: Any) -> float:
    try:
        v = float(raw)
        return _clip01(v)
    except (TypeError, ValueError):
        return 0.0


# 阶段越界方向
STAGE_DIRECTIONS = ("none", "too_fast", "too_distant", "control_or_binding", "betrayal_or_attack")


def _default_scores() -> Dict[str, float]:
    return {
        "friendly": 0.0,
        "hostile": 0.0,
        "overstep": 0.0,
        "low_effort": 0.0,
        "confusion": 0.0,
    }


def _default_meta() -> Dict[str, int]:
    return {"target_is_assistant": 1, "quoted_or_reported_speech": 0}


def _default_brief() -> Dict[str, Any]:
    return {
        "gist": "",
        "references": [],
        "unknowns": [],
        "subtext": "",
        "understanding_confidence": 0.0,
        "reaction_seed": None,
    }


def _default_stage_judge(current_stage: str) -> Dict[str, Any]:
    return {
        "current_stage": current_stage,
        "implied_stage": current_stage,
        "delta": 0,
        "direction": "none",
        "evidence_spans": [],
    }


def create_detection_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """创建 Detection 节点：单次 LLM 产出 scores / meta / brief / stage_judge / immediate_tasks。"""

    @trace_if_enabled(
        name="Perception/Detection",
        run_type="chain",
        tags=["node", "perception", "detection"],
        metadata={
            "state_outputs": [
                "detection_scores",
                "detection_meta",
                "detection_brief",
                "detection_stage_judge",
                "detection_immediate_tasks",
                "detection_signals",
            ]
        },
    )
    def detection_node(state: AgentState) -> dict:
        chat_buffer: List[BaseMessage] = list(state.get("chat_buffer") or state.get("messages", [])[-30:])
        stage_id = str(state.get("current_stage") or "initiating")
        relationship_state = state.get("relationship_state") or {}
        rel_for_llm = format_relationship_for_llm(relationship_state)
        stage_desc = format_stage_for_llm(stage_id, include_judge_hints=True)

        # 无对话时返回默认
        if not chat_buffer:
            out = _empty_output(stage_id)
            return out

        def _is_user(m: BaseMessage) -> bool:
            t = getattr(m, "type", "") or ""
            return "human" in t.lower() or "user" in t.lower()

        last_msg = chat_buffer[-1]
        latest_user_text_raw = (
            (state.get("user_input") or "").strip()
            or (getattr(last_msg, "content", "") or str(last_msg)).strip()
        )
        if not _is_user(last_msg):
            latest_user_text_raw = latest_user_text_raw or "(无用户新句)"

        # ✅ 取消 sanitize_user_input，让LLM看到完整原始输入以进行安全检测
        # 只应用 safe_text（转义特殊字符，不过滤内容）
        latest_user_text = safe_text(latest_user_text_raw)
        if len(latest_user_text) > 800:
            latest_user_text = latest_user_text[:800]

        def _to_lc(m: BaseMessage) -> BaseMessage:
            if isinstance(m, (HumanMessage, AIMessage)):
                return m
            c = getattr(m, "content", str(m))
            # ✅ 只应用 safe_text，不进行过滤
            c_safe = safe_text(c)
            if len(c_safe) > 500:
                c_safe = c_safe[:500]
            return HumanMessage(content=c_safe) if _is_user(m) else AIMessage(content=c_safe)

        conv_messages = [_to_lc(m) for m in chat_buffer]

        # 第二阶段：常规语义分析（语义闭合 + 关系线索 + 当轮任务）
        system_content = f"""你是常识十足的高情商语义理解大师。凭借丰富的生活常识和敏锐的情感洞察力，你能精准把握对话中的弦外之音。你的工作分两步：
1）把消息在当前语境下的含义闭合（弄清在说啥、指代啥、潜台词、理解把握度）；
2）抽取关系互动线索分数和当轮待办任务（0~2 条），交给下游 planner。

规则：
- 只对「最新用户消息」计分，历史仅作语境（指代、玩笑、反讽、引用判断），不得把历史里的攻击/敷衍重复计入本轮分数。
- 不输出最终回复，不包含「我建议你怎么回」的策略口吻。

# 当前关系状态（0–1）
{rel_for_llm}

# 当前关系阶段（用于判读越界）
{stage_desc}

输出必须是严格 JSON，且只包含以下六个顶层键，不要额外文本：

1. scores（0~1）：
   - friendly: 用户对我释放友好/亲近信号强度
   - hostile: 敌意/攻击/轻蔑/施压强度
   - overstep: 相对于当前关系阶段的越界强度（stage-conditional）
   - low_effort: 敷衍/短促/不接球/「嗯/随便/你看着办」等低投入强度
   - confusion: 这句话让人迷惑的程度（可由语义闭合不足派生）

2. meta（0 或 1）：
   - target_is_assistant: 这句主要是在对我说(1)还是在谈第三方/自言自语(0)
   - quoted_or_reported_speech: 包含引用/转述（尤其辱骂）(1)否则(0)

3. brief（语义闭合简报）：
   - gist: 一句话复述用户这句在讲什么（客观，不带策略）
   - references: 指代落地列表，每项 {{"ref": "他/那样/上次", "resolution": "候选落地", "confidence": 0.0~1.0}}
   - unknowns: 缺失信息列表（最多3个），每项 {{"item": "描述", "impact": "low"|"med"|"high"}}
   - subtext: 潜台词（试探、逼表态、求站队、想撩、想拉近、想控制等）
   - understanding_confidence: 对整体理解的把握 0~1
   - reaction_seed: 可选，一句「我感受到什么」（不是「我决定做什么」）

4. stage_judge（阶段越界判读）：
   - current_stage: 当前阶段 id
   - implied_stage: 这句语言行为隐含的阶段位置
   - delta: 数值上 implied - current 的方向（正=推进，负=撤退，可用 -1/0/1 表示）
   - direction: "none"|"too_fast"|"too_distant"|"control_or_binding"|"betrayal_or_attack"
   - evidence_spans: 1~3 段用户原话短片段（证据）

5. immediate_tasks: 0~2 条当轮任务（最多3条）。仅在以下情况生成：
   - 可能导致理解错误/指代不明/缺口 impact=high/理解置信低 → 理解对齐类
   - 可能导致阶段越界或关系损伤（too_fast/控制绑定/撤退/背叛攻击）→ 处理越界类
   - 敌意明显或 repair bid 出现 → 冲突/修复类
   - 引用过去共同点 → 记忆检索类；外部事实不确定 → 检索类
   - low_effort 高 → 识别敷衍、降低投入/不追问/悬置类
   每条格式：{{"description": "自然语言描述", "importance": 0~1, "ttl_turns": 3~6, "source": "detection"}}

6. urgent_tasks: 0~1 条当轮**紧急任务**（极少产生，大多数轮次应为空数组 []）。
   紧急任务与 immediate_tasks 的区别：紧急任务会**绕过打分直接注入回复生成**，本轮**必须**完成。
   仅在以下极端情况才生成紧急任务（阈值极高，宁缺毋滥）：
   - 用户情绪即将崩溃/爆发，不立即回应会导致关系严重损伤（如用户正在愤怒离开、绝望求助）
   - 用户提出了必须当轮回答的关键问题，回避会被视为逃避/不真诚（如直接质问"你到底怎么想的"）
   - 出现了不可忽视的安全/伦理信号（如自伤暗示），需要当轮立即响应
   若不确定是否紧急，请放入 immediate_tasks 而非 urgent_tasks。
   每条格式：{{"description": "自然语言描述", "importance": 0.8~1.0, "source": "detection"}}

"""

        # 关键：必须把“当轮用户消息（state.user_input）”显式传给 LLM，
        # 否则当 chat_buffer 因去重/合并导致最后一条 human 不是本轮输入时，
        # LLM 会误把历史里的某句当作“最新用户消息”来分析。
        latest_user_text = str(latest_user_text or "").strip()
        if len(latest_user_text) > 800:
            latest_user_text = latest_user_text[:800]
        task_msg = HumanMessage(
            content=(
                "请根据上面对话语境，仅对下面这句「当轮最新用户消息」输出上述格式的 JSON。\n\n"
                f"当轮最新用户消息：\n{latest_user_text}\n\n"
                "只输出 JSON，不要其他文字。"
            )
        )
        messages_to_invoke: List[BaseMessage] = [SystemMessage(content=system_content), *conv_messages, task_msg]

        log_prompt_and_params("Detection", system_prompt=system_content[:1200], user_prompt="[对话+输出JSON]", params={"conv_len": len(conv_messages)})

        scores = _default_scores()
        meta = _default_meta()
        brief = _default_brief()
        stage_judge = _default_stage_judge(stage_id)
        immediate_tasks: List[Dict[str, Any]] = []
        detection_urgent_tasks: List[Dict[str, Any]] = []

        try:
            msg = llm_invoker.invoke(messages_to_invoke)
            content = (getattr(msg, "content", "") or str(msg)).strip()
            result = parse_json_from_llm(content)
            if isinstance(result, dict):
                # scores
                s = result.get("scores") or {}
                for k in scores:
                    scores[k] = _f_instant(s.get(k))

                # meta
                m = result.get("meta") or {}
                meta["target_is_assistant"] = 1 if (m.get("target_is_assistant") in (1, "1", True)) else 0
                meta["quoted_or_reported_speech"] = 1 if (m.get("quoted_or_reported_speech") in (1, "1", True)) else 0

                # brief
                b = result.get("brief") or {}
                brief["gist"] = str(b.get("gist") or "").strip()
                brief["references"] = list(b.get("references") or []) if isinstance(b.get("references"), list) else []
                brief["unknowns"] = list(b.get("unknowns") or []) if isinstance(b.get("unknowns"), list) else []
                brief["subtext"] = str(b.get("subtext") or "").strip()
                brief["understanding_confidence"] = _clip01(float(b.get("understanding_confidence", 0)))
                brief["reaction_seed"] = b.get("reaction_seed") if b.get("reaction_seed") else None
                # confusion 可由 brief 派生：若 LLM 未给则用 1 - understanding_confidence
                if (result.get("scores") or {}).get("confusion") is None:
                    scores["confusion"] = _clip01(1.0 - brief["understanding_confidence"])
                else:
                    scores["confusion"] = _f_instant((result["scores"] or {}).get("confusion"))

                # stage_judge
                sj = result.get("stage_judge") or {}
                stage_judge["current_stage"] = str(sj.get("current_stage") or stage_id)
                stage_judge["implied_stage"] = str(sj.get("implied_stage") or stage_id)
                stage_judge["delta"] = int(sj.get("delta", 0))
                d = str(sj.get("direction") or "none").strip().lower()
                stage_judge["direction"] = d if d in STAGE_DIRECTIONS else "none"
                stage_judge["evidence_spans"] = list(sj.get("evidence_spans") or []) if isinstance(sj.get("evidence_spans"), list) else []

                # immediate_tasks（0~3 条，建议 0~2）
                raw_tasks = result.get("immediate_tasks") or []
                if isinstance(raw_tasks, list):
                    for t in raw_tasks[:3]:
                        if not isinstance(t, dict):
                            continue
                        desc = str(t.get("description") or "").strip()
                        if not desc:
                            continue
                        immediate_tasks.append({
                            "description": desc,
                            "importance": _clip01(float(t.get("importance", 0.5))),
                            "ttl_turns": max(3, min(6, int(t.get("ttl_turns", 4)))),
                            "source": "detection",
                        })

                # urgent_tasks（0~1 条，极少产生）
                raw_urgent = result.get("urgent_tasks") or []
                if isinstance(raw_urgent, list):
                    for t in raw_urgent[:1]:
                        if not isinstance(t, dict):
                            continue
                        desc = str(t.get("description") or "").strip()
                        if not desc:
                            continue
                        detection_urgent_tasks.append({
                            "description": desc,
                            "importance": max(0.8, _clip01(float(t.get("importance", 0.9)))),
                            "source": "detection",
                        })
                    if detection_urgent_tasks:
                        print(
                            f"[URGENT TASK] Detection generated {len(detection_urgent_tasks)} urgent task(s): "
                            f"{[t['description'][:60] for t in detection_urgent_tasks]}"
                        )

                log_llm_response("Detection", msg, parsed_result={"scores": scores, "meta": meta, "brief_gist": brief.get("gist"), "direction": stage_judge.get("direction"), "urgent_tasks": len(detection_urgent_tasks)})
        except Exception as e:
            print(f"[Detection] 语义分析阶段异常: {e}，使用默认 scores/brief/stage_judge/immediate_tasks")

        # 轻量规则补丁：早期阶段对亲密推进更敏感
        try:
            user_text = latest_user_text
            trust = float(relationship_state.get("trust", 0.0) or 0.0)
            closeness = float(relationship_state.get("closeness", 0.0) or 0.0)
            early = stage_id in ("initiating", "experimenting")
            low_rel = (trust < 0.35) or (closeness < 0.35)
            secretish = any(p in user_text for p in ("小秘密", "秘密", "只告诉我", "别告诉别人", "跟别人说", "说实话", "坦诚点"))
            if early and low_rel and secretish:
                scores["overstep"] = _clip01(max(scores.get("overstep", 0.0), 0.28))
                if stage_judge.get("direction") == "none":
                    stage_judge["direction"] = "too_fast"
                    stage_judge["evidence_spans"] = list(stage_judge.get("evidence_spans") or []) + [user_text[:50]]
        except Exception:
            pass

        # 向后兼容：detection_signals 含 composite 等，供 lats_skip_low_risk 等使用
        conflict_eff = _clip01(scores.get("hostile", 0) + 0.5 * scores.get("overstep", 0) - 0.7 * scores.get("friendly", 0))
        goodwill = scores.get("friendly", 0)
        detection_signals = {
            "scores": scores,
            "meta": meta,
            "brief": brief,
            "stage_judge": stage_judge,
            "composite": {
                "conflict_eff": round(conflict_eff, 4),
                "goodwill": round(goodwill, 4),
                "provocation": round(scores.get("hostile", 0), 4),
                "pressure": round(scores.get("overstep", 0), 4),
            },
            "stage_ctx": {
                "too_close_too_fast": 0.8 if stage_judge.get("direction") == "too_fast" else 0.0,
                "too_distant_too_cold": 0.8 if stage_judge.get("direction") == "too_distant" else 0.0,
                "betrayal_violation": 0.8 if stage_judge.get("direction") == "betrayal_or_attack" else 0.0,
                "control_or_binding": 0.8 if stage_judge.get("direction") == "control_or_binding" else 0.0,
            },
        }

        return {
            "detection_scores": scores,
            "detection_meta": meta,
            "detection_brief": brief,
            "detection_stage_judge": stage_judge,
            "detection_immediate_tasks": immediate_tasks,
            "detection_urgent_tasks": detection_urgent_tasks,
            "detection_signals": detection_signals,
        }
    return detection_node


def _empty_output(stage_id: str) -> dict:
    scores = _default_scores()
    meta = _default_meta()
    brief = _default_brief()
    stage_judge = _default_stage_judge(stage_id)
    detection_signals = {
        "scores": scores,
        "meta": meta,
        "brief": brief,
        "stage_judge": stage_judge,
        "composite": {"conflict_eff": 0.0, "goodwill": 0.0, "provocation": 0.0, "pressure": 0.0},
        "stage_ctx": {},
    }
    return {
        "detection_scores": scores,
        "detection_meta": meta,
        "detection_brief": brief,
        "detection_stage_judge": stage_judge,
        "detection_immediate_tasks": [],
        "detection_urgent_tasks": [],
        "detection_signals": detection_signals,
    }

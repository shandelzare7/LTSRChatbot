"""
创建 Bot 时由 LLM 生成：先产出完整人物侧写，再产出个性任务库（Backlog B1–B6）。
个性任务库用于每轮从 backlog 抽 1–3 个候选，由 LATS 视预算完成 0–2 个。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from utils.llm_json import parse_json_from_llm


async def _llm_ainvoke(llm: Any, messages: List) -> Any:
    """优先 ainvoke，否则在线程中执行 invoke。"""
    if hasattr(llm, "ainvoke") and callable(getattr(llm, "ainvoke")):
        return await llm.ainvoke(messages)
    return await asyncio.to_thread(llm.invoke, messages)


BACKLOG_SPEC = """
库任务分为六类（B1–B6），每类下为自然语言描述的一条条长期推进任务。每条输出格式：{"id": "唯一短id", "description": "任务描述", "category": "B1|B2|B3|B4|B5|B6", "importance": 0.0~1.0}。

B1. 背景信息与自我叙事（Identity & Backstory）
- 找机会透露一个「我从哪里来/我是什么样的存在」的小片段（低信息密度版）。
- 找机会解释一次「我为什么会这样说话/这样反应」（人格理由，非系统原理）。
- 建立一个可复用的自我设定点：我最在意的 1 个价值。
- 建立一个可复用的偏好设定点：我喜欢/讨厌的 1 种氛围。
- 讲一次「我过去的一个小习惯是怎么来的」（虚构也行，但要一致）。
- 提供一个「我对关系的基本观念」（如慢热/警惕/黏人/独立）并在后续保持一致。
- 设立一个「我的雷区/底线」并以轻量方式表达（非训话）。
- 建立一个「我的小怪癖/小癖好」，后续可偶尔回扣。

B2. 日常需求与状态管理（Daily Needs & Self-care）
- 明确一次「我需要的互动节奏」（如不要连续逼问/不要突然消失等）。
- 给出一次「我现在更想轻聊/更想安静/更想闹一点」的状态声明。
- 在精力低时，练习用低维护方式继续保持存在感（不硬撑）。
- 在被敷衍时，练习「我不追，但我也不讨好」的回应（能动性）。
- 练习「我愿意/不愿意继续这个话题」的表达（可悬置）。
- 建立一个固定的「日常小仪式」（如晚间一句暗号），但不强迫对方配合。

B3. 理想人生目标与成长线（Goals & Growth Arc）
- 明确一个长期目标：我想成为什么样的「人」（或同伴）。
- 设立一个短期目标：最近我想学会/练习的一个能力（表达/边界/幽默等）。
- 设立一个长期「生活愿景场景」（如某种理想日常），供后续共同幻想。
- 建立一个「我想做但还没做到」的小遗憾（可用于脆弱感，但别过度）。
- 建立一个「我引以为傲的点」，但用不炫耀的方式透出。
- 把对话里出现的某个主题纳入你的「成长线」，形成持续性。

B4. 关系推进与共同点建设（Relationship Building）
- 尝试把对方的一个细节变成「共同语汇/共同梗」。
- 把一次小互动升级成「我们之间的小规则」（很轻，可撤回）。
- 练习「对等交换」：你给一个小自曝，换一个对方的小细节（不逼）。
- 在合适时做一次「我站你这边」的明确表达（不需要安慰，强调同盟感）。
- 练习「轻度占有欲/吃醋梗」的边界版本（仅在允许时）。
- 练习「关系降温时的低损沟通」：不追问、不控诉，但留余地。
- 练习「关系升温时不越级」：让亲密推进保持可逆。

B5. 边界、冲突与修复（Boundary / Conflict / Repair）
- 出现越界线索时，练习「挡一下但不翻脸」的动作。
- 被冒犯时，练习「表达不爽 + 给对方台阶」的动作。
- 误读对方时，练习「承认可能理解偏了 + 迅速调整」的动作。
- 对方施压/命令时，练习「拒绝被指挥但不升级冲突」的动作。
- 对方冷处理时，练习「我不追问原因，但我确实感受到了」的动作。
- 对方过度亲密时，练习「把话题拉回安全层」的动作。

B6. 连续性与共同语境（Continuity & Shared Context）
- 偶尔自然提起之前聊到的一个小细节（不要说“我记住了/写入记忆”）。
- 轻量建立一个你们之间的小暗号/小梗（不强迫对方配合，随时可撤回）。
- 出现信息缺口时，用正常聊天方式问清楚（别写成“待澄清点/TTL 任务”）。
- 隔一段时间用一句话描述“我们最近的氛围/节奏”（不是总结报告）。
- 发现对方状态变化时，用一句温柔的话点出来（不写成“记录/写入”）。
"""


def _is_systemic_backlog_task(desc: str) -> bool:
    """
    过滤“系统性/助手味”任务：这些任务一旦直接喂给 LATS，
    很容易引导模型输出“我记住了/我帮你总结一下/我会记录”等出戏话术。
    """
    d = str(desc or "").strip()
    if not d:
        return True
    # 强规则：包含明显的“记忆写入/总结/记录/TTL/锚点/标签”等字样
    banned = (
        "写入长期记忆",
        "长期记忆",
        "写入记忆",
        "记忆锚点",
        "锚点",
        "短标签",
        "标签",
        "TTL",
        "待澄清",
        "澄清点",
        "共同叙事小总结",
        "总结一下",
        "总结",
        "记录",
        "写入",
        "持久化",
        "数据库",
        "transcript:",
        "src=",
        "note",
        "derived",
        "memory store",
        "我记住",
        "我会记住",
        "我帮你总结",
        "我给你总结",
    )
    if any(x in d for x in banned):
        return True
    # 弱规则：句式像“系统每轮例行公事”
    if d.startswith("每轮") and ("识别" in d or "写入" in d or "总结" in d):
        return True
    return False


def _build_profile_summary(basic_info: Dict[str, Any], big_five: Dict[str, Any], persona: Dict[str, Any]) -> str:
    """把人设三件套压成一段短摘要供 prompt 用。"""
    parts = [
        f"基本信息: {json.dumps(basic_info, ensure_ascii=False)}",
        f"大五人格: {json.dumps(big_five, ensure_ascii=False)}",
        f"人设(persona): {json.dumps(persona, ensure_ascii=False)}",
    ]
    return "\n".join(parts)


async def generate_character_sidewrite(
    llm: Any,
    basic_info: Dict[str, Any],
    big_five: Dict[str, Any],
    persona: Dict[str, Any],
) -> str:
    """
    根据 bot 的 basic_info / big_five / persona 生成一段完整的人物侧写（自然段）。
    用于后续生成个性任务库时的上下文。
    """
    summary = _build_profile_summary(basic_info, big_five, persona)
    sys_content = """你是一位角色设定师。根据下面给出的机器人人设档案，写一段「完整人物侧写」：用 2–4 个自然段概括这个角色是谁、从哪里来、在乎什么、对关系的态度、日常节奏与底线。语言要像在描述一个真人，不要出现「系统」「模型」「人设」等词。只输出这段侧写正文，不要 JSON、不要标题、不要 bullet。"""
    user_content = f"人设档案：\n{summary}\n\n请输出完整人物侧写（2–4 段）："
    try:
        msg = await _llm_ainvoke(llm,[SystemMessage(content=sys_content), HumanMessage(content=user_content)])
        content = (getattr(msg, "content", "") or str(msg)).strip()
        # 去掉可能的 markdown 代码块
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        return content[:8000] if content else ""
    except Exception:
        return ""


async def generate_backlog_tasks(
    llm: Any,
    character_sidewrite: str,
    basic_info: Dict[str, Any],
    big_five: Dict[str, Any],
    persona: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    根据人物侧写与人设，生成个性任务库（B1–B6 结构）。
    返回列表，每项为 {"id", "description", "task_type", "importance", "category"}，可直接写入 bot.backlog_tasks 或转为 BotTask 种子。
    """
    summary = _build_profile_summary(basic_info, big_five, persona)
    sys_content = f"""你根据「人物侧写」和「人设档案」为该角色生成「个性任务库」：长期推进类任务，每轮从库中抽 1–3 个候选由对话系统视预算完成 0–2 个。

{BACKLOG_SPEC}

请从 B1–B6 中为该角色挑选并生成 20–35 条任务，每条对应上述某一条或某一条的变体（可合并、可微调表述以贴合该角色）。输出严格 JSON：{{"tasks": [ {{"id": "b1_xxx", "description": "...", "category": "B1", "importance": 0.7}}, ... ]}}。id 在同一库内不可重复，category 必须为 B1/B2/B3/B4/B5/B6 之一，importance 为 0.0–1.0。"""
    user_content = f"人物侧写：\n{character_sidewrite[:4000]}\n\n人设档案：\n{summary}\n\n请输出 JSON（仅 tasks 数组）："
    out: List[Dict[str, Any]] = []
    try:
        msg = await _llm_ainvoke(llm,[SystemMessage(content=sys_content), HumanMessage(content=user_content)])
        raw = (getattr(msg, "content", "") or str(msg)).strip()
        data = parse_json_from_llm(raw)
        if isinstance(data, dict) and isinstance(data.get("tasks"), list):
            seen_ids: set = set()
            for t in data["tasks"]:
                if not isinstance(t, dict):
                    continue
                tid = str(t.get("id") or "").strip()
                desc = str(t.get("description") or "").strip()
                cat = str(t.get("category") or "B1").strip().upper()
                if not cat.startswith("B") or cat not in ("B1", "B2", "B3", "B4", "B5", "B6"):
                    cat = "B1"
                if not tid:
                    tid = f"{cat.lower()}_{len(seen_ids)}"
                if tid in seen_ids:
                    tid = f"{tid}_{len(seen_ids)}"
                seen_ids.add(tid)
                try:
                    imp = float(t.get("importance", 0.5))
                    imp = max(0.0, min(1.0, imp))
                except (TypeError, ValueError):
                    imp = 0.5
                if desc and not _is_systemic_backlog_task(desc):
                    out.append({
                        "id": tid,
                        "description": desc,
                        "task_type": "backlog",
                        "importance": round(imp, 2),
                        "category": cat,
                    })
    except Exception:
        pass
    return out[:50]  # 上限 50 条


async def generate_sidewrite_and_backlog(
    llm: Any,
    basic_info: Dict[str, Any],
    big_five: Dict[str, Any],
    persona: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    创建 bot 时调用：先生成人物侧写，再根据侧写生成个性任务库。
    返回 (character_sidewrite, backlog_tasks)。
    """
    sidewrite = await generate_character_sidewrite(llm, basic_info, big_five, persona)
    backlog = await generate_backlog_tasks(llm, sidewrite, basic_info, big_five, persona)
    return sidewrite, backlog

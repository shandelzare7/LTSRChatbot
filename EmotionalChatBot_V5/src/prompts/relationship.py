from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from utils.yaml_loader import get_project_root


RUBRIC_PATH = Path(get_project_root()) / "config" / "relationship_signals.yaml"


def load_rubric_str() -> str:
    if not RUBRIC_PATH.exists():
        return "Error: Rubric file not found."
    with open(RUBRIC_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return yaml.dump(data, allow_unicode=True, sort_keys=False)


# 静态加载 YAML (只做一次)
STATIC_RUBRIC = load_rubric_str()


ANALYZER_SYSTEM_PROMPT = """
你是常识经验丰富的语言学专家。凭借你对人际语言互动的深刻理解和丰富的生活常识，根据下述信号标准，动态上下文，记忆，分析指令，校准标准，任务完成判定来分析用户最新输入对 6 维关系的影响。

### 1. 信号标准（静态知识库）
以下为 6 维关系的判断标准（dimensions：每维含 name、definition、anchors）。anchors 中 +3 为极强正向、-3 为极强负向，请根据用户输入与各锚点描述的匹配程度，为每个维度输出整数 delta（0 表示无变化）。
{rubric}

### 2. 动态上下文（当前情境）
* **当前分数**: {current_scores}（0-1 范围）
    - 注意: 分数 > 0.8 时较为稳固（难以继续提升）。
    - 注意: 分数 < 0.3 时较为脆弱（容易波动）。
* **当前阶段**: {current_stage}
* **我的情绪（PAD 为 [-1,1]，0 为中性；busyness 为 [0,1]）**: {mood_state}
* **用户画像**: {user_profile}（据此校准判断强度）。

### 2.5 记忆（摘要 + 检索）
{memory_block}

### 3. 分析指令
分析下方的「用户输入」。
1. **语境检查**: 用户的输入是否适合当前关系阶段？
2. **信号匹配**: 将输入匹配到各维度 anchors 中的描述（+3 至 -3 对应不同强度）。
3. **增量赋值**: 为每个维度分配整数变化（-3 到 +3），与 anchors 强度一致。
    - 0: 无变化 / 无相关信息。
    - +1/+2/+3: 正向影响（轻微→强烈），-1/-2/-3: 负向影响（轻微→强烈）。

### 4. 校准规则
- **边际递减**: 若某维度已经很高（>0.8），普通正面信号只给 +1，不给 +2。
- **背叛惩罚**: 若信任/亲密度很高（>0.8），负面信号应加重惩罚（-2 或 -3）。

### 5. 任务完成判定（由你根据语义判断，不设固定关键词）
* **本轮任务列表（tasks_for_lats）**: {tasks_for_lats_str}
* **本轮 bot 的回复**: {bot_reply_this_turn}

根据上述任务列表与 bot 的回复，判断哪些任务在本轮被**完成**、哪些被**尝试**。
- **完成**：bot 在回复中实际执行了该任务（例如确实向用户发问或收集了信息）。
- **尝试**：任务在本轮范围内被涉及但未完成，或仅部分涉及。
输出为 completed_task_ids（已完成的任务 id 数组）与 attempted_task_ids（被尝试的任务 id 数组）。只填 tasks_for_lats 中存在的 id，不确定则留空数组。

（输出格式由系统约束。）"""


def build_analyzer_prompt(state: Dict[str, Any]) -> str:
    """将 State 和 Static YAML 组装成最终 Prompt（含 summary + retrieved 记忆，不含 chat_buffer；chat_buffer 由调用方放正文）"""
    # 优先使用 user_inferred_profile；没有则兼容 loader 的 user_profile
    user_profile = state.get("user_inferred_profile") or state.get("user_profile") or {}
    summary = state.get("conversation_summary") or ""
    retrieved = state.get("retrieved_memories") or []
    memory_parts = []
    if summary:
        memory_parts.append("近期对话摘要：\n" + summary)
    if retrieved:
        memory_parts.append("相关记忆片段：\n" + "\n".join(retrieved))
    memory_block = "\n\n".join(memory_parts) if memory_parts else "（无）"

    tasks_for_lats = state.get("tasks_for_lats") or []
    if isinstance(tasks_for_lats, list) and tasks_for_lats:
        parts = []
        for t in tasks_for_lats:
            if not isinstance(t, dict):
                continue
            tid = t.get("id") or ""
            desc = t.get("description") or ""
            if tid:
                parts.append(f"id={tid}" + (f", 描述={desc}" if desc else ""))
        tasks_for_lats_str = "; ".join(parts) if parts else "（无）"
    else:
        tasks_for_lats_str = "（无）"
    bot_reply_this_turn = (state.get("final_response") or state.get("draft_response") or "").strip() or "（无）"

    return ANALYZER_SYSTEM_PROMPT.format(
        rubric=STATIC_RUBRIC,
        current_scores=state.get("relationship_state") or {},
        current_stage=state.get("current_stage") or "experimenting",
        mood_state=state.get("mood_state") or {},
        user_profile=user_profile,
        memory_block=memory_block,
        tasks_for_lats_str=tasks_for_lats_str,
        bot_reply_this_turn=bot_reply_this_turn[:800] + "…" if len(bot_reply_this_turn) > 800 else bot_reply_this_turn,
    )


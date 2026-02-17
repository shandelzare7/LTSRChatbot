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
你是常识经验丰富的语言学专家。凭借你对人际语言互动的深刻理解和丰富的生活常识，请分析用户最新输入对 6 维关系的影响。

### 1. 信号标准（静态知识库）
以下信号分类标准为你的判断依据：
{rubric}

### 2. 动态上下文（当前情境）
* **当前分数**: {current_scores}（0-1 范围）
    - 注意: 分数 > 0.8 时较为稳固（难以继续提升）。
    - 注意: 分数 < 0.3 时较为脆弱（容易波动）。
* **当前阶段**: {current_stage}
* **我的情绪**: {mood_state}
* **用户画像**: {user_profile}（据此校准判断强度）。

### 2.5 记忆（摘要 + 检索）
{memory_block}

### 3. 分析指令
分析下方的「用户输入」。
1. **语境检查**: 用户的输入是否适合当前关系阶段？
2. **信号匹配**: 将输入匹配到信号标准中的类目。
3. **增量赋值**: 为每个维度分配分数变化（-3 到 +3）。
    - 0: 无变化 / 中性。
    - 1/-1: 轻微影响 / 隐含。
    - 2/-2: 中等影响 / 显式。
    - 3/-3: 强烈影响 / 情感突破或崩塌。

### 4. 校准规则
- **边际递减**: 若某维度已经很高（>0.8），普通正面信号只给 +1，不给 +2。
- **背叛惩罚**: 若信任/亲密度很高（>0.8），负面信号应加重惩罚（-2 或 -3）。

### 5. 用户画像提取（从对话中抽取用户信息）
在关系分析的同时，从对话中提取你能发现的用户信息。

* **当前 user_basic_info**: {user_basic_info}
* **缺失字段**（优先级顺序）: {missing_basic_fields}

规则：
- 如果对话中提到或可以高置信度推断出缺失的 basic_info 字段，填入 `basic_info_updates`。
  - `name`: 字符串; `age`: 整数; `gender`: 字符串; `occupation`: 字符串; `location`: 字符串。
- 只填你有把握的字段，不要猜测。
- 同时捕捉其他用户画像观察（兴趣、性格特征、习惯、偏好、生活事件等），以键值对形式填入 `new_inferred_entries`。键 = 特征/属性名（简洁），值 = 描述。
- 如果没有新信息可提取，两者都留空对象 `{{}}`。

### 6. 输出格式（严格 JSON，不要其他文字）
返回如下 JSON 结构：
{{
  "thought_process": "...",
  "detected_signals": ["...","..."],
  "deltas": {{
    "closeness": 0,
    "trust": 0,
    "liking": 0,
    "respect": 0,
    "warmth": 0,
    "power": 0
  }},
  "basic_info_updates": {{}},
  "new_inferred_entries": {{}}
}}
""".strip()


_BASIC_INFO_FIELD_NAMES = ("name", "age", "gender", "occupation", "location")


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

    user_basic_info = state.get("user_basic_info") or {}
    missing = [
        f for f in _BASIC_INFO_FIELD_NAMES
        if not user_basic_info.get(f) or (isinstance(user_basic_info.get(f), str) and not user_basic_info[f].strip())
    ]
    missing_str = ", ".join(missing) if missing else "(all complete)"

    return ANALYZER_SYSTEM_PROMPT.format(
        rubric=STATIC_RUBRIC,
        current_scores=state.get("relationship_state") or {},
        current_stage=state.get("current_stage") or "experimenting",
        mood_state=state.get("mood_state") or {},
        user_profile=user_profile,
        memory_block=memory_block,
        user_basic_info=user_basic_info,
        missing_basic_fields=missing_str,
    )


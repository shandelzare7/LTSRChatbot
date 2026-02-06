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
You are the psychological engine of an AI companion.
Your task is to analyze the User's latest input and determine how it impacts the relationship across 6 dimensions.

### 1. STATIC KNOWLEDGE (The Signal Rubric)
Use these signals as your ground truth for classification:
{rubric}

### 2. DYNAMIC CONTEXT (Current Situation)
* **Current Scores**: {current_scores}
    - Note: Scores > 80 are resilient (hard to increase).
    - Note: Scores < 30 are fragile (easy to fluctuate).
* **Current Stage**: {current_stage}
* **My Mood**: {mood_state}
* **User Profile**: {user_profile} (Calibrate intensity based on this).

### 3. ANALYSIS INSTRUCTIONS
Analyze the `User Input` below.
1. **Context Check**: Does the user's input fit the current stage?
2. **Signal Matching**: Match input to the Rubric signals.
3. **Delta Assignment**: Assign a score change (-3 to +3) for each dimension.
    - 0: No change / Neutral.
    - 1/-1: Slight impact / Implied.
    - 2/-2: Moderate impact / Explicit.
    - 3/-3: Major impact / Emotional breakthrough or breakdown.

### 4. CALIBRATION RULES
- **Diminishing Returns**: If a dimension is ALREADY High (>80), standard compliments only give +1, not +2.
- **Betrayal**: If Trust/Closeness is High (>80), negative signals should be penalized heavily (-2 or -3).

### 4.5 SPT & TOPIC TAGGING (Fuel Inputs)
- You MUST tag the message with a coarse `topic_category` for breadth tracking.
- You MUST estimate the user's self-disclosure depth (SPT) as `self_disclosure_depth_level`:
  1=Public small talk, 2=Preferences/opinions, 3=Private personal info, 4=Core/trauma/identity secrets
- You MUST decide if the message is `is_intellectually_deep` (true/false):
  - true when it contains reflective reasoning, abstract thinking, moral/intellectual exploration, or deep analysis.

### 5. OUTPUT FORMAT (STRICT JSON ONLY)
Return JSON with the following shape:
{{
  "thought_process": "...",
  "detected_signals": ["...","..."],
  "topic_category": "general|work|family|love|health|hobbies|finance|study|life_goals|other",
  "self_disclosure_depth_level": 1,
  "is_intellectually_deep": false,
  "deltas": {{
    "closeness": 0,
    "trust": 0,
    "liking": 0,
    "respect": 0,
    "warmth": 0,
    "power": 0
  }}
}}
""".strip()


def build_analyzer_prompt(state: Dict[str, Any]) -> str:
    """将 State 和 Static YAML 组装成最终 Prompt"""
    # 优先使用 user_inferred_profile；没有则兼容 loader 的 user_profile
    user_profile = state.get("user_inferred_profile") or state.get("user_profile") or {}
    return ANALYZER_SYSTEM_PROMPT.format(
        rubric=STATIC_RUBRIC,
        current_scores=state.get("relationship_state") or {},
        current_stage=state.get("current_stage") or "experimenting",
        mood_state=state.get("mood_state") or {},
        user_profile=user_profile,
    )


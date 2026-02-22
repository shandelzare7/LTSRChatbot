# 请求：简化 Reply Planner 的 LLM 提示词方案

## 一、当前 prompt 结构（reply_planner 的 system）

1. **TIME_CONTEXT 块**（`build_time_context_block`）
   - now_local, now_day_part, weekday
   - since_last_user_msg_sec, since_last_assistant_msg_sec
   - session_mode (continuation / long_gap / new_session)

2. **header**  
   - "你是 {bot_name}，正在和 {user_name} 对话。"

3. **background**（大段“可用背景信息”）
   - bot_basic_info, bot_persona, user_basic_info, user_profile（本轮选中）
   - 【memory】conversation_summary + retrieved_memories
   - 【state_snapshot】bot_name, stage, mood_state, relationship_state
   - 【style_profile】12D 或 V5 自然语言 prompt 字符串
   - 【requirements】脱敏后的整份（不含 tasks_for_lats 内部字段）

4. **Hard Targets**
   - max_messages, plan_goals.must_cover_points, plan_goals.avoid_points
   - style_targets(12D), stage_targets
   - task_budget_max, word_budget

5. **core_rules**
   - 写作要求（自然、不称 AI、TIME_* 不复述）
   - TIME_SLICE_BEHAVIOR_RULES 常量
   - 内容要求：覆盖 must_cover、避开 avoid、遵守 style/stage、字数≤word_budget、task_budget_max
   - 本轮必须完成（required_tasks 转自然语言）
   - 特殊问法（“你是谁”等）

6. **output**  
   - 单条：只输出 JSON {"reply": "..."}

## 二、requirements 内容（compile_requirements 产出）

- must_have, forbidden, safety_notes, first_message_rule
- max_messages, min_first_len, max_message_len
- stage_pacing_notes, plan_goals, latest_user_text, user_asks_advice
- style_targets（12 维数值）, stage_targets（stage, pacing_notes, allowed_acts, forbidden_acts, violation_sensitivity）
- tasks_for_lats, task_budget_max, word_budget

## 三、相关文件路径

- `EmotionalChatBot_V5/app/lats/reply_planner.py`：_build_system_prompt_b、plan_reply_via_llm
- `EmotionalChatBot_V5/app/lats/requirements.py`：compile_requirements、build_requirements
- `EmotionalChatBot_V5/utils/time_context.py`：build_time_context_block、TIME_SLICE_BEHAVIOR_RULES
- `EmotionalChatBot_V5/app/lats/prompt_utils.py`：build_style_profile、build_system_memory_block、summarize_state_for_planner

## 四、问题

- background 与 Hard Targets 有重合（style、requirements、stage 等既在 background 里全文出现，又在 Hard Targets 里摘成清单）。
- requirements 和 prompt 块较多，希望在不损失必要约束的前提下简化 LLM 看到的提示词，便于模型遵守、也便于维护。

请基于以上结构和相关文件，提出一个**简化 LLM 提示词**的具体方案（例如：合并/删减哪些块、如何避免重复、保留哪些硬约束、建议的 prompt 结构），并把方案写清楚便于落地实现。最后请把方案单独总结成“简化方案”一节。

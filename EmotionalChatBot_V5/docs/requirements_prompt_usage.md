# requirements 提供的提示词长什么样、用在哪些 LLM、与别处是否重合

## 一、requirements 的“完整提示词”长什么样

**requirements** 由 `app/lats/requirements.py` 的 `compile_requirements(state)` 产出，是一个 **dict**。真正作为“一段提示词文本”喂给 LLM 的只有 **reply_planner** 里那一处；其它地方只是**按字段读 requirements**，再自己拼 prompt。

### 1. 在 reply_planner 里：整份 requirements 如何进 prompt

在 **`app/lats/reply_planner.py`** 的 `_build_system_prompt_b` 里，requirements 会**以三种形式**进入 system 提示词：

| 形式 | 内容 | 在 prompt 里的位置 |
|-----|------|---------------------|
| **① 全文 dump** | `requirements_for_prompt = _sanitize_requirements_for_prompt(requirements)`：整份 dict 保留，仅把 `tasks_for_lats` 替换成一句占位「（已在“本轮必须完成”中明确列出；不在此重复内部任务列表）」后，用 `safe_text(requirements_for_prompt)` 转成字符串 | **background** 里的一整段：`【requirements（原文保留；内部任务列表已脱敏）】` + 上述字符串 |
| **② 硬指标清单** | 从 requirements 里取出：`max_messages`、`plan_goals.must_cover_points`、`plan_goals.avoid_points`、`style_targets`、`stage_targets`、`task_budget_max`、`word_budget` | **Hard Targets** 块，逐条列出 |
| **③ 本轮必做** | `required_tasks = _extract_required_tasks(requirements)`：从 `requirements["tasks_for_lats"]` 里筛出 is_urgent 的任务，转成自然语言列表 | **core_rules** 里的「本轮必须完成：」+ 列表 |

也就是说，**“完整 requirements 提供的提示词”**在 reply_planner 里 = 上述 **① 的整段字符串**（即一整个脱敏后的 requirements dict 的文本形式）。下面给出这个 dict 的**字段清单**（即你看到的那段“提示词”里会出现的键和大致含义）。

### 2. requirements dict 的完整字段（compile_requirements 产出）

进入 **① 全文 dump** 的，就是下面这一整份（脱敏后）被 `str()`/`safe_text()` 打出来的样子：

| 键 | 含义 / 来源 |
|----|-------------|
| **must_have** | 必做项列表（当前多为 []） |
| **forbidden** | 违禁词/违禁表述（含沉浸破坏词：设定、人设、虚拟、角色、剧本、配置、模型、系统、作为一个 等） |
| **safety_notes** | 安全底线（不得自称 AI、不得违法/自残/暴力、尊重边界等） |
| **first_message_rule** | 首条必须先回应用户/先给态度或结论 |
| **max_messages** | 最多几条消息（3~5，会按 word_budget 微调） |
| **min_first_len** | 首条最少字数 |
| **max_message_len** | 单条最大字数 |
| **stage_pacing_notes** | 按 current_stage 写的节奏说明 |
| **must_have_policy** | "soft" |
| **must_have_min_coverage** | 0.75 |
| **allow_short_reply** | False |
| **allow_empty_reply** | False |
| **plan_goals** | `{ must_cover_points: [], avoid_points: [] }` |
| **latest_user_text** | 本轮用户输入原文 |
| **user_asks_advice** | 是否在“要建议” |
| **style_targets** | 从 state["style"] 抽的 12 维（verbal_length, tone_temperature 等） |
| **stage_targets** | stage、pacing_notes、allowed_acts、forbidden_acts、violation_sensitivity 等 |
| **tasks_for_lats** | 脱敏后这里是占位字符串，不是原始列表 |
| **task_budget_max** | 本轮最多完成几个任务 |
| **word_budget** | 字数上限 |

所以你在「完整 requirements 提供的提示词」里看到的，就是**上述这一大坨键值对**被转成一段可读文本（类似 Python 的 dict 打印），出现在 **【requirements（原文保留；内部任务列表已脱敏）】** 下面。

---

## 二、requirements 被用到哪些文件的 LLM 里

| 文件 | 用法 | 是否把 requirements 整份当“提示词”发给 LLM |
|------|------|---------------------------------------------|
| **app/lats/reply_planner.py** | ① background 里 `safe_text(requirements_for_prompt)` 整段；② Hard Targets 里摘出的 max_messages / plan_goals / style_targets / stage_targets / task_budget_max / word_budget；③ core_rules 里的 required_tasks（来自 requirements.tasks_for_lats） | **是**：唯一把「整份 requirements（脱敏）当一段提示词」塞进 LLM 的地方 |
| **app/lats/evaluator.py** | **hard_gate**：只读 requirements 的 allow_empty_reply、allow_short_reply、max_messages、max_message_len、min_first_len、forbidden，做规则判断，**不调 LLM**。**soft_score_via_llm / soft_score_batch_via_llm**：参数里有 requirements，但拼 system_prompt 时**没有**把 requirements 写进提示词，只用 bot_basic_info、user_basic_info、对话与候选。**judge_dimension_relationship/stage/mood_busy_batch**：从 requirements 里取 **stage_targets["stage"]** 拼进 prompt（stage_id）。**judge_dimension_task_completion_batch**：从 requirements 里取 **tasks_for_lats**，格式化成「本轮任务列表」字符串拼进 system_prompt。 | **否**：evaluator 里没有任何一处把整份 requirements 当提示词；只用了 **stage_targets.stage** 和 **tasks_for_lats** 这两部分来拼 LLM 的 prompt。 |
| **app/lats/search.py** | 从 state 取 requirements，传给 evaluator 的各 judge（同上）。 | **否**：不直接拼 prompt，只是传参。 |
| **app/nodes/lats_search.py** | 调用 `compile_requirements(state)` 得到 requirements 写入 state；**hard_gate(proc, requirements)** 只做规则检查，不调 LLM。 | **否**：不把 requirements 当 LLM 提示词。 |
| **app/nodes/final_validator.py** | 读 requirements 的 max_messages、min_first_len、max_message_len 做长度/条数修补，不调 LLM。 | **否**。 |

结论：**真正“把 requirements 当作一整段提示词”用到的 LLM，只有 reply_planner 的 system（background 那一块）**；evaluator/search 里只是用 requirements 的**个别字段**（stage、tasks_for_lats）去拼各自的 prompt，没有整份塞进去。

---

## 三、和其它地方有没有重合

有，主要集中在 **reply_planner 的 system** 内部，以及和 **state 里其它字段** 的重复。

### 1. reply_planner 内部：background vs Hard Targets vs core_rules

- **style**  
  - **background** 里已有 **【style_profile（12D）】**：来自 `build_style_profile(state)`，现在是 **V5 自然语言** 那一大段。  
  - **Hard Targets** 里又列了 **style_targets(12D)**：来自 `requirements["style_targets"]`，即从 **state["style"]** 抽的 12 维数字。  
  → 同一轮里，**风格既用自然语言描述了一遍，又用 12 维数字列了一遍**，两处都来自同一份 state["style"]，只是形态不同，属于**重复约束**。

- **stage**  
  - **state_snapshot**（在 background 里）里包含 stage、mood、relationship。  
  - **requirements** 全文（在 background 里）里又包含 **stage_targets**（含 stage、pacing_notes、allowed_acts、forbidden_acts 等）。  
  - **Hard Targets** 再列一遍 **stage_targets**。  
  → stage 相关至少出现 **三次**：state_snapshot、requirements 全文、Hard Targets。

- **word_budget / task_budget_max / max_messages**  
  - 在 **requirements 全文**（background）里出现。  
  - 在 **Hard Targets** 里又单独列出。  
  - 在 **core_rules** 的叙述里再次出现（“字数不超过 word_budget…”）。  
  → 三处重复。

- **本轮必做任务**  
  - **requirements.tasks_for_lats** 在脱敏后不再在 requirements 全文里展示，而是用一句占位代替。  
  - 同一份 **tasks_for_lats** 通过 **required_tasks** 在 **core_rules** 里以「本轮必须完成：」自然语言列表形式出现。  
  → 任务列表只在这里出现一次（core_rules），但和 requirements 的“任务列表”概念同源，逻辑上算**同源不同形态**，不算重复 dump，但和 background 里的 requirements 占位语会一起出现。

### 2. 与其它模块的“重合”

- **style_profile**  
  - reply_planner 的 **style_profile** = `build_style_profile(state)`，优先用 `state["style_profile"]`，否则用 **state["llm_instructions"]**（即 style 节点产出的 V5 自然语言）。  
  - **requirements["style_targets"]** = 从 **state["style"]** 抽的 12 维。  
  → **state["style"]** 和 **state["llm_instructions"]** 同源（style 节点同时写这两个），所以在 reply_planner 里等于**同一套风格信息**：一次以自然语言（style_profile），一次以 12 维数字（requirements.style_targets），**两处重合**。

- **stage / relationship / mood**  
  - **state_snapshot** 里已有 stage、mood_state、relationship_state 的摘要。  
  - **requirements** 里又有 **stage_targets**、以及（通过 plan_goals 等）间接依赖同一批 state。  
  → 和上面 stage 重复一致，**evaluator 的 judge** 只用 requirements 的 **stage_targets["stage"]** 和 **tasks_for_lats**，不再读 requirements 全文，所以**和 reply_planner 的“整段 requirements 提示词”没有重复**，只是数据同源（都来自 state/requirements）。

总结：**“完整 requirements 提供的提示词”** = reply_planner 的 system 里 **【requirements（原文保留；内部任务列表已脱敏）】** 下面的那一整段；**只在这一个 LLM（reply 生成）里被当成整段提示词用**；和别处重合主要发生在 **reply_planner 自己的 background / Hard Targets / core_rules 之间**（style、stage、word_budget 等重复），以及 style 与 requirements.style_targets 同源导致的**风格双重表述**。

---

## 四、为什么显得乱 + 简化建议

**乱的原因**：同一轮里既有「整份 requirements dict 全文 dump」，又有从里摘出来的 Hard Targets，还有 core_rules 里的复述；风格既用自然语言又用 12 维数字；stage/字数/条数在多处重复。结果是 token 浪费、重点不清晰、模型可能不知道以哪处为准。

**简化方向（可任选或组合）：**

1. **不再在 background 里贴 requirements 全文**  
   只保留 Hard Targets + core_rules（含「本轮必须完成」）。生成约束以 Hard Targets 为唯一清单，background 里不再出现 `【requirements（原文保留…）】` 整段。若仍需要少量「原文」给模型参考，可只贴 1～2 条关键字段（如 `plan_goals`、`forbidden` 的简短列表），而不是整 dict。

2. **风格只保留一种形态**  
   要么只用 **style_profile（自然语言）**，Hard Targets 里不再列 style_targets(12D)；要么只用 12D，不再在 background 里贴 style_profile。推荐保留自然语言、删 12D，可读性更好。

3. **stage 只出现一次**  
   state_snapshot 里已有 stage 摘要时，Hard Targets 里的 stage_targets 可改为一句引用（如「遵守当前 stage 与节奏，见 state_snapshot」），不再把整份 stage_targets 再列一遍。

4. **数字类约束只在一处写清**  
   max_messages、word_budget、task_budget_max、min_first_len、max_message_len 等，只在 Hard Targets 里列一次，core_rules 里用「见 Hard Targets」或简短一句带过，避免三处重复。

按上面收紧后，reply_planner 的 system 会变成：**header + background（无 requirements 全文，风格/阶段不重复）+ Hard Targets（唯一约束清单）+ core_rules（写作要求 + 本轮必须完成）**，提示词会更干净、好维护。
